#!/usr/bin/env python3
"""Own one local vLLM server at a time; paired fixed-dev, greedy dynamic-tool episodes."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

@contextmanager
def server(cmd,log_path,port,gpu_indices,timeout):
    from dental_agent.harness.specdec import http_json
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Port already occupied; refusing to touch an existing server')
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu_indices
    log_path=Path(log_path);log_path.parent.mkdir(parents=True,exist_ok=True)
    with log_path.open('w') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=os.name!='nt')
        try:
            deadline=time.monotonic()+timeout
            while time.monotonic()<deadline:
                if proc.poll() is not None:raise RuntimeError(f'Server exited {proc.returncode}; inspect {log_path}')
                try:
                    with __import__('urllib.request',fromlist=['urlopen']).urlopen(f'http://127.0.0.1:{port}/health',timeout=2):break
                except Exception:time.sleep(1)
            else:raise TimeoutError(f'Server startup timeout; inspect {log_path}')
            yield f'http://127.0.0.1:{port}'
        finally:
            # Only our server process group. Never pkill or terminate an unrelated training job.
            if proc.poll() is None:
                if os.name!='nt':
                    import signal
                    os.killpg(proc.pid,signal.SIGTERM)
                else:proc.terminate()
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if os.name!='nt':os.killpg(proc.pid,signal.SIGKILL)
                    else:proc.kill()
                    proc.wait(timeout=10)
            elif os.name!='nt':
                import signal
                try: os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError: pass

class MemoryMonitor:
    def __init__(self,gpus):self.gpus=gpus;self.samples=[];self.stop=threading.Event()
    def run(self):
        from dental_agent.harness.specdec import gpu_snapshot
        while not self.stop.is_set():
            snapshot=gpu_snapshot(self.gpus)
            if snapshot:self.samples.append(snapshot)
            self.stop.wait(.2)
    def __enter__(self):self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start();return self
    def __exit__(self,*_):self.stop.set();self.thread.join(timeout=12)
    def peaks(self):
        if not self.samples:return None
        out={}
        for sample in self.samples:
            for gpu in sample:
                if gpu['index'] not in out or gpu['used_mib']>out[gpu['index']]['used_mib']:out[gpu['index']]=gpu
        return list(out.values())

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True,help='Local merged BF16 model; never an adapter-only folder')
    p.add_argument('--dev-split',default='data/splits/dentex_dev.json');p.add_argument('--data-dir',default='data')
    p.add_argument('--split',choices=['dev'],default='dev');p.add_argument('--limit',type=int)
    p.add_argument('--conditions',nargs='+',choices=['baseline','mtp_1','mtp_2','mtp_3','ngram_2','ngram_3','ngram_5'],default=['baseline','mtp_1','mtp_2','mtp_3','ngram_2','ngram_3','ngram_5'])
    p.add_argument('--output',default='paper/generated/spec_decode_bench.json')
    p.add_argument('--vllm-executable',default='vllm');p.add_argument('--port',type=int,default=8910)
    p.add_argument('--gpu-indices',default='0');p.add_argument('--tensor-parallel',type=int,default=1)
    p.add_argument('--dtype',choices=['float16','bfloat16'],default='bfloat16')
    p.add_argument('--max-model-len',type=int,default=16384);p.add_argument('--gpu-memory-utilization',type=float,default=.9)
    p.add_argument('--max-new-tokens',type=int,default=2048);p.add_argument('--max-tool-calls',type=int,default=50);p.add_argument('--max-turns',type=int)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--startup-timeout',type=int,default=600)
    p.add_argument('--plan-only',action='store_true',help='Write exact server commands without loading data/model')
    p.add_argument('--warmup-episodes',type=int,default=1,help='Separate full dev episodes excluded from timing; 0 disables')
    return p.parse_args(argv)

def main(argv=None):
    args=parse_args(argv)
    from dental_agent.harness.specdec import server_command,ServingReply,episode_measurements,equivalence,gpu_snapshot
    from dental_agent.harness.evaluation import atomic_json,load_dev,evaluate_episode,pooled,digest
    output=Path(args.output)
    if output.exists():raise FileExistsError('Use a fresh output path; never overwrite benchmark evidence')
    if not args.conditions or args.conditions[0]!='baseline' or len(set(args.conditions))!=len(args.conditions):
        raise ValueError('Conditions must start with baseline and be unique')
    if args.warmup_episodes<0:raise ValueError('warmup-episodes must be nonnegative')
    if args.tensor_parallel!=len(args.gpu_indices.split(',')):raise ValueError('tensor parallel must equal selected GPU count')
    commands={c:server_command(args.vllm_executable,args.checkpoint,c,args.port,args.max_model_len,args.gpu_memory_utilization,args.tensor_parallel,args.dtype) for c in args.conditions}
    packages={}
    for name in ['vllm','sglang','torch','transformers','speculators']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    result=dict(schema_version=1,status='planned' if args.plan_only else 'running',settings=vars(args),software=packages,python=sys.version,platform=platform.platform(),
                hardware=gpu_snapshot(args.gpu_indices),commands=commands,conditions={},gates={},measurements_pending=True)
    if args.plan_only:atomic_json(output,result);print(json.dumps(commands,indent=2));return
    if not shutil.which(args.vllm_executable):raise FileNotFoundError('Install/pin vLLM in a separate serving environment')
    ck=Path(args.checkpoint).resolve()
    if not ck.is_dir() or (ck/'adapter_config.json').exists():raise ValueError('Use a local merged checkpoint')
    from dental_agent.harness.checkpoints import tensor_headers
    headers=tensor_headers(ck)
    if any(c.startswith('mtp') for c in args.conditions) and not any(k.startswith('mtp.') for k in headers):raise ValueError('MTP head is absent; run merge/audit before speculation')
    immutable={str(p.relative_to(ck)):[p.stat().st_size,p.stat().st_mtime_ns] for p in ck.glob('*') if p.is_file()}
    ids,images,annotations,categories=load_dev(args.data_dir,args.dev_split,args.limit,args.seed)
    result.update(image_ids=ids,dev_split_sha256=digest(args.dev_split),checkpoint_files=immutable)
    atomic_json(output,result)
    for condition,cmd in commands.items():
        entry=dict(status='running',episodes=[]);result['conditions'][condition]=entry
        try:
            with server(cmd,output.parent/(output.stem+'_'+condition+'.log'),args.port,args.gpu_indices,args.startup_timeout) as url:
                from dental_agent.harness.specdec import http_json
                try:entry['server_version']=http_json(url+'/version')
                except Exception:entry['server_version']=None
                # Same selected warmup episodes and geometry for every condition; excluded from metrics and gates.
                for j in range(args.warmup_episodes):
                    evaluate_episode(ids[j%len(ids)],images,annotations,categories,reply_fn=ServingReply(url,args.seed),seed=args.seed,
                                     max_new_tokens=args.max_new_tokens,max_tool_calls=args.max_tool_calls,max_turns=args.max_turns)
                for iid in ids:
                    reply=ServingReply(url,args.seed);tools=[]
                    with MemoryMonitor(args.gpu_indices) as memory:
                        start=time.perf_counter()
                        record,_=evaluate_episode(iid,images,annotations,categories,reply_fn=reply,seed=args.seed,
                            max_new_tokens=args.max_new_tokens,max_tool_calls=args.max_tool_calls,max_turns=args.max_turns,
                            tool_observer=lambda name,seconds:tools.append(dict(name=name,seconds=seconds)))
                        wall=time.perf_counter()-start
                    record['measurements']=episode_measurements(reply.requests,wall,sum(t['seconds'] for t in tools))
                    record['requests']=reply.requests;record['tool_timings']=tools;record['sampled_gpu_peak']=memory.peaks()
                    entry['episodes'].append(record);atomic_json(output,result)
            entry['summary']=pooled(entry['episodes']);entry['status']='complete'
            if condition!='baseline':
                gate=equivalence(result['conditions']['baseline']['episodes'],entry['episodes'])
                base_f1=result['conditions']['baseline']['summary']['pooled_exact_f1'];f1=entry['summary']['pooled_exact_f1']
                gate['exact_f1_unchanged']=base_f1==f1 and base_f1 is not None
                gate['passed']=gate['passed'] and gate['exact_f1_unchanged'];result['gates'][condition]=gate
        except Exception as error:
            entry.update(status='failed',error=f'{type(error).__name__}: {error}')
            if condition=='baseline':
                result['status']='failed';atomic_json(output,result);raise
        current={str(p.relative_to(ck)):[p.stat().st_size,p.stat().st_mtime_ns] for p in ck.glob('*') if p.is_file()}
        if current!=immutable:raise RuntimeError('Checkpoint changed during benchmark')
        atomic_json(output,result)
    complete=all(e['status']=='complete' for e in result['conditions'].values())
    passed=complete and all(g['passed'] for g in result['gates'].values())
    result.update(status='complete' if passed else 'failed_or_divergent',measurements_pending=not complete)
    atomic_json(output,result)
    print('Result:',output,'status:',result['status'])
    if not passed:raise SystemExit(2)
if __name__=='__main__':main()
