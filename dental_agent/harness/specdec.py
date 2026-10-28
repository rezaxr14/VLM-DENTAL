"""Serial serving benchmark utilities. Missing observations remain None."""
from __future__ import annotations
import base64
import io
import hashlib
import json
import math
import re
import subprocess
import time
import urllib.request

CONDITIONS = ['baseline','mtp_1','mtp_2','mtp_3','ngram_2','ngram_3','ngram_5']

def speculative_config(name):
    if name=='baseline':return None
    if name not in CONDITIONS:raise ValueError('Unknown speculation condition')
    method,k=name.split('_');k=int(k)
    result=dict(method=method,num_speculative_tokens=k)
    if method=='ngram':result.update(prompt_lookup_min=1,prompt_lookup_max=k)
    return result

def server_command(executable,checkpoint,condition,port,max_model_len=16384,gpu_memory_utilization=.9,tensor_parallel=1,dtype="bfloat16"):
    cmd=[executable,'serve',str(checkpoint),'--served-model-name','dental-benchmark','--host','127.0.0.1','--port',str(port),
         '--dtype',dtype,'--tensor-parallel-size',str(tensor_parallel),'--max-model-len',str(max_model_len),
         '--max-num-seqs','1','--gpu-memory-utilization',str(gpu_memory_utilization),'--no-enable-prefix-caching',
         '--enable-per-request-metrics','--limit-mm-per-prompt',json.dumps({'image':128})]
    config=speculative_config(condition)
    if config is not None:cmd+=['--speculative-config',json.dumps(config),'--per-request-spec-decode-metrics','summary']
    return cmd

def http_json(url,payload=None,timeout=600):
    data=None if payload is None else json.dumps(payload).encode()
    req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:return json.load(response)

def prometheus(text):
    values={}
    for line in text.splitlines():
        if not line or line.startswith('#'):continue
        match=re.match(r'([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{.*\})?\s+([-+0-9.eE]+)(?:\s|$)',line)
        if match:
            value=float(match[2])
            if math.isfinite(value):values[match[1]]=values.get(match[1],0)+value
    return values

def counters(url):
    try:
        with urllib.request.urlopen(url+'/metrics',timeout=10) as f:return prometheus(f.read().decode())
    except Exception:return {}

def delta(before,after,*names):
    for name in names:
        if name in before and name in after and after[name]>=before[name]:return after[name]-before[name]
    return None

def encode_messages(messages):
    """Lossless PNG transport of the exact PIL view; no extra resizing or prompt text."""
    output=[]
    for msg in messages:
        content=msg['content']
        if isinstance(content,str):output.append(dict(role=msg['role'],content=content));continue
        converted=[]
        for part in content:
            if part['type']=='text':converted.append(dict(type='text',text=part['text']))
            elif part['type']=='image':
                image=part['image'];buffer=io.BytesIO();image.save(buffer,format='PNG')
                converted.append(dict(type='image_url',image_url=dict(url='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode())))
            else:raise ValueError('Unsupported message part')
        output.append(dict(role=msg['role'],content=converted))
    return output

class ServingReply:
    def __init__(self,url,seed=42,timeout=600,model_id="dental-benchmark"):
        self.url=url.rstrip('/');self.seed=seed;self.timeout=timeout;self.requests=[];self.model_id=model_id
    def __call__(self,model,processor,messages,**kwargs):
        payload=dict(model=self.model_id,messages=encode_messages(messages),temperature=0.0,
                     seed=self.seed,max_tokens=kwargs['max_new_tokens'],n=1,stream=False)
        before=counters(self.url);started=time.perf_counter()
        response=http_json(self.url+'/v1/chat/completions',payload,self.timeout)
        wall=time.perf_counter()-started;after=counters(self.url)
        text=response['choices'][0]['message'].get('content')
        if not isinstance(text,str):raise ValueError('Server did not return raw text content')
        metrics=response.get('metrics') or {};spec=metrics.get('speculative_decoding') or {};usage=response.get('usage') or {}
        prefill=delta(before,after,'vllm:request_prefill_time_seconds_sum')
        decode=delta(before,after,'vllm:request_decode_time_seconds_sum')
        if decode is None and metrics.get('generation_time_ms') is not None:decode=metrics['generation_time_ms']/1000
        drafted=spec.get('num_draft_tokens');accepted=spec.get('num_accepted_draft_tokens')
        if drafted is None:drafted=delta(before,after,'vllm:spec_decode_num_draft_tokens_total','vllm:spec_decode_num_draft_tokens')
        if accepted is None:accepted=delta(before,after,'vllm:spec_decode_num_accepted_tokens_total','vllm:spec_decode_num_accepted_tokens')
        if prefill == 0 and usage.get('prompt_tokens',0)>0:prefill=None  # Counter export may lag request completion.
        if decode == 0 and usage.get('completion_tokens',0)>1:decode=None
        self.requests.append(dict(input_sha256=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),wall_seconds=wall,prefill_seconds=prefill,decode_seconds=decode,
                                  ttft_ms=metrics.get('time_to_first_token_ms'),prompt_tokens=usage.get('prompt_tokens'),
                                  generated_tokens=usage.get('completion_tokens'),drafted_tokens=drafted,accepted_tokens=accepted,
                                  acceptance_rate=accepted/drafted if drafted and accepted is not None else None,
                                  raw_metrics=metrics,finish_reason=response['choices'][0].get('finish_reason')))
        # Token IDs are not exposed by the chat endpoint; never invent IDs by re-tokenizing the decoded text.
        return text,usage.get('prompt_tokens',0),[],None,None

def sum_complete(records,key):
    if not records or any(x.get(key) is None for x in records):return None
    return sum(x[key] for x in records)

def episode_measurements(requests,wall,tool_seconds):
    generated=sum_complete(requests,'generated_tokens');drafted=sum_complete(requests,'drafted_tokens');accepted=sum_complete(requests,'accepted_tokens')
    return dict(generated_tokens=generated,wall_seconds=wall,tool_seconds=tool_seconds,
                request_wall_seconds=sum_complete(requests,'wall_seconds'),prefill_seconds=sum_complete(requests,'prefill_seconds'),
                decode_seconds=sum_complete(requests,'decode_seconds'),accepted_tokens=accepted,drafted_tokens=drafted,
                acceptance_rate=accepted/drafted if drafted and accepted is not None else None,
                tokens_per_second=generated/wall if generated is not None and wall>0 else None)

def equivalence(baseline,candidate):
    a={x['image_id']:x for x in baseline};b={x['image_id']:x for x in candidate}
    if a.keys()!=b.keys() or len(a)!=len(baseline) or len(b)!=len(candidate):
        return dict(passed=False,reason='Missing, extra or duplicate episodes',divergences=[])
    diffs=[]
    for iid in sorted(a):
        left=a[iid];right=b[iid]
        same_final=left['final_answer']==right['final_answer']
        action=lambda rec:[[dict(tool_name=c.get('tool_name'),tool_args=c.get('tool_args'),tool_ok=c.get('tool_ok'),tool_error=c.get('tool_error')) for c in t.get('tool_calls',[])] for t in rec['turns']]
        same_calls=action(left)==action(right)
        same_raw=[t['raw_output'] for t in left['turns']]==[t['raw_output'] for t in right['turns']]
        valid=left['format_ok'] and right['format_ok']
        if not(same_final and same_calls and same_raw and valid):
            diffs.append(dict(image_id=iid,same_final=same_final,same_tool_calls=same_calls,same_raw_text=same_raw,both_valid=valid))
    return dict(passed=not diffs,divergences=diffs,token_ids_verified=False,
                comparison='Exact decoded text, tool actions/arguments/outcomes and final_answer; chat API exposes no token IDs')

def gpu_snapshot(indices='0'):
    try:
        result=subprocess.run(['nvidia-smi','--id='+indices,'--query-gpu=index,name,memory.total,memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
        return [dict(index=line.split(',')[0].strip(),name=line.split(',')[1].strip(),total_mib=float(line.split(',')[2]),used_mib=float(line.split(',')[3])) for line in result.stdout.splitlines() if line.strip()]
    except Exception:return None
