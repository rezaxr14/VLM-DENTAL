#!/usr/bin/env python3
"""Read-only local serving capability inventory: never loads weights or launches a server."""
import argparse,importlib.metadata,json,platform,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--vllm-executable',default='vllm');p.add_argument('--output',required=True);p.add_argument('--gpu-indices',default='0');a=p.parse_args()
    from dental_agent.harness.evaluation import atomic_json
    from dental_agent.harness.specdec import gpu_snapshot
    packages={}
    for name in ['vllm','sglang','speculators','torch','transformers','peft']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    report=dict(python=sys.version,platform=platform.platform(),packages=packages,gpus=gpu_snapshot(a.gpu_indices),
                available_vllm=shutil.which(a.vllm_executable),supported_cli_flags={},model_runtime_verified=False)
    if report['available_vllm']:
        for label,cmd in [('version',[a.vllm_executable,'--version']),('help',[a.vllm_executable,'serve','--help'])]:
            done=subprocess.run(cmd,capture_output=True,text=True,timeout=60)
            report[label]=dict(returncode=done.returncode,stdout=done.stdout,stderr=done.stderr)
        text=report['help']['stdout']+report['help']['stderr']
        report['supported_cli_flags']={flag:flag in text for flag in ['--speculative-config','--enable-per-request-metrics','--per-request-spec-decode-metrics']}
    # Analytical lower bound only, not a 4090 fit guarantee.
    report['reference_memory_lower_bound']=dict(checkpoint_bytes=19306216416,checkpoint_gib=19306216416/2**30,
        full_attention_kv_gib_at_16384=8*2*4*256*16384*2/2**30,
        excludes=['GDN state','vision/image activations','prefill workspace','CUDA graphs','MTP activations','allocator overhead'],
        reference='https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/model.safetensors.index.json')
    atomic_json(a.output,report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
