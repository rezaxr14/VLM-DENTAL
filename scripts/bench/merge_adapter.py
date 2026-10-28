#!/usr/bin/env python3
"""Explicit CPU/BF16 merge into a NEW directory, preserving original MTP bytes."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ['base','adapter','output']:p.add_argument('--'+flag,required=True)
    a=p.parse_args();base=Path(a.base).resolve();out=Path(a.output).resolve()
    if out.exists():raise FileExistsError('Output must be a new directory')
    if not base.is_dir() or not Path(a.adapter).is_dir():raise ValueError('Use local base and adapter paths')
    from dental_agent.harness.checkpoints import tensor_headers,preserve_mtp
    if not any(k.startswith('mtp.') for k in tensor_headers(base)):raise ValueError('Base lacks MTP provenance')
    import torch
    from transformers import Qwen3_5ForConditionalGeneration,AutoProcessor
    from peft import PeftModel
    model=Qwen3_5ForConditionalGeneration.from_pretrained(str(base),torch_dtype=torch.bfloat16,device_map='cpu',local_files_only=True)
    model=PeftModel.from_pretrained(model,a.adapter,is_trainable=False).merge_and_unload(safe_merge=True)
    out.mkdir(parents=True)
    model.save_pretrained(out,safe_serialization=True)
    AutoProcessor.from_pretrained(base,local_files_only=True).save_pretrained(out)
    del model
    report=preserve_mtp(base,out)
    (out/'mtp_merge_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    # Config retains native MTP metadata even if the HF loader discarded its unknown tensors.
    original=json.loads((base/'config.json').read_text());config=json.loads((out/'config.json').read_text())
    for key,value in original.get('text_config',{}).items():
        if key.startswith('mtp_'):config.setdefault('text_config',{})[key]=value
    (out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    if not report['ok']:raise SystemExit('MTP audit failed')
    print('Merged checkpoint:',out)
if __name__=='__main__':main()
