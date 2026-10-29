#!/usr/bin/env python3
"""Stitch an already-downloaded trained head onto its exact target in a new directory."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,help='Local final merged harness-GRPO target')
    parser.add_argument('--draft-dir',required=True,help='Local downloaded mtp/ artifact directory')
    parser.add_argument('--output',required=True)
    parser.add_argument('--harness-mode',choices=['rules_skills_index','rules_skills_full'],required=True)
    parser.add_argument('--speculators-executable',default='speculators')
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    from dental_agent.harness.configuration import require_harness_grpo_target,assert_harness_identity
    from dental_agent.harness.checkpoints import target_tensor_fingerprint,mtp_parameter_inventory
    target=require_harness_grpo_target(args.checkpoint,args.harness_mode)
    folder=Path(args.draft_dir).resolve();out=Path(args.output).resolve()
    if out.exists(): raise FileExistsError('Stitch into a new output directory')
    data=json.loads((folder/'mtp_draft_manifest.json').read_text())
    assert_harness_identity(data.get('harness'),args.harness_mode)
    if data.get('target_manifest')!=target or data.get('draft_variant')!='finetuned':
        raise ValueError('Draft artifact belongs to a different adapter or is not fine-tuned')
    if data.get('target_tensor_sha256')!=target_tensor_fingerprint(args.checkpoint):
        raise ValueError('Draft belongs to different target weights')
    command=[args.speculators_executable,'stitch-mtp',str(folder),args.checkpoint,'--output-path',str(out)]
    print(json.dumps(dict(command=command,harness=data['harness']),indent=2))
    if args.execute:
        subprocess.run(command,check=True)
        if target_tensor_fingerprint(out)!=data['target_tensor_sha256']:
            raise ValueError('Stitch changed target tensors; do not benchmark this output')
        if mtp_parameter_inventory(out)!=data['mtp_inventory']:
            raise ValueError('Restored MTP structure differs from the trained draft')
        shutil.copy2(Path(args.checkpoint)/'dental_target.json',out/'dental_target.json')
        shutil.copy2(folder/'mtp_draft_manifest.json',out/'mtp_draft_manifest.json')
        print('Verified fine-tuned draft:',out)


if __name__=='__main__': main()
