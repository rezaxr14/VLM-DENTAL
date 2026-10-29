#!/usr/bin/env python3
"""Publish only a trained MTP head and its provenance to an existing model repository."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))


def draft_files(workflow):
    from dental_agent.harness.checkpoints import tensor_headers
    root = Path(workflow).resolve()
    manifest = root/'stitched'/'mtp_draft_manifest.json'
    if not manifest.is_file(): raise ValueError('Stitch and audit the fine-tuned draft before publishing')
    data = json.loads(manifest.read_text())
    if data.get('draft_variant') != 'finetuned' or not data.get('changed_mtp_tensors'):
        raise ValueError('No verified fine-tuned MTP head')
    head = root/'checkpoints'/'checkpoint_best'
    tensors = tensor_headers(head)
    count = sum(math.prod(v['shape']) for v in tensors.values())
    expected = data['mtp_inventory']['parameters']
    if count <= 0 or count > expected*1.1:
        raise ValueError('Draft checkpoint contains unexpected extra weights; refusing to upload a target model')
    files = [(p.relative_to(head).as_posix(),p) for p in sorted(head.rglob('*'))
             if p.is_file() and p.suffix in {'.json','.safetensors'}]
    if any(not p.resolve().is_relative_to(head.resolve()) for _,p in files):
        raise ValueError('Draft artifact escapes its checkpoint directory')
    files.append(('mtp_draft_manifest.json',manifest))
    return files,data


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workflow-dir',required=True)
    parser.add_argument('--repo',required=True)
    parser.add_argument('--path-in-repo',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    prefix=args.path_in_repo.replace('\\','/').strip('/')
    if not prefix.startswith('mtp/') or any(p in {'','..','.'} for p in prefix.split('/')):
        raise ValueError('Use a named mtp/ subfolder in the existing model repository')
    files,data=draft_files(args.workflow_dir)
    print(json.dumps(dict(repo=args.repo,path=prefix,files=[n for n,_ in files],harness=data['harness']),indent=2))
    if args.execute:
        from huggingface_hub import HfApi,CommitOperationAdd
        api=HfApi();info=api.model_info(args.repo)
        result=api.create_commit(args.repo,repo_type='model',parent_commit=info.sha,
            operations=[CommitOperationAdd(path_in_repo=prefix+'/'+name,path_or_fileobj=str(path)) for name,path in files],
            commit_message='Save trained dental MTP draft and exact target/harness provenance')
        print(result.commit_url)


if __name__=='__main__': main()
