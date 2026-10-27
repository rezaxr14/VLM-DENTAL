#!/usr/bin/env python3
"""Repair and upload an existing adapter checkpoint on CPU, without loading model weights."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-dir',required=True)
    p.add_argument('--hf-repo',default=None)
    p.add_argument('--hub-base-model-id',default=None)
    p.add_argument('--track',choices=['with_tools','no_tools'])
    p.add_argument('--stage',choices=['dentex_alone','dentex_tufts_overlap','multicohort_all'])
    p.add_argument('--precision',choices=['qlora','bf16','fp16'])
    p.add_argument('--hf-run-tag',default=None)
    p.add_argument('--path-in-repo',default=None,help='Explicit destination, e.g. a GRPO checkpoint subfolder')
    p.add_argument('--prepare-only',action='store_true',help='Repair local metadata and print destination; no network upload')
    a=p.parse_args()
    folder=Path(a.checkpoint_dir).expanduser().resolve()
    if not folder.is_dir():p.error(f'Checkpoint directory not found: {folder}')
    configs=sorted(folder.rglob('adapter_config.json'))
    if not configs:p.error('No adapter_config.json found; select the actual saved checkpoint folder')
    for config in configs:
        if not config.resolve().is_relative_to(folder):p.error('Adapter metadata points outside the checkpoint folder')
        weights=[config.parent/name for name in ['adapter_model.safetensors','adapter_model.bin']]
        weights+=list(config.parent.glob('adapter_model-*.safetensors'))
        if not any(f.is_file() and f.stat().st_size for f in weights):
            p.error(f'No nonempty adapter weights next to {config}')
    meta_path=folder/'sft_run_meta.json'
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    fields={}
    for name in ['track','stage','precision']:
        supplied=getattr(a,name);saved=meta.get(name)
        if supplied and saved and supplied!=saved:p.error(f'--{name}={supplied} conflicts with saved metadata {saved}')
        fields[name]=supplied or saved
    from dental_agent.model.hub_artifacts import prepare_hub_metadata,sft_hub_subfolder,upload_adapter_folder
    if a.path_in_repo:
        target=a.path_in_repo.replace('\\','/').strip('/')
        if not target or any(s in {'.','..',''} for s in target.split('/')):p.error('Invalid --path-in-repo')
    else:
        if not all(fields.values()):p.error('Provide --track --stage --precision, or --path-in-repo, when sft_run_meta.json is missing')
        target=sft_hub_subfolder(**fields,run_tag=a.hf_run_tag or meta.get('hf_run_tag'))
    base=a.hub_base_model_id or meta.get('hub_base_model_id')
    print(f'[CHECKPOINT] {folder}\n[HUB TARGET] {a.hf_repo or "(prepare-only)"}/{target}',flush=True)
    if a.prepare_only:
        print(json.dumps(prepare_hub_metadata(folder,base),indent=2))
    else:
        if not a.hf_repo:p.error('--hf-repo is required unless --prepare-only is set')
        result=upload_adapter_folder(folder,a.hf_repo,target,base_model_id=base,
                                     commit_message='Recover saved adapter checkpoint with valid Hub metadata')
        print(f'[UPLOADED] {a.hf_repo}/{target}\n{result}',flush=True)


if __name__=='__main__':main()
