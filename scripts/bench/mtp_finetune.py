#!/usr/bin/env python3
"""Plan or execute the official Speculators offline MTP workflow; no base-model fine-tuning."""
import argparse,json,shlex,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--conversations',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--endpoint',default='http://127.0.0.1:8910');p.add_argument('--speculators-executable',default='speculators')
    p.add_argument('--seq-length',type=int,default=16384);p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--hidden-states-dtype',choices=['float32','bfloat16'],default='bfloat16')
    p.add_argument('--stage',choices=['prepare','extract','train','stitch','all'],default='all');p.add_argument('--execute',action='store_true')
    from dental_agent.harness.configuration import HARNESS_MODES,require_harness_grpo_target,assert_harness_identity,harness_identity
    p.add_argument('--harness-mode',choices=HARNESS_MODES,required=True)
    a=p.parse_args();out=Path(a.output_dir);exe=a.speculators_executable
    target_manifest = require_harness_grpo_target(a.checkpoint,a.harness_mode)
    from dental_agent.harness.evaluation import atomic_json,digest
    from dental_agent.harness.checkpoints import tensor_headers
    if not any(k.startswith('mtp.') for k in tensor_headers(a.checkpoint)):raise ValueError('Native MTP required')
    provenance=Path(a.conversations+'.provenance.json')
    if not provenance.is_file():raise ValueError('Use regenerate_train.py output with train/dev-exclusion provenance')
    meta=json.loads(provenance.read_text())
    if meta.get('split')!='train-minus-fixed-dev' or meta.get('conversations_sha256')!=digest(a.conversations):raise ValueError('Unexpected or stale training provenance')
    from dental_agent.harness.notebook_eval import merged_signature
    from dental_agent.harness.checkpoints import mtp_parameter_inventory,target_tensor_fingerprint
    if meta.get('checkpoint_sha256') != merged_signature(a.checkpoint):
        raise ValueError('Regenerated conversations do not belong to this exact final target checkpoint')
    if meta.get('target_manifest') != target_manifest: raise ValueError('Conversation target adapter provenance differs')
    assert_harness_identity(meta.get('harness'),a.harness_mode)
    inventory = mtp_parameter_inventory(a.checkpoint)

    commands={
        'prepare':[exe,'prepare-data','--model',a.checkpoint,'--data',a.conversations,'--render-endpoint',a.endpoint,
                   '--output',str(out),'--seq-length',str(a.seq_length)],
        'extract':[exe,'generate-offline-data','--preprocessed-data',str(out),'--endpoint',a.endpoint+'/v1',
                   '--output',str(out/'hidden_states'),'--concurrency','1','--validate-outputs'],
        'train':[exe,'train','--verifier-name-or-path',a.checkpoint,'--data-path',str(out),'--save-path',str(out/'checkpoints'),
                 '--epochs',str(a.epochs),'--total-seq-len',str(a.seq_length),'--speculator-type','mtp','--target-layer-ids','32',
                 '--hidden-states-path',str(out/'hidden_states'),'--hidden-states-dtype',a.hidden_states_dtype,'--on-missing','raise'],
        'stitch':[exe,'stitch-mtp',str(out/'checkpoints/checkpoint_best'),a.checkpoint,'--output-path',str(out/'stitched')],
    }
    selected=list(commands) if a.stage=='all' else [a.stage]
    plan=dict(status='planned',settings=vars(a),commands={s:commands[s] for s in selected},
              mtp_inventory=inventory,harness=harness_identity(a.harness_mode),target_manifest=target_manifest,
              warning='Use a separate serving runtime with Speculators launch_vllm.py --target-layer-ids 32. For one GPU, execute prepare/extract, stop that owned server, then execute train/stitch. Multimodal render/extraction compatibility is version-gated: do not strip images if a version rejects them.')
    atomic_json(out/'mtp_workflow.json',plan)
    for stage in selected:
        print(stage+': '+shlex.join(commands[stage]),flush=True)
        if a.execute:
            subprocess.run(commands[stage],check=True)
            plan['status']='completed_'+stage;atomic_json(out/'mtp_workflow.json',plan)
    if a.execute and 'stitch' in selected:
        # A trained MTP head MUST differ; audit all target tensors instead of comparing MTP to the original head.
        from dental_agent.harness.checkpoints import tensor_hash
        base=tensor_headers(a.checkpoint);new=tensor_headers(out/'stitched')
        target={k for k in base if not k.startswith('mtp.')}
        if target!={k for k in new if not k.startswith('mtp.')} or any(base[k]['shape']!=new[k]['shape'] or base[k]['dtype']!=new[k]['dtype'] or tensor_hash(base[k])!=tensor_hash(new[k]) for k in target):
            raise SystemExit('Stitch changed target tensors; before/after benchmark is invalid')
        if not all(k in new for k in base if k.startswith('mtp.')):raise SystemExit('Stitch lost MTP tensors')
        changed = [k for k in base if k.startswith('mtp.') and tensor_hash(base[k]) != tensor_hash(new[k])]
        if not changed: raise SystemExit('No MTP tensors changed; this is not a fine-tuned draft')
        import shutil
        shutil.copy2(Path(a.checkpoint)/'dental_target.json',out/'stitched'/'dental_target.json')
        draft_manifest = dict(draft_variant='finetuned',harness=harness_identity(a.harness_mode),
            target_tensor_sha256=target_tensor_fingerprint(a.checkpoint),target_manifest=target_manifest,
            conversations_sha256=meta['conversations_sha256'],changed_mtp_tensors=changed,mtp_inventory=inventory)
        atomic_json(out/'stitched'/'mtp_draft_manifest.json',draft_manifest)
        plan['target_tensor_bytes_unchanged']=True;atomic_json(out/'mtp_workflow.json',plan)
if __name__=='__main__':main()
