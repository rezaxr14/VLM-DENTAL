#!/usr/bin/env python3
"""Regenerate on-policy with-tools episodes on TRAIN images selected by training traces, excluding dev."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traces',nargs='+',required=True);p.add_argument('--endpoint',default='http://127.0.0.1:8910')
    p.add_argument('--data-dir',default='data');p.add_argument('--dev-split',default='data/splits/dentex_dev.json')
    p.add_argument('--output',required=True);p.add_argument('--limit',type=int,default=None);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-new-tokens',type=int,default=2048);p.add_argument('--max-tool-calls',type=int,default=50)
    from dental_agent.harness.configuration import HARNESS_MODES,harness_options,harness_identity,require_harness_grpo_target
    p.add_argument('--checkpoint',required=True,help='Exact final harness-trained merged GRPO target served at endpoint')
    p.add_argument('--harness-mode',choices=HARNESS_MODES,required=True)
    a=p.parse_args()
    target_manifest = require_harness_grpo_target(a.checkpoint,a.harness_mode)
    from dental_agent.harness.notebook_eval import merged_signature
    checkpoint_sha = merged_signature(a.checkpoint)
    harness = harness_options(a.harness_mode)
    from dental_agent.data.dentex import load_dentex_dataset
    from dental_agent.data.dev_split import load_dev_keys,assert_not_test_split,image_key
    from dental_agent.harness.specdec import ServingReply,encode_messages,http_json
    from dental_agent.harness.evaluation import digest,atomic_json
    from dental_agent.agent.loop import run_agent
    from dental_agent.utils.canonical import set_letterbox
    import random
    dev=load_dev_keys(a.dev_split);ids=set()
    for filename in a.traces:
        path=Path(filename)
        if not path.name.startswith('train_cot_traces') or path.suffix!='.jsonl':raise ValueError('Use canonical TRAIN trace files only')
        for line in path.read_text().splitlines():
            if not line.strip():continue
            record=json.loads(line)
            if (record.get('dataset') or 'dentex')!='dentex':raise ValueError('This regeneration workflow selects DENTEX train images only')
            if record.get('track') not in (None,'with_tools'):raise ValueError('Use with-tools trace files')
            iid=int(record['image_id'])
            if image_key('dentex',iid) not in dev:ids.add(iid)
    images,_,_=load_dentex_dataset(data_dir=a.data_dir,split_name='train');assert_not_test_split(images)
    if 'source_file' not in images:raise ValueError('Train provenance required')
    if not ids.issubset(set(int(i) for i in images['id'])):raise ValueError('A trace does not belong to the available TRAIN pool')
    chosen=sorted(ids);random.Random(a.seed).shuffle(chosen)
    if a.limit is not None:
        if a.limit<=0:raise ValueError('limit must be positive, or omit it for the full train pool')
        chosen=chosen[:a.limit]
    if not chosen:raise ValueError('No eligible train images')
    out=Path(a.output)
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);set_letterbox(False)
    metadata=dict(split='train-minus-fixed-dev',source_trace_sha256={str(p):digest(p) for p in a.traces},dev_split_sha256=digest(a.dev_split),
                  selected_image_ids=chosen,settings=vars(a),episodes=[],checkpoint_sha256=checkpoint_sha,
                  target_manifest=target_manifest,harness=harness_identity(a.harness_mode))
    models=http_json(a.endpoint.rstrip('/')+'/v1/models')['data']
    if len(models)!=1:raise ValueError('Use an exclusive one-model regeneration server')
    target_name=models[0]['id']
    if str(Path(target_name).resolve()) != str(Path(a.checkpoint).resolve()):
        raise ValueError('The endpoint is not serving the selected final GRPO checkpoint path')
    with out.open('x') as f:
        for iid in chosen:
            reply=ServingReply(a.endpoint,a.seed,model_id=target_name)
            traj=run_agent(iid,images,None,None,canonical_resize=True,temperature=0.0,max_new_tokens=a.max_new_tokens,
                           max_tool_calls=a.max_tool_calls,reply_fn=reply,harness=harness,verbose=False)
            complete=traj.format_ok and all(r.get('finish_reason')!='length' for r in reply.requests)
            metadata['episodes'].append(dict(image_id=iid,complete=complete,requests=reply.requests))
            if not complete:continue  # explicitly counted in provenance; never train a truncated turn as complete
            f.write(json.dumps(dict(id=f'dentex_train_{iid}',messages=encode_messages(traj.messages),
                                    metadata=dict(split='train',image_id=iid,on_policy=True,endpoint=a.endpoint,harness_mode=a.harness_mode)))+'\n');f.flush()
    metadata['conversations_sha256']=digest(out)
    atomic_json(str(out)+'.provenance.json',metadata)
    if not any(x['complete'] for x in metadata['episodes']):raise SystemExit('No complete target episodes; inspect provenance')
    print('On-policy multimodal conversations:',out)
if __name__=='__main__':main()
