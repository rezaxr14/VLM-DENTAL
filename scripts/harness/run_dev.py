#!/usr/bin/env python3
"""Opt-in paired harness evaluation: fixed dev split ONLY. Never retrains weights."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-id',required=True)
    p.add_argument('--adapter-path',required=True)
    p.add_argument('--checkpoint-kind',choices=['sft','grpo'],default='sft')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--dev-split',default='data/splits/dentex_dev.json')
    p.add_argument('--split',choices=['dev'],default='dev')
    p.add_argument('--output',default='paper/generated/harness_dev.json')
    p.add_argument('--conditions',nargs='+',choices=['baseline','rules_only','rules_skills_index','rules_skills_full'],default=['baseline','rules_only','rules_skills_index','rules_skills_full'])
    p.add_argument('--existing-results',choices=['error','resume'],default='error')
    p.add_argument('--limit',type=int)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--max-new-tokens',type=int,default=2048)
    p.add_argument('--max-tool-calls',type=int,default=50)
    p.add_argument('--max-turns',type=int)
    p.add_argument('--precision',choices=['qlora','bf16','fp16','fp32'],default='qlora')
    from dental_agent.model.local_runtime import add_runtime_args
    add_runtime_args(p)
    return p.parse_args(argv)

def main(argv=None):
    args=parse_args(argv)
    from dental_agent.harness.evaluation import CONDITIONS,load_dev,evaluate_episode,pooled,compare_paired,digest,atomic_json
    from dental_agent.harness.skills import SkillLibrary
    from dental_agent.model.local_runtime import runtime_from_args
    from scripts import evaluate_models as ev
    saved = json.loads(Path(args.output).read_text()) if Path(args.output).exists() else None
    if saved and args.existing_results != 'resume':raise SystemExit('Output exists; use --existing-results resume for this exact experiment')
    if not Path(args.adapter_path).is_dir():raise FileNotFoundError(args.adapter_path)
    from dental_agent.harness.notebook_eval import adapter_signature
    adapter_sha256=adapter_signature(args.adapter_path)
    if saved and saved.get('adapter_sha256') != adapter_sha256: raise ValueError('Saved harness adapter content differs or lacks provenance; results preserved')
    ids,images,annotations,categories=load_dev(args.data_dir,args.dev_split,args.limit,args.seed)
    if saved:
        from dental_agent.harness.notebook_eval import validate_resume_settings
        validate_resume_settings(saved, vars(args), ids)
        if saved.get('status') == 'complete':
            print('Complete harness result retained:', args.output); return
    device,dtype,_=ev.setup_eval_hardware(args.precision,args.qlora_compute_dtype)
    model,processor=ev.load_eval_model(args.model_id,args.adapter_path,device,dtype,args.checkpoint_kind+'_with_tools',
                                      quantize_4bit=args.precision=='qlora',runtime_options=runtime_from_args(args))
    root=Path(__file__).resolve().parents[2]
    result=dict(schema_version=1,status='running',split='dev',image_ids=ids,settings=vars(args),adapter_sha256=adapter_sha256,
                dev_split_sha256=digest(args.dev_split),rules_sha256=digest(root/'dental_agent/harness/rules.md'),
                skills=[dict(name=s.name,sha256=s.sha256) for s in SkillLibrary().all()],conditions={},comparisons={})
    if saved: result = saved
    result['settings'] = vars(args); result['status'] = 'running'
    for condition in args.conditions:
        entry = result['conditions'].setdefault(condition, dict(options=asdict(CONDITIONS[condition]),records=[]))
        records = entry['records']; done = {int(r['image_id']) for r in records}
        if len(done) != len(records) or not done.issubset(ids): raise ValueError('Invalid saved harness cohort')
        for image_id in ids:
            if image_id in done: continue
            rec,_=evaluate_episode(image_id,images,annotations,categories,model,processor,harness=CONDITIONS[condition],
                                  seed=args.seed,max_new_tokens=args.max_new_tokens,max_tool_calls=args.max_tool_calls,max_turns=args.max_turns)
            records.append(rec);atomic_json(args.output,result)
        result['conditions'][condition]['summary']=pooled(records)
    reference='baseline' if 'baseline' in result['conditions'] else args.conditions[0]
    for name in args.conditions:
        if name!=reference:result['comparisons'][name]=compare_paired(result['conditions'][reference]['records'],result['conditions'][name]['records'],args.seed)
    result['status']='complete';atomic_json(args.output,result)
    print(json.dumps({k:v['summary'] for k,v in result['conditions'].items()},indent=2))

if __name__=='__main__':main()
