"""Dev-only harness experiment orchestration, separate from the frozen evaluation runner."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import random
import tempfile
from .skills import HarnessOptions

CONDITIONS = {
    'baseline': HarnessOptions(),
    'rules_only': HarnessOptions(rules_from_file=True),
    'rules_skills_index': HarnessOptions(rules_from_file=True,skills=True,skills_mode='index_only'),
    'rules_skills_full': HarnessOptions(rules_from_file=True,skills=True,skills_mode='full_inject'),
}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,delete=False) as f:
        json.dump(value,f,indent=2,allow_nan=False); f.write('\n'); tmp=Path(f.name)
    tmp.replace(path)

def pooled(records):
    tp=sum(x['exact_tp'] for x in records);fp=sum(x['exact_fp'] for x in records);fn=sum(x['exact_fn'] for x in records)
    denom=2*tp+fp+fn
    return dict(exact_tp=tp,exact_fp=fp,exact_fn=fn,
                pooled_exact_f1=2*tp/denom if denom else None,
                invented_findings_rate=fp/(tp+fp) if tp+fp else None,
                format_failures=sum(not x['format_ok'] for x in records),episodes=len(records))

def compare_paired(left,right,seed=42,n_boot=2000):
    a={x['image_id']:x for x in left};b={x['image_id']:x for x in right}
    if not a or a.keys()!=b.keys() or len(a)!=len(left) or len(b)!=len(right):
        raise ValueError('Paired conditions require the same unique dev images')
    ids=sorted(a);rng=random.Random(seed);out={}
    for metric in ['pooled_exact_f1','invented_findings_rate']:
        av=pooled(left)[metric];bv=pooled(right)[metric];samples=[]
        for _ in range(n_boot):
            chosen=rng.choices(ids,k=len(ids));va=pooled([a[i] for i in chosen])[metric];vb=pooled([b[i] for i in chosen])[metric]
            if va is not None and vb is not None:samples.append(vb-va)
        samples.sort()
        out[metric]=dict(delta=None if av is None or bv is None else bv-av,
                         ci95=[samples[int(.025*(len(samples)-1))],samples[int(.975*(len(samples)-1))]] if samples else None,
                         valid_bootstraps=len(samples))
    return out

def load_dev(data_dir,dev_split,limit=None,seed=42):
    if limit is not None and limit<=0:raise ValueError('limit must be positive')
    # Existing loader insists on train provenance, exact fixed dev membership and all matching annotations.
    from scripts import evaluate_models as ev
    from argparse import Namespace
    images,annotations,categories=ev.load_eval_split(Namespace(split='dev',dataset='dentex',dev_split=dev_split,data_dir=data_dir))
    if 'source_file' not in images.columns or images['source_file'].isna().any():
        raise ValueError('Train provenance is required, not just numeric image IDs')
    from dental_agent.data.dev_split import assert_not_test_split
    assert_not_test_split(images)
    ids=sorted(int(x) for x in images['id']);random.Random(seed).shuffle(ids)
    if limit is not None:ids=ids[:limit]
    if not ids:raise ValueError('No dev episodes')
    subset=images[images['id'].isin(ids)].copy()
    for row in subset.itertuples():
        if not isinstance(row.local_path,str) or not Path(row.local_path).is_file():
            raise FileNotFoundError(f'Missing dev image: {row.id}')
    return ids,subset,annotations,categories

def evaluate_episode(image_id,images,annotations,categories,model=None,processor=None,harness=None,reply_fn=None,
                     seed=42,max_new_tokens=2048,max_tool_calls=50,max_turns=None,registry=None,tool_observer=None):
    import numpy as np
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    from dental_agent.agent.loop import run_agent
    from dental_agent.evaluation.ground_truth import extract_ground_truth
    from dental_agent.evaluation.metrics import extract_predicted_findings,match_multi_findings
    from dental_agent.utils.canonical import set_letterbox
    set_letterbox(False)
    traj=run_agent(image_id,images,model,processor,registry=registry,canonical_resize=True,temperature=0.0,
                   max_new_tokens=max_new_tokens,max_tool_calls=max_tool_calls,max_turns=max_turns,
                   harness=harness,reply_fn=reply_fn,tool_observer=tool_observer,verbose=False)
    gt=extract_ground_truth(image_id,annotations,categories)
    pred=extract_predicted_findings(traj.final_answer)
    score=match_multi_findings(gt,pred)
    record=dict(image_id=image_id,predictions=pred,ground_truth=gt,format_ok=traj.format_ok,
                exact_tp=score['exact_tp'],exact_fp=score['exact_fp'],exact_fn=score['exact_fn'],
                turns=[dict(turn=t['turn'],raw_output=t['raw_output'],status=t.get('status'),tool_calls=t.get('tool_calls_this_turn',[])) for t in traj.turns],
                final_answer=traj.final_answer,harness_events=traj.harness_events)
    return record,traj
