"""Generate evidence-qualified paper tables from companion harness JSON, or pending cells.

Independent of model imports. Never writes to the frozen evaluation assets.
No-argument invocation initializes missing assets; --reset-pending is explicit.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
PENDING=r'\todo{pending}'
HARNESS={'baseline':'Baseline','rules_only':'Rules only (identity control)',
         'rules_skills_index':'Rules + on-demand skills','rules_skills_full':'Rules + full skills'}
SERVING={'baseline':'Baseline','mtp_1':'MTP, 1 token','mtp_2':'MTP, 2 tokens',
         'mtp_3':'MTP, 3 tokens','ngram_2':'Lookup, 2 tokens','ngram_3':'Lookup, 3 tokens','ngram_5':'Lookup, 5 tokens'}


def number(value, field, *, integer=False):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
        raise ValueError(f'{field}: expected finite nonnegative number')
    if integer and int(value)!=value:
        raise ValueError(f'{field}: expected integer')
    return value


def ratio(a,b):
    return a/b if b else None


def fmt(value):
    return PENDING if value is None else f'{value:.3f}'


def cohort(data, kind):
    if data.get('schema_version')!=1 or data.get('status') not in (('complete','failed_or_divergent') if kind=='serving' else ('complete',)):
        raise ValueError('Requires complete schema_version=1 results')
    split=data.get('split') if kind=='harness' else data.get('settings',{}).get('split')
    if split!='dev':
        raise ValueError('Only training-derived dev results may enter extension tables')
    ids=data.get('image_ids',[])
    if not ids or len(set(ids))!=len(ids):
        raise ValueError('image_ids must be nonempty and unique')
    if not data.get('dev_split_sha256'):
        raise ValueError('Missing dev split fingerprint')
    conditions=data.get('conditions',{})
    if 'baseline' not in conditions:
        raise ValueError('Missing baseline')
    for name,condition in conditions.items():
        records=condition.get('records' if kind=='harness' else 'episodes',[])
        actual=[r['image_id'] for r in records]
        if len(actual)!=len(ids) or len(set(actual))!=len(ids) or set(actual)!=set(ids):
            raise ValueError(f'{name}: conditions must use exactly the shared image IDs')
        counts(records)
        if kind=='serving' and condition.get('status')!='complete':
            raise ValueError(f'{name}: incomplete serving condition; retain raw JSON and report the failure separately')
    return conditions


def counts(records):
    values={key:sum(number(r[key],key,integer=True) for r in records) for key in ('exact_tp','exact_fp','exact_fn')}
    tp,fp,fn=(values[k] for k in ('exact_tp','exact_fp','exact_fn'))
    return ratio(2*tp,2*tp+fp+fn),ratio(fp,tp+fp)


def table(columns, rows):
    return '\n'.join([r'\begin{tabular}{@{}l'+'r'*(len(columns)-1)+r'@{}}',r'\toprule',
                      ' & '.join(columns)+r' \\',r'\midrule',
                      *(' & '.join(row)+r' \\' for row in rows),r'\bottomrule',r'\end{tabular}'])


def render_harness(data=None):
    conditions=cohort(data,'harness') if data is not None else {}
    rows=[]
    for key,label in HARNESS.items():
        f1,invented=counts(conditions[key]['records']) if key in conditions else (None,None)
        rows.append([label,fmt(f1),fmt(invented)])
    return '\n'.join([r'\begin{table}[tbp]\centering\small',
        r'\caption{Exploratory, paired dev-only harness comparison on one frozen checkpoint. '
        r'Finding invention is $FP/(TP+FP)$, distinct from the primary healthy-image metric. '
        r'Pending cells are not measurements.}',r'\label{tab:harness-dev}',
        table(['Context condition','Exact F1','Finding invention'],rows),r'\end{table}',''])


def signature(episode):
    return {'turns':[{'raw_output':t['raw_output'], 'tool_calls':[{k:c.get(k) for k in ('tool_name','tool_args','tool_ok','tool_error')} for c in t.get('tool_calls',[])]} for t in episode['turns']], 'final_answer':episode['final_answer'],'format_ok':episode['format_ok']}


def serving_stats(condition):
    episodes=condition['episodes']
    def total(key):
        vals=[e['measurements'].get(key) for e in episodes]
        return None if any(v is None for v in vals) else sum(number(v,key) for v in vals)
    wall=total('wall_seconds');request=total('request_wall_seconds')
    tokens=total('generated_tokens');accepted=total('accepted_tokens');drafted=total('drafted_tokens')
    prefill=total('prefill_seconds');decode=total('decode_seconds')
    if accepted is not None and drafted is not None and accepted>drafted:
        raise ValueError('accepted_tokens cannot exceed drafted_tokens')
    return {'wall':wall,'tokens_s':None if tokens is None or request is None else ratio(tokens,request),
            'latency':None if wall is None else wall/len(episodes),
            'acceptance':None if accepted is None or drafted is None else ratio(accepted,drafted),
            'prefill_share':None if prefill is None or decode is None else ratio(prefill,prefill+decode)}


def render_serving(data=None):
    conditions=cohort(data,'serving') if data is not None else {}
    base=conditions.get('baseline')
    base_stats=serving_stats(base) if base else None
    rows=[]
    for key,label in SERVING.items():
        values=[PENDING]*5; gate='pending'
        if key in conditions:
            current=conditions[key];stats=serving_stats(current)
            if key=='baseline':
                passed=True;gate='reference'
            else:
                reported=data.get('gates',{}).get(key,{})
                old={e['image_id']:e for e in base['episodes']}
                same=all(e['format_ok'] and old[e['image_id']]['format_ok'] and signature(e)==signature(old[e['image_id']]) for e in current['episodes'])
                passed=(reported.get('passed') is True and reported.get('exact_f1_unchanged') is True
                        and not reported.get('divergences') and same
                        and counts(base['episodes'])[0] is not None
                        and counts(current['episodes'])[0]==counts(base['episodes'])[0])
                gate='pass' if passed else 'FAIL'
            # Ineligible conditions retain raw stats in source JSON, never an asserted speedup.
            if passed:
                speed=ratio(base_stats['wall'],stats['wall']) if base_stats['wall'] is not None and stats['wall'] is not None else None
                values=[fmt(speed),fmt(stats['acceptance']) if key!='baseline' else '--',
                        fmt(stats['tokens_s']),fmt(stats['latency']),fmt(stats['prefill_share'])]
        rows.append([label,gate,*values])
    return '\n'.join([r'\begin{table}[tbp]\centering\scriptsize',r'\setlength{\tabcolsep}{3pt}',
        r'\caption{Serving efficiency on shared dev episodes. Speedup is baseline total wall time '
        r'divided by condition total wall time; tokens/s uses request time. Latency is seconds '
        r'per episode. Prefill share is prefill/(prefill+decode), only when separately measured. '
        r'Failed gates suppress efficiency claims; missing counters remain pending.}',
        r'\label{tab:serving-efficiency}',table(['Serving','Gate','Speedup','Accept.','Tokens/s','Latency','Prefill share'],rows),r'\end{table}',''])


def generate(out_dir, harness_path=None, serving_path=None, reset=False):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    sources={'harness':harness_path,'serving':serving_path}
    renderers={'harness':render_harness,'serving':render_serving}
    filenames={'harness':'table_harness_dev.tex','serving':'table_serving_efficiency.tex'}
    manifest_path=out_dir/'harness_evidence.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {'schema_version':1}
    # Validate and render all requested inputs before any mutation.
    updates={}
    for kind,path in sources.items():
        target=out_dir/filenames[kind]
        if path:
            raw=Path(path).read_bytes();data=json.loads(raw)
            updates[kind]=(renderers[kind](data),{'status':'result_file','input_sha256':hashlib.sha256(raw).hexdigest(),
                           'dev_split_sha256':data.get('dev_split_sha256'),'image_ids':data.get('image_ids'),
                           'settings':data.get('settings'),'software':data.get('software'),
                           'hardware':data.get('hardware'),'commands':data.get('commands')})
        elif reset or not target.exists():
            updates[kind]=(renderers[kind](),{'status':'pending','reason':'No measured result file supplied'})
    for kind,(content,evidence) in updates.items():
        (out_dir/filenames[kind]).write_text(content,encoding='utf-8');manifest[kind]=evidence
    manifest_path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--harness-results',type=Path)
    parser.add_argument('--serving-results',type=Path)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'paper/generated')
    parser.add_argument('--reset-pending',action='store_true',help='Explicitly replace unsupplied tables with pending cells')
    args=parser.parse_args();generate(args.output_dir,args.harness_results,args.serving_results,args.reset_pending)


if __name__=='__main__':
    main()
