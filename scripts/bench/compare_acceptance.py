#!/usr/bin/env python3
"""Compare native vs fine-tuned MTP only when dev settings, outputs and F1 agree."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

def compare(before,after):
    from dental_agent.harness.specdec import equivalence,sum_complete
    settings=lambda r:{k:v for k,v in r['settings'].items() if k not in {'checkpoint','output','plan_only'}}
    if settings(before)!=settings(after) or before['image_ids']!=after['image_ids'] or before['dev_split_sha256']!=after['dev_split_sha256']:
        raise ValueError('Before/after runs are not paired with identical settings')
    rows={}
    for name in ['mtp_1','mtp_2','mtp_3']:
        a=before.get('conditions',{}).get(name,{});b=after.get('conditions',{}).get(name,{})
        if a.get('status')!='complete' or b.get('status')!='complete':rows[name]=dict(status='pending');continue
        gate=equivalence(a['episodes'],b['episodes'])
        f1a=a['summary']['pooled_exact_f1'];f1b=b['summary']['pooled_exact_f1']
        valid=gate['passed'] and f1a==f1b and before.get('gates',{}).get(name,{}).get('passed',False) and after.get('gates',{}).get(name,{}).get('passed',False)
        def rate(entry):
            measurements=[e['measurements'] for e in entry['episodes']]
            drafted=sum_complete(measurements,'drafted_tokens');accepted=sum_complete(measurements,'accepted_tokens')
            return accepted/drafted if drafted and accepted is not None else None
        rows[name]=dict(status='valid' if valid else 'divergent',before_acceptance=rate(a),after_acceptance=rate(b),
                        dev_exact_f1_before=f1a,dev_exact_f1_after=f1b,equivalence=gate)
    return rows

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['before','after','output']:p.add_argument('--'+name,required=True)
    a=p.parse_args();report=compare(json.loads(Path(a.before).read_text()),json.loads(Path(a.after).read_text()))
    from dental_agent.harness.evaluation import atomic_json
    atomic_json(a.output,report);print(json.dumps(report,indent=2))
    if any(x['status']!='valid' for x in report.values()):raise SystemExit(2)
if __name__=='__main__':main()
