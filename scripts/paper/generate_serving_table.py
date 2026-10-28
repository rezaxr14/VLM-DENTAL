#!/usr/bin/env python3
"""Generate the serving-efficiency table; missing/failed evidence stays red todo."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from dental_agent.harness.specdec import CONDITIONS,sum_complete

def value(x):
    return r'\textcolor{red}{\todo{pending}}' if x is None else f'{x:.3f}'

def rows(report):
    result=[];base=report.get('conditions',{}).get('baseline',{})
    base_wall=sum_complete([x['measurements'] for x in base.get('episodes',[])],'wall_seconds')
    for name in CONDITIONS:
        entry=report.get('conditions',{}).get(name,{})
        measured=[x['measurements'] for x in entry.get('episodes',[])]
        gate=name=='baseline' or report.get('gates',{}).get(name,{}).get('passed',False)
        valid=entry.get('status')=='complete' and gate and bool(measured)
        wall=sum_complete(measured,'wall_seconds') if valid else None
        tokens=sum_complete(measured,'generated_tokens') if valid else None
        pref=sum_complete(measured,'prefill_seconds') if valid else None
        dec=sum_complete(measured,'decode_seconds') if valid else None
        drafted=sum_complete(measured,'drafted_tokens') if valid else None;accepted=sum_complete(measured,'accepted_tokens') if valid else None
        result.append(dict(condition=name,status=entry.get('status','pending'),equivalence=gate if entry else None,
                           speedup=base_wall/wall if valid and wall and base_wall else None,
                           acceptance=accepted/drafted if drafted and accepted is not None else None,
                           tokens_per_second=tokens/wall if tokens is not None and wall else None,
                           episode_latency=wall/len(measured) if wall and measured else None,
                           prefill_share=pref/(pref+dec) if pref is not None and dec is not None and pref+dec>0 else None,
                           decode_share=dec/(pref+dec) if pref is not None and dec is not None and pref+dec>0 else None))
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input');p.add_argument('--output',default='paper/generated/serving_efficiency.tex');a=p.parse_args()
    report=json.loads(Path(a.input).read_text()) if a.input else {}
    data=rows(report)
    lines=[r'% Generated from measured benchmark evidence only; never fill missing observations.',r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Condition & Speedup & Acceptance & Tokens/s & Episode s & Prefill share & Decode share \\',r'\midrule']
    for row in data:
        lines.append(row['condition'].replace('_',r'\_')+' & '+' & '.join(value(row[k]) for k in ['speedup','acceptance','tokens_per_second','episode_latency','prefill_share','decode_share'])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}']
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);out.write_text('\n'.join(lines)+'\n')
    out.with_suffix('.json').write_text(json.dumps(data,indent=2)+'\n')
    print(out)
if __name__=='__main__':main()
