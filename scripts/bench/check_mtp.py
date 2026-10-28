#!/usr/bin/env python3
"""Compare real MTP tensor bytes before and after merging/stitching."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--output',required=True);p.add_argument('--shapes-only',action='store_true');a=p.parse_args()
    from dental_agent.harness.checkpoints import audit_mtp
    from dental_agent.harness.evaluation import atomic_json
    report=audit_mtp(a.base,a.checkpoint,not a.shapes_only);atomic_json(a.output,report);print(json.dumps(report,indent=2))
    if not report['ok']:raise SystemExit(1)
if __name__=='__main__':main()
