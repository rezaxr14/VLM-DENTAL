#!/usr/bin/env python3
"""Pick the longest traces for a sequence-length (memory) probe, and show how many traces fit each ceiling.

The longest trace sets the peak memory of a dynamically padded (GPU) run, so a few steps on only the longest traces
tell you whether a given ``--max-seq-len`` fits. Lengths come from the manifest written by
``compute_exact_trace_lengths.py`` (use the one measured with the same ``--canonical-resize`` setting as training).

    python scripts/select_longest_traces.py --traces data/traces/train_cot_traces.jsonl \
        --manifest data/traces/trace_token_lengths.json --top 4 --output data/traces/probe_longest.jsonl \
        --ceilings 8192 10240 12288 14336 16384

    # probe a lower candidate ceiling: the 4 longest traces that still fit under it
    python scripts/select_longest_traces.py ... --at-most 12288 --output data/traces/probe_12288.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def length_of(rec: dict, fname: str, lengths: dict[str, int]) -> int | None:
    rid, ds = str(rec.get("image_id", "")), str(rec.get("dataset", "default"))
    for key in (f"{fname}::{ds}::{rid}", f"{ds}::{rid}", f"{fname}::{rid}", rid):
        if key in lengths:
            return int(lengths[key])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--traces", required=True, nargs="+", help="With-tools trace JSONL file(s)")
    ap.add_argument("--manifest", required=True, help="trace_token_lengths.json")
    ap.add_argument("--top", type=int, default=4, help="How many of the longest traces to write")
    ap.add_argument("--at-most", type=int, default=None,
                    help="Only consider traces with length <= this (probe a candidate --max-seq-len: its longest survivors)")
    ap.add_argument("--output", default=None, help="Write the probe JSONL here")
    ap.add_argument("--ceilings", type=int, nargs="*", default=[8192, 10240, 12288, 14336, 16384],
                    help="Report how many traces have length <= each value")
    args = ap.parse_args()

    manifest = json.load(open(args.manifest, encoding="utf-8"))
    lengths = manifest.get("lengths_by_file_and_id") or manifest.get("lengths_by_image_id") or {}
    rows: list[tuple[int, dict]] = []
    missing = 0
    for path in args.traces:
        fname = Path(path).name
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            n = length_of(rec, fname, lengths)
            if n is None:
                missing += 1
            else:
                rows.append((n, rec))
    if not rows:
        raise SystemExit("No trace found in the manifest; was it computed for these files?")

    print(f"canonical_resize in manifest: {manifest.get('canonical_resize')}  traces measured: {len(rows)}  not in manifest: {missing}")
    print("Traces that fit each --max-seq-len (dynamic padding, no static vision slots):")
    for c in sorted(args.ceilings):
        k = sum(1 for n, _ in rows if n <= c)
        print(f"  <= {c:>6}: {k:>5} / {len(rows)}")
    rows.sort(key=lambda x: -x[0])
    if args.at_most is not None:
        rows = [r for r in rows if r[0] <= args.at_most]
        if not rows:
            raise SystemExit(f"No trace has length <= {args.at_most}.")
    top = rows[: args.top]
    print(f"Longest {len(top)} traces (tokens): {[n for n, _ in top]}")
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            for _, rec in top:
                f.write(json.dumps(rec) + "\n")
        print(f"Wrote probe set: {args.output}")


if __name__ == "__main__":
    main()
