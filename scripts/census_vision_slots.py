#!/usr/bin/env python3
"""Census of vision-input diversity and static-slot coverage for SFT traces (no model / GPU needed).

Answers, from the real trace files, the questions the TPU design rests on:
  * how many distinct vision-patch shapes would XLA see WITHOUT slot padding (the "91 shapes" claim)?
  * does the static [5 FULL, 10 CROP, 4 COMPARE] budget cover every trace?
  * with a token-length manifest: how many traces fit max_seq_len when slot-padded?

Usage:
  python scripts/census_vision_slots.py data/traces/train_cot_traces_dentex.jsonl [more.jsonl ...] \
      [--manifest data/traces/trace_token_lengths.json] [--seq-lens 10240 16384] [--json out.json]

Family attribution mirrors DentalSFTDataset: turn-1 image = FULL; each later image is attributed via the
"Result of <tool>:" text that follows it (fallback: next pending tool call), using canonical.TOOL_FAMILY.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dental_agent.utils.canonical import (  # noqa: E402
    SLOT_BUDGET,
    TOOL_FAMILY,
    parse_slot_budget,
    patches_per_image,
    slot_totals,
)


def count_families(rec: dict[str, Any]) -> dict[str, int]:
    """Number of images per canonical family a trace presents to the model.

    Mirrors DentalSFTDataset exactly: the FIRST user message always gets exactly one FULL base image
    (any image placeholder stored there is ignored); every image item in a LATER user message is rendered
    by the tool named in the "Result of <tool>:" text that follows it (fallback: next pending call of the
    preceding assistant turn, then zoom_crop). Older observation turns carry "[Earlier tool result omitted]"
    text instead of an image item, so they contribute nothing.
    """
    counts = {f: 0 for f in SLOT_BUDGET}
    counts["FULL"] = 1
    msgs = rec.get("messages") or []
    pending: list[str | None] = []
    first_user_seen = False
    for m in msgs:
        role, content = m.get("role"), m.get("content")
        if role == "assistant":
            pending = []
            try:
                parsed = json.loads(content) if isinstance(content, str) else (content or {})
                for c in (parsed.get("tool_calls") or []) if isinstance(parsed, dict) else []:
                    pending.append(c.get("tool"))
            except Exception:
                pass
        elif role == "user":
            if not first_user_seen:
                first_user_seen = True
                continue
            if not isinstance(content, list):
                continue
            cursor = 0
            for i, item in enumerate(content):
                if not (isinstance(item, dict) and item.get("type") == "image"):
                    continue
                tool = None
                nxt = content[i + 1] if i + 1 < len(content) else None
                if isinstance(nxt, dict) and nxt.get("type") == "text" and str(nxt.get("text", "")).startswith("Result of "):
                    tool = str(nxt["text"]).split(":")[0].replace("Result of ", "").strip()
                elif cursor < len(pending):
                    tool = pending[cursor]
                    cursor += 1
                counts[TOOL_FAMILY.get(tool or "", "CROP")] += 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--manifest", default=None, help="trace_token_lengths.json (computed with --canonical-resize)")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[10240, 16384])
    ap.add_argument("--vision-slots", type=int, nargs=3, metavar=("FULL", "CROP", "COMPARE"),
                    default=[SLOT_BUDGET["FULL"], SLOT_BUDGET["CROP"], SLOT_BUDGET["COMPARE"]],
                    help="Static slot budget to evaluate (same flag as train_sft.py).")
    ap.add_argument("--json", default=None, help="Write the full report here")
    args = ap.parse_args()

    budget = parse_slot_budget(*args.vision_slots)
    static_tokens = slot_totals(budget)["tokens"]
    shapes: collections.Counter = collections.Counter()
    mixes: collections.Counter = collections.Counter()
    over_budget: list[str] = []
    per_family_max = {f: 0 for f in SLOT_BUDGET}
    n = 0
    padded_lengths: list[int] = []
    manifest = json.load(open(args.manifest)) if args.manifest else None
    lens = (manifest or {}).get("lengths_by_file_and_id", {})
    vis = (manifest or {}).get("vision_tokens_by_file_and_id", {})

    for tp in args.traces:
        fname = Path(tp).name
        for line in open(tp, encoding="utf-8"):
            if not line.strip():
                continue
            rec = json.loads(line)
            c = count_families(rec)
            n += 1
            patches = sum(c[f] * patches_per_image(f) for f in c)
            shapes[patches] += 1
            mixes[(c["FULL"], c["CROP"], c["COMPARE"])] += 1
            for f in c:
                per_family_max[f] = max(per_family_max[f], c[f])
            if any(c[f] > budget[f] for f in c):
                over_budget.append(f"{fname}::{rec.get('dataset', 'default')}::{rec.get('image_id')}")
            key = f"{fname}::{rec.get('dataset', 'default')}::{rec.get('image_id')}"
            if key in lens and key in vis:
                padded_lengths.append(lens[key] - vis[key] + static_tokens)

    report = {
        "traces": n,
        "distinct_patch_totals": len(shapes),
        "distinct_family_mixes": len(mixes),
        "max_images_per_family": per_family_max,
        "slot_budget": dict(budget),
        "static_vision_tokens": static_tokens,
        "traces_over_budget": len(over_budget),
        "over_budget_examples": over_budget[:10],
    }
    if padded_lengths:
        report["slot_padded_length_fit"] = {
            str(L): sum(1 for x in padded_lengths if x <= L) for L in args.seq_lens
        } | {"with_manifest_entry": len(padded_lengths)}
    print(json.dumps(report, indent=2))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
    if over_budget:
        sys.exit(2)


if __name__ == "__main__":
    main()
