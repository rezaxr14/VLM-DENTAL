"""Extract a real paired with-tools / no-tools trace example for the paper comparison figure.

    python scripts/paper/extract_trace_examples.py [--with-tools train_cot_traces_dentex.jsonl]

Finds an image that has both a with-tools interactive trace and a single-turn no-tools trace,
preferring images with 1-2 findings and a with-tools trace that demonstrates the full arc
(locate -> zoom -> nudge -> verify -> answer). Prints trimmed excerpts for the figure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
TRACES = REPO_ROOT / 'data' / 'traces'


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding='utf-8') as fh:
        return [json.loads(l) for l in fh if l.strip()]


def thoughts(trace: dict) -> list[str]:
    """Assistant thought strings, one per turn, in order."""
    msgs = trace.get('messages', [])
    out = []
    for i, m in enumerate(msgs):
        if m.get('role') != 'assistant':
            continue
        c = m.get('content')
        if isinstance(c, str):
            try:
                out.append(json.loads(c).get('thought', c))
            except (json.JSONDecodeError, ValueError):
                out.append(c)
    return out


def final_answer(trace: dict):
    return trace.get('final_answer')


def brief_calls(turn: dict) -> list[str]:
    out = []
    for c in turn.get('tool_calls_this_turn', []):
        args = {k: v for k, v in (c.get('tool_args') or {}).items()}
        out.append(f"{c.get('tool_name')}({json.dumps(args) if args else ''})")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-tools', default='train_cot_traces_dentex.jsonl')
    ap.add_argument('--no-tools', default='train_cot_traces_dentex_no_tools.jsonl')
    args = ap.parse_args()

    with_tools = {r['image_id']: r for r in read_jsonl(TRACES / args.with_tools)}
    no_tools = {r['image_id']: r for r in read_jsonl(TRACES / args.no_tools)}
    shared = sorted(set(with_tools) & set(no_tools))

    # Rank candidates: few findings, nudge present, moderate turns, contralateral use.
    best = []
    for img in shared:
        wt = with_tools[img]
        n_find = len(wt.get('ground_truth') or [])
        if not 1 <= n_find <= 2:
            continue
        tools_used = [c['tool_name'] for t in wt['turns'] for c in t.get('tool_calls_this_turn', [])]
        score = 0
        score += (n_find == 1) * 3
        score += min(len(wt['turns']), 10)
        for want, pts in (('nudge_crop', 4), ('contralateral_compare', 2), ('locate_tooth', 2), ('zoom_crop', 2), ('window_level', 1), ('fdi_label', 1)):
            score += pts * (want in tools_used)
        best.append((score, img))
    best.sort(reverse=True)
    print('top candidates (score, image_id):', best[:8])

    img = best[0][1]
    wt, nt = with_tools[img], no_tools[img]
    print(f"\n===== IMAGE {img} =====")
    print('ground truth:', json.dumps(wt.get('ground_truth')))

    print(f"\n--- WITH-TOOLS ({len(wt['turns'])} turns) ---")
    th = thoughts(wt)
    for i, turn in enumerate(wt['turns']):
        thought = th[i] if i < len(th) else ''
        thought = re.sub(r'\s+', ' ', thought).strip()
        print(f"\n[turn {i}] THOUGHT: {thought[:400]}")
        for call in brief_calls(turn):
            print(f"    CALL: {call}")
    print(f"\nFINAL ANSWER: {json.dumps(final_answer(wt))}")

    print(f"\n--- NO-TOOLS ({len(nt['turns'])} turns) ---")
    nth = thoughts(nt)
    for i, turn in enumerate(nt['turns']):
        thought = nth[i] if i < len(nth) else ''
        print(f"\n[turn {i}] THOUGHT: {re.sub(r'\s+', ' ', thought)[:900]}")
    print(f"\nFINAL ANSWER: {json.dumps(final_answer(nt))}")


if __name__ == '__main__':
    main()
