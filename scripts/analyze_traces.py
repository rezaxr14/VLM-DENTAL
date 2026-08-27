#!/usr/bin/env python3
"""
analyze_traces.py - VLM-DENTAL Trace Dataset Analysis & Cleanup

Reads train_cot_traces_unverified.jsonl, removes failed/duplicate traces,
then prints rich tables and generates matplotlib charts covering:
  - Dataset overview
  - Per-trace quality metrics
  - Tool usage frequency & percentile breakdown
  - Token usage estimates
  - Diagnosis distribution

Run at any point during or after trace generation:
    python scripts/analyze_traces.py
    python scripts/analyze_traces.py --input data/traces/train_cot_traces_unverified.jsonl
    python scripts/analyze_traces.py --no-clean       # skip overwriting the file
    python scripts/analyze_traces.py --no-charts      # skip matplotlib output
"""

import argparse
import json
import os
import statistics as stats
from collections import Counter, defaultdict
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TOOL_NAMES = [
    "window_level",
    "locate_tooth",
    "zoom_crop",
    "nudge_crop",
    "fdi_label",
    "denoise",
    "enhance_contrast",
    "contralateral_compare",
]

DIAGNOSIS_NAMES = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth"]

BAR  = "█"
HALF = "▌"


def _bar(value: float, max_val: float, width: int = 30) -> str:
    """ASCII progress bar."""
    filled = int(round(value / max_val * width)) if max_val else 0
    return BAR * filled + "·" * (width - filled)


def _pct(value, total):
    return 100.0 * value / total if total else 0.0


def _percentile(data, p):
    if not data:
        return 0
    s = sorted(data)
    k = (len(s) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def estimate_tokens(obj) -> int:
    """Rough token estimate: 1 token ≈ 4 chars of JSON."""
    return len(json.dumps(obj, ensure_ascii=False)) // 4


# ---------------------------------------------------------------------------
# Loading & Cleaning
# ---------------------------------------------------------------------------

def load_and_clean(path: str, do_clean: bool):
    with open(path, encoding="utf-8") as f:
        raw_lines = [l.strip() for l in f if l.strip()]

    parsed, corrupt = [], 0
    for line in raw_lines:
        try:
            parsed.append(json.loads(line))
        except json.JSONDecodeError:
            corrupt += 1

    # Remove failed (no final_answer)
    good = [t for t in parsed if t.get("trajectory", {}).get("final_answer")]
    failed = len(parsed) - len(good)

    # Deduplicate – keep richest trace (most tool calls)
    groups: dict = defaultdict(list)
    for t in good:
        groups[t["image_id"]].append(t)

    deduped, dup_removed, dup_details = [], 0, []
    for img_id, traces in groups.items():
        if len(traces) == 1:
            deduped.append(traces[0])
        else:
            best = max(traces, key=lambda t: t.get("trajectory", {}).get("tool_calls", 0))
            deduped.append(best)
            dup_removed += len(traces) - 1
            dup_details.append((img_id, len(traces), best.get("trajectory", {}).get("tool_calls", 0)))

    if do_clean:
        with open(path, "w", encoding="utf-8") as f:
            for t in deduped:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")

    return {
        "raw": len(raw_lines),
        "corrupt": corrupt,
        "failed": failed,
        "dup_removed": dup_removed,
        "dup_details": dup_details,
        "traces": deduped,
    }


# ---------------------------------------------------------------------------
# Statistics extraction
# ---------------------------------------------------------------------------

def extract_stats(traces):
    per_trace = []
    tool_call_counts: Counter = Counter()   # total calls per tool
    tool_trace_counts: Counter = Counter()  # traces that used each tool ≥1 time
    diag_counter: Counter = Counter()

    for t in traces:
        traj = t.get("trajectory", {})
        turns = traj.get("turns", [])
        tool_calls_total = traj.get("tool_calls", 0)
        final_answer = traj.get("final_answer", [])

        # Per-turn tool breakdown
        tools_in_trace: Counter = Counter()
        for turn in turns:
            for tc in turn.get("tool_calls_this_turn", []):
                name = tc.get("tool_name", "unknown")
                tools_in_trace[name] += 1
                tool_call_counts[name] += 1

        for name in tools_in_trace:
            tool_trace_counts[name] += 1

        # Findings
        for finding in final_answer:
            diag_counter[finding.get("diagnosis", "Unknown")] += 1

        # Token estimates (input = everything before last assistant turn, output = assistant turns)
        total_bytes = len(json.dumps(t, ensure_ascii=False))
        input_tok_est = total_bytes // 5
        output_tok_est = total_bytes // 20

        per_trace.append({
            "image_id": t["image_id"],
            "n_turns": len(turns),
            "n_tool_calls": tool_calls_total,
            "n_unique_tools": len(tools_in_trace),
            "n_findings": len(final_answer),
            "input_tokens": input_tok_est,
            "output_tokens": output_tok_est,
            "tools_used": dict(tools_in_trace),
        })

    return per_trace, tool_call_counts, tool_trace_counts, diag_counter


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

W = 65  # console width

def section(title: str):
    print(f"\n{'═' * W}")
    pad = (W - len(title) - 2) // 2
    print(f"{'═' * pad} {title} {'═' * (W - pad - len(title) - 2)}")
    print(f"{'═' * W}")


def print_overview(info, per_trace):
    section("DATASET OVERVIEW")
    n = len(per_trace)
    print(f"  Raw lines in file:        {info['raw']:>6}")
    print(f"  Corrupt / unparseable:    {info['corrupt']:>6}")
    print(f"  Failed (no final_answer): {info['failed']:>6}")
    print(f"  Duplicates removed:       {info['dup_removed']:>6}")
    print(f"  ─────────────────────────────────")
    print(f"  Clean traces remaining:   {n:>6}  ✓")
    if info["dup_details"]:
        print(f"\n  Dedup detail:")
        for img_id, cnt, tc in info["dup_details"]:
            print(f"    img_id={img_id}: {cnt} versions found, kept richest ({tc} tool calls)")


def print_quality_table(per_trace):
    section("QUALITY METRICS (per trace)")
    n = len(per_trace)

    def row(label, data, fmt=".1f", unit=""):
        mn, avg, mx = min(data), stats.mean(data), max(data)
        p25 = _percentile(data, 25)
        p75 = _percentile(data, 75)
        p90 = _percentile(data, 90)
        print(f"  {label:<24}  {mn:>5{fmt}}  {avg:>7{fmt}}  {mx:>5{fmt}}  {p25:>6{fmt}}  {p75:>6{fmt}}  {p90:>6{fmt}}")

    print(f"\n  {'Metric':<24}  {'Min':>5}  {'Mean':>7}  {'Max':>5}  {'P25':>6}  {'P75':>6}  {'P90':>6}")
    print(f"  {'─'*24}  {'─'*5}  {'─'*7}  {'─'*5}  {'─'*6}  {'─'*6}  {'─'*6}")
    row("Turns / trace",        [q["n_turns"]        for q in per_trace])
    row("Tool calls / trace",   [q["n_tool_calls"]   for q in per_trace])
    row("Unique tools / trace", [q["n_unique_tools"] for q in per_trace])
    row("Findings / trace",     [q["n_findings"]     for q in per_trace])
    row("Input tokens (est)",   [q["input_tokens"]   for q in per_trace], fmt=".0f")
    row("Output tokens (est)",  [q["output_tokens"]  for q in per_trace], fmt=".0f")

    # Low-quality flag
    low_q = [q for q in per_trace if q["n_unique_tools"] < 3 or q["n_tool_calls"] < 5]
    print(f"\n  Low-quality (< 3 unique tools OR < 5 tool calls): {len(low_q)}", end="")
    if low_q:
        ids = ", ".join(str(q["image_id"]) for q in low_q[:5])
        print(f"  → img_ids: {ids}" + (" ..." if len(low_q) > 5 else ""))
    else:
        print("  ✓ None found")


def print_tool_stats(tool_call_counts, tool_trace_counts, per_trace):
    section("TOOL USAGE BREAKDOWN")
    n = len(per_trace)
    total_calls = sum(tool_call_counts.values())

    # Calls-per-trace for each tool
    tool_per_trace = defaultdict(list)
    for q in per_trace:
        for tool in TOOL_NAMES:
            tool_per_trace[tool].append(q["tools_used"].get(tool, 0))

    print(f"\n  {'Tool':<24}  {'Total':>6}  {'% calls':>7}  {'% traces':>8}  {'Avg/trace':>9}  {'P50':>4}  {'P90':>4}  Chart")
    print(f"  {'─'*24}  {'─'*6}  {'─'*7}  {'─'*8}  {'─'*9}  {'─'*4}  {'─'*4}  {'─'*20}")
    for tool in TOOL_NAMES:
        total = tool_call_counts.get(tool, 0)
        used_in = tool_trace_counts.get(tool, 0)
        avg_pt = stats.mean(tool_per_trace[tool]) if tool_per_trace[tool] else 0
        p50 = _percentile(tool_per_trace[tool], 50)
        p90 = _percentile(tool_per_trace[tool], 90)
        pct_calls = _pct(total, total_calls)
        pct_traces = _pct(used_in, n)
        bar = _bar(total, max(tool_call_counts.values()))
        print(f"  {tool:<24}  {total:>6,}  {pct_calls:>6.1f}%  {pct_traces:>7.1f}%  {avg_pt:>9.1f}  {p50:>4.1f}  {p90:>4.1f}  {bar}")


def print_diagnosis_stats(diag_counter, per_trace):
    section("DIAGNOSIS DISTRIBUTION")
    total_findings = sum(diag_counter.values())
    n = len(per_trace)
    print(f"\n  Total findings across {n} traces: {total_findings}")
    print(f"\n  {'Diagnosis':<22}  {'Count':>6}  {'%':>6}  {'Avg/trace':>9}  Chart")
    print(f"  {'─'*22}  {'─'*6}  {'─'*6}  {'─'*9}  {'─'*30}")
    for diag in DIAGNOSIS_NAMES + [k for k in diag_counter if k not in DIAGNOSIS_NAMES]:
        cnt = diag_counter.get(diag, 0)
        pct = _pct(cnt, total_findings)
        avg = cnt / n
        bar = _bar(cnt, max(diag_counter.values()))
        print(f"  {diag:<22}  {cnt:>6,}  {pct:>5.1f}%  {avg:>9.2f}  {bar}")


def print_token_summary(per_trace):
    section("TOKEN USAGE ESTIMATE")
    n = len(per_trace)
    total_in  = sum(q["input_tokens"]  for q in per_trace)
    total_out = sum(q["output_tokens"] for q in per_trace)
    total     = total_in + total_out
    print(f"\n  Per-trace averages:")
    print(f"    Input  tokens : ~{stats.mean([q['input_tokens']  for q in per_trace]):,.0f}")
    print(f"    Output tokens : ~{stats.mean([q['output_tokens'] for q in per_trace]):,.0f}")
    print(f"\n  Cumulative totals ({n} traces):")
    print(f"    Input  tokens : ~{total_in:>12,}")
    print(f"    Output tokens : ~{total_out:>12,}")
    print(f"    Combined      : ~{total:>12,}")
    # Rough cost estimate at typical free/low-cost model rates
    print(f"\n  Cost estimate (if charged at $0.15/M input, $0.60/M output):")
    cost_in  = total_in  / 1_000_000 * 0.15
    cost_out = total_out / 1_000_000 * 0.60
    print(f"    Input  : ${cost_in:>8.4f}")
    print(f"    Output : ${cost_out:>8.4f}")
    print(f"    Total  : ${cost_in + cost_out:>8.4f}")


# ---------------------------------------------------------------------------
# Charts (matplotlib)
# ---------------------------------------------------------------------------

def make_charts(per_trace, tool_call_counts, diag_counter, output_dir: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("\n  [charts] matplotlib not available – skipping chart generation")
        return

    os.makedirs(output_dir, exist_ok=True)
    plt.rcParams.update({
        "figure.facecolor": "#1a1a2e",
        "axes.facecolor":   "#16213e",
        "axes.edgecolor":   "#e94560",
        "text.color":       "#eaeaea",
        "axes.labelcolor":  "#eaeaea",
        "xtick.color":      "#eaeaea",
        "ytick.color":      "#eaeaea",
        "axes.titlecolor":  "#e94560",
        "grid.color":       "#0f3460",
        "grid.linestyle":   "--",
    })
    ACCENT = "#e94560"
    BLUE   = "#0f3460"
    TEAL   = "#16c79a"

    # 1. Tool usage bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    tools  = TOOL_NAMES
    counts = [tool_call_counts.get(t, 0) for t in tools]
    colors = [ACCENT if c == max(counts) else TEAL for c in counts]
    bars = ax.barh(tools, counts, color=colors, edgecolor="#1a1a2e", linewidth=0.5)
    ax.bar_label(bars, fmt="%d", color="#eaeaea", padding=3, fontsize=9)
    ax.set_xlabel("Total Calls")
    ax.set_title("Tool Usage – Total Calls Across All Traces")
    ax.grid(axis="x", alpha=0.4)
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "tool_usage.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/tool_usage.png")

    # 2. Diagnosis pie chart
    labels = list(diag_counter.keys())
    sizes  = list(diag_counter.values())
    palette = ["#e94560", "#16c79a", "#f5a623", "#7b68ee"][:len(labels)]
    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, autopct="%1.1f%%",
        colors=palette, startangle=140,
        wedgeprops=dict(edgecolor="#1a1a2e", linewidth=1.5)
    )
    for at in autotexts:
        at.set_color("#1a1a2e"); at.set_fontsize(9)
    ax.set_title("Diagnosis Distribution")
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "diagnosis_dist.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/diagnosis_dist.png")

    # 3. Tool calls per trace histogram
    calls = [q["n_tool_calls"] for q in per_trace]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(calls, bins=20, color=TEAL, edgecolor="#1a1a2e", linewidth=0.5)
    ax.axvline(np.mean(calls), color=ACCENT, linestyle="--", linewidth=1.5, label=f"mean={np.mean(calls):.1f}")
    ax.axvline(np.median(calls), color="#f5a623", linestyle=":", linewidth=1.5, label=f"median={np.median(calls):.1f}")
    ax.set_xlabel("Tool Calls per Trace")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Tool Calls per Trace")
    ax.legend()
    ax.grid(alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "tool_calls_hist.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/tool_calls_hist.png")

    # 4. Findings per trace histogram
    findings = [q["n_findings"] for q in per_trace]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(findings, bins=range(1, max(findings) + 2), color=ACCENT, edgecolor="#1a1a2e", linewidth=0.5, align="left")
    ax.set_xlabel("Findings per Trace")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Findings per Trace")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "findings_hist.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/findings_hist.png")

    # 5. Unique tools used per trace (stacked‑ish bar)
    unique = [q["n_unique_tools"] for q in per_trace]
    cnt = Counter(unique)
    x = sorted(cnt.keys())
    y = [cnt[k] for k in x]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, y, color=TEAL, edgecolor="#1a1a2e", linewidth=0.5)
    ax.bar_label(ax.containers[0], fmt="%d", color="#eaeaea", padding=2, fontsize=8)
    ax.set_xlabel("Unique Tools Used in Trace")
    ax.set_ylabel("Number of Traces")
    ax.set_title("Unique Tool Diversity per Trace")
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "tool_diversity.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved: {output_dir}/tool_diversity.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Analyze and clean VLM-DENTAL trace files")
    ap.add_argument("--input",     default="data/traces/train_cot_traces_unverified.jsonl")
    ap.add_argument("--no-clean",  action="store_true", help="Do not overwrite input file")
    ap.add_argument("--no-charts", action="store_true", help="Skip matplotlib chart generation")
    ap.add_argument("--chart-dir", default="data/traces/analysis_charts", help="Where to save charts")
    args = ap.parse_args()

    print(f"\n{'█'*W}")
    print(f"  VLM-DENTAL Trace Analyzer  |  {args.input}")
    print(f"{'█'*W}")

    info = load_and_clean(args.input, do_clean=not args.no_clean)
    if not args.no_clean:
        print(f"\n  [clean] File updated: {len(info['traces'])} traces written (removed "
              f"{info['failed']} failed + {info['dup_removed']} dups)")

    per_trace, tool_call_counts, tool_trace_counts, diag_counter = extract_stats(info["traces"])

    print_overview(info, per_trace)
    print_quality_table(per_trace)
    print_tool_stats(tool_call_counts, tool_trace_counts, per_trace)
    print_diagnosis_stats(diag_counter, per_trace)
    print_token_summary(per_trace)

    if not args.no_charts:
        section("GENERATING CHARTS")
        make_charts(per_trace, tool_call_counts, diag_counter, args.chart_dir)
        print(f"\n  All charts saved to: {args.chart_dir}/")

    section("DONE")
    print(f"\n  {len(per_trace)} clean traces ready for verification → SFT → GRPO\n")


if __name__ == "__main__":
    main()
