"""Paper comparison table: hosted models vs the VLM-DENTAL agent, with the harness effect measured as paired differences.

Input is ``cells``: ``{label: [evaluated records]}`` (hosted-model cells from ``importer.import_transport`` and the project's
own cells from ``matrix.load_records``). Metrics come from ``matrix.summarize_cell`` and ``paired_bootstrap_diff`` unmodified.
"""
from __future__ import annotations

from typing import Any

from dental_agent.evaluation.matrix import paired_bootstrap_diff, summarize_cell


def _f(x, nd=3):
    return "n/a" if x is None else f"{x:.{nd}f}"


def table_rows(cells: dict[str, list[dict[str, Any]]], n_boot: int = 2000, seed: int = 0) -> list[dict[str, Any]]:
    rows = []
    for label, recs in cells.items():
        s = summarize_cell(recs, n_boot=n_boot, seed=seed)
        ex, fd = s["exact"], s["fdi"]
        rows.append({"cell": label, "n": s["n_images"], "format_ok": s["format_ok_rate"], "fdi_f1": fd["f1"], "exact_precision": ex["precision"],
                     "exact_recall": ex["recall"], "exact_f1": ex["f1"], "exact_f1_lo": ex["f1_ci_low"], "exact_f1_hi": ex["f1_ci_high"],
                     "healthy_fp_rate": s["healthy"]["false_positive_image_rate"], "mean_turns": s["mean_turns"],
                     "mean_tool_calls": s["mean_tool_calls"], "mean_seconds": s["mean_seconds"]})
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    head = "| Cell | N | Format OK | FDI F1 | Exact P | Exact R | Exact F1 [95% CI] | Healthy FP rate | Turns | Tool calls |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    body = "".join(f"| {r['cell']} | {r['n']} | {r['format_ok']:.1%} | {_f(r['fdi_f1'])} | {_f(r['exact_precision'])} | {_f(r['exact_recall'])} | "
                   f"{_f(r['exact_f1'])} [{_f(r['exact_f1_lo'])}, {_f(r['exact_f1_hi'])}] | {r['healthy_fp_rate']:.1%} | {_f(r['mean_turns'], 1)} | {_f(r['mean_tool_calls'], 1)} |\n"
                   for r in rows)
    return head + body


def harness_effect(cells: dict[str, list[dict[str, Any]]], pairs: list[tuple[str, str, str]], n_boot: int = 2000, seed: int = 0) -> list[dict[str, Any]]:
    """``pairs`` = (label, A, B): the pooled exact-F1 of A minus B on the images both cover, with a paired bootstrap CI.
    Pairs with a missing cell are skipped (never an empty comparison)."""
    out = []
    for label, a, b in pairs:
        if a in cells and b in cells:
            out.append({"label": label, "a": a, "b": b, **paired_bootstrap_diff(cells[a], cells[b], level="exact", metric="f1", n_boot=n_boot, seed=seed)})
    return out


def standard_pairs(model: str) -> list[tuple[str, str, str]]:
    """Per hosted model: what the tools add, and what the rules+skills harness adds on top."""
    z, t, h = f"{model}/kaggle_zero_shot", f"{model}/kaggle_tools", f"{model}/kaggle_tools_rules_skills"
    return [(f"{model}: tools vs zero-shot", t, z), (f"{model}: rules+skills vs tools", h, t), (f"{model}: tools+harness vs zero-shot", h, z)]


def markdown_effects(effects: list[dict[str, Any]]) -> str:
    head = "| Comparison | N paired | Exact F1 A | Exact F1 B | Diff [95% CI] | p |\n|---|---:|---:|---:|---:|---:|\n"
    return head + "".join(f"| {e['label']} | {e['n_paired']} | {_f(e['a_value'])} | {_f(e['b_value'])} | {e['diff']:+.3f} [{e['ci_low']:+.3f}, {e['ci_high']:+.3f}] | {e['p_value']:.3f} |\n" for e in effects)
