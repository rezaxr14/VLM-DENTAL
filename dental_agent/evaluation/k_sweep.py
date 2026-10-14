"""GRPO group-size (K) sweep, read off the dev split: one pooled score per K, paired against the SFT reference and against
the other K, next to what each K cost to train.

Layout on disk (nothing here is chosen for you; every folder and condition is a flag of ``scripts/aggregate_k_sweep.py``):

    <sweep-dir>/k1/eval_dentex_dev_grpo_with_tools_canonical.jsonl      one folder per K, written by
    <sweep-dir>/k2/...                                                  ``run_eval_matrix.py --output-dir <sweep-dir>/k<K>
    <sweep-dir>/k4/...                                                   --conditions grpo_with_tools --sizes canonical``
    <reference-dir>/eval_dentex_dev_sft_with_tools_canonical.jsonl      the SFT reference, from the ordinary dev matrix
    <log template with {k}>                                             ``grpo_training_log.jsonl`` of each K (optional)

Every K is scored on the same dev images with the pooled finding-level F1 of ``matrix.py`` (exact = tooth and diagnosis,
FDI = tooth only; every finding counts, a healthy image has ground truth ``[]``). Intervals resample whole images.

Three comparisons, all paired (same resample for both sides, so image difficulty cancels):

* K against the SFT reference: does GRPO at this K beat the model it started from?
* K against the best K (highest exact F1): is this K distinguishable from the best one? Worked example: K=2, 4, 8 pool to
  exact F1 0.50, 0.58, 0.60 on 70 images; the paired interval of K=4 minus K=8 is [-0.09, +0.05], which contains 0, so K=4
  is not distinguishable from K=8; K=2 minus K=8 is [-0.19, -0.02] and is.
* each K against the next smaller one: does doubling the group still pay?

``within_noise_of_best`` is a fact about those intervals. ``cheapest_within_noise`` names the smallest such K (K sets the
rollouts per image, so the smallest K costs least), but nothing selects it: the choice is the thesis author's and is
reported with the table. Training cost comes from the GRPO step log: rollouts per step are K by construction, hours are the
sum of the logged rollout and update seconds, and the tie rate is the share of groups whose K rewards were all equal
(K >= 2), whose rollouts carried no learning signal (every advantage is zero). Worked example: 40 steps at K=2 with 14 ties
is a tie rate of 0.35, so about a third of the K=2 rollouts taught nothing.

Limit to state in the paper: one training run per K. The intervals describe the sampling of evaluation images, not the
variation between training runs with another seed.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from dental_agent.evaluation import matrix

K_DIR = re.compile(r"^k(\d+)$")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def out_dir_name(dataset: str, split: str, condition: str, size: str) -> str:
    """Folder (inside the sweep folder) that holds the tables of one condition at one image size, so two sizes or two tracks of
    the same sweep never overwrite each other."""
    return f"ksweep_{dataset}_{split}_{condition}_{size}"


def discover_k_dirs(sweep_dir: str | Path) -> dict[int, Path]:
    """``{K: folder}`` for every ``k<K>`` sub-folder of the sweep folder, ascending in K."""
    found = {int(m.group(1)): p for p in Path(sweep_dir).iterdir() if p.is_dir() and (m := K_DIR.match(p.name))}
    return dict(sorted(found.items()))


def load_k_cells(sweep_dir: str | Path, dataset: str, split: str, condition: str, size: str,
                 ks: list[int] | None = None) -> dict[int, list[dict[str, Any]]]:
    """Per-image records of ``condition`` at ``size`` for every K that has a result file (all K folders, or only ``ks``)."""
    out: dict[int, list[dict[str, Any]]] = {}
    for k, folder in discover_k_dirs(sweep_dir).items():
        if ks is not None and k not in ks:
            continue
        path = matrix.cell_path(folder, dataset, split, condition, size)
        if path.is_file():
            recs = matrix.load_records(path)
            if recs:
                out[k] = recs
    return out


def load_reference(reference_dir: str | Path, dataset: str, split: str, condition: str, size: str) -> list[dict[str, Any]]:
    path = matrix.cell_path(reference_dir, dataset, split, condition, size)
    if not path.is_file():
        raise FileNotFoundError(f"reference results not found: {path}")
    return matrix.load_records(path)


def load_training_log(path: str | Path) -> list[dict[str, Any]]:
    """Step records of one GRPO run. A step that appears twice (a resumed run) keeps its last record."""
    by_step: dict[int, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            by_step[int(rec.get("step", i + 1))] = rec
    return [by_step[s] for s in sorted(by_step)]


# ---------------------------------------------------------------------------
# Training cost and signal
# ---------------------------------------------------------------------------

def summarize_training_log(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Cost and learning-signal facts of one run (see the module docstring for the tie rate)."""
    n = len(rows)
    quarter = max(1, n // 4)

    def mean(vals: list[float]) -> float | None:
        return float(np.mean(vals)) if vals else None

    rewards = [float(r["mean_reward"]) for r in rows if r.get("mean_reward") is not None]
    kls = [float(r["kl_divergence"]) for r in rows if r.get("kl_divergence") is not None]
    timed = [r for r in rows if "rollout_s" in r and "update_s" in r]
    with_groups = [r for r in rows if r.get("n_groups")]
    n_groups = sum(int(r["n_groups"]) for r in with_groups)
    return {
        "steps": n,
        "rollouts_per_step": k,
        "rollouts_total": n * k,
        "hours": sum(float(r["rollout_s"]) + float(r["update_s"]) for r in timed) / 3600.0 if timed else None,
        "steps_timed": len(timed),
        "mean_step_s": mean([float(r["rollout_s"]) + float(r["update_s"]) for r in timed]),
        "reward_first_quarter": mean(rewards[:quarter]),
        "reward_last_quarter": mean(rewards[-quarter:]),
        "kl_mean": mean(kls),
        "kl_last_quarter": mean(kls[-quarter:]),
        # a tie needs at least two rollouts; K = 1 has none by definition, and old logs do not carry the counts
        "tie_rate": (sum(int(r.get("n_tied_groups", 0)) for r in with_groups) / n_groups) if (k >= 2 and n_groups) else None,
        "format_ok_rate": mean([float(r["format_ok_rate"]) for r in rows if r.get("format_ok_rate") is not None]),
        "mean_tool_calls": mean([float(r["mean_tool_calls"]) for r in rows if r.get("mean_tool_calls") is not None]),
        "dropped_overlength": sum(int(r.get("n_dropped_overlength", 0)) for r in rows),
    }


# ---------------------------------------------------------------------------
# The sweep summary
# ---------------------------------------------------------------------------

def _diff(a: list[dict[str, Any]], b: list[dict[str, Any]], level: str, n_boot: int, seed: int) -> dict[str, Any]:
    r = matrix.paired_bootstrap_diff(a, b, level=level, metric="f1", n_boot=n_boot, seed=seed)
    return {k: r[k] for k in ("n_paired", "a_value", "b_value", "diff", "ci_low", "ci_high", "p_value")}


def _excludes_zero(d: dict[str, Any]) -> bool:
    return d["ci_low"] > 0 or d["ci_high"] < 0


def build_k_sweep(
    cells: dict[int, list[dict[str, Any]]],
    reference: list[dict[str, Any]] | None = None,
    logs: dict[int, list[dict[str, Any]]] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    levels: tuple[str, ...] = matrix.LEVELS,
) -> dict[str, Any]:
    """Pooled score, paired comparisons and training cost for every K. ``cells`` maps K to its per-image records."""
    ks = sorted(cells)
    if not ks:
        raise ValueError("no K has result records")
    keys = {k: {matrix.record_key(r) for r in cells[k]} for k in ks}
    union = set().union(*keys.values())
    out: dict[str, Any] = {
        "n_boot": n_boot, "seed": seed, "ks": ks, "levels": list(levels),
        "coverage": {"images_in_union": len(union), "k_missing_images": {k: len(union - keys[k]) for k in ks if union - keys[k]}},
        "reference": None, "rows": [], "adjacent": [], "candidates": {}, "training": {},
    }
    if reference:
        out["reference"] = matrix.summarize_cell(reference, n_boot=n_boot, seed=seed)
    summaries = {k: matrix.summarize_cell(cells[k], n_boot=n_boot, seed=seed) for k in ks}

    for level in levels:
        best = max(ks, key=lambda k: (summaries[k][level]["f1"], -k))     # a tie goes to the smaller K
        within = []
        for k in ks:
            s = summaries[k]
            row: dict[str, Any] = {
                "k": k, "level": level, "images": s["n_images"],
                "precision": s[level]["precision"], "recall": s[level]["recall"], "f1": s[level]["f1"],
                "f1_ci_low": s[level]["f1_ci_low"], "f1_ci_high": s[level]["f1_ci_high"],
                "healthy_fp_image_rate": s["healthy"]["false_positive_image_rate"], "format_ok_rate": s["format_ok_rate"],
                "mean_tool_calls": s["mean_tool_calls"], "is_best": k == best,
                "vs_reference": _diff(cells[k], reference, level, n_boot, seed) if reference else None,
                "vs_best": None if k == best else _diff(cells[k], cells[best], level, n_boot, seed),
            }
            row["within_noise_of_best"] = k == best or not _excludes_zero(row["vs_best"])
            if row["within_noise_of_best"]:
                within.append(k)
            out["rows"].append(row)
        out["candidates"][level] = {"best_k": best, "within_noise_of_best": within, "cheapest_within_noise": min(within)}
        for lo, hi in zip(ks, ks[1:]):
            out["adjacent"].append({"level": level, "k": hi, "k_smaller": lo, **_diff(cells[hi], cells[lo], level, n_boot, seed)})
    for k, rows in (logs or {}).items():
        if k in cells and rows:
            out["training"][k] = summarize_training_log(rows, k)
    return out


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def _d(d: dict[str, Any] | None) -> str:
    return "-" if d is None else f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def summary_markdown(summary: dict[str, Any], condition: str = "", reference_label: str = "SFT reference") -> str:
    lines = ["# GRPO group-size sweep (dev split)", "",
             "Pooled finding-level F1 per K on the same dev images (every finding counts, healthy images included); intervals "
             "resample images (95%). Differences are paired. One training run per K: the intervals do not include "
             "training-run variation. See `dental_agent/evaluation/k_sweep.py`.", ""]
    if condition:
        lines += [f"Condition: `{condition}`.", ""]
    miss = summary["coverage"]["k_missing_images"]
    if miss:
        lines += ["**Coverage warning:** these K lack images that other K have (paired comparisons use shared images only): "
                  + "; ".join(f"K={k} lacks {v}" for k, v in miss.items()), ""]
    for level in summary["levels"]:
        rows = [r for r in summary["rows"] if r["level"] == level]
        cand = summary["candidates"][level]
        lines += [f"## {'Exact match (tooth and diagnosis)' if level == 'exact' else 'Tooth identified (FDI)'}", "",
                  f"| K | images | precision | recall | F1 [95% CI] | vs {reference_label} [95% CI] | vs best K (K={cand['best_k']}) [95% CI] | within noise of best |",
                  "|---|---|---|---|---|---|---|---|"]
        if summary["reference"]:
            ref = summary["reference"][level]
            lines.append(f"| {reference_label} | {summary['reference']['n_images']} | {_f(ref['precision'])} | {_f(ref['recall'])} | "
                         f"{_f(ref['f1'])} [{_f(ref['f1_ci_low'])}, {_f(ref['f1_ci_high'])}] | - | - | - |")
        for r in rows:
            lines.append(f"| {r['k']}{' (best)' if r['is_best'] else ''} | {r['images']} | {_f(r['precision'])} | {_f(r['recall'])} | "
                         f"{_f(r['f1'])} [{_f(r['f1_ci_low'])}, {_f(r['f1_ci_high'])}] | {_d(r['vs_reference'])} | "
                         f"{_d(r['vs_best'])} | {'yes' if r['within_noise_of_best'] else 'no'} |")
        lines += ["", f"Not distinguishable from the best K (paired interval contains 0): K = {cand['within_noise_of_best']}; "
                      f"the smallest, and so cheapest, of them is K = {cand['cheapest_within_noise']}. Reported, not applied.", ""]
        adj = [a for a in summary["adjacent"] if a["level"] == level]
        if adj:
            lines += ["Doubling check (each K minus the next smaller K):", "",
                      "| K | minus K | difference [95% CI] | p |", "|---|---|---|---|"]
            for a in adj:
                lines.append(f"| {a['k']} | {a['k_smaller']} | {a['diff']:+.3f} [{a['ci_low']:+.3f}, {a['ci_high']:+.3f}] | {a['p_value']:.3f} |")
            lines.append("")
    if summary["training"]:
        lines += ["## Training cost and learning signal", "",
                  "| K | steps | rollouts per step | hours | reward first quarter | reward last quarter | KL last quarter | tie rate | format ok | tool calls | dropped over-length |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, t in summary["training"].items():
            lines.append(f"| {k} | {t['steps']} | {t['rollouts_per_step']} | {_f(t['hours'], 1)} | {_f(t['reward_first_quarter'])} | "
                         f"{_f(t['reward_last_quarter'])} | {_f(t['kl_last_quarter'], 4)} | {_f(t['tie_rate'])} | "
                         f"{_f(t['format_ok_rate'])} | {_f(t['mean_tool_calls'], 1)} | {t['dropped_overlength']} |")
        lines += ["", "Tie rate = share of groups whose K rewards were all equal (K >= 2): their rollouts gave no gradient. "
                      "n/a for K = 1 (no group to tie) and for logs written before the counts existed.", ""]
    return "\n".join(lines)


def write_k_sweep_outputs(summary: dict[str, Any], out_dir: str | Path, condition: str = "", reference_label: str = "SFT reference") -> dict[str, Path]:
    """Write ``k_sweep_summary.json``, ``k_sweep_table.csv`` (one row per K and level), ``k_sweep_adjacent.csv``,
    ``k_sweep_training.csv`` and ``k_sweep_summary.md`` into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / v for k, v in {"json": "k_sweep_summary.json", "table": "k_sweep_table.csv", "adjacent": "k_sweep_adjacent.csv",
                                     "training": "k_sweep_training.csv", "markdown": "k_sweep_summary.md"}.items()}
    paths["json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    ref = summary.get("reference")
    with open(paths["table"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k", "level", "images", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high", "healthy_fp_image_rate",
                    "format_ok_rate", "mean_tool_calls", "is_best", "ref_f1", "ref_diff", "ref_ci_low", "ref_ci_high", "ref_p_value",
                    "best_diff", "best_ci_low", "best_ci_high", "best_p_value", "within_noise_of_best"])
        for r in summary["rows"]:
            vr, vb = r["vs_reference"], r["vs_best"]
            w.writerow([r["k"], r["level"], r["images"], r["precision"], r["recall"], r["f1"], r["f1_ci_low"], r["f1_ci_high"],
                        r["healthy_fp_image_rate"], r["format_ok_rate"], r["mean_tool_calls"], int(r["is_best"]),
                        ref[r["level"]]["f1"] if ref else "", *( [vr["diff"], vr["ci_low"], vr["ci_high"], vr["p_value"]] if vr else [""] * 4),
                        *([vb["diff"], vb["ci_low"], vb["ci_high"], vb["p_value"]] if vb else [""] * 4), int(r["within_noise_of_best"])])
    with open(paths["adjacent"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["level", "k", "k_smaller", "n_paired", "k_f1", "k_smaller_f1", "diff", "ci_low", "ci_high", "p_value"])
        for a in summary["adjacent"]:
            w.writerow([a["level"], a["k"], a["k_smaller"], a["n_paired"], a["a_value"], a["b_value"], a["diff"], a["ci_low"], a["ci_high"], a["p_value"]])
    fields = ["steps", "rollouts_per_step", "rollouts_total", "hours", "steps_timed", "mean_step_s", "reward_first_quarter",
              "reward_last_quarter", "kl_mean", "kl_last_quarter", "tie_rate", "format_ok_rate", "mean_tool_calls", "dropped_overlength"]
    with open(paths["training"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["k"] + fields)
        for k, t in summary["training"].items():
            w.writerow([k] + [t[x] for x in fields])
    paths["markdown"].write_text(summary_markdown(summary, condition, reference_label), encoding="utf-8")
    return paths
