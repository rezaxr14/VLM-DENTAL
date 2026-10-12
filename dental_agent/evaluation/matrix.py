"""Finding-level results of the evaluation matrix: pooled precision/recall/F1, per-class and healthy-image results, and
paired bootstrap comparisons between conditions.

Everything here is computed from the per-image JSONL records that ``scripts/evaluate_models.py`` writes. Each record
stores its ground-truth findings and the model's predicted findings, so the counts are recomputed with
``match_multi_findings`` (Rule 13: every finding, never the first annotation row) instead of trusting stored scores.

Pooled ("micro") metrics add the true positives, predictions and ground-truth findings of all images before dividing.
Worked example, three images:

    image   ground truth   predicted   matched exactly
      A         2             2             1
      B         1             0             0
      C         0 (healthy)   1             0

    TP = 1, predicted = 3, ground truth = 3  ->  precision 1/3, recall 1/3, F1 = 2*TP / (predicted + ground truth) = 2/6.

The healthy image C adds a false positive to the pool, so over-calling on healthy scans lowers precision. Healthy
images are also reported on their own, and the pooled numbers are repeated for pathological images only.

Confidence intervals resample whole IMAGES (all findings of an image move together), because findings of one image are
not independent. A paired comparison between two conditions uses the same resample for both, so image difficulty
cancels out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from dental_agent.data.dev_split import record_key
from dental_agent.evaluation.metrics import match_multi_findings, normalize_dental_diagnosis

LEVELS = ("exact", "fdi")  # exact = right tooth AND right diagnosis; fdi = right tooth only
DIAGNOSES = ("Caries", "Deep Caries", "Periapical Lesion", "Impacted")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Read a per-image JSONL file. When an image appears more than once the LAST record wins (an interrupted run that
    was re-run, or a file that older code appended to), so every image counts once."""
    by_key: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            by_key[record_key(rec)] = rec
    return list(by_key.values())


# ---------------------------------------------------------------------------
# Per-image counts
# ---------------------------------------------------------------------------

def image_counts(rec: dict[str, Any]) -> dict[str, Any]:
    """TP / predicted / ground-truth counts of one image at both levels, plus per-diagnosis exact counts."""
    gt = list(rec.get("ground_truth") or [])
    pred = list(rec.get("predictions") or [])
    m = match_multi_findings(gt, pred)

    per_class = {c: {"gt": 0, "pred": 0, "tp": 0} for c in DIAGNOSES}
    for g in gt:
        per_class.setdefault(normalize_dental_diagnosis(g.get("diagnosis")), {"gt": 0, "pred": 0, "tp": 0})["gt"] += 1
    for p in pred:
        per_class.setdefault(normalize_dental_diagnosis(p.get("diagnosis")), {"gt": 0, "pred": 0, "tp": 0})["pred"] += 1
    for pair in m.get("matched_pairs", []):
        if pair.get("exact_match"):
            per_class.setdefault(normalize_dental_diagnosis(pair["gt"].get("diagnosis")), {"gt": 0, "pred": 0, "tp": 0})["tp"] += 1

    return {
        "key": record_key(rec),
        "n_gt": len(gt),
        "n_pred": len(pred),
        "exact_tp": int(m["exact_tp"]),
        "fdi_tp": int(m["fdi_tp"]),
        # every finding right and nothing extra; on a healthy image that is a well-formed empty answer
        "all_exact": bool(m["exact_tp"] == len(gt) == len(pred) and (len(gt) > 0 or rec.get("format_ok", True))),
        "format_ok": bool(rec.get("format_ok", True)),
        "per_class": per_class,
    }


def counts_for(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [image_counts(r) for r in records]


# ---------------------------------------------------------------------------
# Pooled metrics
# ---------------------------------------------------------------------------

def prf(tp: float, n_pred: float, n_gt: float) -> dict[str, float]:
    """Precision, recall and F1 from pooled counts. A zero denominator gives 0.0 (no predictions = nothing was found)."""
    p = tp / n_pred if n_pred > 0 else 0.0
    r = tp / n_gt if n_gt > 0 else 0.0
    f1 = 2.0 * tp / (n_pred + n_gt) if (n_pred + n_gt) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def pooled_prf(counts: list[dict[str, Any]], level: str = "exact") -> dict[str, float]:
    tp = sum(c[f"{level}_tp"] for c in counts)
    return {**prf(tp, sum(c["n_pred"] for c in counts), sum(c["n_gt"] for c in counts)), "tp": tp,
            "n_pred": sum(c["n_pred"] for c in counts), "n_gt": sum(c["n_gt"] for c in counts)}


def _metric_from_sums(tp: np.ndarray, n_pred: np.ndarray, n_gt: np.ndarray, metric: str) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        if metric == "f1":
            out = np.where(n_pred + n_gt > 0, 2.0 * tp / (n_pred + n_gt), 0.0)
        elif metric == "precision":
            out = np.where(n_pred > 0, tp / n_pred, 0.0)
        elif metric == "recall":
            out = np.where(n_gt > 0, tp / n_gt, 0.0)
        else:
            raise ValueError(f"unknown metric '{metric}'")
    return out


def _arrays(counts: list[dict[str, Any]], level: str) -> np.ndarray:
    return np.array([[c[f"{level}_tp"], c["n_pred"], c["n_gt"]] for c in counts], dtype=np.float64).reshape(-1, 3)


def _percentile_ci(values: np.ndarray, ci: float) -> tuple[float, float]:
    lo, hi = np.percentile(values, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(lo), float(hi)


def bootstrap_pooled(
    counts: list[dict[str, Any]], level: str = "exact", metric: str = "f1",
    n_boot: int = 2000, ci: float = 0.95, seed: int = 0,
) -> dict[str, float]:
    """Point estimate and image-resampling confidence interval of a pooled metric."""
    arr = _arrays(counts, level)
    n = len(arr)
    if n == 0:
        return {"value": 0.0, "ci_low": 0.0, "ci_high": 0.0}
    point = float(_metric_from_sums(*arr.sum(axis=0), metric))
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    sums = arr[idx].sum(axis=1)  # (n_boot, 3)
    boot = _metric_from_sums(sums[:, 0], sums[:, 1], sums[:, 2], metric)
    lo, hi = _percentile_ci(boot, ci)
    return {"value": point, "ci_low": lo, "ci_high": hi}


def paired_bootstrap_diff(
    records_a: list[dict[str, Any]], records_b: list[dict[str, Any]], level: str = "exact", metric: str = "f1",
    n_boot: int = 2000, ci: float = 0.95, seed: int = 0,
) -> dict[str, Any]:
    """Pooled metric of A minus B on the images both conditions cover, with a paired image-resampling CI.

    ``p_value`` is the two-sided bootstrap p-value of "no difference": twice the smaller tail share of the resampled
    differences. Images covered by only one condition are dropped and counted (``n_only_a`` / ``n_only_b``).
    """
    ca = {c["key"]: c for c in counts_for(records_a)}
    cb = {c["key"]: c for c in counts_for(records_b)}
    keys = sorted(set(ca) & set(cb))
    out: dict[str, Any] = {"n_paired": len(keys), "n_only_a": len(set(ca) - set(cb)), "n_only_b": len(set(cb) - set(ca)),
                           "level": level, "metric": metric}
    if not keys:
        return {**out, "a_value": 0.0, "b_value": 0.0, "diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    arr_a, arr_b = _arrays([ca[k] for k in keys], level), _arrays([cb[k] for k in keys], level)
    a = float(_metric_from_sums(*arr_a.sum(axis=0), metric))
    b = float(_metric_from_sums(*arr_b.sum(axis=0), metric))
    idx = np.random.default_rng(seed).integers(0, len(keys), size=(n_boot, len(keys)))
    sa, sb = arr_a[idx].sum(axis=1), arr_b[idx].sum(axis=1)
    diffs = _metric_from_sums(sa[:, 0], sa[:, 1], sa[:, 2], metric) - _metric_from_sums(sb[:, 0], sb[:, 1], sb[:, 2], metric)
    lo, hi = _percentile_ci(diffs, ci)
    p = min(1.0, 2.0 * min(float((diffs <= 0).mean()), float((diffs >= 0).mean())))
    return {**out, "a_value": a, "b_value": b, "diff": a - b, "ci_low": lo, "ci_high": hi, "p_value": p}


# ---------------------------------------------------------------------------
# Per-class and healthy-image results
# ---------------------------------------------------------------------------

def per_class_prf(counts: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Exact-level precision/recall/F1 per diagnosis, pooled over images. A finding is a true positive only when the
    tooth AND the diagnosis are right, so a correct tooth with the wrong diagnosis is a miss for the true class and a
    false positive for the predicted class."""
    classes = list(DIAGNOSES) + sorted({c for x in counts for c in x["per_class"]} - set(DIAGNOSES))
    out = {}
    for cls in classes:
        tp = sum(x["per_class"].get(cls, {}).get("tp", 0) for x in counts)
        n_pred = sum(x["per_class"].get(cls, {}).get("pred", 0) for x in counts)
        n_gt = sum(x["per_class"].get(cls, {}).get("gt", 0) for x in counts)
        if n_pred == 0 and n_gt == 0:
            continue
        out[cls] = {**prf(tp, n_pred, n_gt), "tp": tp, "n_pred": n_pred, "n_gt": n_gt}
    return out


def healthy_report(counts: list[dict[str, Any]]) -> dict[str, float]:
    """Behaviour on images without any annotated finding: how often the model invents findings."""
    healthy = [c for c in counts if c["n_gt"] == 0]
    n = len(healthy)
    fp_images = sum(1 for c in healthy if c["n_pred"] > 0)
    return {
        "n_healthy": n,
        "false_positive_images": fp_images,
        "false_positive_image_rate": fp_images / n if n else 0.0,
        "false_positive_findings": sum(c["n_pred"] for c in healthy),
        "correct_empty_answers": sum(1 for c in healthy if c["n_pred"] == 0 and c["format_ok"]),
        "format_failures": sum(1 for c in healthy if not c["format_ok"]),
    }


# ---------------------------------------------------------------------------
# One matrix cell
# ---------------------------------------------------------------------------

def summarize_cell(records: list[dict[str, Any]], n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    """Everything the paper reports for one (condition, image size) cell."""
    counts = counts_for(records)
    n = len(counts)
    pathological = [c for c in counts if c["n_gt"] > 0]
    summary: dict[str, Any] = {
        "n_images": n,
        "n_healthy": n - len(pathological),
        "n_pathological": len(pathological),
        "format_ok_rate": sum(1 for c in counts if c["format_ok"]) / n if n else 0.0,
        "image_exact_all": sum(1 for c in counts if c["all_exact"]) / n if n else 0.0,
        "mean_turns": float(np.mean([r.get("turns", 0) for r in records])) if records else 0.0,
        "mean_tool_calls": float(np.mean([r.get("tool_calls", 0) for r in records])) if records else 0.0,
        "mean_seconds": float(np.mean([r["seconds"] for r in records if r.get("seconds") is not None]))
        if any(r.get("seconds") is not None for r in records) else None,
        "prompt": sorted({str(r.get("prompt", "unified")) for r in records}),
    }
    for level in LEVELS:
        pooled = pooled_prf(counts, level)
        boot = bootstrap_pooled(counts, level, "f1", n_boot=n_boot, seed=seed)
        summary[level] = {**pooled, "f1_ci_low": boot["ci_low"], "f1_ci_high": boot["ci_high"]}
        summary[f"{level}_pathological_only"] = pooled_prf(pathological, level)
    summary["per_class"] = per_class_prf(counts)
    summary["healthy"] = healthy_report(counts)
    return summary


# ---------------------------------------------------------------------------
# The whole matrix: files -> cells -> comparisons -> tables
# ---------------------------------------------------------------------------

CONDITIONS = (
    "base_no_tools", "base_with_tools",
    "sft_no_tools", "sft_with_tools",
    "grpo_no_tools", "grpo_with_tools",
)
SIZES = ("canonical", "original")  # canonical is the primary result; original is reported alongside it


def cell_name(condition: str, size: str) -> str:
    return f"{condition}@{size}"


def cell_path(output_dir: str | Path, dataset: str, split: str, condition: str, size: str) -> Path:
    """File ``evaluate_models.py`` writes for one cell (original size has no suffix, canonical size has ``_canonical``)."""
    tag = "_canonical" if size == "canonical" else ""
    return Path(output_dir) / f"eval_{dataset}_{split}_{condition}{tag}.jsonl"


def load_matrix(output_dir: str | Path, dataset: str, split: str) -> dict[str, list[dict[str, Any]]]:
    """Every cell that has a result file, keyed ``condition@size``."""
    cells: dict[str, list[dict[str, Any]]] = {}
    for size in SIZES:
        for cond in CONDITIONS:
            path = cell_path(output_dir, dataset, split, cond, size)
            if path.is_file():
                recs = load_records(path)
                if recs:
                    cells[cell_name(cond, size)] = recs
    return cells


def default_comparisons(cells: Iterable[str]) -> list[tuple[str, str, str]]:
    """(label, A, B) pairs, each reported as A minus B on the same images: tools vs no tools, SFT vs base, GRPO vs SFT
    (each with and without tools) at one image size, and canonical vs original size for one condition."""
    present = set(cells)
    pairs: list[tuple[str, str, str]] = []
    for size in SIZES:
        for model in ("base", "sft", "grpo"):
            pairs.append((f"tools vs no tools ({model}, {size})", cell_name(f"{model}_with_tools", size), cell_name(f"{model}_no_tools", size)))
        for tools in ("no_tools", "with_tools"):
            pairs.append((f"SFT vs base ({tools}, {size})", cell_name(f"sft_{tools}", size), cell_name(f"base_{tools}", size)))
            pairs.append((f"GRPO vs SFT ({tools}, {size})", cell_name(f"grpo_{tools}", size), cell_name(f"sft_{tools}", size)))
    for cond in CONDITIONS:
        pairs.append((f"canonical vs original ({cond})", cell_name(cond, "canonical"), cell_name(cond, "original")))
    return [p for p in pairs if p[1] in present and p[2] in present]


def build_summary(cells: dict[str, list[dict[str, Any]]], n_boot: int = 2000, seed: int = 0) -> dict[str, Any]:
    keys = {name: {record_key(r) for r in recs} for name, recs in cells.items()}
    union = set().union(*keys.values()) if keys else set()
    summary: dict[str, Any] = {
        "n_boot": n_boot,
        "seed": seed,
        "cells": {name: summarize_cell(recs, n_boot=n_boot, seed=seed) for name, recs in cells.items()},
        "coverage": {
            "images_in_union": len(union),
            "cells_missing_images": {name: sorted(union - k) for name, k in keys.items() if union - k},
        },
        "comparisons": [],
    }
    for label, a, b in default_comparisons(cells):
        for level in LEVELS:
            res = paired_bootstrap_diff(cells[a], cells[b], level=level, metric="f1", n_boot=n_boot, seed=seed)
            summary["comparisons"].append({"label": label, "a": a, "b": b, **res})
    return summary


def _fmt(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:.{nd}f}"


def summary_markdown(summary: dict[str, Any]) -> str:
    cells = summary["cells"]
    lines = ["# Evaluation matrix", "",
             "Pooled finding-level metrics over all images (see `dental_agent/evaluation/matrix.py`). Exact = right tooth and "
             "right diagnosis; FDI = right tooth. Intervals resample images (95%).", ""]
    for size in SIZES:
        rows = [(n, c) for n, c in cells.items() if n.endswith(f"@{size}")]
        if not rows:
            continue
        lines += [f"## {size.capitalize()} size" + (" (primary)" if size == "canonical" else ""), "",
                  "| condition | images | exact P | exact R | exact F1 [95% CI] | FDI F1 | healthy false-positive rate | format ok | turns | tool calls |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for name, c in rows:
            e, f = c["exact"], c["fdi"]
            note = " *" if any("zero_shot_cached" in p for p in c["prompt"]) else ""
            lines.append(
                f"| {name.split('@')[0]}{note} | {c['n_images']} | {_fmt(e['precision'])} | {_fmt(e['recall'])} | "
                f"{_fmt(e['f1'])} [{_fmt(e['f1_ci_low'])}, {_fmt(e['f1_ci_high'])}] | {_fmt(f['f1'])} | "
                f"{_fmt(c['healthy']['false_positive_image_rate'])} ({c['healthy']['false_positive_images']}/{c['healthy']['n_healthy']}) | "
                f"{_fmt(c['format_ok_rate'])} | {_fmt(c['mean_turns'], 1)} | {_fmt(c['mean_tool_calls'], 1)} |")
        lines.append("")
    if any("zero_shot_cached" in p for c in cells.values() for p in c["prompt"]):
        lines += ["(*) The cached zero-shot run used a different prompt; it is not comparable to the other conditions.", ""]
    miss = summary["coverage"]["cells_missing_images"]
    if miss:
        lines += ["**Coverage warning:** these cells lack images that other cells have, so pooled numbers are not over identical "
                  "images (paired comparisons use the shared images only): " + "; ".join(f"{k} lacks {len(v)}" for k, v in miss.items()), ""]
    exact_cmp = [c for c in summary["comparisons"] if c["level"] == "exact"]
    if exact_cmp:
        lines += ["## Paired comparisons (exact F1, A minus B)", "",
                  "| comparison | paired images | A | B | difference [95% CI] | p |", "|---|---|---|---|---|---|"]
        for c in exact_cmp:
            lines.append(f"| {c['label']} | {c['n_paired']} | {_fmt(c['a_value'])} | {_fmt(c['b_value'])} | "
                         f"{c['diff']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}] | {c['p_value']:.3f} |")
        lines.append("")
    lines += ["## Per-class exact F1", "", "| cell | " + " | ".join(DIAGNOSES) + " |", "|---|" + "---|" * len(DIAGNOSES)]
    for name, c in cells.items():
        lines.append(f"| {name} | " + " | ".join(
            f"{_fmt(c['per_class'][d]['f1'])} ({c['per_class'][d]['tp']}/{c['per_class'][d]['n_gt']})" if d in c["per_class"] else "-"
            for d in DIAGNOSES) + " |")
    lines.append("")
    return "\n".join(lines)


def write_matrix_outputs(summary: dict[str, Any], out_dir: str | Path) -> dict[str, Path]:
    """Write ``matrix_summary.json``, ``matrix_pooled.csv``, ``matrix_per_class.csv``, ``matrix_comparisons.csv`` and
    ``matrix_summary.md`` into ``out_dir``."""
    import csv

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {k: out / v for k, v in {
        "json": "matrix_summary.json", "pooled": "matrix_pooled.csv", "per_class": "matrix_per_class.csv",
        "comparisons": "matrix_comparisons.csv", "markdown": "matrix_summary.md"}.items()}
    paths["json"].write_text(json.dumps(summary, indent=2), encoding="utf-8")

    with open(paths["pooled"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "images", "healthy_images", "level", "precision", "recall", "f1", "f1_ci_low", "f1_ci_high",
                    "pathological_only_f1", "image_exact_all", "format_ok_rate", "healthy_fp_image_rate", "mean_turns",
                    "mean_tool_calls", "mean_seconds"])
        for name, c in summary["cells"].items():
            for level in LEVELS:
                m = c[level]
                w.writerow([name, c["n_images"], c["n_healthy"], level, m["precision"], m["recall"], m["f1"], m["f1_ci_low"],
                            m["f1_ci_high"], c[f"{level}_pathological_only"]["f1"], c["image_exact_all"], c["format_ok_rate"],
                            c["healthy"]["false_positive_image_rate"], c["mean_turns"], c["mean_tool_calls"], c["mean_seconds"]])
    with open(paths["per_class"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cell", "diagnosis", "precision", "recall", "f1", "tp", "n_pred", "n_gt"])
        for name, c in summary["cells"].items():
            for cls, m in c["per_class"].items():
                w.writerow([name, cls, m["precision"], m["recall"], m["f1"], m["tp"], m["n_pred"], m["n_gt"]])
    with open(paths["comparisons"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "a", "b", "level", "n_paired", "a_value", "b_value", "diff", "ci_low", "ci_high", "p_value"])
        for c in summary["comparisons"]:
            w.writerow([c["label"], c["a"], c["b"], c["level"], c["n_paired"], c["a_value"], c["b_value"], c["diff"],
                        c["ci_low"], c["ci_high"], c["p_value"]])
    paths["markdown"].write_text(summary_markdown(summary), encoding="utf-8")
    return paths
