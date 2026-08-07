"""
Evaluation metrics: diagnostic accuracy, FDI localization, calibration (ECE),
and bootstrap confidence intervals (§18, §21, §27).

Includes:
- Expected Calibration Error (`expected_calibration_error`, `compute_ece`)
- Comprehensive Batch Evaluation Metrics (`compute_evaluation_metrics`, `compute_diagnostic_metrics`)
- Paired bootstrap difference CI (`bootstrap_paired_diff_ci`, `bootstrap_metric_ci`)
"""

from __future__ import annotations

from typing import Any, Callable, Sequence
import numpy as np
from sklearn.metrics import f1_score, balanced_accuracy_score


def expected_calibration_error(
    confidences: Sequence[float],
    correctness: Sequence[bool | int | float],
    n_bins: int = 10,
) -> float:
    """Standard ECE: bin predictions by stated confidence, compare each bin's mean
    confidence to its actual accuracy, weight by bin size."""
    conf_arr = np.array(confidences, dtype=np.float32)
    corr_arr = np.array(correctness, dtype=np.float32)
    if len(conf_arr) == 0:
        return 0.0

    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (conf_arr > lo) & (conf_arr <= hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / len(conf_arr)) * abs(conf_arr[mask].mean() - corr_arr[mask].mean())
    return float(ece)


# Alias
compute_ece = expected_calibration_error


def compute_evaluation_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Core §6 metrics from a run_agent_batch() results list: FDI (quadrant + tooth
    position) accuracy, per-diagnosis F1, balanced accuracy, format-compliance rate,
    mean reward, and confidence calibration (ECE) when confidence scores were reported."""
    if not results:
        return {}

    y_true_diag: list[str] = []
    y_pred_diag: list[str] = []
    fdi_correct = 0
    format_ok_count = 0
    confidences: list[float] = []
    correctness: list[int] = []

    for r in results:
        gt = r.get("ground_truth", {})
        ans = r.get("final_answer") or {}
        y_true_diag.append(str(gt.get("diagnosis", "")).lower())
        y_pred_diag.append(str(ans.get("diagnosis", "")).lower() if ans else "none")

        quad_ok = ans.get("quadrant") == gt.get("quadrant")
        tooth_ok = ans.get("tooth_position") == gt.get("tooth_position")
        fdi_correct += int(quad_ok and tooth_ok)
        format_ok_count += int(r.get("format_ok", False))

        if ans and "confidence" in ans and ans["confidence"] is not None:
            diag_ok = str(ans.get("diagnosis", "")).lower() == str(gt.get("diagnosis", "")).lower()
            try:
                confidences.append(float(ans["confidence"]))
                correctness.append(int(quad_ok and tooth_ok and diag_ok))
            except (ValueError, TypeError):
                pass

    n = len(results)
    labels = sorted(set(y_true_diag) | set(y_pred_diag))
    per_class_f1 = dict(zip(labels, f1_score(y_true_diag, y_pred_diag, labels=labels,
                                              average=None, zero_division=0).tolist()))

    rewards = [r.get("reward", 0.0) for r in results]
    mean_reward = float(np.mean(rewards)) if rewards else 0.0

    metrics: dict[str, Any] = {
        "n_examples": n,
        "fdi_accuracy": fdi_correct / n,
        "diagnosis_balanced_accuracy": float(balanced_accuracy_score(y_true_diag, y_pred_diag)),
        "diagnosis_per_class_f1": per_class_f1,
        "diagnosis_macro_f1": float(f1_score(y_true_diag, y_pred_diag, average="macro", zero_division=0)),
        "format_compliance_rate": format_ok_count / n,
        "mean_reward": mean_reward,
        "expected_calibration_error": (
            expected_calibration_error(confidences, correctness) if len(confidences) >= 5 else None
        ),
    }
    if len(confidences) < 5:
        metrics["_note"] = "fewer than 5 confidence scores available — ECE skipped"
    return metrics


def compute_diagnostic_metrics(
    trajectories: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute comprehensive diagnostic performance across a test cohort (legacy interface)."""
    n = len(trajectories)
    if n == 0:
        return {}

    quad_correct = 0
    pos_correct = 0
    fdi_correct = 0
    exact_correct = 0
    format_ok_count = 0
    tool_calls_total = 0

    combined = []
    for traj, gt in zip(trajectories, ground_truths):
        item = dict(traj)
        item["ground_truth"] = gt
        if "reward" not in item:
            item["reward"] = 0.0
        combined.append(item)

        tool_calls_total += traj.get("tool_calls", 0)
        ans = traj.get("final_answer")
        fmt = traj.get("format_ok", False) and isinstance(ans, dict)
        if fmt:
            format_ok_count += 1
            q_ok = ans.get("quadrant") == gt.get("quadrant") if gt.get("quadrant") is not None else False
            p_ok = ans.get("tooth_position") == gt.get("tooth_position") if gt.get("tooth_position") is not None else False
            d_ok = str(ans.get("diagnosis", "")).strip().lower() == str(gt.get("diagnosis", "")).strip().lower() if gt.get("diagnosis") else False

            if q_ok:
                quad_correct += 1
            if p_ok:
                pos_correct += 1
            if q_ok and p_ok:
                fdi_correct += 1
            if q_ok and p_ok and d_ok:
                exact_correct += 1

    raw_m = compute_evaluation_metrics(combined)
    return {
        "format_adherence": format_ok_count / n,
        "quadrant_accuracy": quad_correct / n,
        "tooth_position_accuracy": pos_correct / n,
        "fdi_localization_accuracy": fdi_correct / n,
        "exact_match_accuracy": exact_correct / n,
        "mean_tool_calls": tool_calls_total / n,
        "pathology_accuracy": raw_m.get("diagnosis_balanced_accuracy", 0.0),
        "pathology_macro_f1": raw_m.get("diagnosis_macro_f1", 0.0),
        "mean_reward": raw_m.get("mean_reward", 0.0),
        "total_samples": float(n),
    }


def bootstrap_paired_diff_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """95% bootstrap CI for the mean paired difference (a - b), e.g. reward with tools
    minus reward without tools, matched by image."""
    rng = np.random.default_rng(seed)
    diffs = np.array(values_a, dtype=np.float64) - np.array(values_b, dtype=np.float64)
    if len(diffs) == 0:
        return 0.0, (0.0, 0.0)

    boot_means = [float(rng.choice(diffs, size=len(diffs), replace=True).mean()) for _ in range(n_boot)]
    lo, hi = np.percentile(boot_means, [(1 - ci) / 2 * 100, (1 + ci) / 2 * 100])
    return float(diffs.mean()), (float(lo), float(hi))


def bootstrap_metric_ci(
    data: Sequence[Any],
    metric_fn: Callable[[Sequence[Any]], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute point estimate and non-parametric bootstrap confidence interval (low, high)."""
    rng = np.random.default_rng(seed)
    n = len(data)
    if n == 0:
        return 0.0, 0.0, 0.0

    point_estimate = float(metric_fn(data))
    boot_stats = []

    for _ in range(n_bootstrap):
        resample_indices = rng.integers(0, n, size=n)
        sample = [data[i] for i in resample_indices]
        boot_stats.append(metric_fn(sample))

    alpha = (1.0 - ci) / 2.0
    low = float(np.percentile(boot_stats, alpha * 100))
    high = float(np.percentile(boot_stats, (1.0 - alpha) * 100))

    return point_estimate, low, high
