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


def normalize_dental_diagnosis(val: Any) -> str:
    """Normalize various model diagnosis outputs to canonical DENTEX categories:
    'Impacted', 'Caries', 'Periapical Lesion', 'Deep Caries'."""
    if val is None:
        return "Unknown"
    s = str(val).strip().lower()
    if "deep" in s and "caries" in s:
        return "Deep Caries"
    if "caries" in s or "carious" in s or "decay" in s:
        return "Caries"
    if "periapical" in s or "apical" in s or "lesion" in s or "radiolucency" in s:
        return "Periapical Lesion"
    if "impact" in s:
        return "Impacted"
    return str(val).strip().title()


def extract_predicted_findings(parsed_output: Any) -> list[dict[str, Any]]:
    """Extract a normalized list of findings from arbitrary VLM outputs."""
    if parsed_output is None:
        return []
    
    raw_list = []
    if isinstance(parsed_output, list):
        raw_list = parsed_output
    elif isinstance(parsed_output, dict):
        if "findings" in parsed_output and isinstance(parsed_output["findings"], list):
            raw_list = parsed_output["findings"]
        elif "final_answer" in parsed_output:
            fa = parsed_output["final_answer"]
            if isinstance(fa, list):
                raw_list = fa
            elif isinstance(fa, dict):
                raw_list = [fa]
        elif "quadrant" in parsed_output or "tooth_position" in parsed_output or "diagnosis" in parsed_output:
            raw_list = [parsed_output]
    
    clean_findings = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        try:
            q = int(item.get("quadrant")) if item.get("quadrant") is not None else None
            pos = int(item.get("tooth_position")) if item.get("tooth_position") is not None else None
        except (ValueError, TypeError):
            q, pos = None, None
        
        diag = item.get("diagnosis")
        norm_diag = normalize_dental_diagnosis(diag) if diag else "Unknown"
        conf = None
        if "confidence" in item and item["confidence"] is not None:
            try:
                conf = float(item["confidence"])
            except (ValueError, TypeError):
                conf = None
        
        clean_findings.append({
            "quadrant": q,
            "tooth_position": pos,
            "diagnosis": norm_diag,
            "raw_diagnosis": str(diag) if diag else "",
            "confidence": conf,
        })
    return clean_findings


def match_multi_findings(
    ground_truths: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Clinically sound set-level matching between ground truth findings and model predictions.
    
    Computes:
    - FDI Localization TP, FP, FN, Precision, Recall, F1
    - Exact Match (Localization + Pathology) TP, FP, FN, Precision, Recall, F1
    - Detailed matched pairs
    """
    n_gt = len(ground_truths)
    n_pred = len(predictions)
    
    if n_gt == 0 and n_pred == 0:
        return {
            "gt_count": 0, "pred_count": 0,
            "fdi_tp": 0, "fdi_fp": 0, "fdi_fn": 0,
            "fdi_precision": 1.0, "fdi_recall": 1.0, "fdi_f1": 1.0,
            "exact_tp": 0, "exact_fp": 0, "exact_fn": 0,
            "exact_precision": 1.0, "exact_recall": 1.0, "exact_f1": 1.0,
            "matched_pairs": [],
        }
    
    used_preds = set()
    used_gt = set()
    matched_pairs = []
    
    # Pass 1: Match Exact Matches (Both tooth FDI and normalized diagnosis match)
    for gt_idx, gt in enumerate(ground_truths):
        gt_q = gt.get("quadrant")
        gt_pos = gt.get("tooth_position")
        gt_diag = normalize_dental_diagnosis(gt.get("diagnosis"))
        
        for pred_idx, pred in enumerate(predictions):
            if pred_idx in used_preds:
                continue
            p_q = pred.get("quadrant")
            p_pos = pred.get("tooth_position")
            p_diag = normalize_dental_diagnosis(pred.get("diagnosis"))
            
            if gt_q is not None and gt_pos is not None and gt_q == p_q and gt_pos == p_pos and gt_diag == p_diag:
                used_preds.add(pred_idx)
                used_gt.add(gt_idx)
                matched_pairs.append({
                    "gt": gt,
                    "pred": pred,
                    "fdi_match": True,
                    "exact_match": True,
                })
                break
                
    # Pass 2: Match Remaining FDI Localization Matches (Correct tooth, incorrect diagnosis)
    for gt_idx, gt in enumerate(ground_truths):
        if gt_idx in used_gt:
            continue
        gt_q = gt.get("quadrant")
        gt_pos = gt.get("tooth_position")
        
        for pred_idx, pred in enumerate(predictions):
            if pred_idx in used_preds:
                continue
            p_q = pred.get("quadrant")
            p_pos = pred.get("tooth_position")
            
            if gt_q is not None and gt_pos is not None and gt_q == p_q and gt_pos == p_pos:
                used_preds.add(pred_idx)
                used_gt.add(gt_idx)
                matched_pairs.append({
                    "gt": gt,
                    "pred": pred,
                    "fdi_match": True,
                    "exact_match": False,
                })
                break
                
    fdi_tp = len(matched_pairs)
    fdi_fp = n_pred - fdi_tp
    fdi_fn = n_gt - fdi_tp
    
    exact_tp = sum(1 for m in matched_pairs if m["exact_match"])
    exact_fp = n_pred - exact_tp
    exact_fn = n_gt - exact_tp
    
    fdi_p = fdi_tp / max(1, n_pred) if n_pred > 0 else 0.0
    fdi_r = fdi_tp / max(1, n_gt) if n_gt > 0 else 0.0
    fdi_f1 = (2 * fdi_p * fdi_r) / (fdi_p + fdi_r) if (fdi_p + fdi_r) > 0 else 0.0
    
    exact_p = exact_tp / max(1, n_pred) if n_pred > 0 else 0.0
    exact_r = exact_tp / max(1, n_gt) if n_gt > 0 else 0.0
    exact_f1 = (2 * exact_p * exact_r) / (exact_p + exact_r) if (exact_p + exact_r) > 0 else 0.0
    
    return {
        "gt_count": n_gt,
        "pred_count": n_pred,
        "fdi_tp": fdi_tp,
        "fdi_fp": fdi_fp,
        "fdi_fn": fdi_fn,
        "fdi_precision": fdi_p,
        "fdi_recall": fdi_r,
        "fdi_f1": fdi_f1,
        "exact_tp": exact_tp,
        "exact_fp": exact_fp,
        "exact_fn": exact_fn,
        "exact_precision": exact_p,
        "exact_recall": exact_r,
        "exact_f1": exact_f1,
        "matched_pairs": matched_pairs,
    }


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
