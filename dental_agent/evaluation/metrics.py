"""
Evaluation metrics: diagnostic accuracy, FDI localization, calibration (ECE),
and bootstrap confidence intervals (§18, §21, §27).

Includes:
- Expected Calibration Error (`expected_calibration_error`, `compute_ece`)
- Comprehensive Batch Evaluation Metrics (`compute_evaluation_metrics`, `compute_diagnostic_metrics`)
- Paired bootstrap difference CI (`bootstrap_paired_diff_ci`, `bootstrap_metric_ci`)
"""

from __future__ import annotations

import re
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

    # 1. Deep Caries variations
    if "deep" in s and ("caries" in s or "decay" in s or "carious" in s or "cavity" in s):
        return "Deep Caries"

    # 2. Periapical / Apical Lesions (distinguished before generic lesion/caries)
    if "periapical" in s or "apical" in s or "abscess" in s or "granuloma" in s or "cyst" in s:
        return "Periapical Lesion"
    if "radiolucen" in s and not ("caries" in s or "decay" in s):
        return "Periapical Lesion"

    # 3. Caries / Decay (handles 'carious lesion' and 'caries lesion' properly)
    if "caries" in s or "carious" in s or "decay" in s or "cavity" in s or "demineraliz" in s:
        return "Caries"

    # 4. Impaction
    if "impact" in s or "unerupted" in s or "embedded" in s:
        return "Impacted"

    # 5. Generic lesion fallback
    if "lesion" in s:
        return "Periapical Lesion"

    return str(val).strip().title()


def extract_predicted_findings(parsed_output: Any) -> list[dict[str, Any]]:
    """Extract a normalized list of findings from arbitrary VLM outputs with robust FDI parsing."""
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
        elif any(k in parsed_output for k in ("quadrant", "tooth_position", "diagnosis", "tooth", "fdi")):
            raw_list = [parsed_output]
    
    clean_findings = []
    for item in raw_list:
        if not isinstance(item, dict):
            continue
        
        q, pos = None, None
        
        # 1. Parse quadrant
        q_raw = item.get("quadrant")
        if q_raw is not None:
            m_q = re.search(r'\d+', str(q_raw))
            if m_q:
                try:
                    q = int(m_q.group())
                except ValueError:
                    q = None
                    
        # 2. Parse tooth position
        pos_raw = item.get("tooth_position") or item.get("position") or item.get("tooth_number")
        if pos_raw is not None:
            m_pos = re.search(r'\d+', str(pos_raw))
            if m_pos:
                try:
                    pos = int(m_pos.group())
                except ValueError:
                    pos = None

        # 3. Fallback: Parse 2-digit combined FDI tooth number (e.g. 48, "48", "T48", "#48", "4.8")
        if q is None or pos is None or pos > 8:
            combined_tooth = item.get("tooth") or item.get("fdi") or item.get("tooth_code") or item.get("fdi_number") or item.get("tooth_id")
            if combined_tooth is not None:
                digits = re.findall(r'\d', str(combined_tooth))
                if len(digits) >= 2:
                    cand_q = int(digits[0])
                    cand_pos = int(digits[1])
                    if 1 <= cand_q <= 4 and 1 <= cand_pos <= 8:
                        q, pos = cand_q, cand_pos
                elif len(digits) == 1 and pos is not None and 1 <= int(digits[0]) <= 4:
                    q = int(digits[0])

        # 4. Normalize diagnosis
        diag = (
            item.get("diagnosis")
            or item.get("condition")
            or item.get("pathology")
            or item.get("disease")
            or item.get("finding")
            or item.get("abnormality")
            or item.get("label")
        )
        norm_diag = normalize_dental_diagnosis(diag) if diag else "Unknown"
        
        # 5. Normalize confidence
        conf = None
        if "confidence" in item and item["confidence"] is not None:
            try:
                c_str = str(item["confidence"]).replace("%", "").strip()
                c_val = float(c_str)
                if c_val > 1.0 and c_val <= 100.0:
                    c_val = c_val / 100.0
                conf = max(0.0, min(1.0, c_val))
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


def compute_finding_closeness(
    gt: dict[str, Any],
    pred: dict[str, Any],
) -> tuple[float, float, float]:
    """Compute continuous closeness score (0.0 to 1.0) between a ground truth finding
    and a predicted finding, decomposing into spatial proximity and diagnostic similarity.
    
    Spatial Proximity (0.0 to 1.0):
    - 1.00: Exact Quadrant and Tooth Position
    - 0.75: Same Quadrant, Adjacent Tooth Position (|pos_diff| == 1)
    - 0.40: Same Quadrant, Non-adjacent Position
    - 0.30: Arch symmetry (e.g. Q1 vs Q2, Q3 vs Q4) on same tooth position
    - 0.00: Completely different tooth location
    
    Diagnostic Similarity (0.0 to 1.0):
    - 1.00: Exact diagnosis match (e.g. Impacted == Impacted)
    - 0.75: Clinical progression spectrum (Caries <-> Deep Caries)
    - 0.40: Endodontic sequelae (Caries/Deep Caries <-> Periapical Lesion)
    - 0.00: Unrelated diagnosis (e.g. Impacted <-> Caries)
    
    Composite Closeness:
    - 0.5 * spatial_proximity + 0.5 * diag_similarity
    """
    gt_q = gt.get("quadrant")
    gt_pos = gt.get("tooth_position")
    gt_diag = normalize_dental_diagnosis(gt.get("diagnosis"))
    
    p_q = pred.get("quadrant")
    p_pos = pred.get("tooth_position")
    p_diag = normalize_dental_diagnosis(pred.get("diagnosis"))
    
    # 1. Spatial Proximity
    spatial = 0.0
    if gt_q is not None and gt_pos is not None and p_q is not None and p_pos is not None:
        if gt_q == p_q and gt_pos == p_pos:
            spatial = 1.0
        elif gt_q == p_q and abs(gt_pos - p_pos) == 1:
            spatial = 0.75
        elif (gt_pos == 1 and p_pos == 1) and (
            (gt_q == 1 and p_q == 2) or (gt_q == 2 and p_q == 1) or
            (gt_q == 3 and p_q == 4) or (gt_q == 4 and p_q == 3)
        ):
            # Cross-midline central incisor adjacency (FDI 11 <-> 21, 41 <-> 31)
            spatial = 0.75
        elif gt_q == p_q:
            spatial = 0.40
        elif ((gt_q in (1, 2) and p_q in (1, 2)) or (gt_q in (3, 4) and p_q in (3, 4))) and gt_pos == p_pos:
            spatial = 0.30
        else:
            spatial = 0.0
            
    # 2. Diagnostic Similarity
    diag_sim = 0.0
    if gt_diag and p_diag:
        if gt_diag == p_diag:
            diag_sim = 1.0
        elif {gt_diag, p_diag} == {"Caries", "Deep Caries"}:
            diag_sim = 0.75
        elif (gt_diag in {"Caries", "Deep Caries"} and p_diag == "Periapical Lesion") or (p_diag in {"Caries", "Deep Caries"} and gt_diag == "Periapical Lesion"):
            diag_sim = 0.40
        else:
            diag_sim = 0.0
            
    composite = 0.5 * spatial + 0.5 * diag_sim
    return float(composite), float(spatial), float(diag_sim)


def match_multi_findings(
    ground_truths: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    """Clinically sound set-level matching and continuous closeness scoring
    between ground truth findings and model predictions.
    
    Uses Hungarian Bipartite Maximum Matching (scipy.optimize.linear_sum_assignment)
    to find the globally optimal pairing that maximizes Exact Matches, FDI Localization,
    and Continuous Closeness without greedy ordering bias.
    
    Computes:
    - FDI Localization TP, FP, FN, Precision, Recall, F1
    - Exact Match (Localization + Pathology) TP, FP, FN, Precision, Recall, F1
    - Balanced Continuous Closeness Score (0.0 to 1.0)
    - Detailed matched pairs with closeness metrics
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
            "closeness_score": 1.0, "spatial_proximity": 1.0, "diagnostic_similarity": 1.0,
            "matched_pairs": [],
        }
    
    if n_gt == 0 or n_pred == 0:
        return {
            "gt_count": n_gt, "pred_count": n_pred,
            "fdi_tp": 0, "fdi_fp": n_pred, "fdi_fn": n_gt,
            "fdi_precision": 0.0,
            "fdi_recall": 0.0,
            "fdi_f1": 0.0,
            "exact_tp": 0, "exact_fp": n_pred, "exact_fn": n_gt,
            "exact_precision": 0.0,
            "exact_recall": 0.0,
            "exact_f1": 0.0,
            "closeness_score": 0.0, "spatial_proximity": 0.0, "diagnostic_similarity": 0.0,
            "matched_pairs": [],
        }

    # Build Weight Matrix for Bipartite Matching
    # Priority: Exact Match (weight 1000) > FDI Localization (weight 100) > Continuous Closeness (weight 0-1)
    try:
        from scipy.optimize import linear_sum_assignment
        has_scipy = True
    except ImportError:
        has_scipy = False

    matched_pairs = []
    fdi_tp = 0
    exact_tp = 0

    if has_scipy:
        cost_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
        closeness_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
        spatial_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
        diag_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)

        for i, gt in enumerate(ground_truths):
            gt_q = gt.get("quadrant")
            gt_pos = gt.get("tooth_position")
            gt_diag = normalize_dental_diagnosis(gt.get("diagnosis"))

            for j, pred in enumerate(predictions):
                p_q = pred.get("quadrant")
                p_pos = pred.get("tooth_position")
                p_diag = normalize_dental_diagnosis(pred.get("diagnosis"))

                c, s, d = compute_finding_closeness(gt, pred)
                closeness_matrix[i, j] = c
                spatial_matrix[i, j] = s
                diag_matrix[i, j] = d

                fdi_match = (gt_q is not None and gt_pos is not None and gt_q == p_q and gt_pos == p_pos)
                exact_match = fdi_match and (gt_diag == p_diag)

                weight = (1000.0 if exact_match else (100.0 if fdi_match else 0.0)) + c
                cost_matrix[i, j] = -weight  # Maximize weight by minimizing negative weight

        row_ind, col_ind = linear_sum_assignment(cost_matrix)

        for r, c in zip(row_ind, col_ind):
            gt = ground_truths[r]
            pred = predictions[c]
            gt_q = gt.get("quadrant")
            gt_pos = gt.get("tooth_position")
            gt_diag = normalize_dental_diagnosis(gt.get("diagnosis"))
            p_q = pred.get("quadrant")
            p_pos = pred.get("tooth_position")
            p_diag = normalize_dental_diagnosis(pred.get("diagnosis"))

            fdi_match = (gt_q is not None and gt_pos is not None and gt_q == p_q and gt_pos == p_pos)
            exact_match = fdi_match and (gt_diag == p_diag)

            if fdi_match:
                fdi_tp += 1
            if exact_match:
                exact_tp += 1

            matched_pairs.append({
                "gt": gt,
                "pred": pred,
                "fdi_match": bool(fdi_match),
                "exact_match": bool(exact_match),
                "closeness": float(closeness_matrix[r, c]),
                "spatial": float(spatial_matrix[r, c]),
                "diag_sim": float(diag_matrix[r, c]),
            })
    else:
        # Fallback Greedy 2-pass matcher if scipy is unavailable
        used_preds = set()
        used_gt = set()
        for gt_idx, gt in enumerate(ground_truths):
            gt_q, gt_pos = gt.get("quadrant"), gt.get("tooth_position")
            gt_diag = normalize_dental_diagnosis(gt.get("diagnosis"))
            for pred_idx, pred in enumerate(predictions):
                if pred_idx in used_preds:
                    continue
                p_q, p_pos = pred.get("quadrant"), pred.get("tooth_position")
                p_diag = normalize_dental_diagnosis(pred.get("diagnosis"))
                if gt_q is not None and gt_pos is not None and gt_q == p_q and gt_pos == p_pos and gt_diag == p_diag:
                    used_preds.add(pred_idx)
                    used_gt.add(gt_idx)
                    fdi_tp += 1
                    exact_tp += 1
                    matched_pairs.append({
                        "gt": gt, "pred": pred, "fdi_match": True, "exact_match": True,
                        "closeness": 1.0, "spatial": 1.0, "diag_sim": 1.0,
                    })
                    break

        for gt_idx, gt in enumerate(ground_truths):
            if gt_idx in used_gt:
                continue
            gt_q, gt_pos = gt.get("quadrant"), gt.get("tooth_position")
            for pred_idx, pred in enumerate(predictions):
                if pred_idx in used_preds:
                    continue
                p_q, p_pos = pred.get("quadrant"), pred.get("tooth_position")
                if gt_q is not None and gt_pos is not None and gt_q == p_q and gt_pos == p_pos:
                    used_preds.add(pred_idx)
                    used_gt.add(gt_idx)
                    fdi_tp += 1
                    c_score, s_score, d_score = compute_finding_closeness(gt, pred)
                    matched_pairs.append({
                        "gt": gt, "pred": pred, "fdi_match": True, "exact_match": False,
                        "closeness": c_score, "spatial": s_score, "diag_sim": d_score,
                    })
                    break

    fdi_fp = n_pred - fdi_tp
    fdi_fn = n_gt - fdi_tp
    exact_fp = n_pred - exact_tp
    exact_fn = n_gt - exact_tp

    fdi_p = fdi_tp / max(1, n_pred) if n_pred > 0 else (1.0 if n_gt == 0 else 0.0)
    fdi_r = fdi_tp / max(1, n_gt) if n_gt > 0 else (1.0 if n_pred == 0 else 0.0)
    fdi_f1 = (2 * fdi_p * fdi_r) / (fdi_p + fdi_r) if (fdi_p + fdi_r) > 0 else 0.0

    exact_p = exact_tp / max(1, n_pred) if n_pred > 0 else (1.0 if n_gt == 0 else 0.0)
    exact_r = exact_tp / max(1, n_gt) if n_gt > 0 else (1.0 if n_pred == 0 else 0.0)
    exact_f1 = (2 * exact_p * exact_r) / (exact_p + exact_r) if (exact_p + exact_r) > 0 else 0.0

    # Symmetric Balanced Continuous Closeness:
    # 1. Recall Closeness (GT coverage)
    gt_closeness_list = []
    gt_spatial_list = []
    gt_diag_list = []
    if n_gt > 0 and n_pred > 0:
        for gt in ground_truths:
            best_c, best_s, best_d = 0.0, 0.0, 0.0
            for pred in predictions:
                c, s, d = compute_finding_closeness(gt, pred)
                if c > best_c:
                    best_c, best_s, best_d = c, s, d
            gt_closeness_list.append(best_c)
            gt_spatial_list.append(best_s)
            gt_diag_list.append(best_d)
        recall_closeness = sum(gt_closeness_list) / len(gt_closeness_list)
        recall_spatial = sum(gt_spatial_list) / len(gt_spatial_list)
        recall_diag = sum(gt_diag_list) / len(gt_diag_list)
    else:
        recall_closeness = 1.0 if (n_gt == 0 and n_pred == 0) else 0.0
        recall_spatial = recall_closeness
        recall_diag = recall_closeness

    # 2. Precision Closeness (Prediction accuracy without hallucination bonus)
    pred_closeness_list = []
    if n_pred > 0 and n_gt > 0:
        for pred in predictions:
            best_c = 0.0
            for gt in ground_truths:
                c, _, _ = compute_finding_closeness(gt, pred)
                if c > best_c:
                    best_c = c
            pred_closeness_list.append(best_c)
        precision_closeness = sum(pred_closeness_list) / len(pred_closeness_list)
    else:
        precision_closeness = 1.0 if (n_gt == 0 and n_pred == 0) else 0.0

    # Harmonic/Balanced composite closeness
    if recall_closeness + precision_closeness > 0:
        balanced_closeness = (2 * recall_closeness * precision_closeness) / (recall_closeness + precision_closeness)
    else:
        balanced_closeness = 0.0
    
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
        "closeness_score": balanced_closeness,
        "recall_closeness": recall_closeness,
        "precision_closeness": precision_closeness,
        "spatial_proximity": recall_spatial,
        "diagnostic_similarity": recall_diag,
        "matched_pairs": matched_pairs,
    }


def compute_evaluation_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Core §6 metrics from a results list: FDI accuracy/F1, per-diagnosis F1,
    balanced accuracy, format-compliance rate, mean reward, continuous closeness,
    and confidence calibration (ECE). Robust to both single-finding dicts and multi-finding lists."""
    if not results:
        return {}

    y_true_diag: list[str] = []
    y_pred_diag: list[str] = []
    fdi_correct_count = 0
    exact_correct_count = 0
    format_ok_count = 0
    confidences: list[float] = []
    correctness: list[int] = []
    closeness_scores: list[float] = []

    for r in results:
        gt_raw = r.get("ground_truth", {})
        gt_list = gt_raw if isinstance(gt_raw, list) else ([gt_raw] if isinstance(gt_raw, dict) and gt_raw else [])
        
        ans_raw = r.get("final_answer") or r.get("predictions") or {}
        pred_list = extract_predicted_findings(ans_raw)
        
        # Multi-finding set matching
        m_res = match_multi_findings(gt_list, pred_list)
        fdi_correct_count += int(m_res["fdi_tp"] > 0)
        exact_correct_count += int(m_res["exact_tp"] > 0)
        closeness_scores.append(m_res["closeness_score"])
        
        format_ok_count += int(r.get("format_ok", len(pred_list) > 0))

        for gt in gt_list:
            y_true_diag.append(normalize_dental_diagnosis(gt.get("diagnosis")).lower())
        if pred_list:
            for p in pred_list:
                y_pred_diag.append(normalize_dental_diagnosis(p.get("diagnosis")).lower())
                if p.get("confidence") is not None:
                    try:
                        confidences.append(float(p["confidence"]))
                        # Confidence correctness: was this pred matched exactly?
                        is_corr = any(m["exact_match"] for m in m_res.get("matched_pairs", []) if m.get("pred") == p)
                        correctness.append(int(is_corr))
                    except (ValueError, TypeError):
                        pass
        else:
            y_pred_diag.append("none")

    n = len(results)
    labels = sorted(set(y_true_diag) | set(y_pred_diag))
    
    # Safely compute classification metrics
    if len(y_true_diag) == len(y_pred_diag) and len(y_true_diag) > 0:
        bal_acc = float(balanced_accuracy_score(y_true_diag, y_pred_diag))
        macro_f1 = float(f1_score(y_true_diag, y_pred_diag, average="macro", zero_division=0))
        per_class_f1 = dict(zip(labels, f1_score(y_true_diag, y_pred_diag, labels=labels,
                                                  average=None, zero_division=0).tolist()))
    else:
        bal_acc = float(exact_correct_count / max(1, n))
        macro_f1 = float(exact_correct_count / max(1, n))
        per_class_f1 = {}

    rewards = [r.get("reward", 0.0) for r in results if r.get("reward") is not None]
    mean_reward = float(np.mean(rewards)) if rewards else 0.0

    metrics: dict[str, Any] = {
        "n_examples": n,
        "fdi_accuracy": fdi_correct_count / max(1, n),
        "exact_match_accuracy": exact_correct_count / max(1, n),
        "closeness_score": float(np.mean(closeness_scores)) if closeness_scores else 0.0,
        "diagnosis_balanced_accuracy": bal_acc,
        "diagnosis_per_class_f1": per_class_f1,
        "diagnosis_macro_f1": macro_f1,
        "format_compliance_rate": format_ok_count / max(1, n),
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
    ground_truths: list[dict[str, Any]] | list[list[dict[str, Any]]],
) -> dict[str, float]:
    """Compute comprehensive diagnostic performance across a test cohort."""
    n = len(trajectories)
    if n == 0:
        return {}

    quad_correct = 0
    pos_correct = 0
    fdi_correct = 0
    exact_correct = 0
    format_ok_count = 0
    tool_calls_total = 0
    closeness_list = []

    combined = []
    for traj, gt in zip(trajectories, ground_truths):
        item = dict(traj)
        item["ground_truth"] = gt
        if "reward" not in item:
            item["reward"] = 0.0
        combined.append(item)

        tool_calls_total += traj.get("tool_calls", 0) if isinstance(traj.get("tool_calls"), int) else len(traj.get("tool_calls", []))
        
        gt_list = gt if isinstance(gt, list) else ([gt] if isinstance(gt, dict) and gt else [])
        ans_raw = traj.get("final_answer") or traj.get("predictions")
        pred_list = extract_predicted_findings(ans_raw)
        
        fmt = traj.get("format_ok", False) or len(pred_list) > 0
        if fmt:
            format_ok_count += 1
            
        # Check independent component-level correctness
        if any(p.get("quadrant") is not None and any(g.get("quadrant") == p.get("quadrant") for g in gt_list) for p in pred_list):
            quad_correct += 1
        if any(p.get("tooth_position") is not None and any(g.get("tooth_position") == p.get("tooth_position") for g in gt_list) for p in pred_list):
            pos_correct += 1

        m_res = match_multi_findings(gt_list, pred_list)
        if m_res["fdi_tp"] > 0:
            fdi_correct += 1
        if m_res["exact_tp"] > 0:
            exact_correct += 1
        closeness_list.append(m_res["closeness_score"])

    raw_m = compute_evaluation_metrics(combined)
    return {
        "format_adherence": format_ok_count / max(1, n),
        "quadrant_accuracy": quad_correct / max(1, n),
        "tooth_position_accuracy": pos_correct / max(1, n),
        "fdi_localization_accuracy": fdi_correct / max(1, n),
        "exact_match_accuracy": exact_correct / max(1, n),
        "closeness_score": float(np.mean(closeness_list)) if closeness_list else 0.0,
        "mean_tool_calls": tool_calls_total / max(1, n),
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
