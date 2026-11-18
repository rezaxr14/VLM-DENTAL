"""Mask metrics: IoU, Dice, one-to-one matching of predicted to ground-truth masks, and pooled precision / recall / F1 at an
IoU threshold. Matching is Hungarian on IoU (same spirit as the findings matcher: every ground-truth mask counts, none twice).

``require_label`` (default True) only allows a pair when the diagnosis AND tooth match, so a mask is credited only for the right
finding; set it False for class-agnostic overlap.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def dice(a: np.ndarray, b: np.ndarray) -> float:
    s = a.sum() + b.sum()
    return float(2.0 * np.logical_and(a, b).sum() / s) if s else 0.0


def match_masks(gt: Sequence[dict[str, Any]], pred: Sequence[dict[str, Any]], threshold: float = 0.5, require_label: bool = True) -> dict[str, Any]:
    """``gt`` / ``pred`` items: ``{"mask": bool array, "fdi": int, "diagnosis": str}``."""
    n_gt, n_pred = len(gt), len(pred)
    ious = np.zeros((n_gt, n_pred))
    for i, g in enumerate(gt):
        for j, p in enumerate(pred):
            if require_label and (g.get("fdi") != p.get("fdi") or g.get("diagnosis") != p.get("diagnosis")):
                continue
            ious[i, j] = iou(g["mask"], p["mask"])
    pairs = []
    if n_gt and n_pred:
        rows, cols = linear_sum_assignment(-ious)
        pairs = [(int(r), int(c), float(ious[r, c])) for r, c in zip(rows, cols) if ious[r, c] >= threshold]
    tp = len(pairs)
    return {"tp": tp, "fp": n_pred - tp, "fn": n_gt - tp, "n_gt": n_gt, "n_pred": n_pred,
            "pairs": [{"gt": r, "pred": c, "iou": v, "dice": dice(gt[r]["mask"], pred[c]["mask"])} for r, c, v in pairs]}


def pooled(results: Sequence[dict[str, Any]]) -> dict[str, float]:
    tp, fp, fn = (sum(r[k] for r in results) for k in ("tp", "fp", "fn"))
    pairs = [p for r in results for p in r["pairs"]]
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": tp / (tp + fp) if tp + fp else 0.0, "recall": tp / (tp + fn) if tp + fn else 0.0,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
            "mean_matched_iou": float(np.mean([p["iou"] for p in pairs])) if pairs else 0.0,
            "mean_matched_dice": float(np.mean([p["dice"] for p in pairs])) if pairs else 0.0}
