"""
Failure analysis and error taxonomy logger (§28, §29).

Includes:
- Single-case classifier (`categorize_failure`)
- Breakdown statistics generator (`failure_mode_breakdown`)
- Multi-category grouping (`categorize_failures`)
- JSON log exporter (`log_failure_cases`)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import pandas as pd


def categorize_failure(result: dict[str, Any]) -> str:
    """Tags one evaluation result into a standardized failure category."""
    if not result.get("format_ok"):
        return "format_failure"
    ans = result.get("final_answer") or {}
    gt = result.get("ground_truth", {})
    quad_ok = ans.get("quadrant") == gt.get("quadrant")
    tooth_ok = ans.get("tooth_position") == gt.get("tooth_position")
    diag_ok = str(ans.get("diagnosis", "")).lower() == str(gt.get("diagnosis", "")).lower()

    if quad_ok and tooth_ok and diag_ok:
        return "correct"
    if quad_ok and tooth_ok and not diag_ok:
        return "wrong_diagnosis_right_location"
    if not (quad_ok and tooth_ok) and diag_ok:
        return "right_diagnosis_wrong_location"
    return "wrong_location_and_diagnosis"


def failure_mode_breakdown(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Counts + fraction per category across a results list."""
    if not results:
        return pd.DataFrame()
    categories = [categorize_failure(r) for r in results]
    counts = pd.Series(categories).value_counts()
    breakdown = pd.DataFrame({"count": counts, "fraction": counts / len(categories)})
    print(breakdown)
    return breakdown


def categorize_failures(
    trajectories: list[dict[str, Any]],
    ground_truths: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Sort trajectories into mutually exclusive diagnostic failure categories."""
    categories: dict[str, list[dict[str, Any]]] = {
        "success": [],
        "format_failure": [],
        "tool_error": [],
        "fdi_localization_error": [],
        "pathology_misclassification": [],
        "early_termination": [],
    }

    for traj, gt in zip(trajectories, ground_truths):
        final = traj.get("final_answer")
        img_id = traj.get("image_id")
        record = {"image_id": img_id, "trajectory": traj, "ground_truth": gt}

        if not traj.get("format_ok") and not (isinstance(final, dict) and "diagnosis" in final):
            categories["format_failure"].append(record)
            continue

        if not isinstance(final, dict):
            categories["early_termination"].append(record)
            continue

        tool_turns = [t for t in traj.get("turns", []) if t.get("tool_name")]
        has_failed_tools = any(not t.get("tool_ok", True) for t in tool_turns)
        if has_failed_tools:
            categories["tool_error"].append(record)

        q_ok = int(final.get("quadrant") == gt.get("quadrant")) if gt.get("quadrant") is not None else 0
        p_ok = int(final.get("tooth_position") == gt.get("tooth_position")) if gt.get("tooth_position") is not None else 0
        fdi_ok = bool(q_ok and p_ok)

        gt_d = str(gt.get("diagnosis", "")).strip().lower()
        pr_d = str(final.get("diagnosis", "")).strip().lower()
        diag_ok = bool(gt_d == pr_d) if gt_d else False

        if not fdi_ok:
            categories["fdi_localization_error"].append(record)
        elif not diag_ok:
            categories["pathology_misclassification"].append(record)
        else:
            categories["success"].append(record)

    return categories


def log_failure_cases(
    failures: dict[str, list[dict[str, Any]]],
    output_json: str | Path = "experiments/failure_log.json",
) -> None:
    """Save structured failure analysis taxonomy report."""
    summary = {
        category: len(records) for category, records in failures.items()
    }
    print("\n--- Failure Analysis Taxonomy Breakdown ---")
    for cat, count in summary.items():
        print(f"  {cat:<30}: {count}")

    out_p = Path(output_json)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        json.dump(failures, f, indent=2, default=str)
    print(f"Detailed failure logs written to {out_p}")
