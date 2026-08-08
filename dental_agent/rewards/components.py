"""
Individual reward component functions for dental agent evaluation and GRPO training (§5.5).
"""

from __future__ import annotations

from typing import Any, Mapping


def reward_format(trajectory: Mapping[str, Any]) -> float:
    """Format adherence reward (R_format in §5.5).

    Returns 1.0 if the agent produced a valid JSON object matching the expected schema,
    0.0 otherwise.
    """
    if trajectory.get("format_ok"):
        return 1.0
    ans = trajectory.get("final_answer")
    if isinstance(ans, dict) and "diagnosis" in ans:
        return 1.0
    return 0.0


def reward_tool_validity(trajectory: Mapping[str, Any]) -> float:
    """Tool-call validity reward (R_tool in §5.5).

    Returns 1.0 if no tools were called, or the fraction of attempted tool calls
    that had valid arguments and succeeded without error.
    """
    turns = trajectory.get("turns", [])
    tool_turns = [t for t in turns if isinstance(t, dict) and t.get("parsed", {}).get("tool")]
    if not tool_turns:
        return 1.0
    valid_count = sum(1 for t in tool_turns if t.get("tool_ok", False))
    return float(valid_count / len(tool_turns))


def reward_efficiency(trajectory: Mapping[str, Any], max_calls: int = 4) -> float:
    """Tool efficiency reward (R_efficiency in §5.5).

    Rewards succinct, non-redundant tool usage. Starts at 1.0 and deducts specific
    calibrated penalties for each tool invoked. Heavy computational tools or 
    redundant calls are penalized more.
    """
    turns = trajectory.get("turns", [])
    tool_turns = [t for t in turns if isinstance(t, dict) and t.get("parsed", {}).get("tool")]
    
    if not tool_turns:
        return 1.0
        
    TOOL_PENALTIES = {
        "zoom_crop": 0.05,
        "window_level": 0.05,
        "denoise": 0.05,
        "contralateral_compare": 0.08,
        "fdi_label": 0.02,
        "locate_abnormal_teeth": 0.10,
    }
    
    score = 1.0
    for t in tool_turns:
        tool_name = t.get("parsed", {}).get("tool", "")
        penalty = TOOL_PENALTIES.get(tool_name, 0.05)
        score -= penalty
        
    # Cap between 0.0 and 1.0
    return max(0.0, min(1.0, score))


def reward_accuracy(
    trajectory: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
) -> float:
    """Graded diagnostic accuracy reward (R_accuracy in §5.5).

    Graded breakdown:
    - +0.25: Correct dental quadrant (1-4)
    - +0.25: Correct tooth position (1-8)  -> total +0.50 for exact FDI tooth localization
    - +0.50: Correct pathology diagnosis class (Caries, Deep Caries, Periapical Lesion, Impacted Tooth)
    - Total: 1.0 for perfect FDI localization + disease diagnosis.
    """
    ans = trajectory.get("final_answer")
    if not isinstance(ans, dict):
        return 0.0

    score = 0.0
    gt_quad = ground_truth.get("quadrant")
    gt_pos = ground_truth.get("tooth_position")
    gt_diag = str(ground_truth.get("diagnosis", "")).strip().lower()

    # 1. Quadrant check (+0.25)
    pred_quad = ans.get("quadrant")
    if pred_quad is not None and gt_quad is not None and int(pred_quad) == int(gt_quad):
        score += 0.25

    # 2. Tooth position check (+0.25)
    pred_pos = ans.get("tooth_position")
    if pred_pos is not None and gt_pos is not None and int(pred_pos) == int(gt_pos):
        score += 0.25

    # 3. Pathology diagnosis check (+0.50)
    pred_diag = str(ans.get("diagnosis", "")).strip().lower()
    if pred_diag and gt_diag and pred_diag == gt_diag:
        score += 0.50

    return score
