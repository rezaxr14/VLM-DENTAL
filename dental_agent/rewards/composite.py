"""
Composite GRPO reward aggregator (§5.5).
"""

from __future__ import annotations

from typing import Any, Mapping

from dental_agent.config import RewardWeights
from dental_agent.rewards.components import (
    reward_format,
    reward_tool_validity,
    reward_efficiency,
    reward_accuracy,
)


def combine_reward(
    trajectory: Mapping[str, Any],
    ground_truth: Mapping[str, Any],
    weights: RewardWeights | Mapping[str, float] | None = None,
    max_tool_calls: int = 50,
) -> tuple[float, dict[str, float]]:
    """Compute total composite reward R_total = sum(w_i * R_i).

    Parameters
    ----------
    weights : RewardWeights or dict, optional
        Weight mapping with keys: 'acc' (or 'accuracy'), 'fmt' (or 'format'),
        'tool' (or 'tool_validity'), 'eff' (or 'efficiency').
        Defaults to proposal §5.5 values (acc=1.0, fmt=0.2, tool=0.2, eff=0.1).

    Returns
    -------
    total_reward : float
        Scalar reward for GRPO advantage computation.
    components : dict[str, float]
        Breakdown of raw component values.
    """
    if isinstance(weights, RewardWeights):
        w_acc = weights.accuracy
        w_fmt = weights.format
        w_tool = weights.tool_validity
        w_eff = weights.efficiency
    elif isinstance(weights, dict):
        w_acc = weights.get("accuracy", weights.get("acc", 1.0))
        w_fmt = weights.get("format", weights.get("fmt", 0.2))
        w_tool = weights.get("tool_validity", weights.get("tool", 0.2))
        w_eff = weights.get("efficiency", weights.get("eff", 0.1))
    else:
        w_acc, w_fmt, w_tool, w_eff = 1.0, 0.2, 0.2, 0.1

    r_acc = reward_accuracy(trajectory, ground_truth)
    r_fmt = reward_format(trajectory)
    r_tool = reward_tool_validity(trajectory)
    r_eff = reward_efficiency(trajectory, max_calls=max_tool_calls)

    total = (w_acc * r_acc) + (w_fmt * r_fmt) + (w_tool * r_tool) + (w_eff * r_eff)

    components = {
        "accuracy": r_acc,
        "format": r_fmt,
        "tool_validity": r_tool,
        "efficiency": r_eff,
    }
    return float(total), components
