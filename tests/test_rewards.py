"""
Unit tests for reward calculations and composite GRPO aggregation.
"""

from dental_agent.rewards.components import (
    reward_format,
    reward_tool_validity,
    reward_efficiency,
    reward_accuracy,
)
from dental_agent.rewards.composite import combine_reward


def test_reward_format() -> None:
    assert reward_format({"format_ok": True}) == 1.0
    assert reward_format({"format_ok": False, "final_answer": {"diagnosis": "Caries"}}) == 1.0
    assert reward_format({"format_ok": False, "final_answer": None}) == 0.0


def test_reward_tool_validity() -> None:
    # No tool turns -> 1.0
    assert reward_tool_validity({"turns": []}) == 1.0

    # 1 valid tool turn -> 1.0
    traj_valid = {
        "turns": [
            {"parsed": {"tool": "zoom_crop"}, "tool_ok": True}
        ]
    }
    assert reward_tool_validity(traj_valid) == 1.0

    # 1 valid, 1 invalid -> 0.5
    traj_mixed = {
        "turns": [
            {"parsed": {"tool": "zoom_crop"}, "tool_ok": True},
            {"parsed": {"tool": "fake_tool"}, "tool_ok": False},
        ]
    }
    assert reward_tool_validity(traj_mixed) == 0.5


def test_reward_efficiency() -> None:
    assert reward_efficiency({"tool_calls": 0}, max_calls=4) == 1.0
    assert reward_efficiency({"tool_calls": 2}, max_calls=4) == 0.5
    assert reward_efficiency({"tool_calls": 4}, max_calls=4) == 0.0
    assert reward_efficiency({"tool_calls": 6}, max_calls=4) == 0.0


def test_reward_accuracy() -> None:
    gt = {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}

    # Perfect match -> 1.0
    ans_perfect = {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}
    assert reward_accuracy({"final_answer": ans_perfect}, gt) == 1.0

    # Right location, wrong diagnosis -> 0.50
    ans_loc_only = {"quadrant": 1, "tooth_position": 6, "diagnosis": "Impacted Tooth"}
    assert reward_accuracy({"final_answer": ans_loc_only}, gt) == 0.50

    # Right diagnosis, wrong location -> 0.50
    ans_diag_only = {"quadrant": 3, "tooth_position": 8, "diagnosis": "Caries"}
    assert reward_accuracy({"final_answer": ans_diag_only}, gt) == 0.50

    # Right quadrant only -> 0.25
    ans_quad_only = {"quadrant": 1, "tooth_position": 2, "diagnosis": "Periapical Lesion"}
    assert reward_accuracy({"final_answer": ans_quad_only}, gt) == 0.25

    # Completely wrong -> 0.0
    ans_wrong = {"quadrant": 4, "tooth_position": 1, "diagnosis": "Periapical Lesion"}
    assert reward_accuracy({"final_answer": ans_wrong}, gt) == 0.0


def test_combine_reward() -> None:
    gt = {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}
    traj = {
        "final_answer": {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"},
        "format_ok": True,
        "tool_calls": 1,
        "turns": [{"parsed": {"tool": "zoom_crop"}, "tool_ok": True}],
    }

    total, components = combine_reward(traj, gt, weights={"acc": 1.0, "fmt": 0.2, "tool": 0.2, "eff": 0.1})
    assert components["accuracy"] == 1.0
    assert components["format"] == 1.0
    assert components["tool_validity"] == 1.0
    assert components["efficiency"] == 0.75  # 1 - 1/4 = 0.75

    # 1.0*1.0 + 0.2*1.0 + 0.2*1.0 + 0.1*0.75 = 1.0 + 0.2 + 0.2 + 0.075 = 1.475
    assert abs(total - 1.475) < 1e-6
