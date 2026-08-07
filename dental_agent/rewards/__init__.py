"""
Multi-objective GRPO reward functions for dental diagnostic agents (§5.5, §22).
"""

from dental_agent.rewards.components import (
    reward_format,
    reward_tool_validity,
    reward_efficiency,
    reward_accuracy,
)
from dental_agent.rewards.composite import combine_reward
from dental_agent.rewards.judge import reward_judge, evaluate_reasoning_grounding

__all__ = [
    "reward_format",
    "reward_tool_validity",
    "reward_efficiency",
    "reward_accuracy",
    "combine_reward",
    "reward_judge",
    "evaluate_reasoning_grounding",
]
