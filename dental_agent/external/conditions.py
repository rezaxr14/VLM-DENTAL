"""The experiment conditions an external model is evaluated under.

``kaggle_zero_shot``                 no tools, no harness: ``NO_TOOLS_SYSTEM_PROMPT`` + ``FIRST_USER_PROMPT`` (one turn).
``kaggle_tools``                     all tools, harness off.
``kaggle_tools_rules_skills``        all tools, ``harness_options("rules_skills_index")``.
``kaggle_zero_shot_legacy_prompt``   optional supplement: ``ZERO_SHOT_PROMPT`` (the older Baseline #1 prompt), no tools.

The image view is always the GPU-profile canonical view (``canonical_resize=True``, ``letterbox=False``), greedy decoding.
"""
from __future__ import annotations

from dataclasses import dataclass

PROVIDER = "kaggle"
IMAGE_VIEW = {"canonical_resize": True, "letterbox": False}
LOOP_DEFAULTS = {"max_tool_calls": 50, "max_new_tokens": 16384, "malformed_retries": 0, "temperature": 0.0}


@dataclass(frozen=True)
class Condition:
    name: str
    with_tools: bool
    harness_mode: str          # 'off' | 'rules_skills_index'
    prompt: str                # 'unified' | 'legacy_zero_shot'
    required: bool = True
    pack: str = "default"      # tool pack (tool_packs.py): which tools the model is given
    evaluator: str = "dentex_findings"   # how a transport record is scored (evaluators.py)

    @property
    def track(self) -> str:
        return "with_tools" if self.with_tools else "no_tools"


CONDITIONS: dict[str, Condition] = {c.name: c for c in (
    Condition("kaggle_zero_shot", False, "off", "unified"),
    Condition("kaggle_tools", True, "off", "unified"),
    Condition("kaggle_tools_rules_skills", True, "rules_skills_index", "unified"),
    Condition("kaggle_zero_shot_legacy_prompt", False, "off", "legacy_zero_shot", required=False),
)}


def register_condition(condition: Condition, replace: bool = False) -> Condition:
    """Add an experiment condition (a new tool pack, a different harness mode, a segmentation task, ...). Existing names are protected."""
    if condition.name in CONDITIONS and not replace:
        raise ValueError(f"condition {condition.name!r} is already registered")
    CONDITIONS[condition.name] = condition
    return condition


def get_condition(name: str) -> Condition:
    try:
        return CONDITIONS[name]
    except KeyError:
        raise ValueError(f"Unknown condition '{name}'. Known: {', '.join(CONDITIONS)}") from None


def required_conditions() -> list[str]:
    return [n for n, c in CONDITIONS.items() if c.required]


def max_turns(max_tool_calls: int = LOOP_DEFAULTS["max_tool_calls"]) -> int:
    """The loop's own default bound (``max_tool_calls + 15``), so a hosted run has the same turn budget as GRPO/eval."""
    return int(max_tool_calls) + 15
