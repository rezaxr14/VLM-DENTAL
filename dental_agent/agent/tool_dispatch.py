"""
Shared tool-dispatch helper used by BOTH agent loops:

  - langgraph_loop.py's trace-gen graph (ground-truth-directed, hinted,
    used to generate SFT demonstrations)
  - loop.py's run_agent (real, unhinted policy rollout, used by GRPO)

Two loops exist because they do genuinely different jobs, not by accident:
langgraph_loop.py needs LangGraph's explicit state-graph/routing structure to
manage retries, parse-error recovery, and ground-truth-conditioned hinting
during one-off trace generation; loop.py's run_agent is a plain, lightweight
for-loop used at GRPO's hot path, where a training step samples group_size
rollouts per image across a batch and graph-compilation overhead isn't worth
paying repeatedly. Keep both -- but they must never each carry their own copy
of "which tools need the image, and which image" again. That exact drift
(loop.py executing against a compounding current_image while langgraph_loop.py
correctly used base_image) was a real train/rollout mismatch, fixed in Phase 1.
This module is the single source of truth going forward; both loops call
execute_tool_call() instead of re-implementing this dispatch themselves.
"""

from __future__ import annotations

from typing import Any
from PIL import Image

# Tools that consume the source image. Always call with the ORIGINAL image
# (never a previously-cropped/compounded one) -- see module docstring.
IMAGE_CONSUMING_TOOLS = {
    "zoom_crop",
    "window_level",
    "denoise",
    "contralateral_compare",
    "locate_tooth",
    "nudge_crop",
}


def execute_tool_call(
    registry: Any,
    tool_name: str,
    tool_args: dict[str, Any],
    image: Image.Image,
) -> Any:
    """Execute one registered tool call with the correct calling convention.

    Tools in IMAGE_CONSUMING_TOOLS receive `image=image` in addition to their
    other args; every other registered tool is called with its args alone.
    `image` should always be the trace/rollout's original source image, not a
    previously-returned crop -- pass base_image, not a running "current view."
    """
    if tool_name in IMAGE_CONSUMING_TOOLS:
        return registry.execute(tool_name, image=image, **tool_args)
    return registry.execute(tool_name, **tool_args)
