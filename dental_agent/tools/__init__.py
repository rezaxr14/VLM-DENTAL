"""
Agent tool suite: zoom/crop, contrast enhancement, FDI numbering,
grounding tools, and registry.
"""

from dental_agent.tools.zoom_crop import tool_zoom_crop, box_out_of_bounds
from dental_agent.tools.contrast import tool_enhance_contrast
from dental_agent.tools.fdi import (
    fdi_encode,
    fdi_decode,
    tool_fdi_label,
    get_anatomical_name,
    flip_quadrant,
    QUADRANT_NAMES,
    TOOTH_NAMES,
)
from dental_agent.tools.grounding import (
    compute_iou,
    tool_locate_abnormal_teeth,
    OracleGroundingTool,
    LearnedGroundingTool,
)
from dental_agent.tools.synthetic import make_synthetic_dental_image
from dental_agent.tools.registry import ToolRegistry, ToolDefinition, register_tool

__all__ = [
    "tool_zoom_crop",
    "box_out_of_bounds",
    "tool_enhance_contrast",
    "fdi_encode",
    "fdi_decode",
    "tool_fdi_label",
    "get_anatomical_name",
    "flip_quadrant",
    "QUADRANT_NAMES",
    "TOOTH_NAMES",
    "compute_iou",
    "tool_locate_abnormal_teeth",
    "OracleGroundingTool",
    "LearnedGroundingTool",
    "make_synthetic_dental_image",
    "ToolRegistry",
    "ToolDefinition",
    "register_tool",
]
