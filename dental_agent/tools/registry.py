"""
Tool registry for agent tool registration, dispatch, and system prompt generation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Optional
from PIL import Image

from dental_agent.tools.zoom_crop import tool_zoom_crop
from dental_agent.tools.contrast import tool_enhance_contrast
from dental_agent.tools.fdi import tool_fdi_label

_GLOBAL_REGISTRY: ToolRegistry | None = None


def register_tool(
    name: str,
    func: Callable[..., Any],
    description: str,
    schema: dict[str, Any] | None = None,
) -> None:
    """Module-level helper to register a tool in the global default registry."""
    global _GLOBAL_REGISTRY
    if _GLOBAL_REGISTRY is None:
        _GLOBAL_REGISTRY = ToolRegistry.create_default()
    _GLOBAL_REGISTRY.register(name, func, description, schema)


@dataclass
class ToolDefinition:
    name: str
    description: str
    schema: dict[str, Any]
    func: Callable[..., Any]


class ToolRegistry:
    """Registry of tools available to the diagnostic agent."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(
        self,
        name: str,
        func: Callable[..., Any],
        description: str,
        schema: dict[str, Any] | None = None,
    ) -> None:
        """Register a new tool callable with metadata and argument schema."""
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            schema=schema or {},
            func=func,
        )

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def format_tool_descriptions(self) -> str:
        """Generate human/model-readable documentation of all registered tools."""
        lines = []
        for t in self._tools.values():
            schema_str = json.dumps(t.schema) if t.schema else ""
            lines.append(f"- `{t.name}`: {t.description} {schema_str}".strip())
        return "\n".join(lines)

    def execute(self, name: str, **kwargs: Any) -> Any:
        """Execute a registered tool by name with keyword arguments."""
        tool = self.get(name)
        if not tool:
            raise KeyError(f"Tool '{name}' is not registered in ToolRegistry.")
        return tool.func(**kwargs)

    @classmethod
    def create_default(
        cls,
        grounding_tool: Optional[Callable[[int], list[dict[str, Any]]]] = None,
    ) -> "ToolRegistry":
        """Instantiate a registry populated with default diagnostic tools."""
        registry = cls()
        
        # Late imports to avoid circular dependencies
        from dental_agent.tools.windowing import tool_window_level
        from dental_agent.tools.denoise import tool_denoise
        from dental_agent.tools.contralateral import tool_contralateral_compare
        from dental_agent.tools.grounding import tool_locate_tooth

        # 1. zoom_crop
        registry.register(
            name="zoom_crop",
            func=tool_zoom_crop,
            description="Crops around a bounding box [x, y, w, h] with context padding to provide a zoomed view. padding_frac controls how much context around the box to include (default 0.25 = 25% of box size on each side; use higher for more surrounding anatomy, lower for a tighter view once you're confident in the box).",
            schema={"bbox": [0, 0, 100, 100], "padding_frac": 0.25},
        )

        # 2. window_level
        registry.register(
            name="window_level",
            func=tool_window_level,
            description="Applies medical intensity windowing to reveal specific density structures. preset gives a starting point (bone, enamel, soft_tissue, metal_reduction); pass center and/or width to override the preset's values exactly if a finding needs a differently-placed window.",
            schema={"preset": "bone", "center": None, "width": None},
        )

        # 3. locate_tooth (YOLOv8 grounding) -- moved up from position 7. Locating a
        # tooth is a prerequisite for genuinely inspecting it (not asserting bboxes
        # from nowhere), so it belongs right after basic image prep, not after the
        # more situational tools below.
        registry.register(
            name="locate_tooth",
            func=tool_locate_tooth,
            description="Locates a specific tooth using a trained object detector and returns its bounding box.",
            schema={"tooth": 38},
        )

        # 4. fdi_label -- moved up from position 5, pairs naturally with locate_tooth
        # (convert quadrant/position to the FDI number locate_tooth needs, or back).
        registry.register(
            name="fdi_label",
            func=tool_fdi_label,
            description="Converts quadrant (1-4) and tooth position (1-8) into standard 2-digit FDI label.",
            schema={"quadrant": 1, "tooth_position": 6},
        )

        # 5. denoise
        registry.register(
            name="denoise",
            func=tool_denoise,
            description="Applies edge-preserving noise reduction to distinguish real pathology from sensor grain/noise. strength (0.0-1.0, default 0.6) controls how aggressively to smooth -- low to preserve fine detail, high for visibly grainy crops.",
            schema={"method": "bilateral", "strength": 0.6},
        )

        # 6. contralateral_compare
        registry.register(
            name="contralateral_compare",
            func=tool_contralateral_compare,
            description="Crops a region in one quadrant and its anatomical mirror in the opposite quadrant, returning a side-by-side composite for symmetry comparison.",
            schema={"bbox": [0, 0, 100, 100], "quadrant": 1},
        )

        # 7. enhance_contrast -- was previously unregistered (imported at the top
        # of this file, but never actually added to create_default()'s tool set,
        # same dead-tool pattern locate_abnormal_teeth had). Already had a
        # well-designed continuous `factor` parameter; just needed wiring in.
        registry.register(
            name="enhance_contrast",
            func=tool_enhance_contrast,
            description="Adjusts contrast by a multiplicative factor (default 1.5; >1 increases contrast, <1 decreases it, 1.0 is unchanged). Useful for subtle enamel demineralization or periapical radiolucency that's hard to see at normal contrast.",
            schema={"factor": 1.5},
        )

        # 7. locate_abnormal_teeth (optional grounding backend) -- still last/optional:
        # its backing Faster R-CNN detector has never been trained (no checkpoint
        # exists anywhere on disk), so grounding_tool is never actually passed in by
        # any current call site. Deferred, not wired up -- see trace_generation.py's
        # ToolRegistry.create_default() call sites, all of which still pass none.
        if grounding_tool is not None:
            registry.register(
                name="locate_abnormal_teeth",
                func=grounding_tool,
                description="Specialist detector returning candidate bounding boxes and FDI positions for abnormal teeth.",
                schema={"image_id": 0},
            )

        # 8. nudge_crop -- lets the agent correct a bbox it was already given
        # (from locate_tooth or a prior nudge_crop) when the detector's box
        # doesn't actually center the tooth it asked for. Data-only, like
        # locate_tooth: returns adjusted coordinates, not an image -- pair
        # with zoom_crop on the returned bbox to see the corrected region.
        from dental_agent.tools.nudge import tool_nudge_crop

        registry.register(
            name="nudge_crop",
            func=tool_nudge_crop,
            description="Adjusts a bounding box you were already given (shift + rescale) without re-running detection. Use when a returned crop is off-target.",
            schema={"bbox": [0, 0, 100, 100], "dx_frac": 0.0, "dy_frac": 0.0, "scale": 1.0},
        )

        return registry
