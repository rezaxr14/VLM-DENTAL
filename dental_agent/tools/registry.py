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
            description="Crops around a bounding box [x, y, w, h] with context padding to provide a zoomed view.",
            schema={"bbox": [0, 0, 100, 100]},
        )

        # 2. window_level
        registry.register(
            name="window_level",
            func=tool_window_level,
            description="Applies medical intensity windowing to reveal specific density structures (e.g. bone, enamel, soft_tissue, metal_reduction).",
            schema={"preset": "bone"},
        )
        
        # 3. denoise
        registry.register(
            name="denoise",
            func=tool_denoise,
            description="Applies edge-preserving noise reduction to distinguish real pathology from sensor grain/noise.",
            schema={"method": "bilateral"},
        )
        
        # 4. contralateral_compare
        registry.register(
            name="contralateral_compare",
            func=tool_contralateral_compare,
            description="Crops a region in one quadrant and its anatomical mirror in the opposite quadrant, returning a side-by-side composite for symmetry comparison.",
            schema={"bbox": [0, 0, 100, 100], "quadrant": 1},
        )

        # 5. fdi_label
        registry.register(
            name="fdi_label",
            func=tool_fdi_label,
            description="Converts quadrant (1-4) and tooth position (1-8) into standard 2-digit FDI label.",
            schema={"quadrant": 1, "tooth_position": 6},
        )

        # 6. locate_abnormal_teeth (optional grounding backend)
        if grounding_tool is not None:
            registry.register(
                name="locate_abnormal_teeth",
                func=grounding_tool,
                description="Specialist detector returning candidate bounding boxes and FDI positions for abnormal teeth.",
                schema={"image_id": 0},
            )

        # 7. locate_tooth (YOLOv8 grounding)
        registry.register(
            name="locate_tooth",
            func=tool_locate_tooth,
            description="Locates a specific tooth using a trained object detector and returns its bounding box.",
            schema={"tooth": 38},
        )

        return registry
