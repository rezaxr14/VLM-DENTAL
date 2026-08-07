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

        # 1. zoom_crop
        registry.register(
            name="zoom_crop",
            func=tool_zoom_crop,
            description="Crops around a bounding box [x, y, w, h] with context padding to provide a zoomed view.",
            schema={"bbox": [0, 0, 100, 100]},
        )

        # 2. enhance_contrast
        registry.register(
            name="enhance_contrast",
            func=tool_enhance_contrast,
            description="Increases visual contrast (factor default 1.5) for subtle lesion detection.",
            schema={"factor": 1.5},
        )

        # 3. fdi_label
        registry.register(
            name="fdi_label",
            func=tool_fdi_label,
            description="Converts quadrant (1-4) and tooth position (1-8) into standard 2-digit FDI label.",
            schema={"quadrant": 1, "tooth_position": 6},
        )

        # 4. locate_abnormal_teeth (optional grounding backend)
        if grounding_tool is not None:
            registry.register(
                name="locate_abnormal_teeth",
                func=grounding_tool,
                description="Specialist detector returning candidate bounding boxes and FDI positions for abnormal teeth.",
                schema={"image_id": 0},
            )

        return registry
