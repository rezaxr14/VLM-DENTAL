"""Tool packs: which tools a condition's model is given.

The ``default`` pack is the project's ``ToolRegistry.create_default()``. A new capability (a segmentation tool, step-by-step image
editing tools such as contrast / filters / zoom / kernels, a re-look tool) is added as a pack: a function returning a ``ToolRegistry``,
registered once, then named by a ``Condition(pack=...)``. Nothing else changes: the prompt hash (which contains the tool
descriptions) and the pack name enter the run identity, so results produced with different tools can never be mixed, and the same
pack name travels to hosted workers through ``ShardSpec.pack`` / ``plugins``.

    from dental_agent.external import tool_packs
    from dental_agent.external.conditions import Condition, register_condition

    def my_pack():
        reg = ToolRegistry.create_default()
        reg.register("sharpen", tool_sharpen, "Unsharp mask ...", {"amount": 1.0})
        return reg

    tool_packs.register_pack("edit_v1", my_pack)
    register_condition(Condition("kaggle_tools_edit_v1", True, "off", "unified", required=False, pack="edit_v1"))
"""
from __future__ import annotations

import hashlib
from typing import Callable

_PACKS: dict[str, Callable[[], "object"]] = {}


def _default_pack():
    from dental_agent.tools.registry import ToolRegistry
    return ToolRegistry.create_default()


_PACKS["default"] = _default_pack


def register_pack(name: str, factory: Callable[[], "object"], replace: bool = False) -> None:
    if name in _PACKS and not replace:
        raise ValueError(f"tool pack {name!r} is already registered")
    _PACKS[name] = factory


def get_registry(name: str = "default"):
    """A fresh ``ToolRegistry`` for the pack (a run never shares mutable tool state with another)."""
    try:
        return _PACKS[name]()
    except KeyError:
        raise ValueError(f"Unknown tool pack '{name}'. Known: {', '.join(_PACKS)}. Did the plugin module that registers it get imported?") from None


def packs() -> list[str]:
    return sorted(_PACKS)


def tools_sha256(name: str = "default") -> str:
    return hashlib.sha256(get_registry(name).format_tool_descriptions().encode("utf-8")).hexdigest()
