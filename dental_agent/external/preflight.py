"""Checks that must pass before any model is called. Each returns a dict; ``require`` raises with every failure at once.

The tool-using conditions depend on a working ``locate_tooth`` grounder, which silently degrades to an error observation when
ultralytics or the checkpoint is missing (a run would then measure a broken tool, not the model), so it is loaded and its
SHA-256 pinned here.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


class PreflightFailed(RuntimeError):
    pass


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_grounder(get_instance=None) -> dict[str, Any]:
    try:
        if get_instance is None:
            from dental_agent.tools.grounding import ToothGrounder
            get_instance = ToothGrounder.get_instance
        g = get_instance()
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    if getattr(g, "model", None) is None:
        return {"ok": False, "error": f"grounding model not loaded (path {getattr(g, 'model_path', None)})"}
    return {"ok": True, "path": str(g.model_path), "sha256": file_sha256(g.model_path)}


def check_images(images_df, ids) -> dict[str, Any]:
    present = set(int(i) for i in images_df["id"])
    missing_rows = [int(i) for i in ids if int(i) not in present]
    missing_files = [int(r.id) for r in images_df[images_df["id"].isin(list(ids))].itertuples()
                     if not isinstance(r.local_path, str) or not Path(r.local_path).is_file()]
    return {"ok": not missing_rows and not missing_files, "n": len(ids), "missing_rows": missing_rows[:20], "missing_files": missing_files[:20]}


def check_yolo_overlap(yolo_train_ids, test_ids) -> dict[str, Any]:
    """F10: the grounder must not have trained on a test image (it would leak localisation into the tool)."""
    overlap = sorted(set(int(i) for i in yolo_train_ids) & set(int(i) for i in test_ids))
    return {"ok": not overlap, "n_overlap": len(overlap), "overlap": overlap[:20]}


def check_prompts_render(split: str | None = None, dataset: str = "dentex") -> dict[str, Any]:
    """The prompts the run will use render (the cohort prompts on the 250-image split, which raise when a base prompt's class list drifts)."""
    try:
        from dental_agent.agent.prompts import FIRST_USER_PROMPT, build_agent_system_prompt
        from dental_agent.external.cohort import prompts_for
        from dental_agent.tools.registry import ToolRegistry
        texts = prompts_for(split, dataset)
        text = build_agent_system_prompt(ToolRegistry.create_default().format_tool_descriptions(), dataset=texts.agent_dataset)
        return {"ok": bool(text and texts.no_tools and texts.zero_shot and FIRST_USER_PROMPT)}
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def require(results: dict[str, dict[str, Any]]) -> None:
    bad = {k: v for k, v in results.items() if not v.get("ok")}
    if bad:
        raise PreflightFailed("; ".join(f"{k}: {v.get('error') or {kk: vv for kk, vv in v.items() if kk != 'ok'}}" for k, v in bad.items()))
