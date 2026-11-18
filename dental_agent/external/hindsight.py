"""Hindsight export: every external-model episode, successes AND failures, in one table a hindsight-RAG can be trained from.

A row is the scored record's story of one image: what was asked, what the model did turn by turn (tools, harness events), what it
answered, what the truth was, and a coarse ``outcome`` label. Rows are pushed to the project's EXISTING private traces repository
(``HF_TRACES_REPO``) through the same atomic, identity-checked bundles as the results; no repository is ever created and nothing
is exported unless this module is called explicitly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from dental_agent.external.identity import fingerprint
from dental_agent.external.importer import safe_name
from dental_agent.external.persist import ExternalHub

ROOT = "hindsight/external"
OUTCOMES = ("success", "partial", "failure", "format_failure", "error")


def outcome(rec: dict[str, Any]) -> str:
    """success: every finding exactly right; partial: something right (tooth number or exact label); failure: nothing right;
    format_failure: the answer could not be parsed; error: the run itself failed (stored by the transport, never scored)."""
    if (rec.get("external") or {}).get("error") or rec.get("error"):
        return "error"
    if not rec.get("format_ok"):
        return "format_failure"
    if rec.get("all_exact_match"):
        return "success"
    if rec.get("exact_match") or rec.get("fdi_correct") or rec.get("matched_pairs"):
        return "partial"
    return "failure"


def hindsight_row(rec: dict[str, Any], settings: dict[str, Any]) -> dict[str, Any]:
    ext = rec.get("external") or {}
    return {
        "image_id": int(rec["image_id"]), "outcome": outcome(rec),
        "model": settings["model_slug"], "condition": rec["condition"], "dataset": rec.get("dataset"), "split": rec.get("split"),
        "run_fingerprint": fingerprint(settings), "pack": settings.get("pack", "default"), "harness": settings.get("harness"),
        "ground_truth": rec.get("ground_truth"), "predictions": rec.get("predictions"), "matched_pairs": rec.get("matched_pairs"),
        "scores": {k: rec.get(k) for k in ("fdi_f1", "exact_f1", "closeness_score", "fdi_correct", "exact_match", "all_exact_match")},
        "trajectory": rec.get("trajectory"), "raw_output": rec.get("raw_output"), "turns": rec.get("turns"), "tool_calls": rec.get("tool_calls"),
        "harness_events": ext.get("harness_events"), "usage": ext.get("usage"), "seconds": rec.get("seconds"),
    }


def hindsight_prefix(settings: dict[str, Any], shard: str | None = None) -> str:
    parts = [ROOT, settings.get("dataset", "dentex"), settings["split"], safe_name(settings["model_slug"]), settings["condition"],
             "run_" + fingerprint(settings)[:16]]
    if shard:
        parts.append(safe_name(shard))
    return "/".join(parts)


def export_hindsight(hub: ExternalHub | None, folder: str | Path, settings: dict[str, Any], evaluated: list[dict[str, Any]], *,
                     complete: bool = True, shard: str | None = None) -> dict[str, Any]:
    """Push the hindsight rows of one cell to the traces repository. ``hub`` comes from
    ``ExternalHub.from_env(os.environ['HF_TRACES_REPO'], repo_type='dataset')``; with ``None`` (no token / no repository) nothing is pushed."""
    rows = [hindsight_row(r, settings) for r in evaluated]
    counts = {o: sum(1 for r in rows if r["outcome"] == o) for o in OUTCOMES}
    if hub is None or not rows:
        return {"pushed": False, "rows": len(rows), "outcomes": counts}
    identity = {"profile": settings["model_slug"], "kind": settings["condition"], "stage": "hindsight", "shard": shard, "settings": settings}
    hub.save_rows(folder, hindsight_prefix(settings, shard), identity, rows, complete=complete)
    return {"pushed": True, "rows": len(rows), "outcomes": counts}
