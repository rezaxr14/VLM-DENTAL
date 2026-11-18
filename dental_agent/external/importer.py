"""Transport <-> evaluated records.

Transport files (``<out>/<model>/<condition>.transport.jsonl``) come from wherever the model ran (this machine, a Kaggle
notebook, a hosted task's output). Importing validates them and scores them LOCALLY against the dataset's annotations, so
ground truth never has to leave this machine and the scoring code is the project's own.
"""
from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any, Iterable

from dental_agent.external.identity import IdentityMismatch, assert_same, fingerprint

LEAK_KEYS = ("ground_truth", "exact_f1", "exact_tp", "fdi_f1", "matched_pairs")


def safe_name(slug: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", slug).strip("_") or "model"


def cell_paths(out_dir: str | Path, model_slug: str, condition: str) -> dict[str, Path]:
    base = Path(out_dir) / safe_name(model_slug)
    return {"transport": base / f"{condition}.transport.jsonl", "evaluated": base / f"{condition}.jsonl",
            "usage": base / f"{condition}.usage.jsonl", "meta": base / f"{condition}.meta.json",
            "journal": base / f"{condition}.journal.jsonl"}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    out = []
    p = Path(path)
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue          # a torn last line from an interrupted run: that image is simply re-run
    return out


_APPEND_LOCK = threading.Lock()          # worker threads append to the same files


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    p = Path(path)
    line = json.dumps(record) + "\n"
    with _APPEND_LOCK:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()


def latest_by_image(records: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    """Last record per image wins (a re-run after an infrastructure error supersedes the error record)."""
    return {int(r["image_id"]): r for r in records}


def completed(records: Iterable[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {i: r for i, r in latest_by_image(records).items() if not r.get("error")}


def validate_transport(rec: dict[str, Any], *, cohort_ids: set[int] | None, settings: dict[str, Any] | None) -> None:
    if rec.get("kind") != "transport":
        raise ValueError("not a transport record")
    leaked = [k for k in LEAK_KEYS if k in rec]
    if leaked:
        raise ValueError(f"transport record carries scoring fields {leaked}; refusing to trust it")
    if cohort_ids is not None and int(rec["image_id"]) not in cohort_ids:
        raise ValueError(f"image {rec['image_id']} is not in the cohort")
    if settings is not None:
        saved = rec.get("identity")
        try:
            assert_same(saved, settings)
        except IdentityMismatch as exc:
            raise IdentityMismatch(f"image {rec['image_id']}: {exc}") from exc


def import_transport(transport_path: str | Path, annots_df, cats_df, *, settings: dict[str, Any], dataset: str = "dentex",
                     split: str = "test", cohort_ids: set[int] | None = None, evaluated_path: str | Path | None = None,
                     evaluator=None) -> dict[str, Any]:
    """Score every completed transport record. Returns ``{'records', 'pending', 'errors'}``; writes ``evaluated_path`` when given."""
    if evaluator is None:
        from dental_agent.external.evaluators import get_evaluator
        evaluator = get_evaluator(settings.get("evaluator", "dentex_findings"))
    evaluate = evaluator
    latest = latest_by_image(read_jsonl(transport_path))
    records, errors = [], {}
    for image_id, rec in sorted(latest.items()):
        if rec.get("error"):
            errors[image_id] = rec["error"]
            continue
        validate_transport(rec, cohort_ids=cohort_ids, settings=settings)
        records.append(evaluate(rec, annots_df, cats_df, dataset=dataset, split=split, settings=settings,
                                model_slug=settings.get("model_slug")))
    pending = sorted(cohort_ids - set(latest) - set(errors)) if cohort_ids is not None else []
    if evaluated_path is not None:
        p = Path(evaluated_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return {"records": records, "pending": pending, "errors": errors, "fingerprint": fingerprint(settings)}
