"""Resumable per-image and per-question phases with the model injected as a callable (so they run without a GPU in tests).

Phase 1 (``run_policy_phase``): the trained agent runs once per image with its native prompt and the frozen protocol; its findings
are cached. Phase 2 (``run_answer_phase``): a model answers every question with the official prompt, optionally with the cached findings.
Both write one JSON line per item, record the settings that decide what a number means, and resume only under identical settings.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from dental_agent.benchmarks import mmoral_opg as mm

NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]*")


def safe_name(name: str, what: str = "name") -> str:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name) or ".." in name:
        raise ValueError(f"{what} {name!r} must use letters, digits, _ . + - only")
    return name


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def resume_records(path: Path, existing: str, settings: dict[str, Any], key: str) -> dict[Any, dict[str, Any]]:
    """Apply ``existing`` (error | resume | overwrite) to a result file; returns the finished records by ``key``."""
    if existing not in {"error", "resume", "overwrite"}:
        raise ValueError("existing must be error, resume or overwrite")
    rows = read_jsonl(path)
    if not rows:
        return {}
    if existing == "overwrite":
        path.unlink()
        return {}
    if existing == "error":
        raise SystemExit(f"{path} already has {len(rows)} records. Resume it (existing='resume') or overwrite it.")
    done = {}
    for rec in rows:
        diff = {k: (rec.get("settings", {}).get(k), v) for k, v in settings.items() if rec.get("settings", {}).get(k) != v}
        if diff:
            raise SystemExit(f"{path} was produced with different settings {diff}; resuming would mix them. Use overwrite.")
        done[rec[key]] = rec
    return done


def run_policy_phase(images: list[dict[str, Any]], out_path: str | Path, run_one: Callable[[int, str], dict[str, Any]],
                     settings: dict[str, Any], existing: str = "resume", limit: int | None = None) -> list[dict[str, Any]]:
    """``images``: dicts with ``id`` and ``local_path``. ``run_one(image_id, path)`` returns the record fields (``findings``,
    ``answered``, ``format_ok``, ``turns``, ``tool_calls``, ``raw_output``, ``trajectory``)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = resume_records(out_path, existing, settings, "image_id")
    chosen = images[:limit] if limit else images
    records = []
    for n, img in enumerate(chosen, 1):
        iid = int(img["id"])
        if iid in done:
            records.append(done[iid])
            continue
        t0 = time.time()
        rec = dict(image_id=iid, **run_one(iid, str(img["local_path"])), seconds=time.time() - t0, settings=settings, timestamp=time.time())
        with open(out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        records.append(rec)
        print(f"[POLICY {n}/{len(chosen)}] image {iid}: answered={rec.get('answered')} findings={len(rec.get('findings') or [])} "
              f"turns={rec.get('turns')} tools={rec.get('tool_calls')}", flush=True)
    return records


def load_findings(policy_path: str | Path, needed_image_ids: set[int]) -> dict[int, dict[str, Any]]:
    """Cached policy findings by image id. Every needed image must be there: no silent fallback to the plain prompt."""
    rows = {int(r["image_id"]): r for r in read_jsonl(Path(policy_path))}
    missing = sorted(needed_image_ids - set(rows))
    if missing:
        raise SystemExit(f"{policy_path}: no policy result for {len(missing)} image(s), e.g. {missing[:5]}. Finish the policy phase first.")
    return rows


def build_questions(closed, opened, kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    """Flat list of question dicts, closed first then open, each with its official prompt."""
    out = []
    if "closed" in kinds:
        for r in closed.to_dict("records"):
            out.append(dict(kind="closed", index=int(r["index"]), image_id=int(r["image_id"]), category=r["category"], answer=r["answer"],
                            question=r["question"], options=[r[c] for c in mm.OPTION_COLUMNS],
                            prompt=mm.closed_prompt(r["question"], [r[c] for c in mm.OPTION_COLUMNS])))
    if "open" in kinds:
        for r in opened.to_dict("records"):
            out.append(dict(kind="open", index=int(r["index"]), image_id=int(r["image_id"]), category=r["category"], answer=r["answer"],
                            question=r["question"], prompt=mm.open_prompt(r["question"])))
    return out


def run_answer_phase(questions: list[dict[str, Any]], out_dir: str | Path, answer_one: Callable[[dict[str, Any], str], dict[str, Any]],
                     settings: dict[str, Any], findings: dict[int, dict[str, Any]] | None = None, existing: str = "resume",
                     limit: int | None = None) -> dict[str, list[dict[str, Any]]]:
    """``answer_one(question, prompt) -> {'raw': text, 'new_tokens': int, 'hit_token_cap': bool}``.
    With ``findings`` (Option A) the findings block is put before the official prompt, and the findings source is part of the settings.
    One file per kind: ``<out_dir>/closed.jsonl`` and ``open.jsonl``. Returns the records by kind."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, list[dict[str, Any]]] = {}
    for kind in ("closed", "open"):
        qs = [q for q in questions if q["kind"] == kind]
        if not qs:
            continue
        path = out_dir / f"{kind}.jsonl"
        kind_settings = {**settings, "kind": kind}
        done = resume_records(path, existing, kind_settings, "index")
        chosen = qs[:limit] if limit else qs
        recs = []
        for n, q in enumerate(chosen, 1):
            if q["index"] in done:
                recs.append(done[q["index"]])
                continue
            prompt, ctx = q["prompt"], None
            if findings is not None:
                f = findings[q["image_id"]]
                prompt = mm.with_findings(q["prompt"], f.get("findings"), bool(f.get("answered")))
                ctx = dict(findings_answered=bool(f.get("answered")), n_findings=len(f.get("findings") or []))
            t0 = time.time()
            out = answer_one(q, prompt)
            prediction, flags = mm.strip_thinking(out["raw"])
            rec = dict(index=q["index"], image_id=q["image_id"], kind=kind, category=q["category"], answer=q["answer"], question=q["question"],
                       **({f"option{i+1}": o for i, o in enumerate(q["options"])} if kind == "closed" else {}),
                       prediction=prediction, raw=out["raw"], new_tokens=out.get("new_tokens"), hit_token_cap=bool(out.get("hit_token_cap")),
                       **flags, **(ctx or {}), seconds=time.time() - t0, settings=kind_settings, timestamp=time.time())
            with open(path, "a", encoding="utf-8") as f_:
                f_.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            recs.append(rec)
            if n % 25 == 0 or n == len(chosen):
                print(f"[ANSWER {kind} {n}/{len(chosen)}]", flush=True)
        result[kind] = recs
    return result
