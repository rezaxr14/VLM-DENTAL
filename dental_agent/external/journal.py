"""Turn-level journal: every model reply is saved the moment it arrives, so an interrupted image resumes for free.

A paused or crashed run (daily cap, expired key, closed session) can stop in the middle of an image that already paid for
several turns. The journal stores each turn (request fingerprint, reply text, usage); on resume the loop replays the stored
turns without a request and only the first unanswered turn goes to the provider. A request that no longer matches the stored
one fails closed (``HistoryMismatch``): replaying an answer to a different question would silently corrupt the record.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

from dental_agent.external.importer import append_jsonl, read_jsonl


def request_key(model_slug: str, system: str | None, parts: list[dict[str, Any]], max_tokens: int, temperature: float) -> str:
    """Fingerprint of one request's NEW content (earlier turns are already pinned by their own, replayed, entries)."""
    norm = []
    for p in parts:
        if p.get("type") == "text":
            norm.append(["text", p.get("text", "")])
        else:
            norm.append(["image", hashlib.sha256(str(p.get("png_b64", "")).encode()).hexdigest()])
    blob = json.dumps({"model": model_slug, "system": system, "parts": norm, "max_tokens": int(max_tokens),
                       "temperature": float(temperature)}, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class TurnJournal:
    """Append-only JSONL, keyed by ``(image_id, turn)``. Safe to share between worker threads."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._turns: dict[tuple[int, int], dict[str, Any]] = {}
        for rec in read_jsonl(self.path):
            if "image_id" in rec and "turn" in rec and "request_sha" in rec:
                self._turns[(int(rec["image_id"]), int(rec["turn"]))] = rec

    def get(self, image_id: int, turn: int) -> dict[str, Any] | None:
        with self._lock:
            return self._turns.get((int(image_id), int(turn)))

    def record(self, image_id: int, turn: int, request_sha: str, text: str, usage: dict[str, Any]) -> None:
        rec = {"image_id": int(image_id), "turn": int(turn), "request_sha": request_sha, "text": text, "usage": usage}
        with self._lock:
            self._turns[(int(image_id), int(turn))] = rec
            append_jsonl(self.path, rec)

    def turns_for(self, image_id: int) -> int:
        with self._lock:
            return sum(1 for (i, _t) in self._turns if i == int(image_id))

    def __len__(self) -> int:
        with self._lock:
            return len(self._turns)
