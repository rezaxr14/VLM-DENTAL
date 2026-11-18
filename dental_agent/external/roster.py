"""Which hosted models are available, and which of them can see an image.

Kaggle's listing format and vision metadata are gate G1 items, so discovery is tolerant and nothing is assumed: a model is
usable only when vision is confirmed, either by the listing or by ``probe_vision`` (a tiny coloured-squares image the model
must describe). The roster is saved as JSON so a run uses a fixed, reviewable list.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


@dataclass
class RosterEntry:
    slug: str
    display: str | None = None
    vision: bool | None = None          # None = unknown, must be probed
    source: str = "manual"
    note: str | None = None


def parse_listing(text: str) -> list[RosterEntry]:
    """JSON (list of strings / objects, or {models:[...]}) first; otherwise one slug per line (first whitespace-separated
    token, header and separator lines skipped)."""
    text = text.strip()
    try:
        data = json.loads(text)
    except Exception:
        data = None
    if data is not None:
        if isinstance(data, dict):
            data = data.get("models") or data.get("data") or []
        out = []
        for item in data:
            if isinstance(item, str):
                out.append(RosterEntry(item, source="listing"))
            elif isinstance(item, dict):
                slug = item.get("slug") or item.get("id") or item.get("name")
                if slug:
                    vis = item.get("vision", item.get("multimodal", item.get("supports_images")))
                    out.append(RosterEntry(str(slug), item.get("display_name") or item.get("title"), vis if isinstance(vis, bool) else None, "listing"))
        return out
    out = []
    for line in text.splitlines():
        line = line.strip()
        if not line or set(line) <= set("-=+| "):
            continue
        tok = re.split(r"[\s|]+", line.strip("| "))[0]
        if tok.lower() in {"slug", "model", "models", "name", "id"} or not re.match(r"^[\w./:-]+$", tok):
            continue
        out.append(RosterEntry(tok, source="listing"))
    return out


def discover(run: Callable[[list[str]], str] | None = None, command: tuple[str, ...] = ("kaggle", "benchmarks", "tasks", "models")) -> list[RosterEntry]:
    """Run the Kaggle CLI listing (``run`` is injectable for tests) and parse it. Needs credentials; never called implicitly."""
    run = run or (lambda cmd: subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120).stdout)
    return parse_listing(run(list(command)))


def probe_image():
    """Left half red, right half blue: an unambiguous two-colour image."""
    from PIL import Image
    img = Image.new("RGB", (256, 128), (220, 20, 20))
    img.paste((20, 20, 220), (128, 0, 256, 128))
    return img


PROBE_PROMPT = 'Which two colours do you see, left and right? Use plain English colour names (not hex codes). Reply with JSON only: {"left": "<colour>", "right": "<colour>"}'


def probe_vision(backend, max_tokens: int = 64) -> bool:
    """True when the model reports red-left / blue-right. A wrong answer or an empty reply is False (not usable)."""
    from dental_agent.external.transcript_sync import to_parts
    chat = backend.open_chat(None)
    parts = to_parts({"role": "user", "content": [{"type": "image", "image": probe_image()}, {"type": "text", "text": PROBE_PROMPT}]})
    text = chat.send(parts, max_tokens, 0.0).text.lower()
    return "red" in text and "blue" in text and text.find("red") < text.find("blue")


def save(entries: list[RosterEntry], path: str | Path) -> None:
    Path(path).write_text(json.dumps([asdict(e) for e in entries], indent=2))


def load(path: str | Path) -> list[RosterEntry]:
    return [RosterEntry(**d) for d in json.loads(Path(path).read_text())]


def usable(entries: list[RosterEntry]) -> list[RosterEntry]:
    """Confirmed-vision models only."""
    return [e for e in entries if e.vision is True]
