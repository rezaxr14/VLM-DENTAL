"""Small run-metadata file written next to an SFT adapter, read by the GRPO preflight.

GRPO must run on the same image distribution the SFT adapter was trained on (canonical views, the same held-out dev
split, the same track). None of that was recorded with the adapter, so a mismatch could only be found by reading the
SFT notebook. ``sft_run_meta.json`` records it; ``scripts/run_grpo.py --preflight`` compares it with the GRPO command.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

SFT_META_FILE = "sft_run_meta.json"


def keys_fingerprint(keys: Iterable[str]) -> str:
    """Order-independent hash of a set of ``dataset::image_id`` keys. The empty set (no dev split) is ``"none"``."""
    ks = sorted(str(k) for k in keys)
    if not ks:
        return "none"
    return hashlib.sha1("\n".join(ks).encode("utf-8")).hexdigest()[:16]


def write_sft_run_meta(out_dir: str | Path, **fields: Any) -> Path:
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    target = path / SFT_META_FILE
    with open(target, "w", encoding="utf-8") as f:
        json.dump(fields, f, indent=2, sort_keys=True, default=str)
        f.write("\n")
    return target


def read_sft_run_meta(adapter_dir: str | Path) -> dict[str, Any] | None:
    """Metadata next to the adapter, or None for an adapter trained before this file existed."""
    p = Path(adapter_dir) / SFT_META_FILE
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
