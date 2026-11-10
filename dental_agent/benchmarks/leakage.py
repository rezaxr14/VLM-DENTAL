"""Screen an external benchmark's images against the DENTEX images the model was trained or evaluated on.

Two stages: a 256-bit difference hash finds candidates (small Hamming distance), then a normalised cross-correlation of 128x64
grayscale thumbnails confirms them. Exact byte duplicates (SHA-256) are reported separately. This is a SCREEN: it catches copies, rescaled
copies and re-encodings, not crops, rotations or flips. The closest pairs are always listed, so the margin is visible.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

HASH_W, HASH_H, THUMB = 16, 16, (128, 64)


def _gray(path: str | Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.convert("L")


def dhash_bits(path: str | Path) -> np.ndarray:
    from PIL import Image
    small = np.asarray(_gray(path).resize((HASH_W + 1, HASH_H), Image.LANCZOS), dtype=np.float32)
    return (small[:, 1:] > small[:, :-1]).ravel()


def thumbnail(path: str | Path) -> np.ndarray:
    from PIL import Image
    t = np.asarray(_gray(path).resize(THUMB, Image.LANCZOS), dtype=np.float32).ravel()
    t = t - t.mean()
    n = np.linalg.norm(t)
    return t / n if n > 0 else t


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def screen(bench: Mapping[Any, str], reference: Mapping[Any, str], *, max_hamming: int = 24, min_ncc: float = 0.95, top: int = 10) -> dict[str, Any]:
    """``bench``/``reference``: {key: image path}. Returns exact duplicates, near duplicates (Hamming <= max_hamming AND NCC >= min_ncc),
    and the ``top`` closest pairs by Hamming distance whatever their value."""
    if not bench or not reference:
        raise ValueError("Both image sets must be non-empty")
    ref_keys = list(reference)
    ref_hash = np.stack([dhash_bits(reference[k]) for k in ref_keys])
    ref_sha = {sha256_file(reference[k]): k for k in ref_keys}
    exact, near, closest = [], [], []
    for key, path in bench.items():
        sha = sha256_file(path)
        if sha in ref_sha:
            exact.append(dict(bench=key, reference=ref_sha[sha]))
        dist = (ref_hash != dhash_bits(path)).sum(axis=1)
        j = int(dist.argmin())
        ncc = float(thumbnail(path) @ thumbnail(reference[ref_keys[j]]))
        pair = dict(bench=key, reference=ref_keys[j], hamming=int(dist[j]), ncc=ncc)
        closest.append(pair)
        for jj in np.where(dist <= max_hamming)[0]:
            c = float(thumbnail(path) @ thumbnail(reference[ref_keys[int(jj)]]))
            if c >= min_ncc:
                near.append(dict(bench=key, reference=ref_keys[int(jj)], hamming=int(dist[jj]), ncc=c))
    closest.sort(key=lambda p: (p["hamming"], -p["ncc"]))
    return dict(n_bench=len(bench), n_reference=len(reference), max_hamming=max_hamming, min_ncc=min_ncc, exact=exact, near=near,
                closest=closest[:top], verdict="OVERLAP FOUND" if exact or near else "no copy found by this screen")
