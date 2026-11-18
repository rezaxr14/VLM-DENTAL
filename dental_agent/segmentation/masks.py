"""Mask construction and a tiny run-length encoding (column-major runs, alternating 0/1 starting with 0), so masks can be
stored in JSON without extra dependencies."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from PIL import Image, ImageDraw


def polygon_to_mask(polygon: Sequence[Sequence[float]], width: int, height: int) -> np.ndarray:
    img = Image.new("L", (width, height), 0)
    ImageDraw.Draw(img).polygon([(float(x), float(y)) for x, y in polygon], outline=1, fill=1)
    return np.asarray(img, dtype=bool)


def bbox_to_mask(bbox_xywh: Sequence[float], width: int, height: int) -> np.ndarray:
    x, y, w, h = (float(v) for v in bbox_xywh)
    m = np.zeros((height, width), dtype=bool)
    x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
    x1, y1 = min(width, int(round(x + w))), min(height, int(round(y + h)))
    if x1 > x0 and y1 > y0:
        m[y0:y1, x0:x1] = True
    return m


def rle_encode(mask: np.ndarray) -> dict[str, Any]:
    m = np.asarray(mask, dtype=bool)
    flat = m.T.reshape(-1)                      # column-major
    counts, cur, run = [], False, 0
    for v in flat:
        if bool(v) == cur:
            run += 1
        else:
            counts.append(run)
            cur, run = bool(v), 1
    counts.append(run)
    return {"size": [int(m.shape[0]), int(m.shape[1])], "counts": counts}


def rle_decode(rle: dict[str, Any]) -> np.ndarray:
    h, w = (int(v) for v in rle["size"])
    flat = np.zeros(h * w, dtype=bool)
    pos, cur = 0, False
    for c in rle["counts"]:
        if cur:
            flat[pos: pos + c] = True
        pos += c
        cur = not cur
    if pos != h * w:
        raise ValueError("rle counts do not cover the image")
    return flat.reshape(w, h).T


def finding_mask(finding, width: int, height: int) -> np.ndarray:
    return rle_decode(finding.rle) if finding.rle is not None else polygon_to_mask(finding.polygon, width, height)
