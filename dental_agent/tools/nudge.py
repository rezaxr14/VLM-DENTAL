"""
Nudge tool — lets the agent correct a bounding box it was already given (by
locate_tooth, or a previous nudge_crop) without re-running detection, for
when the returned crop doesn't actually show the tooth it asked for, or
shows it off-center. Deliberately data-only (mirrors locate_tooth's
convention): it returns adjusted coordinates, not an image. Pair with
zoom_crop on the returned bbox to actually see the corrected region.
"""

from __future__ import annotations

from typing import Any
from PIL import Image


def tool_nudge_crop(
    image: Image.Image,
    bbox: list[float],
    dx_frac: float = 0.0,
    dy_frac: float = 0.0,
    scale: float = 1.0,
) -> dict[str, Any]:
    """Adjust a previously-returned bounding box.

    Args:
        image: The panoramic radiograph (PIL.Image) — used only to clamp the
            adjusted box to valid bounds; this tool does not re-run detection.
        bbox: The [x, y, w, h] box to adjust — one you were already given by
            locate_tooth or a prior nudge_crop, never asserted from nowhere.
        dx_frac: Shift right, as a fraction of the box's own width (negative
            shifts left). e.g. 0.5 moves the box right by half its own width —
            fractional so the shift stays proportional at any zoom level.
        dy_frac: Shift down, as a fraction of the box's own height (negative
            shifts up).
        scale: Rescale the box around its new center. >1 zooms out for more
            context (use this if the tooth you expected isn't in frame at
            all); <1 zooms in tighter (use this once you can see it but want
            a closer look).

    Returns:
        Dictionary with the adjusted `bbox` ([x, y, w, h]). This tool only
        computes coordinates — call zoom_crop with the returned bbox to
        actually see the corrected region.
    """
    if not bbox or len(bbox) != 4:
        return {"error": "bbox must be [x, y, w, h]."}

    x, y, w, h = bbox
    if w <= 0 or h <= 0:
        return {"error": "bbox width and height must be positive."}

    cx, cy = x + w / 2.0, y + h / 2.0
    cx += dx_frac * w
    cy += dy_frac * h

    scale = max(0.1, scale)
    new_w = min(w * scale, float(image.width))
    new_h = min(h * scale, float(image.height))

    new_x = cx - new_w / 2.0
    new_y = cy - new_h / 2.0
    new_x = max(0.0, min(new_x, image.width - new_w))
    new_y = max(0.0, min(new_y, image.height - new_h))

    return {
        "bbox": [round(new_x, 1), round(new_y, 1), round(new_w, 1), round(new_h, 1)],
        "note": "Call zoom_crop with this bbox to view the adjusted region.",
    }
