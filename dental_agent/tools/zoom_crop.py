"""
Zoom/crop tool — deterministic function that returns a higher-resolution
crop around a specified bounding box with configurable padding.
"""

from __future__ import annotations

from PIL import Image


def tool_zoom_crop(
    image: Image.Image,
    bbox: list[float],
    padding_frac: float = 0.25,
) -> Image.Image:
    """Return a cropped, zoomed-in view around *bbox* (``[x, y, w, h]``).

    Adds *padding_frac* extra context on each side, clamped to image bounds.
    This is the agent's zoom/crop tool — deterministic, so its behavior is
    fully predictable and auditable.
    """
    x, y, w, h = bbox
    pad_x, pad_y = w * padding_frac, h * padding_frac
    left = max(0, x - pad_x)
    top = max(0, y - pad_y)
    right = min(image.width, x + w + pad_x)
    bottom = min(image.height, y + h + pad_y)
    return image.crop((left, top, right, bottom))


def box_out_of_bounds(bbox: list[float], img_w: int, img_h: int) -> bool:
    """Check if a [x, y, w, h] bounding box exceeds image dimensions or has negative bounds."""
    x, y, w, h = bbox
    return x < 0 or y < 0 or (x + w) > img_w or (y + h) > img_h

