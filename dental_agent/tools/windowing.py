"""
Medical Image Windowing tool (Level/Width) for dental radiographs.
"""
from typing import Literal
from PIL import Image
import numpy as np

WINDOW_PRESETS = {
    "bone": {"center": 128, "width": 100},           # Mid-range to see trabecular patterns
    "enamel": {"center": 200, "width": 100},         # High range to see dense enamel vs caries
    "soft_tissue": {"center": 80, "width": 150},     # Low range to see pulp/periapical tissues
    "metal_reduction": {"center": 100, "width": 200},# Wider width to reduce blowout from crowns
}

def tool_window_level(
    image: Image.Image,
    preset: Literal["bone", "enamel", "soft_tissue", "metal_reduction"] = "bone",
    center: float | None = None,
    width: float | None = None,
) -> Image.Image:
    """
    Apply medical intensity windowing to reveal specific density structures.
    Stretches the chosen intensity range to full black/white, clipping the rest.

    preset gives sensible starting points; pass center and/or width to override
    either one exactly instead of relying on a preset (e.g. if a finding sits
    right at a preset's edge and needs a slightly shifted window to see clearly).
    Any value not given falls back to the preset's value, not the other override.
    """
    if preset not in WINDOW_PRESETS:
        preset = "bone"

    resolved_center = center if center is not None else WINDOW_PRESETS[preset]["center"]
    resolved_width = width if width is not None else WINDOW_PRESETS[preset]["width"]
    resolved_width = max(1.0, resolved_width)  # a zero/negative width would divide by ~0 below

    # Work in grayscale for intensity math
    img_array = np.array(image.convert("L"), dtype=np.float32)

    min_val = resolved_center - (resolved_width / 2.0)
    max_val = resolved_center + (resolved_width / 2.0)

    # Clip to the window
    windowed = np.clip(img_array, min_val, max_val)

    # Scale clipped range linearly to 0-255
    if max_val > min_val:
        windowed = ((windowed - min_val) / (max_val - min_val)) * 255.0

    windowed = windowed.astype(np.uint8)

    # Return as RGB to keep consistent channel depth for VLMs
    return Image.fromarray(windowed).convert("RGB")
