"""
Synthetic dental radiograph generator for unit testing and offline validation.

Creates a synthetic panoramic X-ray with realistic jaw arches, teeth geometry,
and noise patterns without requiring any downloaded assets.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def make_synthetic_dental_image(
    width: int = 1024,
    height: int = 512,
    seed: int = 0,
    findings: list[dict[str, Any]] | None = None,
) -> Image.Image:
    """Create a synthetic panoramic dental radiograph for offline self-testing.

    Generates a dark background with simulated upper/lower dental arches,
    radiolucent jaw regions, and enamel-density tooth outlines.
    """
    rng = np.random.default_rng(seed)
    base = np.zeros((height, width), dtype=np.uint8)

    # Soft gradient representing skull/mandible tissue attenuation
    y, x = np.ogrid[:height, :width]
    mandible_arch = np.exp(-((y - height * 0.65) ** 2) / (2 * (height * 0.25) ** 2))
    maxilla_arch = np.exp(-((y - height * 0.35) ** 2) / (2 * (height * 0.20) ** 2))
    attenuation = (mandible_arch * 0.5 + maxilla_arch * 0.4) * 120
    base = np.clip(base + attenuation, 0, 255).astype(np.uint8)

    # Add Gaussian noise
    noise = rng.normal(0, 10, (height, width))
    base = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(base, mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw simulated upper and lower teeth along parabolic arches
    n_teeth_per_arch = 16
    arch_w = width * 0.4
    center_x = width // 2

    # Upper arch
    for i in range(n_teeth_per_arch):
        t = (i - (n_teeth_per_arch - 1) / 2) / (n_teeth_per_arch / 2)
        tx = center_x + t * arch_w
        ty = height * 0.35 - (1 - t**2) * 20
        tw, th = 18, 35
        draw.rounded_rectangle(
            [tx - tw / 2, ty - th / 2, tx + tw / 2, ty + th / 2],
            radius=4,
            fill=(220, 220, 220),
            outline=(255, 255, 255),
        )

    # Lower arch
    for i in range(n_teeth_per_arch):
        t = (i - (n_teeth_per_arch - 1) / 2) / (n_teeth_per_arch / 2)
        tx = center_x + t * arch_w
        ty = height * 0.65 + (1 - t**2) * 20
        tw, th = 16, 30
        draw.rounded_rectangle(
            [tx - tw / 2, ty - th / 2, tx + tw / 2, ty + th / 2],
            radius=4,
            fill=(210, 210, 210),
            outline=(245, 245, 245),
        )

    if findings:
        for f in findings:
            q = f.get("quadrant", 1)
            pos = f.get("tooth_position", 1)
            px = center_x + (pos - 4.5) * (arch_w / 8) * (1 if q in (1, 4) else -1)
            py = height * 0.35 if q in (1, 2) else height * 0.65
            draw.ellipse([px - 8, py - 8, px + 8, py + 8], fill=(60, 60, 60))

    # Apply light Gaussian blur to simulate X-ray scatter
    return img.filter(ImageFilter.GaussianBlur(radius=1.5))
