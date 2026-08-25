"""
Contrast enhancement tool — deterministic function that applies PIL ImageEnhance.
"""

from __future__ import annotations

from typing import Any
from PIL import Image, ImageEnhance


def tool_enhance_contrast(
    image: Image.Image,
    factor: float = 1.5,
    **kwargs: Any,
) -> Image.Image:
    """Enhance the contrast of a dental X-ray crop by *factor*.

    Useful when examining subtle enamel demineralization (incipient caries)
    or periapical radiolucency. Deterministic and fast.
    """
    if kwargs:
        import warnings
        warnings.warn(f"tool_enhance_contrast ignored unexpected arguments: {list(kwargs.keys())}")
        
    enhancer = ImageEnhance.Contrast(image)
    return enhancer.enhance(factor)
