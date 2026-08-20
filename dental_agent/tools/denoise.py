"""
Edge-preserving noise reduction tool for dental radiographs.
"""
from typing import Literal
import cv2
import numpy as np
from PIL import Image

def tool_denoise(
    image: Image.Image,
    method: Literal["bilateral", "median"] = "bilateral",
    strength: float = 0.6,
) -> Image.Image:
    """
    Applies edge-preserving noise reduction to distinguish real pathology from sensor noise.
    Bilateral filter is recommended for X-rays to preserve bone/enamel boundaries while smoothing grain.

    strength: 0.0-1.0, how aggressively to smooth. Low strength preserves fine detail
    (use when checking for subtle enamel demineralization); high strength more aggressively
    removes grain (use on visibly noisy/grainy crops where grain could be mistaken for
    pathology). Previous fixed behavior (d=15, sigmaColor=sigmaSpace=120 for bilateral;
    9x9 for median) corresponds to roughly strength=0.6, the new default.
    """
    strength = max(0.0, min(1.0, strength))
    img_array = np.array(image)

    if method == "bilateral":
        # d: neighborhood diameter, 5-25px. sigmaColor/sigmaSpace: 20-200.
        d = int(round(5 + strength * 20))
        sigma = int(round(20 + strength * 180))
        denoised = cv2.bilateralFilter(img_array, d=d, sigmaColor=sigma, sigmaSpace=sigma)
    elif method == "median":
        # Kernel must be odd; 3 (barely-there) to 13 (aggressive).
        ksize = int(round(3 + strength * 10))
        if ksize % 2 == 0:
            ksize += 1
        denoised = cv2.medianBlur(img_array, ksize=ksize)
    else:
        denoised = img_array

    return Image.fromarray(denoised)
