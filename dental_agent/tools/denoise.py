"""
Edge-preserving noise reduction tool for dental radiographs.
"""
from typing import Literal
import cv2
import numpy as np
from PIL import Image

def tool_denoise(
    image: Image.Image,
    method: Literal["bilateral", "median"] = "bilateral"
) -> Image.Image:
    """
    Applies edge-preserving noise reduction to distinguish real pathology from sensor noise.
    Bilateral filter is recommended for X-rays to preserve bone/enamel boundaries while smoothing grain.
    """
    # Convert PIL Image to numpy array (RGB)
    img_array = np.array(image)
    
    if method == "bilateral":
        # d=15, sigmaColor=120, sigmaSpace=120 for strong noise reduction on high-res X-rays
        denoised = cv2.bilateralFilter(img_array, d=15, sigmaColor=120, sigmaSpace=120)
    elif method == "median":
        # 9x9 median filter for strong salt-and-pepper noise removal
        denoised = cv2.medianBlur(img_array, ksize=9)
    else:
        denoised = img_array
        
    return Image.fromarray(denoised)
