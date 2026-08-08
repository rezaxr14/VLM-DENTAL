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
        # d=9, sigmaColor=75, sigmaSpace=75 are standard strong but safe smoothing parameters
        denoised = cv2.bilateralFilter(img_array, d=9, sigmaColor=75, sigmaSpace=75)
    elif method == "median":
        # 5x5 median filter
        denoised = cv2.medianBlur(img_array, ksize=5)
    else:
        denoised = img_array
        
    return Image.fromarray(denoised)
