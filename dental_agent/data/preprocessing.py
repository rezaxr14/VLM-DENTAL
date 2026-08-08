"""
Image preprocessing for panoramic dental X-rays.

Handles RGB conversion, aspect-ratio-preserving resize with padding,
and bounding-box coordinate remapping.
"""

from __future__ import annotations

from PIL import Image


def preprocess_image(
    img: Image.Image,
    target_size: tuple[int, int] = (1024, 512),
) -> tuple[Image.Image, float, tuple[int, int]]:
    """Preprocess a panoramic dental X-ray for VLM input.

    1. Convert to RGB (panoramic X-rays are often single-channel;
       the VLM's vision encoder expects 3-channel input).
    2. Resize with aspect ratio preserved, then pad to *target_size*
       (panoramic X-rays are wide and short — a plain square resize
       distorts tooth shape).

    Returns
    -------
    processed : Image
        The preprocessed image.
    scale : float
        The scale factor applied to the original image.
    offset : (int, int)
        The (x, y) paste offset used for padding.
    """
    img = img.convert("RGB")
    target_w, target_h = target_size
    scale = min(target_w / img.width, target_h / img.height)
    new_w, new_h = int(img.width * scale), int(img.height * scale)
    resized = img.resize((new_w, new_h), Image.BICUBIC)

    padded = Image.new("RGB", target_size, (0, 0, 0))
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2
    padded.paste(resized, (paste_x, paste_y))
    return padded, scale, (paste_x, paste_y)


def remap_bbox(
    bbox: list[float],
    scale: float,
    offset: tuple[int, int],
) -> list[float]:
    """Remap a ``[x, y, w, h]`` bounding box from original image coordinates
    into the coordinate space produced by :func:`preprocess_image`.

    Needed to keep ground-truth boxes aligned once images are resized/padded.
    """
    x, y, w, h = bbox
    off_x, off_y = offset
    return [x * scale + off_x, y * scale + off_y, w * scale, h * scale]


import numpy as np

def check_image_quality(img: Image.Image) -> tuple[bool, str]:
    """
    Quality gate for dental X-rays to detect severe corruption before inference/training.
    Returns (is_valid, reason).
    """
    img_array = np.array(img.convert("L"))
    
    # 1. Size check
    if img_array.shape[0] < 100 or img_array.shape[1] < 100:
        return False, "Image too small or unreadable"
        
    # 2. Horizontal banding / scan-line corruption check (like test_5.png)
    # If there are extreme intensity jumps between adjacent rows across the entire image width,
    # it strongly indicates digital scan-line failure/corruption rather than anatomical structure.
    row_means = np.mean(img_array, axis=1)
    row_diffs = np.abs(np.diff(row_means))
    if np.max(row_diffs) > 75:  # 75 pixel intensity jump on average across a whole row is huge
        return False, "Severe horizontal banding / scan failure detected"
        
    # 3. Blank/solid image check
    if np.std(img_array) < 5:
        return False, "Image is nearly blank or solid color"
        
    return True, "OK"
