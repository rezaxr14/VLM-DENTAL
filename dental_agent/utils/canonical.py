"""Canonical vision-input geometry shared by SFT, GRPO rollouts and the XLA warmup.

Every image the model sees is mapped to one of three fixed *families* so the
number of vision tokens per image is a constant, not a function of the scan's
native resolution (native DENTEX panoramics reach ~17k visual tokens).

Geometry (Qwen3.5 vision tower: patch_size=16, 2x2 spatial merge, feature dim 1536):

    family    size (W x H)   grid_thw (t, h, w)   patches   LLM tokens
    FULL      1536 x 768     (1, 48, 96)          4608      1152
    CROP       256 x 384     (1, 24, 16)           384        96
    COMPARE    512 x 384     (1, 24, 32)           768       192

IMPORTANT invariant: tools always execute on the NATIVE image (their bbox
arguments are native pixel coordinates). Canonicalisation only changes the
*view* that is handed to the processor. Never run a tool on a canonical image.

This module is intentionally torch-free so it can be imported anywhere.
"""

from __future__ import annotations

import os
from typing import Mapping

from PIL import Image

PATCH_SIZE = 16
SPATIAL_MERGE = 2
MERGE_AREA = SPATIAL_MERGE * SPATIAL_MERGE
PATCH_FEATURE_DIM = 3 * 2 * PATCH_SIZE * PATCH_SIZE  # C * temporal_patch(2) * 16 * 16 = 1536

# family -> (width, height)
CANONICAL_SIZES: Mapping[str, tuple[int, int]] = {
    "FULL": (1536, 768),
    "CROP": (256, 384),
    "COMPARE": (512, 384),
}

# Static vision-slot budget (covers 100% of the deduplicated corpus per the project census).
SLOT_BUDGET: Mapping[str, int] = {"FULL": 5, "CROP": 10, "COMPARE": 4}

# Tool -> family of the image it returns. Tools not listed return no image.
TOOL_FAMILY: Mapping[str, str] = {
    "zoom_crop": "CROP",
    "contralateral_compare": "COMPARE",
    "denoise": "FULL",
    "window_level": "FULL",
    "enhance_contrast": "FULL",
}


def grid_thw(family: str) -> tuple[int, int, int]:
    w, h = CANONICAL_SIZES[family]
    return (1, h // PATCH_SIZE, w // PATCH_SIZE)


def patches_per_image(family: str) -> int:
    t, h, w = grid_thw(family)
    return t * h * w


def tokens_per_image(family: str) -> int:
    return patches_per_image(family) // MERGE_AREA


TOTAL_SLOTS = sum(SLOT_BUDGET.values())
TOTAL_PATCHES = sum(n * patches_per_image(f) for f, n in SLOT_BUDGET.items())
TOTAL_VISION_TOKENS = sum(n * tokens_per_image(f) for f, n in SLOT_BUDGET.items())

# Guard the arithmetic the rest of the pipeline (and sft.apply_spmd_input_sharding) relies on.
assert TOTAL_SLOTS == 19
assert TOTAL_PATCHES == 29952
assert TOTAL_VISION_TOKENS == 7488


def family_from_grid(thw: tuple[int, int, int]) -> str | None:
    """Return the canonical family whose grid equals ``thw`` (or None)."""
    for fam in CANONICAL_SIZES:
        if tuple(int(x) for x in thw) == grid_thw(fam):
            return fam
    return None


def family_for_tool(tool_name: str | None, default: str = "CROP") -> str:
    return TOOL_FAMILY.get(tool_name or "", default)


def to_canonical(image: Image.Image, family: str) -> Image.Image:
    """Resample ``image`` to the exact canonical size of ``family`` (LANCZOS, RGB).

    Plain resize (no letterboxing): this is the settled project spec. Note it does
    not preserve aspect ratio for CROP/COMPARE outputs of arbitrary shape.
    """
    size = CANONICAL_SIZES[family]
    img = image if image.mode == "RGB" else image.convert("RGB")
    if img.size == size:
        return img
    return img.resize(size, Image.Resampling.LANCZOS)


ENV_CANONICAL_RESIZE = "DENTAL_CANONICAL_RESIZE"


def resolve_canonical_resize(value: bool | None = None) -> bool:
    """Single source of truth for "does the model see canonical views?".

    An explicit ``value`` wins. Otherwise the ``DENTAL_CANONICAL_RESIZE`` env var decides
    (``1/true/yes/on``), defaulting to False (native resolution). Training, GRPO rollouts and
    evaluation MUST resolve to the same answer, otherwise the policy is evaluated on a
    different image distribution than it was trained on.
    """
    if value is not None:
        return bool(value)
    return os.environ.get(ENV_CANONICAL_RESIZE, "").strip().lower() in {"1", "true", "yes", "on"}
