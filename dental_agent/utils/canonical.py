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


def parse_slot_budget(n_full: int, n_crop: int, n_compare: int) -> dict[str, int]:
    """Build a slot budget from CLI integers (``--vision-slots FULL CROP COMPARE``)."""
    budget = {"FULL": int(n_full), "CROP": int(n_crop), "COMPARE": int(n_compare)}
    if budget["FULL"] < 1 or budget["CROP"] < 0 or budget["COMPARE"] < 0:
        raise ValueError(f"Invalid vision slot budget {budget}: need FULL >= 1 and CROP, COMPARE >= 0.")
    return budget


def slot_totals(budget: Mapping[str, int] = SLOT_BUDGET) -> dict[str, int]:
    """Static totals implied by a slot budget: slots, patches (pixel_values rows) and LLM vision tokens."""
    return {
        "slots": sum(budget.values()),
        "patches": sum(n * patches_per_image(f) for f, n in budget.items()),
        "tokens": sum(n * tokens_per_image(f) for f, n in budget.items()),
    }


TOTAL_SLOTS = slot_totals()["slots"]
TOTAL_PATCHES = slot_totals()["patches"]
TOTAL_VISION_TOKENS = slot_totals()["tokens"]  # default budget only; runtime code takes ``slot_totals(budget)``


def family_from_grid(thw: tuple[int, int, int]) -> str | None:
    """Return the canonical family whose grid equals ``thw`` (or None)."""
    for fam in CANONICAL_SIZES:
        if tuple(int(x) for x in thw) == grid_thw(fam):
            return fam
    return None


def family_for_tool(tool_name: str | None, default: str = "CROP") -> str:
    return TOOL_FAMILY.get(tool_name or "", default)


LETTERBOX_FILL = (0, 0, 0)  # radiograph background is dark; constant, so train/GRPO/eval see identical borders


def to_canonical(image: Image.Image, family: str) -> Image.Image:
    """Fit ``image`` into the canonical canvas of ``family`` WITHOUT changing its aspect ratio.

    The image is scaled uniformly (LANCZOS, RGB) to the largest size that fits the canvas and centred on a
    constant-colour canvas. Token counts are identical to a plain resize because the canvas size is fixed, but
    geometry is preserved: a near-square ``zoom_crop`` is NOT stretched to the 2:3 CROP canvas (a plain resize
    stretched it up to 1.7x vertically relative to horizontally). No pixels are cropped away.
    """
    cw, ch = CANONICAL_SIZES[family]
    img = image if image.mode == "RGB" else image.convert("RGB")
    if img.size == (cw, ch):
        return img
    scale = min(cw / img.width, ch / img.height)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    resized = img.resize((nw, nh), Image.Resampling.LANCZOS)
    if (nw, nh) == (cw, ch):
        return resized
    canvas = Image.new("RGB", (cw, ch), LETTERBOX_FILL)
    canvas.paste(resized, ((cw - nw) // 2, (ch - nh) // 2))
    return canvas


def content_fraction(image_size: tuple[int, int], family: str) -> float:
    """Fraction of the canonical canvas occupied by real image content after ``to_canonical``."""
    cw, ch = CANONICAL_SIZES[family]
    w, h = image_size
    scale = min(cw / w, ch / h)
    return (w * scale) * (h * scale) / (cw * ch)
