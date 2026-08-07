"""
FDI World Dental Federation two-digit tooth numbering system helpers.

Maps between (quadrant, tooth_position) pairs and standardized two-digit
FDI labels ("11" to "48"), including anatomy nomenclature and quadrant flipping.
"""

from __future__ import annotations

QUADRANT_NAMES: dict[int, str] = {
    1: "Maxillary Right (Upper Right)",
    2: "Maxillary Left (Upper Left)",
    3: "Mandibular Left (Lower Left)",
    4: "Mandibular Right (Lower Right)",
}

TOOTH_NAMES: dict[int, str] = {
    1: "Central Incisor",
    2: "Lateral Incisor",
    3: "Canine",
    4: "First Premolar",
    5: "Second Premolar",
    6: "First Molar",
    7: "Second Molar",
    8: "Third Molar (Wisdom Tooth)",
}


def fdi_encode(quadrant: int, tooth_position: int) -> int:
    """Combine quadrant (1-4) and tooth position (1-8) into an integer FDI number (11-48)."""
    if quadrant not in (1, 2, 3, 4) or tooth_position not in range(1, 9):
        raise ValueError(
            f"Invalid quadrant ({quadrant}) or tooth position ({tooth_position}). "
            f"Quadrant must be 1-4 and tooth_position must be 1-8."
        )
    return quadrant * 10 + tooth_position


def fdi_decode(fdi_number: int | str) -> tuple[int, int]:
    """Decode an FDI number (11-48) into (quadrant, tooth_position)."""
    num = int(fdi_number)
    quadrant = num // 10
    tooth_position = num % 10
    if quadrant not in (1, 2, 3, 4) or tooth_position not in range(1, 9):
        raise ValueError(f"Invalid FDI number: {fdi_number}")
    return quadrant, tooth_position


def tool_fdi_label(quadrant: int, tooth_position: int) -> str | None:
    """Format (quadrant, tooth_position) as a two-digit FDI string ("36"), or None if invalid."""
    if quadrant in (1, 2, 3, 4) and tooth_position in range(1, 9):
        return f"{quadrant}{tooth_position}"
    return None


def get_anatomical_name(quadrant: int, tooth_position: int) -> str:
    """Return the full anatomical name (e.g., 'Maxillary Right Central Incisor (11)')."""
    q_name = QUADRANT_NAMES.get(quadrant, f"Quadrant {quadrant}")
    t_name = TOOTH_NAMES.get(tooth_position, f"Tooth {tooth_position}")
    label = tool_fdi_label(quadrant, tooth_position) or "??"
    return f"{q_name} {t_name} (FDI #{label})"


def flip_quadrant(quadrant: int) -> int:
    """Swap anatomical left/right quadrants under a horizontal flip (1<->2, 3<->4)."""
    mapping = {1: 2, 2: 1, 3: 4, 4: 3}
    if quadrant not in mapping:
        raise ValueError(f"Invalid quadrant: {quadrant}")
    return mapping[quadrant]
