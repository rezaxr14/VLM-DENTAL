"""What a segmentation answer looks like: one mask per finding, in native image pixel coordinates."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

DIAGNOSES = ("Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth")


class SegmentationContractError(ValueError):
    pass


@dataclass
class SegmentedFinding:
    quadrant: int
    tooth_position: int
    diagnosis: str
    polygon: list[tuple[float, float]] | None = None      # [(x, y), ...] native pixels, >= 3 vertices
    rle: dict[str, Any] | None = None                       # {"size": [h, w], "counts": [...]} (see masks.rle_encode)
    score: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def fdi(self) -> int:
        return self.quadrant * 10 + self.tooth_position


def validate(finding: SegmentedFinding, width: int, height: int) -> SegmentedFinding:
    if finding.quadrant not in (1, 2, 3, 4) or finding.tooth_position not in range(1, 9):
        raise SegmentationContractError(f"invalid FDI {finding.quadrant}{finding.tooth_position}")
    if finding.diagnosis not in DIAGNOSES:
        raise SegmentationContractError(f"unknown diagnosis '{finding.diagnosis}'")
    if (finding.polygon is None) == (finding.rle is None):
        raise SegmentationContractError("exactly one of polygon / rle is required")
    if finding.polygon is not None:
        if len(finding.polygon) < 3:
            raise SegmentationContractError("polygon needs at least 3 vertices")
        for x, y in finding.polygon:
            if not (0 <= x <= width and 0 <= y <= height):
                raise SegmentationContractError(f"vertex ({x}, {y}) outside the {width}x{height} image")
    else:
        if list(finding.rle.get("size", [])) != [height, width]:
            raise SegmentationContractError("rle size does not match the image")
    if finding.score is not None and not 0.0 <= finding.score <= 1.0:
        raise SegmentationContractError("score must be in [0, 1]")
    return finding
