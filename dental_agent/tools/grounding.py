"""
Tooth localization and grounding tools.

Provides:
- OracleGroundingTool: Look up ground-truth bounding boxes from DENTEX annotations (§12).
- LearnedGroundingTool: Run a trained Faster R-CNN detector on an image (§24).
- compute_iou: Box IoU matching geometry helper.
"""

from __future__ import annotations

from typing import Any, Callable, Optional
from PIL import Image
import pandas as pd

from dental_agent.tools.fdi import tool_fdi_label


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


class OracleGroundingTool:
    """Oracle grounding tool backed by DENTEX ground truth annotations (§12).

    Simulates the interface of a specialist detector by returning true bounding
    boxes for abnormal teeth.
    """

    def __init__(
        self,
        annots_df: pd.DataFrame,
        categories_df: pd.DataFrame,
        diag_col: str | None = None,
    ) -> None:
        self.annots_df = annots_df
        self.categories_df = categories_df
        self.diag_col = diag_col or self._detect_diag_col()
        self.diag_lookup = (
            dict(zip(categories_df["id"], categories_df["name"]))
            if len(categories_df) else {}
        )

    def _detect_diag_col(self) -> str | None:
        for c in ("category_id_3", "category_id"):
            if c in self.annots_df.columns:
                return c
        return None

    def __call__(self, image_id: int) -> list[dict[str, Any]]:
        """Return list of abnormal tooth candidate dicts for given image_id."""
        matches = self.annots_df[self.annots_df["image_id"] == image_id]
        results = []
        for _, ann in matches.iterrows():
            quad = ann.get("category_id_1")
            pos = ann.get("category_id_2")
            diag_id = ann.get(self.diag_col) if self.diag_col else None
            diag_name = self.diag_lookup.get(diag_id, "unknown")
            results.append({
                "bbox": list(ann["bbox"]),
                "quadrant": int(quad) if pd.notna(quad) else None,
                "tooth_position": int(pos) if pd.notna(pos) else None,
                "fdi_label": tool_fdi_label(int(quad), int(pos)) if pd.notna(quad) and pd.notna(pos) else None,
                "diagnosis": diag_name,
                "is_oracle": True,
            })
        return results


def tool_locate_abnormal_teeth(
    image_id: int,
    annots_df: pd.DataFrame | None = None,
    categories_df: pd.DataFrame | None = None,
    diag_col: str = "category_id_3",
) -> list[dict[str, Any]]:
    """Functional wrapper for oracle tooth localization tool (§12)."""
    if annots_df is None:
        return []
    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )
    matches = annots_df[annots_df["image_id"] == image_id]
    results = []
    for _, ann in matches.iterrows():
        quad = ann.get("category_id_1")
        pos = ann.get("category_id_2")
        diag_id = ann.get(diag_col)
        diag_name = cat_lookup.get(diag_id, "unknown")
        results.append({
            "bbox": list(ann["bbox"]),
            "quadrant": int(quad) if pd.notna(quad) else None,
            "tooth_position": int(pos) if pd.notna(pos) else None,
            "fdi_label": tool_fdi_label(int(quad), int(pos)) if pd.notna(quad) and pd.notna(pos) else None,
            "diagnosis": diag_name,
            "is_oracle": True,
        })
    return results


class LearnedGroundingTool:
    """Learned grounding tool backed by a trained Stage 0 PyTorch detector (§24).

    Predicts bounding boxes, quadrants, and tooth positions, but leaves diagnosis
    as None so the VLM must reason about pathology after zooming in.
    """

    def __init__(
        self,
        detector: Any,
        images_df: pd.DataFrame,
        score_threshold: float = 0.5,
    ) -> None:
        self.detector = detector
        self.images_df = images_df
        self.score_threshold = score_threshold

    def __call__(self, image_id: int) -> list[dict[str, Any]]:
        import torch
        import torchvision

        row = self.images_df[self.images_df["id"] == image_id].iloc[0]
        image = Image.open(row["local_path"]).convert("RGB")
        device = next(self.detector.parameters()).device
        image_tensor = torchvision.transforms.functional.to_tensor(image).to(device)

        self.detector.eval()
        with torch.no_grad():
            prediction = self.detector([image_tensor])[0]

        results = []
        for box, label, score in zip(
            prediction["boxes"], prediction["labels"], prediction["scores"]
        ):
            if score < self.score_threshold:
                continue
            label_val = int(label.item())
            quadrant = (label_val - 1) // 8 + 1
            tooth_position = (label_val - 1) % 8 + 1
            x1, y1, x2, y2 = box.tolist()
            results.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "quadrant": quadrant,
                "tooth_position": tooth_position,
                "fdi_label": tool_fdi_label(quadrant, tooth_position),
                "score": float(score.item()),
                "diagnosis": None,
                "is_oracle": False,
            })
        return results
