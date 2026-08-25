"""
Stage 0: Faster R-CNN detector architecture (§24).

What remains here after `locate_abnormal_teeth` was removed as a registered
tool (see roadmap.md's changelog and dental_agent/tools/registry.py): the
project decided the agent finds and corrects abnormal-tooth grounding via
`locate_tooth` (the trained, live YOLO detector) + `nudge_crop`'s
self-correction loop, not a second, separate learned detector backend.
`train_stage0_detector` and `tool_locate_abnormal_teeth_learned` (the
FDI-position-specific Faster R-CNN trainer and the tool-integration wrapper
around its output) existed only to serve that removed tool and have been
removed along with it, as has `visualize_detector_predictions` (which only
ever visualized that tool's output). Their sole prior call sites
(`scripts/run_detector.py`, `dental_agent/cli.py`'s `train_detector`
command) were removed too, for the same reason.

What's kept, because it's genuinely reused elsewhere -- specifically by
`dental_agent/evaluation/diagnosis_baseline.py`, which trains and evaluates
a plain supervised object detector directly on diagnosis labels as the
paper's "prior supervised detector" comparison baseline (a real, still-needed
part of the evaluation plan, not related to `locate_abnormal_teeth` at all):
- `build_stage0_detector`: generic Faster R-CNN MobileNetV3 builder,
  parameterized by `num_classes` -- used for 32-class FDI grounding
  originally, now reused as-is for diagnosis_baseline.py's diagnosis-class
  head.
- `detection_collate_fn`: generic torchvision detection collate function.
- `DentexDetectionDataset`/`DentalDetectionDataset`, `flip_quadrant`,
  `compute_iou`: kept as general-purpose detection-dataset/IoU utilities
  (`compute_iou` specifically is imported directly by `tests/test_tools.py`)
  even though their most direct original use (feeding the now-removed
  FDI-position training loop) is gone -- removing them isn't necessary to
  remove `locate_abnormal_teeth` and risks breaking something that reuses
  them without a clear benefit.

`evaluate_stage0_detector` (an FDI-position precision/recall/F1 evaluator)
was removed along with the rest: it called `tool_locate_abnormal_teeth_learned`
directly and its correctness check was hardcoded to that function's
quadrant/tooth_position output shape, so it wasn't actually a reusable
general-purpose evaluator the way `compute_iou` is -- it wasn't reused by
`diagnosis_baseline.py`'s own separate evaluator
(`evaluate_diagnosis_baseline_detector`), which checks diagnosis-category
correctness, not FDI position.

Includes:
- Data augmentation with anatomical quadrant flip (`flip_quadrant`)
- Detection dataset with augmentation (`DentexDetectionDataset`, `DentalDetectionDataset`)
- Collate function (`detection_collate_fn`)
- Detector builder (`build_stage0_detector`)
- IoU math (`compute_iou`)
"""

from __future__ import annotations

import random
from typing import Any, Sequence
from PIL import Image
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from dental_agent.data.dentex import dentex_row_to_fdi


def flip_quadrant(quadrant: int) -> int:
    """A horizontal flip swaps anatomical left/right, so under FDI notation quadrant
    1<->2 (upper right/left) and 3<->4 (lower right/left) swap too.
    Tooth position within a quadrant (1-8) does NOT change under a flip."""
    return {1: 2, 2: 1, 3: 4, 4: 3}.get(quadrant, quadrant)


class DentexDetectionDataset(Dataset):
    """Wraps DENTEX quadrant-enumeration annotations as a torchvision detection dataset:
    each abnormal tooth becomes one box, labeled by its FDI position (class 1-32; class 0 background).
    With augment=True, applies a horizontal flip to ~50% of examples with quadrant relabeling."""

    def __init__(
        self,
        images_df: pd.DataFrame,
        annots_df: pd.DataFrame,
        augment: bool = False,
    ) -> None:
        self.images_lookup = images_df.dropna(subset=["local_path"]).set_index("id")
        self.image_ids = sorted([i for i in annots_df["image_id"].unique() if i in self.images_lookup.index])
        self.annots_df = annots_df
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id = self.image_ids[idx]
        row = self.images_lookup.loc[image_id]
        image = Image.open(row["local_path"]).convert("RGB")
        anns = self.annots_df[self.annots_df["image_id"] == image_id]

        boxes: list[list[float]] = []
        labels: list[int] = []
        for _, ann in anns.iterrows():
            x, y, w, h = ann["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            quad, tooth = dentex_row_to_fdi(ann)
            labels.append((quad - 1) * 8 + tooth)  # 1-32; 0 reserved for background

        image_tensor = torchvision.transforms.functional.to_tensor(image)

        if self.augment and random.random() < 0.5 and boxes:
            image_tensor = torchvision.transforms.functional.hflip(image_tensor)
            img_w = image_tensor.shape[-1]
            flipped_boxes: list[list[float]] = []
            flipped_labels: list[int] = []
            for (x1, y1, x2, y2), label in zip(boxes, labels):
                flipped_boxes.append([float(img_w - x2), y1, float(img_w - x1), y2])
                quad, tooth = (label - 1) // 8 + 1, (label - 1) % 8 + 1
                new_quad = flip_quadrant(quad)
                flipped_labels.append((new_quad - 1) * 8 + tooth)
            boxes, labels = flipped_boxes, flipped_labels

        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([image_id])}
        return image_tensor, target


# Alias for backward compatibility
DentalDetectionDataset = DentexDetectionDataset


def detection_collate_fn(batch: Sequence[Any]) -> tuple[Any, ...]:
    """Collate function for torchvision detection models."""
    return tuple(zip(*batch))


def build_stage0_detector(num_classes: int = 33) -> Any:
    """Instantiate a Faster R-CNN MobileNetV3-Large FPN detector with custom class head (32 FDI classes + background)."""
    try:
        detector = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    except Exception:
        detector = fasterrcnn_mobilenet_v3_large_fpn(weights=None)
    in_features = detector.roi_heads.box_predictor.cls_score.in_features
    detector.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return detector


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """IoU between two [x1, y1, x2, y2] boxes."""
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, ya2 - ya1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0
