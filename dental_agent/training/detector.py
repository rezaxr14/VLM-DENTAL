"""
Stage 0: Faster R-CNN Tooth Localization and FDI Grounding Model (§24).
Replaces the oracle detector tool with a trained specialist object detector.

Includes:
- Data augmentation with anatomical quadrant flip (`flip_quadrant`)
- Detection dataset with augmentation (`DentexDetectionDataset`, `DentalDetectionDataset`)
- Collate function (`detection_collate_fn`)
- Detector builder (`build_stage0_detector`)
- Training loop (`train_stage0_detector`)
- Learned tool integration (`tool_locate_abnormal_teeth_learned`)
- Visualizer (`visualize_detector_predictions`)
- IoU math & greedy evaluator (`compute_iou`, `evaluate_stage0_detector`)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Sequence
from PIL import Image, ImageDraw
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from tqdm import tqdm

from dental_agent.data.dentex import dentex_row_to_fdi

from dental_agent.tools.fdi import tool_fdi_label


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


def train_stage0_detector(
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    output_path: str | Path | None = None,
    epochs: int = 1,
    lr: float = 5e-4,
    batch_size: int = 2,
    subset_n: int | None = None,
    verbose_every: int = 10,
    augment: bool = True,
    device: str | None = None,
) -> Any:
    """Train the Stage 0 specialist tooth detector on DENTEX annotations."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = DentexDetectionDataset(images_df, annots_df, augment=augment)
    if subset_n:
        dataset.image_ids = dataset.image_ids[:subset_n]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=detection_collate_fn,
    )

    detector = build_stage0_detector().to(device)
    optimizer = torch.optim.AdamW([p for p in detector.parameters() if p.requires_grad], lr=lr)

    detector.train()
    for epoch in range(epochs):
        running_loss, n_batches = 0.0, 0
        for step, (images, targets) in enumerate(loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = detector(images, targets)
            loss = sum(loss_dict.values())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            if (step + 1) % verbose_every == 0:
                print(f"  epoch {epoch + 1} step {step + 1}/{len(loader)}  loss={loss.item():.3f}")
        print(f"Epoch {epoch + 1}/{epochs} done — mean loss: {running_loss / max(n_batches, 1):.3f}")

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(detector.state_dict(), output_path)
        print(f"Stage 0 detector saved to: {output_path}")

    return detector


def tool_locate_abnormal_teeth_learned(
    image_id: int,
    detector: Any,
    images_df: pd.DataFrame,
    score_threshold: float = 0.5,
) -> list[dict[str, Any]]:
    """Same return shape as the oracle tool_locate_abnormal_teeth(), but backed by a real
    trained detector. `diagnosis` is always None (Stage 0 only localizes/identifies FDI)."""
    matches = images_df[images_df["id"] == image_id]
    if matches.empty:
        return []
    row = matches.iloc[0]
    image_path = row.get("local_path")
    if not image_path or not os.path.exists(str(image_path)):
        return []

    image = Image.open(image_path).convert("RGB")
    device = next(detector.parameters()).device
    image_tensor = torchvision.transforms.functional.to_tensor(image).to(device)

    detector.eval()
    with torch.no_grad():
        prediction = detector([image_tensor])[0]

    results: list[dict[str, Any]] = []
    for box, label, score in zip(prediction["boxes"], prediction["labels"], prediction["scores"]):
        if score < score_threshold:
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
        })
    return results


def visualize_detector_predictions(
    detector: Any,
    image_id: int,
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    score_threshold: float = 0.5,
    save_path: str | Path | None = None,
) -> None:
    """Red = detector predictions, green = ground truth on the same image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    predictions = tool_locate_abnormal_teeth_learned(image_id, detector, images_df, score_threshold)

    matches = images_df[images_df["id"] == image_id]
    if matches.empty:
        print(f"Image {image_id} not found.")
        return
    row = matches.iloc[0]
    img = Image.open(row["local_path"]).convert("RGB")
    draw = ImageDraw.Draw(img)

    for p in predictions:
        x, y, w, h = p["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline="red", width=3)
        draw.text((x, max(0, y - 12)), f"pred {p['fdi_label']} ({p['score']:.2f})", fill="red")

    gt_anns = annots_df[annots_df["image_id"] == image_id]
    for _, ann in gt_anns.iterrows():
        x, y, w, h = ann["bbox"]
        quad, tooth = dentex_row_to_fdi(ann)
        fdi = tool_fdi_label(quad, tooth)
        draw.rectangle([x, y, x + w, y + h], outline="lime", width=2)
        draw.text((x, y + h + 2), f"gt {fdi}", fill="lime")

    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.title(f"image_id={image_id}: red=predicted, green=ground truth")
    plt.axis("off")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()


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


def evaluate_stage0_detector(
    detector: Any,
    image_ids: list[int],
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    iou_threshold: float = 0.5,
    score_threshold: float = 0.5,
) -> dict[str, float]:
    """Single-operating-point precision/recall/F1 via greedy IoU matching."""
    tp = fp = fn = 0
    for image_id in image_ids:
        predictions = sorted(
            tool_locate_abnormal_teeth_learned(image_id, detector, images_df, score_threshold),
            key=lambda p: -p["score"],
        )
        gt_anns = annots_df[annots_df["image_id"] == image_id]
        gt_boxes = [
            (
                *dentex_row_to_fdi(g),
                [g["bbox"][0], g["bbox"][1], g["bbox"][0] + g["bbox"][2], g["bbox"][1] + g["bbox"][3]],
            )
            for _, g in gt_anns.iterrows()
            if g["bbox"][2] > 0 and g["bbox"][3] > 0
        ]
        matched: set[int] = set()

        for p in predictions:
            p_box = [p["bbox"][0], p["bbox"][1], p["bbox"][0] + p["bbox"][2], p["bbox"][1] + p["bbox"][3]]
            best_iou, best_j = 0.0, -1
            for j, (_, _, gbox) in enumerate(gt_boxes):
                if j in matched:
                    continue
                iou = compute_iou(p_box, gbox)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            is_correct = (
                best_j >= 0 and best_iou >= iou_threshold
                and gt_boxes[best_j][0] == p["quadrant"]
                and gt_boxes[best_j][1] == p["tooth_position"]
            )
            if is_correct:
                tp += 1
                matched.add(best_j)
            else:
                fp += 1
        fn += len(gt_boxes) - len(matched)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    print(f"Stage 0 detector @ IoU>={iou_threshold}, score>={score_threshold}: "
          f"precision={precision:.3f}  recall={recall:.3f}  f1={f1:.3f}  (tp={tp} fp={fp} fn={fn})")
    return {"precision": precision, "recall": recall, "f1": f1, "tp": float(tp), "fp": float(fp), "fn": float(fn)}
