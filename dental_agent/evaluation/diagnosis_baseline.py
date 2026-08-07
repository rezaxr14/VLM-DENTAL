"""
Baseline #5: Supervised Faster R-CNN Diagnosis Detector Baseline (§30).

Trains a standard object detector directly on pathology/diagnosis classes
(rather than FDI numbers) as a supervised non-agent baseline comparison.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from PIL import Image
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader

from dental_agent.training.detector import build_stage0_detector, detection_collate_fn


class DentexDiagnosisDetectionDataset(Dataset):
    """Dataset labeling each tooth bounding box by DIAGNOSIS class instead of FDI position."""

    def __init__(
        self,
        images_df_in: pd.DataFrame,
        annots_df_in: pd.DataFrame,
        diag_col_in: str,
        categories_df_in: pd.DataFrame | None,
    ) -> None:
        self.images_lookup = images_df_in.dropna(subset=["local_path"]).set_index("id")
        self.image_ids = sorted([i for i in annots_df_in["image_id"].unique() if i in self.images_lookup.index])
        self.annots_df = annots_df_in
        self.diag_col = diag_col_in

        cat_lookup = (
            dict(zip(categories_df_in["id"], categories_df_in["name"]))
            if categories_df_in is not None and len(categories_df_in)
            else {}
        )
        self.cat_lookup = cat_lookup

        present_names = (
            sorted(annots_df_in[diag_col_in].map(cat_lookup).dropna().unique())
            if diag_col_in in annots_df_in
            else ["Caries"]
        )
        self.class_to_idx = {name: i + 1 for i, name in enumerate(present_names)}  # 0 = background
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}

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
            diag_name = self.cat_lookup.get(ann.get(self.diag_col), str(ann.get(self.diag_col)))
            if diag_name not in self.class_to_idx:
                continue
            boxes.append([float(x), float(y), float(x + w), float(y + h)])
            labels.append(self.class_to_idx[diag_name])

        boxes_t = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([image_id])}
        image_tensor = torchvision.transforms.functional.to_tensor(image)
        return image_tensor, target


def train_diagnosis_baseline_detector(
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    diag_col: str = "category_id_3",
    categories_df: pd.DataFrame | None = None,
    holdout_ids: set[int] | None = None,
    output_path: str | Path | None = None,
    epochs: int = 1,
    batch_size: int = 2,
    lr: float = 5e-4,
    subset_n: int | None = None,
    verbose_every: int = 10,
    device: str | None = None,
) -> Any:
    """Plain supervised detector trained directly on diagnosis labels."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_annots = annots_df[~annots_df["image_id"].isin(holdout_ids)] if holdout_ids else annots_df
    dataset = DentexDiagnosisDetectionDataset(images_df, train_annots, diag_col, categories_df)
    if subset_n:
        dataset.image_ids = dataset.image_ids[:subset_n]

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=detection_collate_fn)

    num_classes = len(dataset.class_to_idx) + 1
    detector = build_stage0_detector(num_classes=num_classes).to(device)
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

    detector.class_to_idx = dataset.class_to_idx
    detector.idx_to_class = dataset.idx_to_class

    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        torch.save(detector.state_dict(), output_path)
        print(f"Diagnosis detector saved to: {output_path}")

    return detector


def evaluate_diagnosis_baseline_detector(
    detector: Any,
    image_ids: list[int],
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    score_threshold: float = 0.5,
    diag_col: str = "category_id_3",
    categories_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Diagnosis accuracy for the baseline detector — highest-confidence prediction per image."""
    correct = total = 0
    device = next(detector.parameters()).device
    detector.eval()

    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )

    for image_id in image_ids:
        gt_anns = annots_df[annots_df["image_id"] == image_id]
        if gt_anns.empty:
            continue
        gt_diag = cat_lookup.get(gt_anns.iloc[0].get(diag_col), str(gt_anns.iloc[0].get(diag_col)))

        matches = images_df[images_df["id"] == image_id]
        if matches.empty:
            continue
        row = matches.iloc[0]
        image = Image.open(row["local_path"]).convert("RGB")
        image_tensor = torchvision.transforms.functional.to_tensor(image).to(device)

        with torch.no_grad():
            prediction = detector([image_tensor])[0]

        total += 1
        if len(prediction["scores"]) == 0 or prediction["scores"][0] < score_threshold:
            continue
        pred_label = int(prediction["labels"][0].item())
        pred_diag = getattr(detector, "idx_to_class", {}).get(pred_label)
        if pred_diag and pred_diag.lower() == str(gt_diag).lower():
            correct += 1

    accuracy = correct / total if total else 0.0
    print(f"Diagnosis-baseline detector accuracy: {accuracy:.3f} ({correct}/{total})")
    return {"accuracy": accuracy, "n": total, "correct": correct}
