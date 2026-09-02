"""Purely Offline Evaluation of Best DENTEX-Only YOLO Grounding Model on Local Disk.

Zero internet access: strictly uses files on local disk.
Evaluates both:
1. Standard Ultralytics model.val() on local validation split.
2. Target-filtered 1-to-1 greedy bipartite matching & true COCO mAP50-95.
3. Direct validation against raw DENTEX validation_triple.json annotations.
"""

import os
import sys
from pathlib import Path

# Enforce strict offline mode
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import json
import torch
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO

from scripts.train_grounding_tool import (
    evaluate_yolo_labels_target_grounding,
    evaluate_target_grounding,
)


def main():
    print("=" * 80)
    print("  PURE OFFLINE LOCAL DENTEX GROUNDING EVALUATION (Zero Internet Access)")
    print("=" * 80)

    # 1. Locate local model weights
    weight_candidates = [
        Path("data/models/grounding_tool_cv_best/weights/best.pt"),
        Path("data/models/dentex_grounding_tool_cv_best/weights/best.pt"),
        Path("runs/detect/data/models/grounding_tool/weights/best.pt"),
    ]
    weight_path = next((w for w in weight_candidates if w.exists()), None)
    if not weight_path:
        print("❌ Error: No local model weights found on disk.")
        return

    print(f"✅ Local Model Checkpoint: {weight_path} ({weight_path.stat().st_size / 1e6:.1f} MB)")
    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"✅ Inference Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    model = YOLO(str(weight_path))

    # 2. Check local YOLO dataset directories
    val_img_dir = Path("data/yolo_dentex/images/validation")
    val_lbl_dir = Path("data/yolo_dentex/labels/validation")
    yaml_path = Path("data/yolo_dentex/dataset.yaml")

    if val_img_dir.exists() and val_lbl_dir.exists():
        n_imgs = len(list(val_img_dir.glob("*.png"))) + len(list(val_img_dir.glob("*.jpg")))
        n_lbls = len(list(val_lbl_dir.glob("*.txt")))
        print(f"\n📂 Local YOLO Validation Split: {n_imgs} images, {n_lbls} label files")

        # --- Benchmark A: Target-Filtered Evaluation (COCO mAP50-95 & 1-to-1 Nominal) ---
        print("\n" + "-" * 80)
        print("  Benchmark 1: Target-Filtered Grounding (Greedy Nominal 1-to-1 Matching)")
        print("-" * 80)
        res_tf = evaluate_yolo_labels_target_grounding(
            model=model,
            img_dir=val_img_dir,
            label_dir=val_lbl_dir,
            conf_thresh=0.001,
            nominal_conf_thresh=0.25,
            imgsz=640,
            device=device,
        )

        print(f"  Target mAP50:      {res_tf['map50']:.4f}")
        print(f"  Target mAP50-95:   {res_tf['map50_95']:.4f}")
        print(f"  Nominal Precision: {res_tf['precision']:.4f}")
        print(f"  Nominal Rec@0.50:  {res_tf['recall_50']:.4f}")
        print(f"  Nominal Rec@0.75:  {res_tf['recall_75']:.4f}")
        print(f"  Nominal Mean IoU:  {res_tf['mean_iou']:.4f}")
        print(f"  Total GT Targets:  {res_tf['total_targets']}")

        # --- Benchmark B: Standard Ultralytics model.val() ---
        if yaml_path.exists():
            print("\n" + "-" * 80)
            print("  Benchmark 2: Standard Full-Arch model.val() (Unfiltered)")
            print("-" * 80)
            try:
                val_res = model.val(
                    data=str(yaml_path),
                    split="val",
                    imgsz=640,
                    device=device,
                    verbose=False,
                )
                print(f"  Raw mAP50:         {val_res.box.map50:.4f}")
                print(f"  Raw mAP50-95:      {val_res.box.map:.4f}")
                print(f"  Raw Precision:     {val_res.box.mp:.4f}")
                print(f"  Raw Recall:        {val_res.box.mr:.4f}")
            except Exception as e:
                print(f"  Notice during model.val(): {e}")

    # 3. Direct verification against raw DENTEX validation_triple.json (if available)
    raw_json = Path("data/dentex/DENTEX/validation_triple.json")
    raw_xrays = Path("data/dentex/DENTEX/validation_data/quadrant_enumeration_disease/xrays")
    if raw_json.exists() and raw_xrays.exists():
        print("\n" + "-" * 80)
        print("  Benchmark 3: Direct Raw COCO Annotation Evaluation (validation_triple.json)")
        print("-" * 80)
        with open(raw_json, "r", encoding="utf-8") as f:
            coco_data = json.load(f)

        images_df = pd.DataFrame(coco_data.get("images", []))
        annots_df = pd.DataFrame(coco_data.get("annotations", []))
        
        # Map local image paths
        images_df["local_path"] = images_df["file_name"].apply(
            lambda fn: str(raw_xrays / Path(fn).name) if (raw_xrays / Path(fn).name).exists() else None
        )
        valid_df = images_df[images_df["local_path"].notna()].copy()
        print(f"  Found {len(valid_df)} / {len(images_df)} raw test images locally ({len(annots_df)} annotations)")

        if len(valid_df) > 0:
            res_raw = evaluate_target_grounding(
                model=model,
                val_images_df=valid_df,
                val_annots_df=annots_df,
                conf_thresh=0.001,
                nominal_conf_thresh=0.25,
                imgsz=640,
                device=device,
            )
            print(f"  Target mAP50:      {res_raw['map50']:.4f}")
            print(f"  Target mAP50-95:   {res_raw['map50_95']:.4f}")
            print(f"  Nominal Precision: {res_raw['precision']:.4f}")
            print(f"  Nominal Rec@0.50:  {res_raw['recall_50']:.4f}")
            print(f"  Nominal Rec@0.75:  {res_raw['recall_75']:.4f}")
            print(f"  Nominal Mean IoU:  {res_raw['mean_iou']:.4f}")
            print(f"  Total GT Targets:  {res_raw['total_targets']}")

    print("\n" + "=" * 80)
    print("  EVALUATION COMPLETE (100% Offline)")
    print("=" * 80)


if __name__ == "__main__":
    main()
