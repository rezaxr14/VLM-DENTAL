import argparse
import json
import os
import shutil
from pathlib import Path

import pandas as pd
import yaml
from sklearn.model_selection import KFold
from tqdm import tqdm

from dental_agent.data.dentex import load_dentex_dataset


def convert_single_image(img_row, annots_df, images_out, labels_out):
    """Convert a single image's annotations to YOLO format.

    Returns the destination stem (without extension) if annotations were written,
    or None if the image has no valid tooth annotations.
    """
    img_id = img_row["id"]
    local_path = Path(img_row["local_path"])

    if not local_path.exists():
        return None

    img_w = img_row["width"]
    img_h = img_row["height"]

    img_annots = annots_df[annots_df["image_id"] == img_id]

    yolo_lines = []
    for _, ann in img_annots.iterrows():
        if pd.isna(ann.get("category_id_1")) or pd.isna(ann.get("category_id_2")):
            continue

        # DENTEX's category_id_1/category_id_2 are 0-indexed (quadrant 0-3, position 0-7) --
        # convert to FDI (1-4, 1-8) before using them.
        quadrant = int(ann["category_id_1"]) + 1
        position = int(ann["category_id_2"]) + 1

        # Map FDI (Quadrant 1-4, Position 1-8) to YOLO Class (0-31)
        if not (1 <= quadrant <= 4 and 1 <= position <= 8):
            continue

        class_idx = (quadrant - 1) * 8 + (position - 1)

        # COCO bbox: [x_min, y_min, width, height]
        bbox = ann["bbox"]
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue

        x_min, y_min, bw, bh = bbox

        # YOLO format: normalized x_center, y_center, width, height
        x_center = (x_min + bw / 2.0) / img_w
        y_center = (y_min + bh / 2.0) / img_h
        w_norm = bw / img_w
        h_norm = bh / img_h

        # Clamp values between 0 and 1
        x_center = max(0.0, min(1.0, x_center))
        y_center = max(0.0, min(1.0, y_center))
        w_norm = max(0.0, min(1.0, w_norm))
        h_norm = max(0.0, min(1.0, h_norm))

        yolo_lines.append(f"{class_idx} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

    if not yolo_lines:
        return None

    # Create a unique destination stem to prevent cross-folder collisions
    folder_prefix = local_path.parent.parent.name
    dest_stem = f"{folder_prefix}_{local_path.stem}" if folder_prefix else f"img_{img_id}_{local_path.stem}"

    dest_img_path = images_out / f"{dest_stem}{local_path.suffix}"
    if not dest_img_path.exists():
        shutil.copy2(local_path, dest_img_path)

    label_path = labels_out / f"{dest_stem}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines) + "\n")

    return dest_stem


def convert_to_yolo_format(output_dir: str | Path, split: str = "train", data_dir: str = "data"):
    """Convert DENTEX COCO annotations to YOLOv8 txt format (static split mode).

    Output structure:
        output_dir/
            images/
                train/
                val/
            labels/
                train/
                val/
            dataset.yaml
    """
    output_dir = Path(output_dir)
    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {split} split from DENTEX (combine_enumeration_splits={split == 'train'})...")
    try:
        images_df, annots_df, _ = load_dentex_dataset(
            data_dir=data_dir,
            split_name=split,
            combine_enumeration_splits=(split == "train"),
        )
    except Exception as e:
        print(f"Failed to load split '{split}': {e}")
        return

    valid_images = images_df[images_df["local_path"].notna()].copy()
    print(f"Found {len(valid_images)} valid images for split '{split}'. Processing...")

    for _, img_row in tqdm(valid_images.iterrows(), total=len(valid_images)):
        convert_single_image(img_row, annots_df, images_out, labels_out)


def create_dataset_yaml(output_dir: str | Path, val_subdir: str = "images/validation"):
    """Generate the dataset.yaml file required by ultralytics YOLO."""
    output_dir = Path(output_dir)

    names = {}
    for q in range(1, 5):
        for p in range(1, 9):
            idx = (q - 1) * 8 + (p - 1)
            names[idx] = f"Tooth {q}{p}"

    yaml_content = {
        "path": str(output_dir.absolute()),
        "train": "images/train",
        "val": val_subdir,
        "names": names,
    }

    with open(output_dir / "dataset.yaml", "w") as f:
        yaml.dump(yaml_content, f, sort_keys=False)


def prepare_cv_folds(
    output_dir: str | Path,
    n_folds: int = 5,
    data_dir: str = "data",
    seed: int = 42,
):
    """Create k-fold cross-validation splits from DENTEX training data.

    - The 1339 training images are split into k folds via KFold.
    - The 50 validation images are held out permanently as a separate test set.
    - Each fold k gets: fold_k/{images,labels}/{train,val}/ + dataset.yaml
    - The held-out set goes to: test/{images,labels}/ + dataset.yaml
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load training pool (1339 images) ---
    print(f"Loading training split from DENTEX (combine_enumeration_splits=True)...")
    train_images_df, train_annots_df, _ = load_dentex_dataset(
        data_dir=data_dir,
        split_name="train",
        combine_enumeration_splits=True,
    )
    train_images_df = train_images_df[train_images_df["local_path"].notna()].copy()
    print(f"Training pool: {len(train_images_df)} images")

    # --- Load held-out validation set (50 images) ---
    print(f"Loading validation split from DENTEX (held-out test set)...")
    val_images_df, val_annots_df, _ = load_dentex_dataset(
        data_dir=data_dir,
        split_name="validation",
        combine_enumeration_splits=False,
    )
    val_images_df = val_images_df[val_images_df["local_path"].notna()].copy()
    print(f"Held-out test set: {len(val_images_df)} images")

    # --- Create k-fold splits ---
    image_ids = train_images_df["id"].unique()
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_ids = list(kf.split(image_ids))

    fold_summary = {"n_folds": n_folds, "seed": seed, "folds": [], "test_count": len(val_images_df)}

    for fold_idx, (train_idx, val_idx) in enumerate(fold_ids):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        fold_dir = output_dir / f"fold_{fold_idx}"
        train_img_dir = fold_dir / "images" / "train"
        train_lbl_dir = fold_dir / "labels" / "train"
        val_img_dir = fold_dir / "images" / "val"
        val_lbl_dir = fold_dir / "labels" / "val"

        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Get image IDs for this fold's train/val splits
        train_ids = set(image_ids[train_idx])
        val_ids = set(image_ids[val_idx])

        fold_train_df = train_images_df[train_images_df["id"].isin(train_ids)]
        fold_val_df = train_images_df[train_images_df["id"].isin(val_ids)]

        # Convert train images
        train_count = 0
        for _, img_row in tqdm(fold_train_df.iterrows(), total=len(fold_train_df), desc=f"Fold {fold_idx} train"):
            result = convert_single_image(img_row, train_annots_df, train_img_dir, train_lbl_dir)
            if result is not None:
                train_count += 1

        # Convert val images
        val_count = 0
        for _, img_row in tqdm(fold_val_df.iterrows(), total=len(fold_val_df), desc=f"Fold {fold_idx} val"):
            result = convert_single_image(img_row, train_annots_df, val_img_dir, val_lbl_dir)
            if result is not None:
                val_count += 1

        # Generate dataset.yaml for this fold
        create_dataset_yaml(fold_dir, val_subdir="images/val")

        print(f"Fold {fold_idx}: {train_count} train images, {val_count} val images")
        fold_summary["folds"].append({
            "fold": fold_idx,
            "train_count": train_count,
            "val_count": val_count,
        })

    # --- Copy held-out test set (50 val images, never trained on) ---
    print(f"\n--- Held-out test set ---")
    test_dir = output_dir / "test"
    test_img_dir = test_dir / "images"
    test_lbl_dir = test_dir / "labels"
    test_img_dir.mkdir(parents=True, exist_ok=True)
    test_lbl_dir.mkdir(parents=True, exist_ok=True)

    test_count = 0
    for _, img_row in tqdm(val_images_df.iterrows(), total=len(val_images_df), desc="Test set"):
        result = convert_single_image(img_row, val_annots_df, test_img_dir, test_lbl_dir)
        if result is not None:
            test_count += 1

    create_dataset_yaml(test_dir, val_subdir="images")
    print(f"Test set: {test_count} images (held out, never used in training)")

    # --- Write fold summary ---
    fold_summary_path = output_dir / "fold_summary.json"
    with open(fold_summary_path, "w") as f:
        json.dump(fold_summary, f, indent=2)
    print(f"\nFold summary saved to {fold_summary_path}")
    print(f"Done! CV dataset ready at {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from DENTEX.")
    parser.add_argument(
        "--mode",
        choices=["train", "cv"],
        default="train",
        help="train = static train/val split, cv = k-fold cross-validation",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (only used with --mode cv)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for KFold shuffle")
    args = parser.parse_args()

    if args.mode == "train":
        yolo_dir = Path("data/yolo_dentex")
        print("Preparing YOLO Dataset (static split mode)...")
        convert_to_yolo_format(yolo_dir, split="validation")
        convert_to_yolo_format(yolo_dir, split="train")
        create_dataset_yaml(yolo_dir, val_subdir="images/validation")
        print(f"\nDone! Dataset ready at {yolo_dir}/dataset.yaml")
    else:
        yolo_dir = Path("data/yolo_dentex_cv")
        print(f"Preparing YOLO Dataset ({args.folds}-fold cross-validation mode)...")
        prepare_cv_folds(yolo_dir, n_folds=args.folds, seed=args.seed)
