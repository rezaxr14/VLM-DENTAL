import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Callable

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import pandas as pd
import yaml
from sklearn.model_selection import KFold
from tqdm import tqdm

from dental_agent.data.dentex import load_dentex_dataset, dentex_row_to_fdi
from dental_agent.data.tufts import load_tufts_tooth_boxes

# Which loader + FDI-conversion function each supported dataset uses.
# quadrant_position_fn takes one annotation row and returns (quadrant,
# tooth_position) already in proper FDI (1-4, 1-8) form. This is
# deliberately NOT the same function for every dataset: DENTEX's raw
# category_id_1/category_id_2 are 0-indexed (the documented "0-Index
# Quirk"), an artifact of DENTEX's own JSON encoding -- NOT a universal
# convention. A dataset whose own loader already outputs correct 1-indexed
# FDI values (as tufts.py's does) must NOT be run through dentex_row_to_fdi
# a second time, or it'd be double-incremented. Add a new dataset by adding
# one entry here with its own loader and its own (possibly identity)
# conversion function -- don't hardcode a new dataset's quirks into
# convert_single_image itself.
DATASET_LOADERS: dict[str, dict] = {
    "dentex": {
        "load": load_dentex_dataset,
        "quadrant_position_fn": dentex_row_to_fdi,
    },
    "tufts": {
        # load_tufts_tooth_boxes, NOT load_tufts_dataset -- convert_single_image
        # below only ever reads category_id_1/category_id_2/bbox, never
        # diagnosis, so the full ~25,000-box grounding corpus (every
        # annotated tooth, not just the ~200 images with a periapical
        # finding) is what you actually want feeding YOLO. See tufts.py's
        # module docstring for why these are two separate functions.
        "load": load_tufts_tooth_boxes,
        "quadrant_position_fn": lambda row: (int(row["category_id_1"]), int(row["category_id_2"])),
    },
    # "tunisia": {
    #     "load": load_tunisia_dataset,
    #     "quadrant_position_fn": lambda row: (int(row["category_id_1"]), int(row["category_id_2"])),
    # },
    # Uncomment once dental_agent/data/tunisia_panoramic.py's region-to-FDI
    # mapping is implemented (currently raises NotImplementedError by
    # design -- see that module's _region_to_fdi docstring). Same identity
    # (not +1) reasoning as tufts above. IMPORTANT, unlike tufts: this
    # dataset has has_diagnosis_labels=False in dataset_catalog.py -- it
    # can only ever expand locate_tooth's grounding training corpus here,
    # never feed diagnosis trace-gen. Don't wire it into anything that
    # expects category_id_3 (diagnosis) to exist.
}


def convert_single_image(
    img_row,
    annots_df,
    images_out,
    labels_out,
    quadrant_position_fn: Callable = dentex_row_to_fdi,
    dataset_tag: str | None = None,
):
    """Convert a single image's annotations to YOLO format.

    Returns the destination stem (without extension) if annotations were written,
    or None if the image has no valid tooth annotations.

    quadrant_position_fn: converts one annotation row to (quadrant,
    tooth_position) in proper FDI form -- dataset-specific, see
    DATASET_LOADERS above. Defaults to DENTEX's conversion for backward
    compatibility with existing call sites.

    dataset_tag: if given, guarantees collision-free filenames across
    datasets (e.g. "dentex_123" vs "tufts_123" for the same numeric image
    id) instead of relying on the source folder name happening to be
    unique, which was the previous, DENTEX-only-safe assumption.
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

        quadrant, position = quadrant_position_fn(ann)

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

    # Create a unique destination stem to prevent collisions. With an
    # explicit dataset_tag (multi-dataset runs), that tag plus the numeric
    # image id is guaranteed unique regardless of source folder naming --
    # two datasets can both have an image with id=5 without colliding.
    # Without one (single-dataset runs, unchanged from before), fall back
    # to the source folder name, which was only ever guaranteed unique
    # within DENTEX's own folder structure.
    if dataset_tag:
        dest_stem = f"{dataset_tag}_{img_id}_{local_path.stem}"
    else:
        folder_prefix = local_path.parent.parent.name
        dest_stem = f"{folder_prefix}_{local_path.stem}" if folder_prefix else f"img_{img_id}_{local_path.stem}"

    dest_img_path = images_out / f"{dest_stem}{local_path.suffix}"
    if not dest_img_path.exists():
        shutil.copy2(local_path, dest_img_path)

    label_path = labels_out / f"{dest_stem}.txt"
    with open(label_path, "w") as f:
        f.write("\n".join(yolo_lines) + "\n")

    return dest_stem


def _ensure_images_downloaded(
    images_df: pd.DataFrame,
    dataset_name: str,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Check if images_df has rows with missing local_path (e.g. fresh Colab session).
    If so, auto-download the missing image slices via download_dentex_slice / download_tufts_slice
    and re-resolve local paths so that all annotated images are available for YOLO dataset building."""
    if images_df.empty:
        return images_df

    missing_mask = images_df["local_path"].isna().copy()
    if "local_path" in images_df.columns:
        for idx, row in images_df.iterrows():
            lp = row.get("local_path")
            if lp and not Path(str(lp)).exists():
                missing_mask.loc[idx] = True

    missing_ids = images_df.loc[missing_mask, "id"].dropna().unique().tolist()
    missing_ids = [int(i) for i in missing_ids]

    if not missing_ids:
        return images_df

    print(f"[{dataset_name}] Found {len(missing_ids)} missing local image files out of {len(images_df)} total. Auto-downloading...")

    if dataset_name == "dentex":
        from dental_agent.data.dentex import download_dentex_slice, resolve_image_paths
        repo_id = os.environ.get("DENTEX_IMAGES_REPO")
        if repo_id:
            download_dentex_slice(missing_ids, repo_id=repo_id, cache_dir=str(data_dir) if data_dir else None)
            from dental_agent.data.dentex import find_local_dentex_dir
            local_dentex = find_local_dentex_dir(data_dir, "train")
            images_df = resolve_image_paths(images_df, local_dentex or Path(data_dir or "data"))
        else:
            from dental_agent.data.dentex import download_dentex, extract_dentex_zips, resolve_image_paths
            dentex_path = download_dentex(cache_dir=str(data_dir) if data_dir else None, full=True)
            extract_dentex_zips(dentex_path)
            images_df = resolve_image_paths(images_df, dentex_path)

    elif dataset_name == "tufts":
        from dental_agent.data.tufts import download_tufts_slice, _find_radiograph_dir, find_local_tufts_dir
        repo_id = os.environ.get("TUFTS_IMAGES_REPO")
        if repo_id:
            download_tufts_slice(missing_ids, repo_id=repo_id, cache_dir=str(data_dir) if data_dir else None)
            tufts_root = find_local_tufts_dir()
            rad_dir = _find_radiograph_dir(tufts_root) if tufts_root else None
            for idx, row in images_df.iterrows():
                if pd.isna(row.get("local_path")) or not Path(str(row.get("local_path"))).exists():
                    img_id = int(row["id"])
                    for ext in (".jpg", ".JPG", ".png", ".PNG"):
                        candidate = (rad_dir / f"{img_id}{ext}") if rad_dir else None
                        if candidate and candidate.is_file():
                            images_df.loc[idx, "local_path"] = str(candidate.resolve())
                            break
                        if data_dir:
                            hf_matches = list(Path(data_dir).glob(f"**/{img_id}{ext}"))
                            if hf_matches:
                                images_df.loc[idx, "local_path"] = str(hf_matches[0].resolve())
                                break

    valid_count = len(images_df[images_df["local_path"].notna()])
    print(f"[{dataset_name}] Successfully resolved {valid_count} / {len(images_df)} image paths.")
    return images_df


def convert_to_yolo_format(output_dir: str | Path, split: str = "train", data_dir: str = "data", datasets: list[str] | None = None):
    """Convert one or more datasets' annotations to YOLOv8 txt format (static split mode).

    datasets: names from DATASET_LOADERS to combine (default: ["dentex"], unchanged
    behavior from before this was generalized). Each dataset is loaded and converted
    independently, tagged with its own name for collision-free filenames, then merged
    into the same images_out/labels_out -- so "run YOLO over all of them" just means
    passing more names here once their loaders exist.

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
    datasets = datasets or ["dentex"]
    output_dir = Path(output_dir)
    images_out = output_dir / "images" / split
    labels_out = output_dir / "labels" / split

    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    total_written = 0
    for dataset_name in datasets:
        if dataset_name not in DATASET_LOADERS:
            print(f"Skipping unknown dataset '{dataset_name}' -- not in DATASET_LOADERS.")
            continue
        loader_spec = DATASET_LOADERS[dataset_name]

        print(f"Loading {split} split from {dataset_name} (combine_enumeration_splits={split == 'train'})...")
        try:
            if dataset_name == "dentex":
                images_df, annots_df, _ = loader_spec["load"](
                    data_dir=data_dir, split_name=split, combine_enumeration_splits=(split == "train"),
                )
            else:
                images_df, annots_df, _ = loader_spec["load"](data_dir=data_dir)
        except Exception as e:
            print(f"Failed to load '{dataset_name}' split '{split}': {e}")
            continue

        images_df = _ensure_images_downloaded(images_df, dataset_name, data_dir=data_dir)
        valid_images = images_df[images_df["local_path"].notna()].copy()
        print(f"Found {len(valid_images)} valid images in {dataset_name} for split '{split}'. Processing...")

        written = 0
        for _, img_row in tqdm(valid_images.iterrows(), total=len(valid_images), desc=dataset_name):
            result = convert_single_image(
                img_row, annots_df, images_out, labels_out,
                quadrant_position_fn=loader_spec["quadrant_position_fn"],
                dataset_tag=dataset_name,
            )
            if result is not None:
                written += 1
        print(f"{dataset_name}: {written} images written.")
        total_written += written

    print(f"Total across {len(datasets)} dataset(s): {total_written} images.")


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
    datasets: list[str] | None = None,
):
    """Create k-fold cross-validation splits from the combined training pool.

    - datasets (default ["dentex"], unchanged behavior from before this was
      generalized): which named datasets (from DATASET_LOADERS) contribute
      to the TRAINING pool that gets K-fold split.
    - The held-out TEST set is always DENTEX's own 50 official validation
      images, regardless of `datasets` -- deliberately never mixed with
      other datasets. This is what keeps locate_tooth's reported numbers
      comparable to the official DENTEX challenge leaderboard; adding more
      training data should make the tool better, not change what it's
      being measured against.
    - Each fold k gets: fold_k/{images,labels}/{train,val}/ + dataset.yaml
    - The held-out set goes to: test/{images,labels}/ + dataset.yaml
    """
    datasets = datasets or ["dentex"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load training pool, combined across every requested dataset ---
    # Each entry is (dataset_name, images_df, annots_df, quadrant_position_fn).
    # Kept as separate DataFrames per dataset (not concatenated into one) so
    # numeric image ids from different sources can never collide -- a
    # combined pool is built below using (dataset_name, id) compound keys.
    train_pools: list[tuple[str, pd.DataFrame, pd.DataFrame, Callable]] = []
    for dataset_name in datasets:
        if dataset_name not in DATASET_LOADERS:
            print(f"Skipping unknown dataset '{dataset_name}' -- not in DATASET_LOADERS.")
            continue
        loader_spec = DATASET_LOADERS[dataset_name]
        print(f"Loading training pool from {dataset_name}...")
        try:
            if dataset_name == "dentex":
                images_df, annots_df, _ = loader_spec["load"](
                    data_dir=data_dir, split_name="train", combine_enumeration_splits=True,
                )
            else:
                images_df, annots_df, _ = loader_spec["load"](data_dir=data_dir)
        except Exception as e:
            print(f"Failed to load '{dataset_name}' training pool: {e}")
            continue
        images_df = _ensure_images_downloaded(images_df, dataset_name, data_dir=data_dir)
        images_df = images_df[images_df["local_path"].notna()].copy()
        print(f"  {dataset_name} training pool: {len(images_df)} images")
        train_pools.append((dataset_name, images_df, annots_df, loader_spec["quadrant_position_fn"]))

    if not train_pools:
        print("No datasets loaded successfully -- nothing to do.")
        return

    # --- Load DENTEX's held-out validation set (50 images) -- ALWAYS DENTEX-only ---
    print(f"Loading validation split from DENTEX (held-out test set, never mixed with other datasets)...")
    val_images_df, val_annots_df, _ = load_dentex_dataset(
        data_dir=data_dir,
        split_name="validation",
        combine_enumeration_splits=False,
    )
    val_images_df = _ensure_images_downloaded(val_images_df, "dentex", data_dir=data_dir)
    val_images_df = val_images_df[val_images_df["local_path"].notna()].copy()
    print(f"Held-out test set: {len(val_images_df)} images (DENTEX only)")

    # --- Build a combined (dataset_name, image_id) pool for K-fold splitting ---
    combined_keys = [
        (dataset_name, img_id)
        for dataset_name, images_df, _, _ in train_pools
        for img_id in images_df["id"].unique()
    ]
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_ids = list(kf.split(combined_keys))

    fold_summary = {
        "n_folds": n_folds, "seed": seed, "datasets": datasets, "folds": [],
        "test_count": len(val_images_df),
    }

    for fold_idx, (train_idx, val_idx) in enumerate(fold_ids):
        print(f"\n--- Fold {fold_idx + 1}/{n_folds} ---")

        fold_dir = output_dir / f"fold_{fold_idx}"
        train_img_dir = fold_dir / "images" / "train"
        train_lbl_dir = fold_dir / "labels" / "train"
        val_img_dir = fold_dir / "images" / "val"
        val_lbl_dir = fold_dir / "labels" / "val"

        for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
            d.mkdir(parents=True, exist_ok=True)

        train_keys = {combined_keys[i] for i in train_idx}
        val_keys = {combined_keys[i] for i in val_idx}

        train_count = 0
        val_count = 0
        for dataset_name, images_df, annots_df, qp_fn in train_pools:
            fold_train_ids = {img_id for (dn, img_id) in train_keys if dn == dataset_name}
            fold_val_ids = {img_id for (dn, img_id) in val_keys if dn == dataset_name}

            fold_train_df = images_df[images_df["id"].isin(fold_train_ids)]
            fold_val_df = images_df[images_df["id"].isin(fold_val_ids)]

            for _, img_row in tqdm(fold_train_df.iterrows(), total=len(fold_train_df), desc=f"Fold {fold_idx} train ({dataset_name})"):
                result = convert_single_image(img_row, annots_df, train_img_dir, train_lbl_dir, quadrant_position_fn=qp_fn, dataset_tag=dataset_name)
                if result is not None:
                    train_count += 1

            for _, img_row in tqdm(fold_val_df.iterrows(), total=len(fold_val_df), desc=f"Fold {fold_idx} val ({dataset_name})"):
                result = convert_single_image(img_row, annots_df, val_img_dir, val_lbl_dir, quadrant_position_fn=qp_fn, dataset_tag=dataset_name)
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
        result = convert_single_image(img_row, val_annots_df, test_img_dir, test_lbl_dir, quadrant_position_fn=dentex_row_to_fdi, dataset_tag="dentex")
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
    parser = argparse.ArgumentParser(description="Prepare YOLO dataset from one or more datasets.")
    parser.add_argument(
        "--mode",
        choices=["train", "cv"],
        default="train",
        help="train = static train/val split, cv = k-fold cross-validation",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of CV folds (only used with --mode cv)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for KFold shuffle")
    parser.add_argument(
        "--datasets",
        type=str,
        default="dentex",
        help="Comma-separated dataset names from DATASET_LOADERS to combine (default: dentex only, "
             "unchanged behavior). e.g. --datasets dentex,tufts or --datasets dentex,tunisia once "
             "the respective loader's annotation mapping is implemented (both currently raise "
             "NotImplementedError by design). The held-out CV test set (--mode cv) is always "
             "DENTEX's own 50 official validation images regardless of this flag -- see "
             "prepare_cv_folds' docstring.",
    )
    args = parser.parse_args()
    raw_datasets_str = args.datasets if isinstance(args.datasets, str) else " ".join(args.datasets)
    dataset_list = [d.strip() for d in raw_datasets_str.replace(",", " ").split() if d.strip()]
    dir_suffix = "_".join(dataset_list) if dataset_list != ["dentex"] else "dentex"

    if args.mode == "train":
        yolo_dir = Path(f"data/yolo_{dir_suffix}")
        print(f"Preparing YOLO Dataset (static split mode) for {dataset_list}...")
        convert_to_yolo_format(yolo_dir, split="validation", datasets=dataset_list)
        convert_to_yolo_format(yolo_dir, split="train", datasets=dataset_list)
        create_dataset_yaml(yolo_dir, val_subdir="images/validation")
        print(f"\nDone! Dataset ready at {yolo_dir}/dataset.yaml")
    else:
        yolo_dir = Path(f"data/yolo_{dir_suffix}_cv")
        print(f"Preparing YOLO Dataset ({args.folds}-fold cross-validation mode) for {dataset_list}...")
        prepare_cv_folds(yolo_dir, n_folds=args.folds, seed=args.seed, datasets=dataset_list)
