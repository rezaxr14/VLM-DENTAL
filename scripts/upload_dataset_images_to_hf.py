#!/usr/bin/env python3
"""
Upload one dataset's images (+ an annotation JSON) to its own lightweight
per-image HF dataset repo -- the "upload once, download only what a given
run needs" mechanism every dataset in this project shares (see
dental_agent/data/hf_dataset_utils.py).

This was two near-identical scripts (upload_dentex_images_to_hf.py,
upload_tufts_images_to_hf.py) with the same overall shape -- stage a temp
folder, copy images, write/copy an annotation JSON, one upload_folder commit
-- but different bundle-preparation logic. Unified into one so the SHARED
parts can't independently diverge the way the FDI-index conversion did
across files before that got consolidated into dentex_row_to_fdi(). Add a
new dataset by adding one bundler function + one DATASET_BUNDLERS entry,
not a new copy of this whole script.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
import pandas as pd

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.config import load_config, load_env

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)


def _prepare_dentex_bundle(temp_dir_path: Path, cfg):
    """Copies DENTEX's full training set (all 1,339 images from both
    quadrant-enumeration-disease and quadrant_enumeration) plus validation
    split into temp_dir_path, generating merged train.json.
    """
    from dental_agent.data.dentex import load_dentex_dataset, find_local_dentex_dir

    print(f"Loading DENTEX combined dataset from {cfg.data_dir}...")
    dentex_path = find_local_dentex_dir(cfg.data_dir, split_name="train")
    if not dentex_path:
        print("ERROR: Could not find DENTEX dataset locally. Run download_dataset.py first.")
        sys.exit(1)

    imgs_df, annots_df, cats_df = load_dentex_dataset(
        data_dir=cfg.data_dir, split_name="train", combine_enumeration_splits=True
    )

    coco_json = {
        "images": imgs_df.to_dict(orient="records"),
        "annotations": annots_df.to_dict(orient="records"),
        "categories": cats_df.to_dict(orient="records"),
    }
    train_json_path = temp_dir_path / "train.json"
    print(f"Writing combined train.json ({len(imgs_df)} images, {len(annots_df)} annotations)...")
    with open(train_json_path, "w") as f:
        json.dump(coco_json, f)

    train_eligible = imgs_df[imgs_df["local_path"].notna()].copy()

    val_json_path = dentex_path / "validation_triple.json"
    if not val_json_path.exists():
        val_candidates = sorted(dentex_path.rglob("*validation_triple*.json"))
        if val_candidates:
            val_json_path = val_candidates[0]

    val_eligible = pd.DataFrame()
    if val_json_path.exists():
        print(f"Copying validation JSON ({val_json_path.name}) to temp folder as validation_triple.json...")
        shutil.copy2(val_json_path, temp_dir_path / "validation_triple.json")
        try:
            val_imgs_df, val_annots_df, _ = load_dentex_dataset(
                data_dir=cfg.data_dir, split_name="validation", combine_enumeration_splits=False
            )
            val_eligible = val_imgs_df[val_imgs_df["local_path"].notna()].copy()
            print(f"Found {len(val_eligible)} validation images to include in upload bundle.")
        except Exception as e:
            print(f"Warning: Could not load validation split images for bundling: {e}")
    else:
        print(f"WARNING: No validation_triple.json found under {dentex_path}.")

    return [
        (train_eligible, "png", "train_images"),
        (val_eligible, "png", "validation_images") if not val_eligible.empty else None
    ]


def _prepare_tufts_bundle(temp_dir_path: Path, cfg):
    """Stages Tufts full dataset: 1,000 images under Radiographs/ and full
    annotation JSONs under Segmentation/, Expert/, and Student/.
    """
    from dental_agent.data.tufts import (
        find_local_tufts_dir,
        _find_radiograph_dir,
        _find_annotation_file,
        load_tufts_tooth_boxes,
        load_tufts_dataset,
    )

    tufts_root = find_local_tufts_dir()
    if tufts_root is None:
        print("ERROR: No local Tufts Dental Database directory found.")
        sys.exit(1)

    radiograph_dir = _find_radiograph_dir(tufts_root)
    if radiograph_dir is None:
        print(f"ERROR: Found {tufts_root} but couldn't locate a Radiographs folder with images.")
        sys.exit(1)

    print("Copying Tufts annotations...")
    for sub, fn in [
        ("Segmentation", "teeth_bbox.json"),
        ("Segmentation", "teeth_polygon.json"),
        ("Expert", "expert.json"),
        ("Student", "student.json"),
    ]:
        p = _find_annotation_file(tufts_root, sub, fn)
        if p and p.exists():
            dest = temp_dir_path / sub / fn
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
            print(f"  Staged {sub}/{fn}")

    # Build COCO-shaped train.json from load_tufts_dataset for trace-gen compatibility
    imgs_df, annots_df, cats_df = load_tufts_dataset(data_dir=cfg.data_dir)
    coco_json = {
        "images": imgs_df.to_dict(orient="records"),
        "annotations": annots_df.to_dict(orient="records"),
        "categories": cats_df.to_dict(orient="records"),
    }
    with open(temp_dir_path / "train.json", "w") as f:
        json.dump(coco_json, f)

    # Return all 1,000 images from load_tufts_tooth_boxes
    all_imgs_df, _, _ = load_tufts_tooth_boxes(data_dir=cfg.data_dir)
    return [(all_imgs_df, "JPG", "Radiographs")]


def _prepare_tunisia_bundle(temp_dir_path: Path, cfg):
    """Writes a fresh COCO-shaped train.json built from Tunisia's own
    DataFrames -- same approach as _prepare_tufts_bundle.
    """
    from dental_agent.data.tunisia_panoramic import load_tunisia_dataset

    print("Loading Tunisia (Panoramic Dental Xray Dataset) locally...")
    imgs_df, annots_df, cats_df = load_tunisia_dataset(data_dir=cfg.data_dir)

    coco_json = {
        "images": imgs_df.to_dict(orient="records"),
        "annotations": annots_df.to_dict(orient="records"),
        "categories": cats_df.to_dict(orient="records"),
    }
    with open(temp_dir_path / "train.json", "w") as f:
        json.dump(coco_json, f)

    valid_imgs = imgs_df[imgs_df["local_path"].notna()]
    annotated_ids = set(annots_df["image_id"].unique())
    eligible_imgs = valid_imgs[valid_imgs["id"].isin(annotated_ids)]
    return [(eligible_imgs, "jpg", "images")]


# Add a new dataset by adding one entry here, pointing at a new bundler
# function above.
DATASET_BUNDLERS = {
    "dentex": _prepare_dentex_bundle,
    "tufts": _prepare_tufts_bundle,
    "tunisia": _prepare_tunisia_bundle,
}


def main():
    parser = argparse.ArgumentParser(description="Upload a dataset's images to Hugging Face Hub")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list(DATASET_BUNDLERS.keys()),
        help="Which dataset to upload (dentex, tufts, tunisia).",
    )
    parser.add_argument("--repo-id", type=str, required=True, help="HF Dataset Repo ID (e.g. Reza-Nadimi/dentex-train-images)")
    parser.add_argument("--public", action="store_true", help="Make repo public (default: private)")
    args = parser.parse_args()

    load_env()
    cfg = load_config()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token or hf_token.startswith("your_"):
        print("ERROR: HF_TOKEN is missing or invalid in .env. Requires write access.")
        sys.exit(1)

    api = HfApi(token=hf_token)
    api.create_repo(repo_id=args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)

        bundles = DATASET_BUNDLERS[args.dataset](temp_dir_path, cfg)
        for bundle in bundles:
            if not bundle: continue
            eligible_imgs, image_ext, target_subfolder = bundle
            dest_images_dir = temp_dir_path / target_subfolder
            dest_images_dir.mkdir(parents=True, exist_ok=True)
            print(f"Found {len(eligible_imgs)} valid annotated images to stage into {target_subfolder}/...")

            for idx, row in eligible_imgs.reset_index(drop=True).iterrows():
                img_id = int(row["id"])
                local_path = str(row["local_path"])
                dest_path = dest_images_dir / f"{img_id}.{image_ext}"
                try:
                    os.link(local_path, dest_path)
                except Exception:
                    shutil.copy2(local_path, dest_path)
                if (idx + 1) % 250 == 0 or (idx + 1) == len(eligible_imgs):
                    print(f"  Staged {idx + 1}/{len(eligible_imgs)} images into {target_subfolder}/...")

        print(f"Uploading bundled dataset to {args.repo_id} (private={not args.public})...")
        max_retries = 12
        for attempt in range(max_retries):
            try:
                api.upload_folder(
                    folder_path=str(temp_dir_path),
                    repo_id=args.repo_id,
                    repo_type="dataset",
                )
                print("Upload commit succeeded.")
                break
            except Exception as e:
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate limit" in err_str.lower()
                is_network_drop = any(k in err_str.lower() for k in ["ssl", "eof", "connection", "retry", "timeout", "socket", "uploading"])

                if is_rate_limit:
                    wait_s = 60 * (attempt + 1)
                    print(f"  [HF Rate Limit 429] Waiting {wait_s}s for Hugging Face quota cooldown (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_s)
                elif is_network_drop and attempt < max_retries - 1:
                    wait_s = 5 * (attempt + 1)
                    print(f"  [Network Drop/SSL EOF] Notice: {e}. Resuming upload where it left off in {wait_s}s (attempt {attempt+1}/{max_retries})...")
                    time.sleep(wait_s)
                else:
                    raise e

    print("Upload complete.")


if __name__ == "__main__":
    main()
