#!/usr/bin/env python3
"""
Upload Tufts Dental Database images (+ a COCO-shaped annotation JSON built
from annots_df/categories_df) to a lightweight per-image HF dataset repo --
same one-time, bundle-then-single-commit pattern as
upload_dentex_images_to_hf.py, so download_tufts_slice() can later fetch
only the images a given trace-gen/training run actually needs.

This intentionally raises the same NotImplementedError load_tufts_dataset()
does (see tufts.py's module docstring) until the tooth-position and
diagnosis-category mapping functions are filled in against real, verified
Tufts files. Written now so that once that mapping exists, this script is
already the right shape to just run -- not something to write from scratch
at that point.
"""
import os
import sys
import json
import shutil
import tempfile
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.config import load_config, load_env
from dental_agent.data.tufts import load_tufts_dataset

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upload Tufts images to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="HF Dataset Repo ID (e.g. rezaxr14/tufts-train-images)")
    args = parser.parse_args()

    load_env()
    cfg = load_config()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token or hf_token.startswith("your_"):
        print("ERROR: HF_TOKEN is missing or invalid in .env. Requires write access.")
        sys.exit(1)

    api = HfApi(token=hf_token)

    print("Loading Tufts dataset locally...")
    imgs_df, annots_df, cats_df = load_tufts_dataset(data_dir=cfg.data_dir)

    valid_imgs = imgs_df[imgs_df["local_path"].notna()]
    annotated_ids = set(annots_df["image_id"].unique())
    eligible_imgs = valid_imgs[valid_imgs["id"].isin(annotated_ids)]

    print(f"Found {len(eligible_imgs)} valid annotated images to upload.")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        images_dir = temp_dir_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        # Build a COCO-shaped train.json from the DataFrames (mirrors what
        # DENTEX's own train.json already looks like), so a future HF-fallback
        # read path in load_tufts_dataset (symmetric to DENTEX's) has
        # something in the same shape to read back.
        coco_json = {
            "images": imgs_df.to_dict(orient="records"),
            "annotations": annots_df.to_dict(orient="records"),
            "categories": cats_df.to_dict(orient="records"),
        }
        with open(temp_dir_path / "train.json", "w") as f:
            json.dump(coco_json, f)

        for idx, row in eligible_imgs.iterrows():
            img_id = int(row["id"])
            local_path = str(row["local_path"])
            # .jpg to match Tufts' native format (DENTEX's upload script uses
            # .png to match ITS native format) -- download_tufts_slice's
            # filename_template must stay in sync with this.
            dest_path = images_dir / f"{img_id}.jpg"
            try:
                os.link(local_path, dest_path)
            except Exception:
                shutil.copy2(local_path, dest_path)
            if (idx + 1) % 100 == 0:
                print(f"  Prepared {idx + 1}/{len(eligible_imgs)} images...")

        print(f"Uploading bundled dataset to {args.repo_id}...")
        api.upload_folder(
            folder_path=str(temp_dir_path),
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print("Upload complete.")


if __name__ == "__main__":
    main()
