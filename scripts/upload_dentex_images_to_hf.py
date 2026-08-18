#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path
import tempfile
import shutil

# Ensure dental_agent is importable
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.config import load_config, load_env
from dental_agent.data.dentex import load_dentex_dataset

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Upload DENTEX images to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="HF Dataset Repo ID (e.g. rezaxr14/dentex-train-images)")
    args = parser.add_argument_group()
    args = parser.parse_args()

    load_env()
    cfg = load_config()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token or hf_token.startswith("your_"):
        print("ERROR: HF_TOKEN is missing or invalid in .env. Requires write access.")
        sys.exit(1)

    api = HfApi(token=hf_token)
    
    print(f"Loading dataset from {cfg.data_dir}...")
    imgs_df, annots_df, cats_df = load_dentex_dataset(
        data_dir=cfg.data_dir, split_name="train", combine_enumeration_splits=False
    )
    
    valid_imgs = imgs_df[imgs_df["local_path"].notna()]
    annotated_ids = set(annots_df["image_id"].unique())
    eligible_imgs = valid_imgs[valid_imgs["id"].isin(annotated_ids)]

    print(f"Found {len(eligible_imgs)} valid annotated images to upload.")

    # Prepare a temporary directory to bundle all files into a single commit
    print(f"Preparing {len(eligible_imgs)} images for bulk upload...")
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        images_dir = temp_dir_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Copy annotation JSON. The original DENTEX zip names this file after its own
        # annotation tier (e.g. train_quadrant_enumeration_disease.json), not "train.json" --
        # search for it rather than hardcoding an exact name, same approach dentex.py itself
        # uses elsewhere for this reason. dentex.py's HF-download fallback (load_dentex_dataset /
        # load_combined_dentex_dataset) is hardcoded to request exactly "train.json" from the
        # repo root, though, so the destination name below must stay exactly that.
        quadrant_disease_dir = Path(cfg.data_dir) / "DENTEX" / "training_data" / "quadrant-enumeration-disease"
        json_candidates = sorted(quadrant_disease_dir.glob("*train*.json"))
        if not json_candidates:
            print(f"ERROR: No train annotation JSON found under {quadrant_disease_dir} "
                  f"(looked for *train*.json). Aborting -- uploading images without the "
                  f"annotation JSON produces a repo that download_dentex_slice() can't use "
                  f"(it 404s looking for train.json at the repo root). Run download_and_cleanup.py "
                  f"first if the dataset isn't extracted locally yet.")
            sys.exit(1)
        if len(json_candidates) > 1:
            print(f"ERROR: Multiple candidate JSON files found under {quadrant_disease_dir}: "
                  f"{[p.name for p in json_candidates]}. Pick the right one and set json_path "
                  f"explicitly rather than relying on the glob picking correctly.")
            sys.exit(1)
        json_path = json_candidates[0]
        print(f"Copying annotations JSON ({json_path.name}) to temp folder as train.json...")
        shutil.copy2(json_path, temp_dir_path / "train.json")

        # 2. Copy images
        for idx, row in eligible_imgs.iterrows():
            img_id = int(row["id"])
            local_path = str(row["local_path"])
            dest_path = images_dir / f"{img_id}.png"
            
            # Use os.link if possible for speed, fallback to shutil.copy2
            try:
                os.link(local_path, dest_path)
            except Exception:
                shutil.copy2(local_path, dest_path)
                
            if (idx + 1) % 100 == 0:
                print(f"  Prepared {idx + 1}/{len(eligible_imgs)} images...")

        # 3. Upload the entire folder at once
        print(f"Uploading bundled dataset to {args.repo_id}...")
        api.upload_folder(
            folder_path=str(temp_dir_path),
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print("Upload complete.")

if __name__ == "__main__":
    main()
