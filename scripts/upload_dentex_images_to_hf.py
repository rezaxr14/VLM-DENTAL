#!/usr/bin/env python3
import os
import sys
import time
from pathlib import Path

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

    # Upload annotation JSON
    # Which subfolder does the data come from?
    # Actually, the annotations dataframe was loaded from the JSON.
    # It's better to find the original JSON file and upload it.
    # dentex.py uses: json_path = data_dir / "DENTEX" / "training_data" / "quadrant_enumeration_disease" / "train.json"
    json_path = Path(cfg.data_dir) / "DENTEX" / "training_data" / "quadrant_enumeration_disease" / "train.json"
    if json_path.exists():
        print(f"Uploading annotations JSON...")
        api.upload_file(
            path_or_fileobj=str(json_path),
            path_in_repo="train.json",
            repo_id=args.repo_id,
            repo_type="dataset"
        )
    else:
        print(f"WARNING: Annotation JSON not found at {json_path}")

    # Upload images
    for idx, row in eligible_imgs.iterrows():
        img_id = int(row["id"])
        local_path = str(row["local_path"])
        path_in_repo = f"images/{img_id}.png"
        
        print(f"[{idx+1}/{len(eligible_imgs)}] Uploading {local_path} -> {path_in_repo} ...")
        
        retries = 3
        while retries > 0:
            try:
                api.upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=path_in_repo,
                    repo_id=args.repo_id,
                    repo_type="dataset"
                )
                break
            except Exception as e:
                retries -= 1
                print(f"  Upload failed: {e}. Retrying... ({retries} left)")
                time.sleep(2)
        if retries == 0:
            print(f"  FAILED to upload image {img_id}. Skipping.")

    print("Upload complete.")

if __name__ == "__main__":
    main()
