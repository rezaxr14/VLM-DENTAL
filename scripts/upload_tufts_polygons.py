#!/usr/bin/env python3
"""Upload Tufts annotation files (including teeth_polygon.json, student.json, expert.json)
to Hugging Face Hub dataset repository.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import HfApi

repo_root = Path(__file__).resolve().parent.parent
load_dotenv(repo_root / ".env", override=True)


def upload_tufts_annotations():
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token or hf_token.startswith("YOUR_"):
        print("ERROR: HF_TOKEN is not set or is a placeholder in .env.")
        sys.exit(1)

    repo_id = os.environ.get("TUFTS_IMAGES_REPO", "Reza-Nadimi/tufts-train-images")
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True)

    tufts_dir = repo_root / "data" / "Tufts"
    if not tufts_dir.exists():
        print(f"ERROR: Local Tufts directory not found at {tufts_dir}")
        sys.exit(1)

    files_to_upload = [
        ("Segmentation/teeth_polygon.json", "Segmentation/teeth_polygon.json"),
        ("Segmentation/teeth_bbox.json", "Segmentation/teeth_bbox.json"),
        ("Expert/expert.json", "Expert/expert.json"),
        ("Student/student.json", "Student/student.json"),
    ]

    print(f"Uploading Tufts annotations to Hugging Face repository: {repo_id}...")
    for local_rel, remote_rel in files_to_upload:
        local_path = tufts_dir / local_rel
        if not local_path.exists():
            print(f"  [SKIPPED] {local_rel} does not exist at {local_path}")
            continue

        size_mb = local_path.stat().st_size / (1024 * 1024)
        print(f"  Uploading {local_rel} ({size_mb:.2f} MB) -> {remote_rel}...")
        try:
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=remote_rel,
                repo_id=repo_id,
                repo_type="dataset",
                commit_message=f"Upload {remote_rel} ({size_mb:.1f} MB)",
            )
            print(f"  [SUCCESS] Uploaded {remote_rel}")
        except Exception as e:
            print(f"  [FAILED] Could not upload {remote_rel}: {e}")

    print("\nVerifying uploaded files on Hugging Face...")
    try:
        remote_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        seg_files = [f for f in remote_files if f.startswith("Segmentation/") or f.startswith("Student/") or f.startswith("Expert/")]
        print("Remote annotation files on Hub:")
        for sf in sorted(seg_files):
            print(f"  - {sf}")
    except Exception as e:
        print(f"Could not list remote files: {e}")


if __name__ == "__main__":
    upload_tufts_annotations()
