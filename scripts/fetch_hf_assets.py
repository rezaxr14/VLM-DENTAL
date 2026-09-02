"""Fetch DENTEX/Tufts images and trained YOLO models from Hugging Face Hub."""

import argparse
import os
import shutil
from pathlib import Path
import dotenv
from huggingface_hub import snapshot_download

dotenv.load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Fetch dataset images and model checkpoints from Hugging Face Hub.")
    parser.add_argument("--images", action="store_true", help="Download DENTEX & Tufts dataset images.")
    parser.add_argument("--models", action="store_true", help="Download YOLO model weights from HF models repo.")
    parser.add_argument("--all", action="store_true", help="Download both images and models.")
    args = parser.parse_args()

    # Default to fetching both if no specific flag passed
    fetch_images = args.images or args.all or (not args.images and not args.models)
    fetch_models = args.models or args.all or (not args.images and not args.models)

    token = os.environ.get("HF_TOKEN")
    dentex_repo = os.environ.get("DENTEX_IMAGES_REPO", "Reza-Nadimi/dentex-train-images")
    tufts_repo = os.environ.get("TUFTS_IMAGES_REPO", "Reza-Nadimi/tufts-train-images")
    models_repo = os.environ.get("HF_ARTIFACT_REPO", "Reza-Nadimi/vlm-dental-models")

    if fetch_images:
        print(f"\n📦 Fetching DENTEX images from {dentex_repo}...")
        snapshot_download(
            repo_id=dentex_repo,
            repo_type="dataset",
            local_dir="data/dentex/DENTEX",
            token=token,
        )
        print("✅ DENTEX images ready in data/dentex/DENTEX")

        print(f"\n📦 Fetching Tufts images from {tufts_repo}...")
        snapshot_download(
            repo_id=tufts_repo,
            repo_type="dataset",
            local_dir="data/Tufts",
            token=token,
        )
        print("✅ Tufts images ready in data/Tufts")

    if fetch_models:
        print(f"\n🤖 Fetching YOLO models & checkpoints from {models_repo}/yolo_cv...")
        staging = Path("data/models/_hf_download_temp")
        while True:
            try:
                print("Attempting to download YOLO models...")
                snapshot_download(
                    repo_id=models_repo,
                    repo_type="model",
                    token=token,
                    allow_patterns=["yolo_cv/**/*.pt"],
                    local_dir=str(staging),
                )
                break  # Success
            except Exception as e:
                print(f"Download failed with error: {e}")
                print("Retrying in 5 seconds... (Infinite loop enabled)")
                import time
                time.sleep(5)
                
        yolo_dir = staging / "yolo_cv"
        if yolo_dir.exists():
            target_dir = Path("data/models")
            for item in yolo_dir.iterdir():
                dest = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest)
            shutil.rmtree(staging, ignore_errors=True)
        print("✅ YOLO model checkpoints ready in data/models/")

    print("\n🎉 Download complete!")


if __name__ == "__main__":
    main()
