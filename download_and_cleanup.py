import argparse
import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

from dental_agent.data.dentex import download_dentex, extract_dentex_zips


def main():
    parser = argparse.ArgumentParser(description="Download only the DENTEX data needed by this repo's training and YOLO workflows.")
    parser.add_argument("--split", choices=["train", "validation"], default="train", help="Which DENTEX split to hydrate")
    parser.add_argument("--full", action="store_true", help="Download the full repo snapshot instead of the targeted subset.")
    args = parser.parse_args()

    cache_dir = str(Path("hf_cache").absolute())
    target_data_dir = Path("data/dentex")

    print(f"Using cache directory: {cache_dir}")
    print(f"Downloading the targeted DENTEX files for split '{args.split}' (full={args.full}).")

    downloaded_path = download_dentex(
        repo_id="ibrahimhamamci/DENTEX",
        cache_dir=cache_dir,
        split_name=args.split,
        full=args.full,
    )

    print("\nDownload complete! Extracting zip files...")
    extract_dentex_zips(downloaded_path)

    print(f"\nMoving extracted dataset to {target_data_dir}...")
    target_data_dir.mkdir(parents=True, exist_ok=True)

    source_dentex = downloaded_path / "DENTEX"
    dest_dentex = target_data_dir / "DENTEX"

    if source_dentex.exists():
        shutil.copytree(source_dentex, dest_dentex, dirs_exist_ok=True)
        print(f"Successfully moved files to {dest_dentex}")
    else:
        print(f"Warning: Expected DENTEX folder at {source_dentex} but didn't find it.")

    print(f"\nCleaning up cache directory {cache_dir} to free up space...")
    try:
        shutil.rmtree(cache_dir, ignore_errors=True)
        print("Cache cleanup successful!")
    except Exception as e:
        print(f"Could not completely remove cache dir automatically. You can manually delete {cache_dir} safely. Error: {e}")

    print("\nDone! The notebook-required DENTEX files are ready under data/dentex.")


if __name__ == "__main__":
    main()
