#!/usr/bin/env python3
"""
Standalone script to simply download and extract the DENTEX dataset.
This will NOT generate any CoT traces or consume API keys.
"""

import argparse
from dental_agent.config import load_config
from dental_agent.data.dentex import load_dentex_dataset

def main() -> None:
    parser = argparse.ArgumentParser(description="Download DENTEX dataset")
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Split to download: 'train' (full 10GB dataset) or 'validation' (142MB sample)",
    )
    args = parser.parse_args()

    cfg = load_config()
    print(f"Checking for DENTEX '{args.split}' split in {cfg.data_dir}...")
    
    # This will trigger download & extraction if files are missing locally!
    imgs_df, annots_df, cats_df = load_dentex_dataset(
        data_dir=cfg.data_dir, split_name=args.split
    )

    print("\nDataset successfully loaded and ready!")
    print(f"Total Images   : {len(imgs_df)}")
    print(f"Total Annotations : {len(annots_df)}")

if __name__ == "__main__":
    main()
