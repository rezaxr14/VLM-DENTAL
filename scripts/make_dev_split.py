#!/usr/bin/env python3
"""Create the DENTEX dev split (image keys held out of SFT and GRPO training) once, with a fixed seed.

    python scripts/make_dev_split.py --data-dir data --n 70 --seed 42 --output data/splits/dentex_dev.json

The split is drawn from the 705 DENTEX *train* images only. The official 50-image validation set stays the test set.
Re-running with the same seed and the same train data reproduces the file exactly; commit the file so every
machine (RTX 4090, TPU, Kaggle) uses the identical split.
"""

import argparse
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.data.dentex import load_dentex_dataset
from dental_agent.data.dev_split import (
    DEFAULT_DEV_SEED,
    DEFAULT_DEV_SIZE,
    DEFAULT_DEV_SPLIT_PATH,
    make_dev_split,
    save_dev_split,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create the DENTEX dev split from the train images.")
    parser.add_argument("--data-dir", type=str, default="data", help="Base directory of the DENTEX data")
    parser.add_argument("--n", type=int, default=DEFAULT_DEV_SIZE, help="Number of dev images (default 70)")
    parser.add_argument("--seed", type=int, default=DEFAULT_DEV_SEED, help="Shuffle seed (default 42)")
    parser.add_argument("--output", type=str, default=str(DEFAULT_DEV_SPLIT_PATH), help="Output JSON path")
    parser.add_argument("--force", action="store_true", help="Overwrite an existing split file")
    args = parser.parse_args()

    out = Path(args.output)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} already exists; the dev split must stay fixed. Use --force only to deliberately replace it.")

    images_df, annots_df, _ = load_dentex_dataset(args.data_dir, split_name="train")
    split = make_dev_split(images_df, annots_df, n=args.n, seed=args.seed)
    save_dev_split(split, out)
    print(
        f"Dev split: {split['n']} images ({split['n_pathological']} pathological, {split['n_healthy']} healthy) "
        f"from {len(images_df)} DENTEX train images -> {out}"
    )


if __name__ == "__main__":
    main()
