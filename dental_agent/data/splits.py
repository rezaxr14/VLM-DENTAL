"""
Train / evaluation split management.

Handles DENTEX's official test split detection and fallback to a
fixed-seed held-out slice of the training data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _score_named_split(jf: str, d: dict, split_name: str) -> int:
    """Heuristic: how likely is *jf* to be the *split_name* split?"""
    score = int(split_name in jf.lower())
    ann0 = d["annotations"][0] if d.get("annotations") else {}
    score += sum(
        k in ann0
        for k in ("category_id_1", "category_id_2", "category_id_3", "extra")
    )
    score += 2 * int("diagnosis" in jf.lower() or "disease" in jf.lower())
    return score


def get_holdout_ids(
    images_df: pd.DataFrame,
    all_coco: dict[str, dict],
    best_path: str,
    by_basename: dict[str, str],
    seed: int = 42,
    holdout_fraction: float = 0.20,
) -> tuple[set[int], pd.DataFrame, pd.DataFrame, str]:
    """Determine held-out evaluation image IDs.

    Tries DENTEX's own official test split first; falls back to carving a
    fixed-seed held-out slice out of the train split if the test split is
    unavailable or has no public ground truth.

    Returns
    -------
    holdout_ids : set[int]
        Image IDs reserved for evaluation.
    images_df : DataFrame
        Potentially updated images_df (with test-split rows merged in).
    annots_df : DataFrame
        Potentially updated annots_df (with test-split rows merged in).
    source : str
        Description of which split was used.
    """
    annotated = [
        (jf, d) for jf, d in all_coco.items()
        if isinstance(d, dict) and d.get("annotations")
    ]
    test_ranked = sorted(
        annotated, key=lambda c: _score_named_split(*c, "test"), reverse=True
    )

    if (
        test_ranked
        and "test" in test_ranked[0][0].lower()
        and test_ranked[0][0] != best_path
    ):
        source = "official DENTEX test split"
        _, test_coco = test_ranked[0]
        test_images_df = pd.DataFrame(test_coco["images"])
        test_images_df["local_path"] = test_images_df["file_name"].apply(
            lambda fn: by_basename.get(__import__("os").path.basename(fn))
        )
        test_annots_df = pd.DataFrame(test_coco["annotations"])
        if "bbox" in test_annots_df.columns:
            test_annots_df["bbox"] = test_annots_df["bbox"].apply(list)

        holdout_ids = set(test_images_df["id"])
        # Return the test data for the caller to merge
        return holdout_ids, test_images_df, test_annots_df, source

    source = (
        "held-out slice of the TRAIN split (fixed seed) "
        "— NOT DENTEX's official test set"
    )
    rng = np.random.default_rng(seed)
    all_ids = images_df["id"].unique()
    holdout_ids = set(
        rng.choice(all_ids, size=int(len(all_ids) * holdout_fraction), replace=False)
    )
    return holdout_ids, pd.DataFrame(), pd.DataFrame(), source


def get_training_pool(
    images_df: pd.DataFrame,
    holdout_ids: set[int],
) -> pd.DataFrame:
    """Return the subset of images available for training (excluding holdout)."""
    return images_df[
        ~images_df["id"].isin(holdout_ids)
    ].dropna(subset=["local_path"])
