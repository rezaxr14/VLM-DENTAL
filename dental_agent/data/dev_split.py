"""Development split carved out of the DENTEX *train* images.

Why it exists: the 50 official DENTEX validation images are the held-out TEST set and must never be used for
training, reward shaping or choosing hyper-parameters (GRPO group size K, ablations). Everything that needs a
"validation" signal during development therefore uses this split instead: ~70 of the 705 DENTEX train images,
chosen once with a fixed seed, written to ``data/splits/dentex_dev.json`` and excluded from SFT and GRPO training.

Image ids collide between the train pool (1-705) and the official validation set (1-50), so every image is
identified by the pair ``(dataset, image_id)``, serialised as ``"<dataset>::<image_id>"`` (e.g. ``"dentex::412"``).
This is the same ``dataset::image_id`` key that the token-length manifest uses.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

DEV_SPLIT_VERSION = 1
DEFAULT_DEV_SPLIT_PATH = Path("data/splits/dentex_dev.json")
DEFAULT_DEV_SIZE = 70
DEFAULT_DEV_SEED = 42

# Annotation-file names that mark the official held-out split (see dentex.score_candidate).
_TEST_FILE_PATTERN = re.compile(r"(^|[^a-z])(val|valid|validation|test)([^a-z]|$)", re.IGNORECASE)


def image_key(dataset: str, image_id: Any) -> str:
    """Canonical ``dataset::image_id`` key. ``image_id`` is normalised through int() when it is integral."""
    try:
        iid = str(int(image_id))
    except (TypeError, ValueError):
        iid = str(image_id)
    return f"{str(dataset).lower().strip()}::{iid}"


def record_key(record: dict[str, Any], default_dataset: str = "dentex") -> str:
    """Key of a trace / eval record (uses its ``dataset`` field, falling back to ``default_dataset``)."""
    return image_key(record.get("dataset") or default_dataset, record.get("image_id"))


def make_dev_split(
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    n: int = DEFAULT_DEV_SIZE,
    seed: int = DEFAULT_DEV_SEED,
    dataset: str = "dentex",
) -> dict[str, Any]:
    """Pick ``n`` images from a TRAIN-split dataframe, keeping the healthy/pathological ratio.

    Worked example (DENTEX train: 705 images, 27 without any disease annotation, n=70):
    healthy share = 27/705 = 3.8% -> round(70 * 0.038) = 3 healthy + 67 pathological.
    The draw is deterministic: ids are sorted, then shuffled by ``random.Random(seed)``.
    """
    assert_not_test_split(images_df)
    if n <= 0 or n >= len(images_df):
        raise ValueError(f"dev split size must be in 1..{len(images_df) - 1}, got {n}")

    annotated = set(annots_df["image_id"].unique()) if len(annots_df) else set()
    ids = sorted(int(i) for i in images_df["id"].unique())
    healthy = [i for i in ids if i not in annotated]
    pathological = [i for i in ids if i in annotated]

    n_healthy = int(round(n * len(healthy) / max(len(ids), 1)))
    n_healthy = min(n_healthy, len(healthy))
    n_path = n - n_healthy

    rng = random.Random(seed)
    rng.shuffle(healthy)
    rng.shuffle(pathological)
    chosen_healthy = sorted(healthy[:n_healthy])
    chosen_path = sorted(pathological[:n_path])
    chosen = sorted(chosen_healthy + chosen_path)

    return {
        "version": DEV_SPLIT_VERSION,
        "dataset": dataset,
        "source": "dentex train images (never the official validation/test split)",
        "seed": seed,
        "n": len(chosen),
        "n_healthy": len(chosen_healthy),
        "n_pathological": len(chosen_path),
        "image_ids": chosen,
        "keys": [image_key(dataset, i) for i in chosen],
    }


def save_dev_split(split: dict[str, Any], path: str | Path = DEFAULT_DEV_SPLIT_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)
        f.write("\n")
    return p


def load_dev_keys(path: str | Path | None) -> set[str]:
    """Load the ``dataset::image_id`` keys of a dev split file. ``None`` -> empty set (nothing excluded)."""
    if path is None:
        return set()
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"Dev split file not found: {p}. Create it once with: python scripts/make_dev_split.py "
            f"--data-dir <data> --output {p}"
        )
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    keys = data.get("keys")
    if not keys:
        ds = data.get("dataset", "dentex")
        keys = [image_key(ds, i) for i in data.get("image_ids", [])]
    return {str(k) for k in keys}


def filter_records(records: Iterable[dict[str, Any]], exclude_keys: set[str]) -> tuple[list[dict[str, Any]], int]:
    """Drop records whose ``dataset::image_id`` key is in ``exclude_keys``. Returns (kept, n_dropped)."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for r in records:
        if exclude_keys and record_key(r) in exclude_keys:
            dropped += 1
        else:
            kept.append(r)
    return kept, dropped


def assert_not_test_split(images_df: pd.DataFrame) -> None:
    """Refuse to continue when a dataframe was loaded from the official validation (= test) annotation file.

    Detection uses the ``source_file`` column that ``load_dentex_dataset`` fills in. It cannot be fooled by id
    collisions, which is why ids alone are never used.
    """
    if "source_file" not in images_df.columns:
        return
    bad = sorted({
        Path(str(s)).name for s in images_df["source_file"].dropna().unique()
        if _TEST_FILE_PATTERN.search(Path(str(s)).stem)
    })
    if bad:
        raise SystemExit(
            f"Refusing to train on the held-out test split: images were loaded from {bad}. Training, the K sweep and "
            "the ablations must use the DENTEX train images (load_dentex_dataset(..., split_name='train')) minus the dev split."
        )


def select_training_pool(
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    dataset: str = "dentex",
    dev_keys: set[str] | None = None,
    include_healthy: bool = True,
    max_images: int | None = None,
    seed: int = DEFAULT_DEV_SEED,
) -> pd.DataFrame:
    """Images GRPO may train on: train split, minus dev keys, with a usable file, optional healthy filter.

    The pool is shuffled once with ``seed`` (so ``--max-images`` is an unbiased subset, not "the first N ids")
    and then truncated to ``max_images``. Healthy images (no annotation rows) carry ground truth ``[]``; the
    accuracy reward gives 1.0 only to an empty answer on them.
    """
    assert_not_test_split(images_df)
    pool = images_df.dropna(subset=["local_path"]) if "local_path" in images_df.columns else images_df
    if dev_keys:
        keep = [image_key(dataset, i) not in dev_keys for i in pool["id"]]
        pool = pool[keep]
    if not include_healthy:
        annotated = set(annots_df["image_id"].unique()) if len(annots_df) else set()
        pool = pool[pool["id"].isin(annotated)]
    pool = pool.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    if max_images is not None and max_images > 0:
        pool = pool.iloc[:max_images].reset_index(drop=True)
    return pool
