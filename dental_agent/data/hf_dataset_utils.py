"""
Dataset-agnostic HuggingFace helpers: the "upload once, download only the
images a given slice actually needs" mechanism DENTEX already used
(dentex.py's download_dentex_slice / upload_dentex_images_to_hf.py), pulled
out here so Tufts and any future dataset can reuse the identical mechanism
instead of each dataset module carrying its own copy.

Nothing dataset-specific lives here -- no DENTEX or Tufts imports, no
column-name assumptions. A dataset module (dentex.py, tufts.py, ...) calls
download_dataset_slice() with its own repo-id env var and filename
convention; get_slice_ids/compute_slice_assignment (slicing.py) are already
dataset-agnostic and unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import hf_hub_download


def download_dataset_slice(
    image_ids: list[int],
    repo_id: str | None,
    filename_template: str = "images/{id}.png",
    cache_dir: str | None = None,
) -> dict[int, Path | None]:
    """Download only the given image_ids from a lightweight per-image HF dataset
    repo, skipping every image the current slice doesn't need -- this is what
    lets a 50-image trace-gen run skip fetching an entire multi-GB archive.

    filename_template is filled in with `id=image_id` per file (e.g. Tufts
    might use "images/{id}.jpg" instead of DENTEX's "images/{id}.png" -- the
    per-item repo layout each dataset's upload script actually produces).

    Returns {image_id: local_path}, with None entries for any image that
    failed to fetch (caller decides whether to skip or fall back).
    """
    if not repo_id:
        return {}

    local_paths: dict[int, Path | None] = {}
    for img_id in image_ids:
        filename = filename_template.format(id=img_id)
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            local_paths[img_id] = Path(local_path)
        except Exception as e:
            print(f"Warning: Failed to download targeted slice image {img_id} ({filename}): {e}")
            local_paths[img_id] = None
    return local_paths


def resolve_images_repo_env(dataset_name: str) -> str | None:
    """Look up the per-item images repo env var for a dataset by convention:
    DENTEX -> DENTEX_IMAGES_REPO, TUFTS -> TUFTS_IMAGES_REPO, etc. -- so a new
    dataset only needs its own .env entry, not new lookup code."""
    return os.environ.get(f"{dataset_name.upper()}_IMAGES_REPO")
