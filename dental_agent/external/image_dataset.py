"""Stage the cohort's IMAGES as a private Kaggle dataset, so hosted runs can see them.

Hosted workers mount attached datasets and nothing else; an image set that exists only on the Hugging Face Hub or a local disk
(the official 50-image test set, the 250-image cohort after the merge) has to become a dataset once. Only images are staged:
annotations and ground truth never leave this machine (scoring is local). The dataset is created private (the CLI default); the
folder layout is ``images/<file name>``, so ``sharding.locate_image_datasets`` reads the paths back after it is attached.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def stage_image_dataset(images_df, image_ids, out_dir: str | Path, *, dataset_id: str, title: str | None = None) -> dict[str, Any]:
    """Copy the images into ``out_dir/images`` and write ``dataset-metadata.json``. ``dataset_id`` is ``owner/slug``."""
    owner, _, slug = dataset_id.partition("/")
    if not owner or not slug:
        raise ValueError("dataset_id must look like 'owner/slug'")
    out = Path(out_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    wanted = {int(i) for i in image_ids}
    files: dict[str, str] = {}
    for row in images_df[images_df["id"].isin(wanted)].itertuples():
        src = Path(row.local_path)
        if not src.is_file():
            raise FileNotFoundError(f"image {int(row.id)} is not readable at {src}")
        name = src.name
        if name in files.values() and files.get(str(int(row.id))) != name:
            raise ValueError(f"two images share the file name {name}; rename one before staging")
        shutil.copyfile(src, out / "images" / name)
        files[str(int(row.id))] = name
    missing = sorted(wanted - {int(k) for k in files})
    if missing:
        raise ValueError(f"images not found in the image table: {missing[:10]}")
    (out / "dataset-metadata.json").write_text(json.dumps(
        {"title": title or slug.replace("-", " "), "id": dataset_id, "licenses": [{"name": "other"}]}, indent=2), encoding="utf-8")
    return {"dir": str(out), "dataset_id": dataset_id, "images": len(files), "files": files}
