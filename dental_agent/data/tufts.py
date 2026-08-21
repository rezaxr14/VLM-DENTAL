"""
Tufts Dental Database loader -- mirrors dentex.py's interface exactly
(images_df / annots_df / categories_df with the same column names:
category_id_1=quadrant, category_id_2=tooth_position, category_id_3=
diagnosis category id, bbox=[x,y,w,h]) so it's a drop-in second dataset
for everything downstream (trace-gen, YOLO training, the reward pipeline)
that already reads that shape.

READ THIS BEFORE TRUSTING THE OUTPUT OF THIS MODULE:

Tufts is NOT freely auto-downloadable like DENTEX. It's gated behind a
request form at https://tdd.ece.tufts.edu/ -- there is no HF/Kaggle mirror
this code fetches automatically, and there shouldn't be one wired in without
checking that a given mirror's redistribution terms actually allow it.
find_local_tufts_dir() expects you've already requested access, downloaded,
and extracted the archive yourself; point TUFTS_LOCAL_DIR at it.

Tufts' native annotations are NOT DENTEX-shaped. Multiple independent public
descriptions of the dataset agree on the broad structure -- top-level
Expert/Radiographs/Segmentation/Student folders, radiograph images under
something like Radiographs/Images1, per-tooth segmentation masks under
Segmentation, plus expert eye-tracking maps and free-text descriptions -- but
NOT on the two things that actually matter for training data correctness:
  1. How a mask instance's pixel value or filename maps to a specific FDI
     quadrant+tooth-position (category_id_1/category_id_2 below).
  2. How Tufts' abnormality masks/free-text descriptions map onto DENTEX's
     specific 4-class diagnosis vocabulary (Caries / Deep Caries /
     Periapical Lesion / Impacted Tooth) for category_id_3.

Guessing either of these for a medical training-data pipeline is exactly the
kind of mistake that produces silently-wrong diagnostic labels no later
step would catch. So: image discovery and bbox-from-mask (a real, verifiable
image-processing operation, connected-component bounding boxes) are
implemented and tested below. The tooth-identity and diagnosis-category
mapping are NOT guessed -- `_infer_tooth_position` and
`_map_abnormality_to_dentex_category` raise NotImplementedError with the
specifics needed to fill them in once real extracted files are available to
verify against (either your own inspection, or point me at a sample and I
will).
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


def find_local_tufts_dir(search_roots: list[str] | None = None) -> Path | None:
    """Look for an already-extracted local copy of the Tufts archive.

    Unlike DENTEX, there's no automatic fetch here -- Tufts access is gated
    behind a request form, so this only ever looks at local disk. Set
    TUFTS_LOCAL_DIR in .env to skip the search entirely.
    """
    env_path = os.environ.get("TUFTS_LOCAL_DIR")
    if env_path and os.path.isdir(env_path):
        return Path(env_path)

    search_roots = search_roots or [".", "./data", "/content", "/kaggle/input"]
    # Tolerant of naming variation across different real-world extracted
    # copies (case, underscores vs spaces) -- this part doesn't need the
    # exact folder name verified, just a plausible match to point at.
    patterns = ["*[Tt]ufts*[Dd]ental*", "*TDD*", "*tufts-dental-database*"]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pattern in patterns:
            matches = glob.glob(os.path.join(root, pattern))
            for m in matches:
                if os.path.isdir(m):
                    return Path(m)
    return None


def _find_radiograph_dir(tufts_root: Path) -> Path | None:
    """Locate the raw radiograph images under the Tufts folder tree.
    Tolerant of naming variation (Radiograph vs Radiographs, Images1 vs
    other numbering) since this is a directory-shape search, not a claim
    about file contents."""
    candidates = list(tufts_root.glob("**/Radiograph*/Images*")) + \
                 list(tufts_root.glob("**/radiograph*/images*"))
    for c in candidates:
        if c.is_dir() and any(c.glob("*.jpg")) or any(c.glob("*.png")) or any(c.glob("*.JPG")):
            return c
    return candidates[0] if candidates else None


def _find_mask_dir(tufts_root: Path) -> Path | None:
    """Locate the per-tooth segmentation mask directory."""
    candidates = list(tufts_root.glob("**/Segmentation/**")) + \
                 list(tufts_root.glob("**/segmentation/**"))
    mask_dirs = [c for c in candidates if c.is_dir() and ("mask" in c.name.lower())]
    return mask_dirs[0] if mask_dirs else None


def _infer_tooth_position(mask_path: Path, mask_array: np.ndarray) -> tuple[int, int]:
    """Map a segmented tooth instance to (quadrant, tooth_position), FDI
    convention (quadrant 1-4, tooth_position 1-8) -- the same pair
    category_id_1/category_id_2 hold for DENTEX.

    NOT IMPLEMENTED: this needs verifying against real Tufts files before
    it can be trusted. What would resolve it, in order of preference:
      1. A manifest/README/CSV shipped with the dataset that names each
         mask file's tooth by FDI number or an equivalent scheme -- if
         Tufts ships one, use its numbering directly instead of inferring
         anything from mask geometry.
      2. Confirmation of what a mask instance's pixel VALUE actually
         encodes (tooth ID 1-32? FDI two-digit number directly? an
         arbitrary per-image instance index with no fixed meaning?).
      3. Failing both, geometric inference (top/bottom half of the image
         -> quadrants 1-2 vs 3-4; left-right position along the jaw curve
         -> tooth_position) is possible but is an approximation, not a
         verified ground-truth label, and should be flagged as such
         wherever it's used, not presented as equivalent to DENTEX's
         expert-annotated positions.
    """
    raise NotImplementedError(
        "Tufts tooth-position mapping is not implemented -- see this "
        "function's docstring. Needs real extracted Tufts files (or a "
        "description of the mask numbering scheme) to verify against "
        "before writing this, rather than guessing at a medical label."
    )


def _map_abnormality_to_dentex_category(description: str) -> str | None:
    """Map a Tufts abnormality description onto DENTEX's controlled
    diagnosis vocabulary (Caries, Deep Caries, Periapical Lesion, Impacted
    Tooth), or return None if it doesn't correspond to any of them.

    NOT IMPLEMENTED: same reasoning as _infer_tooth_position. Tufts'
    abnormality labeling is richer than a single category (the dataset
    description mentions axes like anatomical location, radiodensity,
    effect on surrounding structure) and free-text -- collapsing that onto
    DENTEX's 4 classes needs the real label vocabulary in hand, not a
    keyword-matching guess that could silently mislabel a finding.
    """
    raise NotImplementedError(
        "Tufts diagnosis-category mapping is not implemented -- see this "
        "function's docstring. Needs the real abnormality label/taxonomy "
        "files to map correctly."
    )


def _bbox_from_mask(mask_array: np.ndarray, instance_value: int) -> list[float] | None:
    """Compute a [x, y, w, h] bounding box for one labeled instance in a
    mask array. This part IS a standard, verifiable operation -- no dataset-
    specific knowledge needed beyond "this pixel value is one instance."
    """
    ys, xs = np.where(mask_array == instance_value)
    if len(xs) == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def load_tufts_dataset(
    data_dir: str | None = None,
    max_images: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Tufts into the same (images_df, annots_df, categories_df) shape
    dentex.py's load_dentex_dataset returns.

    Currently raises NotImplementedError once it reaches annotation
    construction -- image discovery works, but annots_df needs
    _infer_tooth_position and _map_abnormality_to_dentex_category filled in
    first (see their docstrings). Left as a hard stop rather than returning
    an empty/placeholder annots_df, so a caller can't accidentally train on
    silently-empty ground truth without noticing.
    """
    tufts_root = find_local_tufts_dir()
    if tufts_root is None:
        raise FileNotFoundError(
            "No local Tufts Dental Database directory found. Tufts is access-gated "
            "(request it at https://tdd.ece.tufts.edu/) -- download and extract it "
            "yourself, then set TUFTS_LOCAL_DIR in .env to the extracted folder."
        )

    radiograph_dir = _find_radiograph_dir(tufts_root)
    if radiograph_dir is None:
        raise FileNotFoundError(
            f"Found {tufts_root} but couldn't locate a Radiographs/Images* folder inside it. "
            "The extracted folder structure may not match what this loader expects -- "
            "check the actual layout and adjust _find_radiograph_dir."
        )

    image_files = sorted(
        list(radiograph_dir.glob("*.jpg")) + list(radiograph_dir.glob("*.JPG")) + list(radiograph_dir.glob("*.png"))
    )
    if max_images:
        image_files = image_files[:max_images]

    rows = []
    for i, f in enumerate(image_files):
        with Image.open(f) as im:
            width, height = im.size
        rows.append({"id": i, "file_name": f.name, "local_path": str(f), "width": width, "height": height})
    images_df = pd.DataFrame(rows)

    # This is the hard stop -- see module docstring. Image discovery above is
    # solid and independently useful; annotation construction is not, yet.
    raise NotImplementedError(
        f"Found {len(images_df)} Tufts radiograph images at {radiograph_dir}, but "
        "annots_df construction needs _infer_tooth_position and "
        "_map_abnormality_to_dentex_category implemented first -- both currently "
        "raise NotImplementedError rather than guess. See tufts.py's module "
        "docstring for exactly what's needed to fill them in."
    )


def download_tufts_slice(
    image_ids: list[int],
    repo_id: str | None = None,
    cache_dir: str | None = None,
) -> dict[int, Path | None]:
    """Download only the given image_ids from a lightweight per-image HF repo
    (same mechanism as DENTEX, once Tufts images have actually been uploaded
    there by scripts/upload_tufts_images_to_hf.py -- see hf_dataset_utils.py).
    """
    if repo_id is None:
        repo_id = os.environ.get("TUFTS_IMAGES_REPO")
    from dental_agent.data.hf_dataset_utils import download_dataset_slice
    return download_dataset_slice(image_ids, repo_id=repo_id, filename_template="images/{id}.jpg", cache_dir=cache_dir)
