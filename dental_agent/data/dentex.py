"""
DENTEX dataset loading, annotation parsing, and image-path resolution.

Wraps the DENTEX HuggingFace dataset repo into a clean
(images_df, annots_df, categories_df) triple with resolved local paths,
quality checks, automatic zip extraction, and parquet caching for fast reloads.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import time
from typing import Any
import zipfile

import pandas as pd
from huggingface_hub import hf_hub_download, snapshot_download
from PIL import Image
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Resilient Download Helpers
# ---------------------------------------------------------------------------

def download_dentex_file(
    filename: str,
    repo_id: str = "ibrahimhamamci/DENTEX",
    cache_dir: str | None = None,
    max_retries: int = 5,
    retry_delay: float = 3.0,
) -> Path:
    """Download a specific file from the DENTEX dataset repository with retry logic."""
    # 1. Try local cache first for instant reloads without network roundtrips
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type="dataset",
            cache_dir=cache_dir,
            local_files_only=True,
        )
        return Path(local_path)
    except Exception:
        pass

    # 2. Download with retry if not already present in local cache
    for attempt in range(1, max_retries + 1):
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                cache_dir=cache_dir,
            )
            return Path(local_path)
        except Exception as e:
            if attempt == max_retries:
                raise RuntimeError(
                    f"Failed to download {filename} from {repo_id} after {max_retries} attempts: {e}"
                )
            print(f"Download attempt {attempt}/{max_retries} failed for {filename} ({e}). Retrying in {retry_delay}s...")
            time.sleep(retry_delay * attempt)
    raise RuntimeError(f"Unexpected error downloading {filename}")


def download_dentex(
    repo_id: str = "ibrahimhamamci/DENTEX",
    cache_dir: str | None = None,
    split_name: str = "validation",
) -> Path:
    """Download (or reuse cached) DENTEX dataset files.
    
    If split_name is 'validation' or 'val', downloads validation files directly
    without waiting for the full 10GB training dataset.
    """
    if split_name in ("val", "validation"):
        # Download validation assets directly
        val_zip = download_dentex_file("DENTEX/validation_data.zip", repo_id=repo_id, cache_dir=cache_dir)
        val_json = download_dentex_file("DENTEX/validation_triple.json", repo_id=repo_id, cache_dir=cache_dir)
        return val_zip.parent.parent

    try:
        dentex_path = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            cache_dir=cache_dir,
            max_workers=2,
        )
        return Path(dentex_path)
    except Exception as e:
        print(f"snapshot_download failed ({e}); falling back to file-by-file download...")
        if split_name == "train":
            download_dentex_file("DENTEX/training_data.zip", repo_id=repo_id, cache_dir=cache_dir)
        download_dentex_file("DENTEX/validation_data.zip", repo_id=repo_id, cache_dir=cache_dir)
        val_json = download_dentex_file("DENTEX/validation_triple.json", repo_id=repo_id, cache_dir=cache_dir)
        return val_json.parent.parent


def extract_dentex_zips(root_dir: str | Path, remove_zips: bool = True) -> None:
    """Extract any downloaded .zip archives (validation_data.zip, training_data.zip, etc.) in place
    and remove the .zip files after successful extraction to save disk space."""
    zip_files = glob.glob(os.path.join(str(root_dir), "**", "*.zip"), recursive=True)
    for zf in zip_files:
        extract_target = os.path.splitext(zf)[0]
        already_extracted = (
            os.path.exists(extract_target)
            and os.path.isdir(extract_target)
            and len(os.listdir(extract_target)) > 0
        )
        if not already_extracted:
            try:
                print(f"Extracting {os.path.basename(zf)}...")
                with zipfile.ZipFile(zf, "r") as z:
                    z.extractall(Path(zf).parent)
                print(f"Extracted {os.path.basename(zf)} successfully.")
            except Exception as e:
                print(f"Warning: Failed to extract {zf}: {e}")
                continue

        if remove_zips and os.path.exists(zf):
            try:
                os.remove(zf)
                print(f"Cleaned up {os.path.basename(zf)} to save disk space.")
            except Exception as e:
                print(f"Warning: Could not remove {zf}: {e}")


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def load_coco_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


_load_coco_json = load_coco_json


def discover_annotation_files(dentex_path: str | Path) -> dict[str, dict]:
    """Find and parse all COCO-style JSON annotation files."""
    json_files = sorted(
        glob.glob(os.path.join(str(dentex_path), "**", "*.json"), recursive=True)
    )
    all_coco: dict[str, dict] = {}
    for jf in json_files:
        try:
            all_coco[jf] = load_coco_json(jf)
        except Exception:
            pass  # Skip unparseable files
    return all_coco


def hierarchy_summary(
    all_coco: dict[str, dict], dentex_path: str | Path
) -> pd.DataFrame:
    """Summarise all discovered annotation files (hierarchy levels × splits)."""
    rows = []
    for jf, d in all_coco.items():
        if not isinstance(d, dict):
            continue
        rows.append({
            "file": os.path.relpath(jf, str(dentex_path)),
            "n_images": len(d.get("images", [])),
            "n_annotations": len(d.get("annotations", [])),
            "n_categories": len(d.get("categories", [])),
            "category_names": ", ".join(
                c.get("name", "?") for c in d.get("categories", [])[:8]
            ),
        })
    return pd.DataFrame(rows)


def score_candidate(jf: str, d: dict, split_name: str = "train") -> int:
    """Heuristic score: higher = more likely to be the fully-annotated file."""
    score = 0
    name = jf.lower()
    if split_name in name or (split_name in ("val", "validation") and "val" in name):
        score += 2
    ann0 = d["annotations"][0] if d.get("annotations") else {}
    for key in ("category_id_1", "category_id_2", "category_id_3", "extra"):
        if key in ann0:
            score += 1
    if "diagnosis" in name or "disease" in name or "triple" in name:
        score += 2
    return score


_score_candidate = score_candidate


def pick_best_annotation_file(
    all_coco: dict[str, dict],
    split_name: str = "train",
) -> tuple[str, dict]:
    """Select the fully-annotated (quadrant-enumeration-diagnosis) file."""
    candidates = [
        (jf, d) for jf, d in all_coco.items()
        if isinstance(d, dict) and d.get("annotations")
    ]
    if not candidates:
        raise RuntimeError(
            "No COCO-style JSON with an 'annotations' field was found."
        )
    candidates.sort(
        key=lambda c: _score_candidate(c[0], c[1], split_name), reverse=True
    )
    return candidates[0]


# ---------------------------------------------------------------------------
# DataFrame construction
# ---------------------------------------------------------------------------

def build_dataframes(
    coco: dict[str, Any],
    data_dir: str | None = None,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse COCO JSON into (images_df, annots_df, categories_df).

    Results are cached to parquet under *data_dir* if provided.
    """
    if data_dir and use_cache:
        paths = {
            "images": os.path.join(data_dir, "images_df.parquet"),
            "annots": os.path.join(data_dir, "annots_df.parquet"),
            "categories": os.path.join(data_dir, "categories_df.parquet"),
        }
        if all(os.path.exists(p) for p in paths.values()):
            return (
                pd.read_parquet(paths["images"]),
                pd.read_parquet(paths["annots"]),
                pd.read_parquet(paths["categories"]),
            )

    images_df = pd.DataFrame(coco.get("images", []))
    annots_df = pd.DataFrame(coco.get("annotations", []))
    categories_df = pd.DataFrame(coco.get("categories", []))

    # Ensure bbox is a plain list (for parquet serialization)
    if "bbox" in annots_df.columns:
        annots_df["bbox"] = annots_df["bbox"].apply(list)

    if data_dir and use_cache and len(images_df):
        os.makedirs(data_dir, exist_ok=True)
        try:
            images_df.to_parquet(paths["images"])
            annots_df.to_parquet(paths["annots"])
            categories_df.to_parquet(paths["categories"])
        except Exception:
            pass

    return images_df, annots_df, categories_df


# ---------------------------------------------------------------------------
# Image path resolution & quality checks
# ---------------------------------------------------------------------------

def resolve_image_paths(
    images_df: pd.DataFrame, dentex_path: str | Path
) -> pd.DataFrame:
    """Add a ``local_path`` column mapping each image record to its file."""
    image_files = (
        glob.glob(os.path.join(str(dentex_path), "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(str(dentex_path), "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join("data", "**", "*.png"), recursive=True)
    )
    by_basename = {os.path.basename(p): p for p in image_files}
    images_df = images_df.copy()
    if "file_name" in images_df.columns:
        images_df["local_path"] = images_df["file_name"].apply(
            lambda fn: by_basename.get(os.path.basename(fn))
        )
    return images_df


def validate_images(
    images_df: pd.DataFrame, verbose: bool = True
) -> tuple[list[tuple[str, str]], pd.Series]:
    """Check for corrupt/unreadable images and report colour-mode distribution.

    Returns (bad_files, mode_counts).
    """
    bad_files: list[tuple[str, str]] = []
    modes: list[str] = []
    paths = images_df["local_path"].dropna() if "local_path" in images_df.columns else []
    iterator = tqdm(paths, desc="Validating images") if verbose else paths

    for p in iterator:
        try:
            with Image.open(p) as im:
                im.verify()
            with Image.open(p) as im:
                modes.append(im.mode)
        except Exception as e:
            bad_files.append((str(p), str(e)))

    mode_counts = pd.Series(modes).value_counts()
    return bad_files, mode_counts


def check_bbox_bounds(
    images_df: pd.DataFrame, annots_df: pd.DataFrame
) -> pd.DataFrame | None:
    """Return annotations where the bounding box exceeds image dimensions."""
    if "width" not in images_df.columns or "height" not in images_df.columns:
        return None

    dims = images_df.set_index("id")[["width", "height"]]
    merged = annots_df.join(dims, on="image_id")

    def _out_of_bounds(row: pd.Series) -> bool:
        x, y, w, h = row["bbox"]
        return x < 0 or y < 0 or x + w > row["width"] or y + h > row["height"]

    oob_mask = merged.apply(_out_of_bounds, axis=1)
    return merged[oob_mask][["image_id", "bbox", "width", "height"]]


def list_files(root: str | Path, max_depth: int = 3, max_per_dir: int = 15) -> None:
    """Pretty-print the directory tree (useful for initial dataset inspection)."""
    root = Path(root)
    for dirpath, dirnames, filenames in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        if depth > max_depth:
            dirnames[:] = []
            continue
        indent = "  " * depth
        print(f"{indent}{Path(dirpath).name}/")
        for f in sorted(filenames)[:max_per_dir]:
            print(f"{indent}  {f}")
        if len(filenames) > max_per_dir:
            print(f"{indent}  ... ({len(filenames) - max_per_dir} more files)")


def find_local_dentex_dir(data_dir: str | Path | None = None, split_name: str = "validation") -> Path | None:
    """Find a local directory containing DENTEX dataset files for the requested split."""
    candidates: list[Path] = []
    if data_dir:
        p = Path(data_dir)
        candidates.extend([p, p / "dentex", p / "DENTEX"])
    candidates.extend([
        Path("data/dentex/DENTEX"),
        Path("data/dentex"),
        Path("data"),
    ])
    for c in candidates:
        if c.exists() and c.is_dir():
            # Check for split-specific files
            is_train = split_name in ("train", "training")
            if is_train:
                has_train_data = any(c.rglob("training_data*")) or any(c.rglob("train*")) or any(c.rglob("disease*"))
                if has_train_data:
                    return c
            else:
                has_val_data = any(c.rglob("validation_data*")) or any(c.rglob("val*"))
                if has_val_data:
                    return c
    return None


def load_dentex_dataset(
    data_dir: str | Path | None = None,
    split_name: str = "validation",
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Top-level convenience loader for DENTEX dataset.

    Checks local project data directories first for instant loading. If not found locally,
    downloads or discovers files, auto-extracts zip archives, parses the
    best annotation file, resolves local image paths, and returns
    (images_df, annots_df, categories_df).
    """
    local_dir = find_local_dentex_dir(data_dir, split_name)
    if local_dir is not None:
        dentex_path = local_dir
    else:
        print(f"Dataset for split '{split_name}' not found locally. Triggering download...")
        dentex_path = download_dentex(
            cache_dir=str(data_dir) if data_dir else None,
            split_name=split_name,
        )

    extract_dentex_zips(dentex_path)
    all_coco = discover_annotation_files(dentex_path)
    _, best_coco = pick_best_annotation_file(all_coco, split_name=split_name)
    images_df, annots_df, categories_df = build_dataframes(
        best_coco, data_dir=str(data_dir) if data_dir else None, use_cache=use_cache
    )
    images_df = resolve_image_paths(images_df, dentex_path)
    return images_df, annots_df, categories_df
