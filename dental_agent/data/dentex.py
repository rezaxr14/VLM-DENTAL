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


def dentex_row_to_fdi(row: Any, default: int = 0) -> tuple[int, int]:
    """Convert one DENTEX annotation row's raw category_id_1/category_id_2
    into proper 1-indexed FDI (quadrant 1-4, tooth_position 1-8).

    THE SINGLE SOURCE OF TRUTH for the "0-Index Quirk" (documented as
    CRITICAL in .agents/rules/vlm_dental.md): DENTEX's raw JSON labels
    category_id_1/category_id_2 as 0-indexed (quadrant 0-3, position 0-7),
    but every prompt, tool, and reward in this codebase is written against
    1-indexed FDI notation (quadrant 1-4, position 1-8) -- prompts.py's
    worked examples, _hint_for_tooth's FDI-string parsing, reward_accuracy's
    quadrant/tooth_position comparison, all of it.

    This function exists because that +1 conversion was implemented once,
    correctly, in trace_generation.py -- and then re-implemented, incorrectly
    (i.e. omitted), by hand in seven other files (ablations.py, baselines.py,
    batch_runner.py, judge.py, detector.py, test_aim1_trace.py) that each
    built their own ground-truth dict directly from the raw columns without
    knowing the conversion was needed. The result: every one of those files'
    ground truth fed straight into reward_accuracy/combine_reward with
    quadrant and tooth_position off by one from what a correctly-trained
    model actually outputs -- meaning a perfectly correct answer would score
    as wrong on both fields (0.50 of R_accuracy's 1.0 weight) in the GRPO
    reward, the H1/H2 ablation studies, the baseline comparisons, and the
    batch evaluation runner, every single time. Only the diagnosis-category
    term (a string lookup, unaffected by this indexing) was ever scoring
    correctly in any of them. Call this function instead of reconstructing
    the +1 by hand, so this can't happen a ninth time.
    """
    return int(row.get("category_id_1", default)) + 1, int(row.get("category_id_2", default)) + 1


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
    full: bool = False,
) -> Path:
    """Download the DENTEX files needed for the active workload.

    Default behavior is targeted: only the training/validation archives needed by the notebook
    and the validation triple JSON. Set ``full=True`` to opt into the full repo snapshot when
    you explicitly want the entire dataset.
    """
    split_name = split_name.lower()

    if full:
        try:
            dentex_path = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                cache_dir=cache_dir,
                max_workers=2,
            )
            return Path(dentex_path)
        except Exception as e:
            print(f"snapshot_download failed ({e}); falling back to targeted downloads...")

    if split_name in ("val", "validation"):
        val_zip = download_dentex_file("DENTEX/validation_data.zip", repo_id=repo_id, cache_dir=cache_dir)
        val_json = download_dentex_file("DENTEX/validation_triple.json", repo_id=repo_id, cache_dir=cache_dir)
        return val_zip.parent.parent

    if split_name in ("train", "training"):
        train_zip = download_dentex_file("DENTEX/training_data.zip", repo_id=repo_id, cache_dir=cache_dir)
        val_zip = download_dentex_file("DENTEX/validation_data.zip", repo_id=repo_id, cache_dir=cache_dir)
        val_json = download_dentex_file("DENTEX/validation_triple.json", repo_id=repo_id, cache_dir=cache_dir)
        return train_zip.parent.parent

    # Fallback for any non-standard split name: only grab the targeted validation bundle.
    val_zip = download_dentex_file("DENTEX/validation_data.zip", repo_id=repo_id, cache_dir=cache_dir)
    val_json = download_dentex_file("DENTEX/validation_triple.json", repo_id=repo_id, cache_dir=cache_dir)
    return val_zip.parent.parent


def download_dentex_slice(
    image_ids: list[int],
    repo_id: str | None = None,
    cache_dir: str | None = None,
) -> dict[int, Path | None]:
    """Download only the given image_ids from the lightweight per-image HF repo.
    Returns {image_id: local_path}. Falls back to None entries if fetch fails.

    Thin wrapper over the dataset-agnostic download_dataset_slice (see
    hf_dataset_utils.py) -- kept here under its original name so existing
    call sites (run_trace_gen.py, tests) don't need to change.
    """
    if repo_id is None:
        repo_id = os.environ.get("DENTEX_IMAGES_REPO")
    from dental_agent.data.hf_dataset_utils import download_dataset_slice
    return download_dataset_slice(image_ids, repo_id=repo_id, filename_template="images/{id}.png", cache_dir=cache_dir)


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
    split_name: str = "default",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Parse COCO JSON into (images_df, annots_df, categories_df).

    Results are cached to parquet under *data_dir* if provided.
    """
    if data_dir and use_cache:
        paths = {
            "images": os.path.join(data_dir, f"{split_name}_images_df.parquet"),
            "annots": os.path.join(data_dir, f"{split_name}_annots_df.parquet"),
            "categories": os.path.join(data_dir, f"{split_name}_categories_df.parquet"),
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
    """Add a ``local_path`` column mapping each image record to its file.
    
    Prefers resolving relative paths against ``source_file`` (if present) to prevent
    basename collisions across dataset folders (e.g. quadrant-enumeration vs quadrant-enumeration-disease).
    """
    image_files = (
        glob.glob(os.path.join(str(dentex_path), "**", "*.png"), recursive=True)
        + glob.glob(os.path.join(str(dentex_path), "**", "*.jpg"), recursive=True)
        + glob.glob(os.path.join("data", "**", "*.png"), recursive=True)
    )
    by_basename = {os.path.basename(p): p for p in image_files}
    images_df = images_df.copy()

    def _resolve(row: pd.Series) -> str | None:
        file_name = row.get("file_name")
        if not file_name or pd.isna(file_name):
            return None
        source_file = row.get("source_file")
        if source_file and not pd.isna(source_file):
            src_dir = os.path.dirname(str(source_file))
            cand1 = os.path.normpath(os.path.join(src_dir, str(file_name)))
            if os.path.exists(cand1):
                return cand1
            cand2 = os.path.normpath(os.path.join(src_dir, "xrays", os.path.basename(str(file_name))))
            if os.path.exists(cand2):
                return cand2
        return by_basename.get(os.path.basename(str(file_name)))

    if "file_name" in images_df.columns:
        images_df["local_path"] = images_df.apply(_resolve, axis=1)
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
        candidates.extend([
            p,
            p / "dentex" / "DENTEX",
            p / "dentex",
            p / "DENTEX",
            p / "data" / "dentex" / "DENTEX",
            p / "data" / "dentex",
        ])
    candidates.extend([
        Path("data/dentex/DENTEX"),
        Path("data/dentex"),
        Path("data"),
    ])
    
    # Google Drive paths (auto-detect when Drive is mounted in Colab)
    if Path("/content/drive/MyDrive").exists():
        candidates.extend([
            Path("/content/drive/MyDrive/VLM-DENTAL/data/dentex/DENTEX"),
            Path("/content/drive/MyDrive/VLM-DENTAL/data/dentex"),
            Path("/content/drive/MyDrive/VLM-DENTAL/data"),
            Path("/content/drive/MyDrive/vlmdental/data/dentex/DENTEX"),
            Path("/content/drive/MyDrive/vlmdental/data/dentex"),
            Path("/content/drive/MyDrive/vlmdental/data"),
            Path("/content/drive/MyDrive/dental_agent/data"),
            Path("/content/drive/MyDrive/DENTEX"),
        ])
    for c in candidates:
        if c.exists() and c.is_dir():
            # Check for split-specific files
            is_train = split_name in ("train", "training")
            if is_train:
                has_train_dir = (c / "training_data").exists() or any(d.is_dir() and "training_data" in d.name for d in c.iterdir())
                has_train_json = any(c.rglob("*train*.json")) or any(c.rglob("*disease*.json")) or any(c.rglob("*triple*.json"))
                if has_train_dir and has_train_json:
                    return c
            else:
                has_val_dir = (c / "validation_data").exists() or any(d.is_dir() and "validation_data" in d.name for d in c.iterdir())
                has_val_json = any(c.rglob("*val*.json")) or any(c.rglob("*validation_triple*.json"))
                if has_val_dir and has_val_json:
                    return c
    return None


def load_combined_dentex_dataset(
    data_dir: str | Path | None = None,
    split_name: str = "train",
    use_cache: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and combine all DENTEX datasets for a given split that contain FDI tooth enumeration
    annotations (category_id_1 and category_id_2).

    For 'train', this merges annotations from both 'quadrant-enumeration-disease' and
    'quadrant-enumeration' datasets (~1,294 images total), nearly doubling the training
    data available for YOLO tooth grounding models.
    """
    local_dir = find_local_dentex_dir(data_dir, split_name)
    if local_dir is not None:
        print(f"✅ Found existing DENTEX dataset for '{split_name}' split at: {local_dir}")
        dentex_path = local_dir
    else:
        repo_id = os.environ.get("DENTEX_IMAGES_REPO")
        if repo_id:
            print(f"Dataset not found locally, but DENTEX_IMAGES_REPO={repo_id} is set.")
            print(f"⬇️ Downloading annotation JSON directly from {repo_id}...")
            json_name = "train.json" if split_name in ("train", "training") else "validation_triple.json"
            local_json = hf_hub_download(repo_id=repo_id, filename=json_name, repo_type="dataset", cache_dir=str(data_dir) if data_dir else None)
            dentex_path = Path(local_json).parent
        else:
            print(f"⚠️ WARNING: Dataset for split '{split_name}' NOT found in any local or Colab Drive paths!")
            print(f"⬇️ Initiating HuggingFace download... (Press STOP in Colab now if you want to cancel)")
            dentex_path = download_dentex(
                cache_dir=str(data_dir) if data_dir else None,
                split_name=split_name,
            )

    extract_dentex_zips(dentex_path)
    all_coco = discover_annotation_files(dentex_path)

    matching_coco: list[tuple[str, dict]] = []
    for jf, d in all_coco.items():
        if not isinstance(d, dict) or not d.get("annotations"):
            continue
        name = jf.lower()
        if split_name in ("val", "validation"):
            if "val" not in name and "validation" not in name:
                continue
        elif split_name in ("train", "training"):
            if "train" not in name and "training" not in name:
                continue

        ann0 = d["annotations"][0] if d["annotations"] else {}
        if "category_id_1" in ann0 and "category_id_2" in ann0:
            matching_coco.append((jf, d))

    if not matching_coco:
        print(f"No multi-file enumeration matches for split '{split_name}'. Falling back to single best file.")
        best_path, best_coco = pick_best_annotation_file(all_coco, split_name=split_name)
        matching_coco = [(best_path, best_coco)]

    combined_images: list[dict] = []
    combined_annots: list[dict] = []
    combined_categories: list[dict] = []

    global_img_id = 1
    global_ann_id = 1

    for jf, coco in matching_coco:
        img_id_map: dict[int, int] = {}
        for img in coco.get("images", []):
            old_id = img["id"]
            new_img = dict(img)
            new_img["id"] = global_img_id
            new_img["source_file"] = jf
            img_id_map[old_id] = global_img_id
            combined_images.append(new_img)
            global_img_id += 1

        for ann in coco.get("annotations", []):
            old_img_id = ann.get("image_id")
            if old_img_id not in img_id_map:
                continue
            new_ann = dict(ann)
            new_ann["id"] = global_ann_id
            new_ann["image_id"] = img_id_map[old_img_id]
            combined_annots.append(new_ann)
            global_ann_id += 1

        if not combined_categories and coco.get("categories"):
            combined_categories = coco.get("categories")

    print(f"Combined {len(matching_coco)} annotation file(s) for '{split_name}': {len(combined_images)} images, {len(combined_annots)} annotations.")

    images_df = pd.DataFrame(combined_images)
    annots_df = pd.DataFrame(combined_annots)
    categories_df = pd.DataFrame(combined_categories)

    if "bbox" in annots_df.columns:
        annots_df["bbox"] = annots_df["bbox"].apply(list)

    images_df = resolve_image_paths(images_df, dentex_path)
    return images_df, annots_df, categories_df


def load_dentex_dataset(
    data_dir: str | Path | None = None,
    split_name: str = "validation",
    use_cache: bool = True,
    combine_enumeration_splits: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Top-level convenience loader for DENTEX dataset.

    Checks local project data directories first for instant loading. If not found locally,
    downloads or discovers files, auto-extracts zip archives, parses annotation files,
    resolves local image paths, and returns (images_df, annots_df, categories_df).
    
    Set ``combine_enumeration_splits=True`` to merge all valid quadrant-enumeration subfolders
    for split (e.g. quadrant-enumeration-disease + quadrant-enumeration for train).
    """
    if combine_enumeration_splits:
        return load_combined_dentex_dataset(data_dir=data_dir, split_name=split_name, use_cache=use_cache)

    local_dir = find_local_dentex_dir(data_dir, split_name)
    if local_dir is not None:
        dentex_path = local_dir
    else:
        repo_id = os.environ.get("DENTEX_IMAGES_REPO")
        if repo_id:
            print(f"Dataset not found locally, but DENTEX_IMAGES_REPO={repo_id} is set.")
            print(f"⬇️ Downloading annotation JSON directly from {repo_id}...")
            json_name = "train.json" if split_name in ("train", "training") else "validation_triple.json"
            local_json = hf_hub_download(repo_id=repo_id, filename=json_name, repo_type="dataset", cache_dir=str(data_dir) if data_dir else None)
            dentex_path = Path(local_json).parent
        else:
            print(f"Dataset for split '{split_name}' not found locally. Triggering download...")
            dentex_path = download_dentex(
                cache_dir=str(data_dir) if data_dir else None,
                split_name=split_name,
            )

    extract_dentex_zips(dentex_path)
    all_coco = discover_annotation_files(dentex_path)
    best_path, best_coco = pick_best_annotation_file(all_coco, split_name=split_name)
    images_df, annots_df, categories_df = build_dataframes(
        best_coco, data_dir=str(data_dir) if data_dir else None, use_cache=use_cache, split_name=split_name
    )
    if "source_file" not in images_df.columns:
        images_df["source_file"] = best_path
    images_df = resolve_image_paths(images_df, dentex_path)
    return images_df, annots_df, categories_df

