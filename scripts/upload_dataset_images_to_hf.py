#!/usr/bin/env python3
"""
Upload one dataset's images (+ an annotation JSON) to its own lightweight
per-image HF dataset repo -- the "upload once, download only what a given
run needs" mechanism every dataset in this project shares (see
dental_agent/data/hf_dataset_utils.py).

This was two near-identical scripts (upload_dentex_images_to_hf.py,
upload_tufts_images_to_hf.py) with the same overall shape -- stage a temp
folder, copy images, write/copy an annotation JSON, one upload_folder commit
-- but different bundle-preparation logic. Unified into one so the SHARED
parts can't independently diverge the way the FDI-index conversion did
across files before that got consolidated into dentex_row_to_fdi(). Add a
new dataset by adding one bundler function + one DATASET_BUNDLERS entry,
not a new copy of this whole script.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.config import load_config, load_env

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Please install huggingface_hub: pip install huggingface_hub")
    sys.exit(1)


def _prepare_dentex_bundle(temp_dir_path: Path, cfg):
    """Copies DENTEX's own pre-existing annotation JSON files into
    temp_dir_path (matching dentex.py's HF-download fallback's exact
    expected filenames), and returns (eligible_images_df, image_extension).
    """
    from dental_agent.data.dentex import load_dentex_dataset, find_local_dentex_dir

    print(f"Loading DENTEX dataset from {cfg.data_dir}...")
    dentex_path = find_local_dentex_dir(cfg.data_dir, split_name="train")
    if not dentex_path:
        print("ERROR: Could not find DENTEX dataset locally. Run download_dataset.py first.")
        sys.exit(1)

    imgs_df, annots_df, cats_df = load_dentex_dataset(
        data_dir=cfg.data_dir, split_name="train", combine_enumeration_splits=False
    )

    # DENTEX's own train JSON is named after its annotation tier (e.g.
    # train_quadrant_enumeration_disease.json), not "train.json" -- search
    # for it rather than hardcoding, same approach dentex.py itself uses
    # elsewhere for this reason. dentex.py's HF-download fallback
    # (load_dentex_dataset / load_combined_dentex_dataset) is hardcoded to
    # request exactly "train.json" from the repo root, though, so the
    # destination name below must stay exactly that.
    quadrant_disease_dir = dentex_path / "training_data" / "quadrant-enumeration-disease"
    json_candidates = sorted(quadrant_disease_dir.glob("*train*.json"))
    if not json_candidates:
        print(f"ERROR: No train annotation JSON found under {quadrant_disease_dir} "
              f"(looked for *train*.json). Aborting -- uploading images without the "
              f"annotation JSON produces a repo that download_dentex_slice() can't use "
              f"(it 404s looking for train.json at the repo root). Run download_dataset.py "
              f"first if the dataset isn't extracted locally yet.")
        sys.exit(1)
    if len(json_candidates) > 1:
        print(f"ERROR: Multiple candidate JSON files found under {quadrant_disease_dir}: "
              f"{[p.name for p in json_candidates]}. Pick the right one and set json_path "
              f"explicitly rather than relying on the glob picking correctly.")
        sys.exit(1)
    json_path = json_candidates[0]
    print(f"Copying annotations JSON ({json_path.name}) to temp folder as train.json...")
    shutil.copy2(json_path, temp_dir_path / "train.json")

    valid_imgs = imgs_df[imgs_df["local_path"].notna()]
    annotated_ids = set(annots_df["image_id"].unique())
    train_eligible = valid_imgs[valid_imgs["id"].isin(annotated_ids)]

    val_json_path = dentex_path / "validation_triple.json"
    if not val_json_path.exists():
        val_candidates = sorted(dentex_path.rglob("*validation_triple*.json"))
        if val_candidates:
            val_json_path = val_candidates[0]
    
    val_eligible = pd.DataFrame()
    if val_json_path.exists():
        print(f"Copying validation JSON ({val_json_path.name}) to temp folder as validation_triple.json...")
        shutil.copy2(val_json_path, temp_dir_path / "validation_triple.json")
        try:
            val_imgs_df, val_annots_df, _ = load_dentex_dataset(
                data_dir=cfg.data_dir, split_name="validation"
            )
            val_valid = val_imgs_df[val_imgs_df["local_path"].notna()]
            val_annot_ids = set(val_annots_df["image_id"].unique())
            val_eligible = val_valid[val_valid["id"].isin(val_annot_ids)]
            print(f"Found {len(val_eligible)} validation images to include in upload bundle.")
        except Exception as e:
            print(f"Warning: Could not load validation split images for bundling: {e}")
    else:
        print(f"WARNING: No validation_triple.json found under {dentex_path}. The HF repo will only support training splits.")

    eligible_imgs = pd.concat([train_eligible, val_eligible]).drop_duplicates(subset=["id"]) if not val_eligible.empty else train_eligible
    return eligible_imgs, "png"


def _prepare_tufts_bundle(temp_dir_path: Path, cfg):
    """Writes a fresh COCO-shaped train.json built from Tufts' own
    DataFrames -- unlike DENTEX, there's no pre-existing DENTEX-style JSON
    file to copy, so this constructs one directly (see tufts.py). Returns
    (eligible_images_df, image_extension). Will raise NotImplementedError,
    same as load_tufts_dataset itself, until tufts.py's tooth-position/
    diagnosis mapping is filled in -- see that module's docstring.
    """
    from dental_agent.data.tufts import load_tufts_dataset

    print("Loading Tufts dataset locally...")
    imgs_df, annots_df, cats_df = load_tufts_dataset(data_dir=cfg.data_dir)

    coco_json = {
        "images": imgs_df.to_dict(orient="records"),
        "annotations": annots_df.to_dict(orient="records"),
        "categories": cats_df.to_dict(orient="records"),
    }
    with open(temp_dir_path / "train.json", "w") as f:
        json.dump(coco_json, f)

    valid_imgs = imgs_df[imgs_df["local_path"].notna()]
    annotated_ids = set(annots_df["image_id"].unique())
    eligible_imgs = valid_imgs[valid_imgs["id"].isin(annotated_ids)]
    return eligible_imgs, "jpg"


def _prepare_tunisia_bundle(temp_dir_path: Path, cfg):
    """Writes a fresh COCO-shaped train.json built from Tunisia's own
    DataFrames -- same approach as _prepare_tufts_bundle, since there's no
    pre-existing DENTEX-style JSON to copy. Will raise NotImplementedError,
    same as load_tunisia_dataset itself, until tunisia_panoramic.py's
    region-to-FDI mapping is filled in -- see that module's docstring.
    Note: this dataset has has_diagnosis_labels=False (dataset_catalog.py)
    -- the resulting train.json will never carry a diagnosis category,
    only tooth-position ground truth for locate_tooth.
    """
    from dental_agent.data.tunisia_panoramic import load_tunisia_dataset

    print("Loading Tunisia (Panoramic Dental Xray Dataset) locally...")
    imgs_df, annots_df, cats_df = load_tunisia_dataset(data_dir=cfg.data_dir)

    coco_json = {
        "images": imgs_df.to_dict(orient="records"),
        "annotations": annots_df.to_dict(orient="records"),
        "categories": cats_df.to_dict(orient="records"),
    }
    with open(temp_dir_path / "train.json", "w") as f:
        json.dump(coco_json, f)

    valid_imgs = imgs_df[imgs_df["local_path"].notna()]
    annotated_ids = set(annots_df["image_id"].unique())
    eligible_imgs = valid_imgs[valid_imgs["id"].isin(annotated_ids)]
    return eligible_imgs, "jpg"


# Add a new dataset by adding one entry here, pointing at a new bundler
# function above -- not a new copy of main()'s staging/upload logic below.
DATASET_BUNDLERS = {
    "dentex": _prepare_dentex_bundle,
    "tufts": _prepare_tufts_bundle,
    "tunisia": _prepare_tunisia_bundle,
}


def main():
    parser = argparse.ArgumentParser(description="Upload a dataset's images to Hugging Face Hub")
    parser.add_argument(
        "--dataset", type=str, required=True, choices=list(DATASET_BUNDLERS.keys()),
        help="Which dataset to upload. tufts and tunisia will both raise NotImplementedError "
             "until their annotation mapping is filled in -- see dental_agent/data/tufts.py "
             "and dental_agent/data/tunisia_panoramic.py respectively. tunisia additionally "
             "has has_diagnosis_labels=False (dataset_catalog.py): its bundle will never "
             "carry a diagnosis category, regardless of the mapping question.",
    )
    parser.add_argument("--repo-id", type=str, required=True, help="HF Dataset Repo ID (e.g. rezaxr14/dentex-train-images)")
    args = parser.parse_args()

    load_env()
    cfg = load_config()

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token or hf_token.startswith("your_"):
        print("ERROR: HF_TOKEN is missing or invalid in .env. Requires write access.")
        sys.exit(1)

    api = HfApi(token=hf_token)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        images_dir = temp_dir_path / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        eligible_imgs, image_ext = DATASET_BUNDLERS[args.dataset](temp_dir_path, cfg)
        print(f"Found {len(eligible_imgs)} valid annotated images to upload.")

        for idx, row in eligible_imgs.iterrows():
            img_id = int(row["id"])
            local_path = str(row["local_path"])
            dest_path = images_dir / f"{img_id}.{image_ext}"
            try:
                os.link(local_path, dest_path)
            except Exception:
                shutil.copy2(local_path, dest_path)
            if (idx + 1) % 100 == 0:
                print(f"  Prepared {idx + 1}/{len(eligible_imgs)} images...")

        print(f"Uploading bundled dataset to {args.repo_id}...")
        api.upload_folder(
            folder_path=str(temp_dir_path),
            repo_id=args.repo_id,
            repo_type="dataset",
        )

    print("Upload complete.")


if __name__ == "__main__":
    main()
