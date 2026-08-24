"""
Panoramic Dental Xray Dataset (Tunisia) loader -- mirrors dentex.py's
interface (images_df / annots_df / categories_df, same column names:
category_id_1=quadrant, category_id_2=tooth_position, bbox=[x,y,w,h]) so
it's a drop-in dataset for prepare_yolo_dataset.py's DATASET_LOADERS,
hf_dataset_utils.py's upload/download helpers, and slicing.py -- everything
downstream that already reads that shape.

READ THIS BEFORE TRUSTING THE OUTPUT OF THIS MODULE:

Source paper: Brahmi W, Jdey I. "Automatic tooth instance segmentation and
identification from panoramic X-Ray images using deep CNN." Multimed Tools
Appl 83, 55565-55585 (2024). https://doi.org/10.1007/s11042-023-17568-z
Dataset: Mendeley Data v3, https://data.mendeley.com/datasets/73n3kz2k4k/3,
CC BY 4.0, no registration required -- see dataset_catalog.py's entry
(corrected after direct follow-up from the v2 listing the original Uribe
et al. 2024 systematic review cites).

WHY THIS DATASET, OVER THE OTHER GROUNDING-ONLY CANDIDATES IN THE CATALOG:
DNS (COCO+FDI, exactly the shape locate_tooth needs) and TL-pano (VIA,
FDI+quadrant sub-labels, closest match to fdi_label's output shape) both
looked like better matches for annotation richness -- but DNS is
access-gated (upon request to the authors, same unpredictable-wait problem
as Tufts) and TL-pano is both access-gated AND restricted to non-commercial
research use (a real encumbrance for a project whose outputs -- a trained
model, a published paper, released verified traces -- may not stay
comfortably inside "non-commercial" once published). Tunisia is the only
one of the three with zero access wait (plain public download) and the
cleanest license (CC BY 4.0, no share-alike or non-commercial restriction)
-- worth taking on its one remaining open question (below) rather than
starting a loader that's immediately blocked on a request form.

WHAT'S SETTLED vs. WHAT ISN'T:

Settled: per dataset_catalog.py's has_diagnosis_labels=False, this dataset
has NO pathology/diagnosis labels in any of its three parts (107 images
with tooth-instance segmentation, 60 images labeled by 8 tooth-TYPE classes
-- canine/incisor/molar/premolar variants, morphological type, not FDI
position -- and 54 unannotated images). So this loader only ever targets
the 107-image segmented subset, and it can only ever feed locate_tooth's
grounding generalization, never diagnosis trace-gen. Don't wire its output
into anything that expects a diagnosis category (category_id_3) to exist.

NOT settled: whether the 107-image subset's VIA region_attributes carry a
per-tooth FDI (or equivalent) position label, or are anonymous "this is a
tooth, distinguish it from its neighbors" instances with no semantic
numbering. This matters more than it might look like a minor gap:
prepare_yolo_dataset.py's locate_tooth trains a 32-CLASS detector (one
class per FDI quadrant+position -- see convert_single_image's
`class_idx = (quadrant-1)*8 + (position-1)`), NOT a single-class "is this a
tooth" detector. Anonymous instance masks with no position label literally
cannot feed that 32-class scheme as-is; they'd need either a real
per-instance label in the data (this module's hoped-for case), or a design
change to locate_tooth itself (e.g. a single-class detector plus a separate
numbering step) -- a real, separate design decision this module does not
make unilaterally by guessing.

The paper's own title is a genuine positive signal, not a coincidence:
"Instance Segmentation AND IDENTIFICATION" is the field's standard phrasing
for exactly this (per-tooth numbering assignment, not just telling
instances apart) -- see e.g. Cui et al., "ToothNet: Automatic Tooth
Instance Segmentation and Identification," CVPR 2019, which uses the
identical phrase to mean FDI/Universal numbering. But it is NOT proof: the
paper itself is paywalled beyond its abstract, and the Mendeley listing is
blocked by bot detection from automated fetching, so this has not been
confirmed against the real region_attributes keys/values. That
confirmation takes about thirty seconds once the archive is actually
downloaded and extracted -- see `_region_to_fdi`'s docstring for exactly
what to check.

ENGINEERING DISCIPLINE (same as tufts.py, deliberately): the parts that are
100% verifiable without guessing -- finding the extracted files, parsing
the standard, published VIA2 JSON schema (Dutta A, Zisserman A, "The VIA
Annotation Software for Images, Audio and Video," ACM Multimedia 2019 --
the same tool+paper this dataset's own authors cite for how they built it,
so this isn't a bespoke format to infer), and computing a bounding box from
a polygon's points (plain geometry, not a labeling claim) -- are
implemented and safe to run today. The one dataset-specific SEMANTIC
question, `_region_to_fdi`, raises NotImplementedError with exactly what to
check, rather than guessing at a medical label. This is a genuine step
further than tufts.py could get (Tufts' image-to-mask correspondence
itself was too ambiguous to build anything past image discovery); here,
real bounding boxes ARE produced by `load_tunisia_dataset` below, they just
aren't yet tagged with an FDI quadrant+position.
"""

from __future__ import annotations

import glob
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image


def find_local_tunisia_dir(search_roots: list[str] | None = None) -> Path | None:
    """Look for an already-extracted local copy of the Mendeley archive.

    Unlike Tufts, this needs NO access request -- CC BY 4.0, plain public
    download. This only needs the archive downloaded and unzipped once from
    https://data.mendeley.com/datasets/73n3kz2k4k/3, same manual step as
    Tufts otherwise (no automatic fetch is wired in here, since there's no
    guarantee of a stable direct-download URL across Mendeley dataset
    versions). Set TUNISIA_LOCAL_DIR in .env to skip the search entirely.
    """
    env_path = os.environ.get("TUNISIA_LOCAL_DIR")
    if env_path and os.path.isdir(env_path):
        return Path(env_path)

    search_roots = search_roots or [".", "./data", "/content", "/kaggle/input"]
    # Tolerant of naming variation across different real-world extracted
    # copies (the Mendeley zip's top-level folder name isn't fixed) -- this
    # part doesn't need the exact folder name verified, just a plausible
    # match to point at.
    patterns = ["*[Pp]anoramic*[Dd]ental*[Xx]ray*", "*73n3kz2k4k*", "*[Tt]unisia*[Dd]ental*"]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pattern in patterns:
            matches = glob.glob(os.path.join(root, pattern))
            for m in matches:
                if os.path.isdir(m):
                    return Path(m)
    return None


def _find_via_json(tunisia_root: Path) -> Path | None:
    """Locate the VIA annotation export inside the extracted archive.
    Tolerant of the several conventional filenames VIA2 exports use (the
    tool doesn't force one name) -- this is a filename-shape search, not a
    claim about the JSON's contents.
    """
    candidates = (
        list(tunisia_root.glob("**/via_region_data.json"))
        + list(tunisia_root.glob("**/via_export_json.json"))
        + list(tunisia_root.glob("**/*via*.json"))
    )
    if candidates:
        return candidates[0]
    # Fall back to "the only JSON file in the archive" if none of VIA's
    # usual names are present -- still a filename-shape guess, not a
    # contents guess.
    all_json = list(tunisia_root.glob("**/*.json"))
    return all_json[0] if len(all_json) == 1 else None


def _find_image_dir(tunisia_root: Path) -> Path | None:
    """Locate the panoramic image files. Per the Mendeley listing, this
    release ships JPEG images at 2964x1464 -- confirm orientation/exact
    size against the real files rather than assuming this docstring's
    number is authoritative.
    """
    image_files = (
        list(tunisia_root.glob("**/*.jpg")) + list(tunisia_root.glob("**/*.jpeg"))
        + list(tunisia_root.glob("**/*.JPG")) + list(tunisia_root.glob("**/*.jfif"))
    )
    if not image_files:
        return None
    # Most common parent directory among found images -- tolerant of the
    # archive having a nested images/ folder, a flat layout, or stray
    # duplicate images elsewhere in the tree.
    parent_counts = Counter(f.parent for f in image_files)
    return parent_counts.most_common(1)[0][0]


def _region_to_bbox(region: dict[str, Any]) -> list[float] | None:
    """Compute [x_min, y_min, w, h] from one VIA region's shape_attributes.

    This IS a standard, verifiable operation -- VIA2's rect/polygon/
    polyline/circle/ellipse shape_attributes schema is published (Dutta &
    Zisserman 2019), not specific to this dataset, so no guessing is needed
    to parse it correctly regardless of what the region semantically
    represents. Returns None for shapes this function doesn't handle
    (e.g. a bare "point") rather than guessing a zero-size box.
    """
    shape = region.get("shape_attributes", {})
    name = shape.get("name")

    if name == "rect":
        x, y = float(shape["x"]), float(shape["y"])
        w, h = float(shape["width"]), float(shape["height"])
        return [x, y, w, h]

    if name in ("polygon", "polyline"):
        xs = shape.get("all_points_x", [])
        ys = shape.get("all_points_y", [])
        if not xs or not ys:
            return None
        x0, x1 = float(min(xs)), float(max(xs))
        y0, y1 = float(min(ys)), float(max(ys))
        return [x0, y0, x1 - x0, y1 - y0]

    if name in ("circle", "ellipse"):
        cx, cy = float(shape["cx"]), float(shape["cy"])
        rx = float(shape.get("rx", shape.get("r", 0)))
        ry = float(shape.get("ry", shape.get("r", 0)))
        return [cx - rx, cy - ry, 2 * rx, 2 * ry]

    return None


def _region_to_fdi(region_attributes: dict[str, Any]) -> tuple[int, int]:
    """Map one VIA region's region_attributes to (quadrant, tooth_position)
    in 1-indexed FDI form (quadrant 1-4, position 1-8) -- the same pair
    category_id_1/category_id_2 hold for DENTEX (see dentex_row_to_fdi).

    NOT IMPLEMENTED: this is the ONE open question blocking this loader,
    and per this module's docstring it takes about thirty seconds to
    resolve once the real archive is on disk:

      1. Open the VIA JSON (see _find_via_json) and look at ONE region's
         region_attributes dict -- e.g. in a notebook cell:
         `list(json.load(open(via_json_path))["_via_img_metadata"].values())[0]["regions"][0]["region_attributes"]`
         VIA region_attributes are fully custom per-project; there is no
         fixed schema for what's inside, unlike shape_attributes above.
      2. Does it have a key beyond a generic class label like "tooth"? If
         there IS a numbering field, what convention is it in -- two-digit
         FDI directly (e.g. "36"), Universal Numbering System (1-32,
         see fdi.py for the conversion this project already has for that
         case), or something else entirely?
      3. Does EVERY region in the 107-image segmented subset have this
         field populated, or only some? A partially-missing field can't be
         trusted as ground truth without knowing why it's missing on some
         instances (annotator skipped it vs. genuinely not visible).

    If the answer to (2) is "no identity field, anonymous instances only,"
    this dataset cannot feed prepare_yolo_dataset.py's 32-class
    DATASET_LOADERS scheme as-is (see that module's docstring and this
    module's own docstring) -- it would still be usable for a different
    purpose (a single-class "is this a tooth" pretraining stage for
    locate_tooth), which is a real design change to raise with the rest of
    the project, not something to decide by guessing here.
    """
    raise NotImplementedError(
        "Tunisia region-to-FDI mapping is not implemented -- see this function's "
        "docstring for exactly what to check in the real VIA JSON (three questions, "
        "~30 seconds with the file open) before writing this."
    )


def load_tunisia_dataset(
    data_dir: str | None = None,
    max_images: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the Tunisia Panoramic Dental Xray Dataset into the same
    (images_df, annots_df, categories_df) shape dentex.py's
    load_dentex_dataset returns.

    Gets further than load_tufts_dataset before hitting the open question:
    image discovery, VIA JSON parsing, and bbox-from-region geometry are
    ALL implemented and safe to run today (none of them require knowing
    what a region's identity label means -- see _region_to_bbox's
    docstring). annots_df below already carries real bounding boxes. The
    hard stop is specifically at assigning each region an FDI
    quadrant+position (_region_to_fdi) -- left as a hard stop rather than a
    silent guess, same reasoning as tufts.py.

    `data_dir` is accepted for interface parity with load_dentex_dataset /
    load_tufts_dataset (and so prepare_yolo_dataset.py can call every
    loader the same way) but isn't currently used to locate files here --
    see find_local_tunisia_dir, which searches fixed candidate roots plus
    TUNISIA_LOCAL_DIR instead.
    """
    tunisia_root = find_local_tunisia_dir()
    if tunisia_root is None:
        raise FileNotFoundError(
            "No local Tunisia Panoramic Dental Xray Dataset directory found. Unlike "
            "Tufts this needs no registration -- download "
            "https://data.mendeley.com/datasets/73n3kz2k4k/3 (CC BY 4.0) and extract it "
            "yourself, then set TUNISIA_LOCAL_DIR in .env to the extracted folder."
        )

    via_json_path = _find_via_json(tunisia_root)
    if via_json_path is None:
        raise FileNotFoundError(
            f"Found {tunisia_root} but couldn't locate a VIA annotation JSON inside it "
            "(looked for via_region_data.json, via_export_json.json, *via*.json, or a "
            "single unambiguous *.json). The extracted archive's layout may not match "
            "what this loader expects -- check the actual contents and adjust "
            "_find_via_json."
        )

    image_dir = _find_image_dir(tunisia_root)
    if image_dir is None:
        raise FileNotFoundError(f"Found {tunisia_root} but no .jpg/.jpeg images inside it.")

    with open(via_json_path) as f:
        via_data = json.load(f)

    # VIA2 project-file exports wrap per-image entries in an
    # "_via_img_metadata" dict; plain "via_region_data.json"-style exports
    # ARE that metadata dict directly. Handle both without guessing which
    # one this particular archive uses.
    img_metadata = via_data.get("_via_img_metadata", via_data)

    image_rows: list[dict[str, Any]] = []
    annot_rows: list[dict[str, Any]] = []
    next_annot_id = 0

    for img_idx, (_key, entry) in enumerate(sorted(img_metadata.items())):
        if max_images is not None and img_idx >= max_images:
            break
        filename = entry.get("filename")
        if not filename:
            continue

        local_path = image_dir / filename
        if not local_path.exists():
            # Tolerant of the VIA key including a filesize suffix that
            # doesn't correspond to an on-disk rename -- fall back to
            # searching by basename anywhere under the archive root.
            matches = list(tunisia_root.glob(f"**/{filename}"))
            if not matches:
                continue
            local_path = matches[0]

        with Image.open(local_path) as im:
            width, height = im.size

        image_rows.append({
            "id": img_idx,
            "file_name": filename,
            "local_path": str(local_path),
            "width": width,
            "height": height,
        })

        for region in entry.get("regions", []):
            bbox = _region_to_bbox(region)
            if bbox is None:
                continue
            annot_rows.append({
                "id": next_annot_id,
                "image_id": img_idx,
                "bbox": bbox,
                # Kept raw (not yet converted to category_id_1/2) so
                # _region_to_fdi's three verification questions can be
                # answered directly off this DataFrame once it's real:
                # annots_df.iloc[0]["region_attributes"].
                "region_attributes": region.get("region_attributes", {}),
            })
            next_annot_id += 1

    images_df = pd.DataFrame(image_rows)
    annots_df = pd.DataFrame(annot_rows)
    categories_df = pd.DataFrame()  # No fixed category list for this dataset -- see module docstring.

    if len(annots_df) == 0:
        # Nothing to hard-stop on -- either an empty/unannotated slice was
        # pointed at (e.g. the 54 unannotated images), or region parsing
        # found nothing. Return empty rather than raising, so a caller
        # inspecting image discovery alone (e.g. counting available
        # images) isn't forced through the FDI hard-stop below for no reason.
        return images_df, annots_df, categories_df

    # This is the hard stop -- see module docstring. Everything above
    # (image discovery, VIA parsing, bbox geometry) is solid and
    # independently useful -- annots_df already has real bounding boxes;
    # category_id_1/category_id_2 construction is not, yet.
    raise NotImplementedError(
        f"Found {len(images_df)} Tunisia images with {len(annots_df)} tooth regions at "
        f"{tunisia_root}, but category_id_1/category_id_2 construction needs "
        "_region_to_fdi implemented first -- see that function's docstring for exactly "
        "what to check (three questions, ~30 seconds with the real file open)."
    )


def download_tunisia_slice(
    image_ids: list[int],
    repo_id: str | None = None,
    cache_dir: str | None = None,
) -> dict[int, Path | None]:
    """Download only the given image_ids from a lightweight per-image HF repo
    (same mechanism as DENTEX/Tufts, once Tunisia images have actually been
    uploaded there by `scripts/upload_dataset_images_to_hf.py --dataset tunisia`
    -- see hf_dataset_utils.py).
    """
    if repo_id is None:
        repo_id = os.environ.get("TUNISIA_IMAGES_REPO")
    from dental_agent.data.hf_dataset_utils import download_dataset_slice
    return download_dataset_slice(image_ids, repo_id=repo_id, filename_template="images/{id}.jpg", cache_dir=cache_dir)
