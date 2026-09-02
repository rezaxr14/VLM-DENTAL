"""
Tufts Dental Database loader -- mirrors dentex.py's interface
(images_df / annots_df / categories_df with the same column names:
category_id_1=quadrant, category_id_2=tooth_position, category_id_3=
diagnosis category id, bbox=[x,y,w,h]) so it's a drop-in second dataset
for everything downstream (trace-gen, YOLO training, the reward pipeline)
that already reads that shape.

READ THIS BEFORE TRUSTING THE OUTPUT OF THIS MODULE:

Tufts is NOT freely auto-downloadable like DENTEX. It's gated behind a
request form at https://tdd.ece.tufts.edu/ -- find_local_tufts_dir()
expects you've already requested access, downloaded, and extracted the
archive yourself; point TUFTS_LOCAL_DIR at it.

Everything below was verified against the *real* extracted annotation
files (Segmentation/teeth_bbox.json, Expert/expert.json,
Student/student.json -- all 1,000 records each), not guessed and not
taken on faith from secondary sources. Three things worth knowing before
you use it:

1. IDENTITY, NOT +1 -- unlike dentex_row_to_fdi(). category_id_1/
   category_id_2 below are already 1-indexed FDI (quadrant 1-4, position
   1-8) at construction time, matching prepare_yolo_dataset.py's existing
   "tufts" entry (currently commented out -- this patch fills in the
   loader it's waiting on) which expects an identity quadrant_position_fn,
   not dentex_row_to_fdi's +1. DENTEX's 0-indexing is an artifact of
   DENTEX's own JSON encoding, not a universal convention -- don't run
   Tufts rows through dentex_row_to_fdi directly, that would double-
   increment them. Every annots_df row here carries a "source_dataset":
   "tufts" column for exactly this reason -- reward/eval/training code
   that needs FDI values should call dental_agent.data.fdi_utils.row_to_fdi
   (dispatches on that column) rather than dentex_row_to_fdi directly, so
   this can't silently regress the next time a third dataset shows up.

2. DIAGNOSIS COVERAGE IS PARTIAL, NOT GUESSED TO BE COMPLETE. Verified
   full taxonomy of expert.json's abnormality `title` field, across all
   1,000 images (340 abnormal, 660 normal -- corrects an earlier draft
   estimate of 705/295 that undercounted abnormal cases): None (660),
   Periapical (217), Non-Odontogenic (105), Pericoronal (48),
   Inter-Radicular (4). Only "Periapical" has an unambiguous match to one
   of DENTEX's four classes (Caries, Deep Caries, Periapical Lesion,
   Impacted). Tufts' expert.json simply does not annotate caries or tooth
   impaction as findings -- those two Tufts categories return None from
   _map_tufts_pathology_to_dentex_category rather than being force-fit
   onto the nearest-sounding DENTEX label. If you want Tufts to
   contribute Caries/Deep Caries/Impacted training signal, that needs a
   real decision (a different Tufts annotation layer, or accepting Tufts
   only ever contributes Periapical Lesion examples) -- not something to
   silently paper over here.

3. MULTI-TOOTH LESIONS ARE REAL, NOT A PARSING BUG. expert.json's
   abnormality objects give lesion polygons but no explicit tooth-number
   field -- which tooth(s) a finding belongs to has to be recovered by
   overlapping the lesion polygon's bounding box against
   teeth_bbox.json's per-tooth boxes for that image (a real geometric
   join, implemented below, not a guess). Measured against all 374 real
   non-None findings: 166 (44%) overlap exactly one tooth with clearly
   dominant area, 144 (39%) overlap two or more teeth with comparable
   area (this matches the clinical reality visible in the free-text
   Descriptions -- e.g. "number 25 26" for one lesion spanning two
   adjacent roots), and 64 (17%) don't overlap any tooth box at all
   (plausible for Non-Odontogenic bone lesions away from tooth roots).
   load_tufts_dataset() emits one annotation row per overlapping tooth
   (so a two-tooth lesion becomes two rows sharing the same diagnosis and
   pathology polygon) and tags each with n_teeth_matched so a caller can
   filter to n_teeth_matched == 1 for an unambiguous-only subset. Findings
   with zero tooth overlap are excluded from annots_df (can't populate
   category_id_1/2 without a tooth) but counted in the printed summary,
   not silently dropped.

4. TOOTH-BBOX OVERLAP UNDER-RECALLS FOR TRUE PERIAPICAL LESIONS, BY
   CLINICAL DEFINITION. "Periapical" means at/beyond the root apex --
   which routinely falls just OUTSIDE teeth_bbox.json's tooth box (that
   box covers the visible crown+root, not the periapical bone beyond the
   root tip). Verified example: image 5's Description reads "radiopacity
   apical to teeth number 18 and 19"; its two lesion rings sit entirely
   below (higher y) both teeth 18 and 19's boxes -- zero bbox overlap
   with either, so the geometric join in load_tufts_dataset excludes this
   finding from annots_df entirely, even though the correct tooth
   assignment is stated in plain text. This is the SAFE failure direction
   (silently dropped, not silently wrong), but it does mean recall on
   true periapical-tooth pairs is lower than the "9/217 findings had zero
   overlap" summary line alone would suggest is a healthy number --
   that low count can also hide the opposite failure (a lesion box that
   overlaps a plausible NEIGHBORING tooth by accident rather than the one
   actually named in the Description, which looks like a clean match but
   isn't). Checked 8 real Periapical findings by hand: every explicit
   tooth number in the free-text Description exactly matched a real
   tooth present in that image's teeth_bbox.json (one partial miss where
   a described tooth wasn't in that image's bbox set at all -- plausibly
   a missing/extracted tooth). That's a strong, unexploited signal:
   extracting tooth numbers directly from Description text would likely
   beat this geometric join on both precision and recall, but it's a
   real methodology change (primary source of ground truth, not just an
   implementation detail) worth deciding deliberately rather than
   swapping in silently -- left as bbox-overlap-only for now.

Two more verified, non-obvious facts baked into the constants below:

- Tooth `title` in teeth_bbox.json is not always numeric. Alongside
  Universal Numbering 1-32 (permanent teeth), 20 of the 1,000 images
  carry letter titles A-T -- the standard Universal Numbering System
  letter scheme for PRIMARY (deciduous) teeth, i.e. mixed-dentition
  patients. Primary teeth map to FDI quadrants 5-8 (not 1-4), which is
  outside the range every existing prompt/reward/hint in this codebase
  assumes (dentex_row_to_fdi's docstring is explicit that quadrant is
  1-4). include_primary_teeth defaults to False everywhere in this module
  for exactly that reason -- turning it on hands quadrant-5-8 rows to
  code that has never seen them, silently, which is the same failure
  class as the FDI 0-index bug. Primary-tooth data is still real and
  still returned when the caller explicitly opts in.

- teeth_bbox.json's External IDs are lowercase ("797.jpg"); expert.json's
  and student.json's are uppercase ("245.JPG"). The underlying numeric
  stems are identical across all three files (verified: same 1,000-id
  set), but the ids are NOT a clean 1-1000 range -- 38 numbers in [1,1000]
  are absent and 38 numbers above 1000 (up to 1051) are present instead.
  Everything here joins on the numeric stem, case- and extension-
  insensitive; nothing assumes range(1, 1001).

Segmentation (teeth_polygon.json, the *_mask/ image folders,
maxillomandibular/) is intentionally NOT implemented yet -- see
load_tufts_segmentation()'s docstring below for exactly what's already
usable (expert.json's pathology polygons, verified) vs. what still needs
real files to check pixel-value semantics against (the mask images) before
it can be trusted, following the same policy as everything else in this
module: a real file to verify against, or a documented NotImplementedError,
never a guess.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

# Mirrors dentex.py's fallback categories_df exactly, so category_id_3
# values Tufts emits resolve to the same names/ids DENTEX consumers
# already expect.
DENTEX_CATEGORIES = [
    {"id": 0, "name": "Impacted", "supercategory": "Impacted"},
    {"id": 1, "name": "Caries", "supercategory": "Caries"},
    {"id": 2, "name": "Periapical Lesion", "supercategory": "Periapical Lesion"},
    {"id": 3, "name": "Deep Caries", "supercategory": "Deep Caries"},
]
_PERIAPICAL_LESION_ID = 2


def _permanent_universal_to_fdi(n: int) -> tuple[int, int]:
    """Universal 1-32 (permanent teeth) -> FDI (quadrant 1-4, position 1-8).
    Verified against all 1,000 teeth_bbox.json records: title is a digit
    string "1".."32" for permanent teeth. Formula per quadrant confirmed
    against the real per-record examples and the standard Universal<->FDI
    conversion."""
    if 1 <= n <= 8:
        return 1, 9 - n          # Universal 1-8   -> FDI 18..11
    if 9 <= n <= 16:
        return 2, n - 8          # Universal 9-16  -> FDI 21..28
    if 17 <= n <= 24:
        return 3, 25 - n         # Universal 17-24 -> FDI 38..31
    if 25 <= n <= 32:
        return 4, n - 24         # Universal 25-32 -> FDI 41..48
    raise ValueError(f"Universal tooth number {n} out of permanent-tooth range 1-32")


# Standard Universal Letter System for primary teeth (A-T), FDI quadrants
# 5-8, positions 1-5 (primary dentition has no premolars/3rd molars, so
# only 5 tooth positions per quadrant vs. permanent's 8). This is a
# documented clinical convention, not an inference from this dataset's
# geometry -- listed explicitly rather than derived so it's auditable.
_PRIMARY_LETTER_TO_FDI = {
    "A": (5, 5), "B": (5, 4), "C": (5, 3), "D": (5, 2), "E": (5, 1),
    "F": (6, 1), "G": (6, 2), "H": (6, 3), "I": (6, 4), "J": (6, 5),
    "K": (7, 5), "L": (7, 4), "M": (7, 3), "N": (7, 2), "O": (7, 1),
    "P": (8, 1), "Q": (8, 2), "R": (8, 3), "S": (8, 4), "T": (8, 5),
}

# Verified full taxonomy of expert.json/student.json's abnormality
# `title` field across all 1,000 images (both files, identical vocabulary,
# different counts). Only Periapical maps to one of DENTEX's 4 classes --
# see module docstring point 2 for why the rest deliberately return None.
_TUFTS_TITLE_TO_DENTEX_CATEGORY_ID: dict[str, int | None] = {
    "None": None,
    "Periapical": _PERIAPICAL_LESION_ID,
    "Non-Odontogenic": None,
    "Pericoronal": None,
    "Inter-Radicular": None,
}


def _is_valid_tufts_root(p: Path) -> bool:
    """Validate that a directory is genuinely a Tufts dataset root and not an
    output/intermediate YOLO folder (e.g. yolo_dentex_tufts_cv)."""
    if not p.is_dir():
        return False
    name_lower = p.name.lower()
    if any(ignore_word in name_lower for ignore_word in ("yolo", "runs", "output", "export", "checkpoint", "fold_")):
        return False
    # Check for presence of key Tufts annotation files or radiograph directories
    has_annotations = (
        (p / "Segmentation" / "teeth_bbox.json").exists()
        or (p / "segmentation" / "teeth_bbox.json").exists()
        or (p / "teeth_bbox.json").exists()
        or (p / "Expert" / "expert.json").exists()
        or (p / "expert" / "expert.json").exists()
        or (p / "expert.json").exists()
        or any(p.glob("**/teeth_bbox.json"))
        or any(p.glob("**/expert.json"))
    )
    has_radiographs = (
        (p / "Radiographs").is_dir()
        or (p / "radiographs").is_dir()
        or any(p.glob("**/Radiographs/*.jpg"))
        or any(p.glob("**/radiographs/*.jpg"))
        or any(p.glob("**/Radiographs/*.png"))
        or any(p.glob("**/radiographs/*.png"))
    )
    return bool(has_annotations or has_radiographs)


def find_local_tufts_dir(search_roots: list[str] | None = None) -> Path | None:
    """Look for an already-extracted local copy of the Tufts archive.

    If not found locally, falls back to downloading the full repository snapshot
    from TUFTS_IMAGES_REPO if configured in .env.
    """
    env_path = os.environ.get("TUFTS_LOCAL_DIR")
    if env_path and os.path.isdir(env_path) and _is_valid_tufts_root(Path(env_path)):
        return Path(env_path)

    search_roots = search_roots or [".", "./data", "/content", "/kaggle/input"]
    patterns = ["*[Tt]ufts*", "*TDD*", "*tufts-dental-database*"]
    for root in search_roots:
        if not os.path.isdir(root):
            continue
        for pattern in patterns:
            matches = glob.glob(os.path.join(root, pattern))
            for m in matches:
                candidate = Path(m)
                if candidate.is_dir() and _is_valid_tufts_root(candidate):
                    return candidate

    # HF snapshot fallback if dataset is uploaded to Hugging Face
    repo_id = os.environ.get("TUFTS_IMAGES_REPO", "Reza-Nadimi/tufts-train-images")
    if repo_id:
        try:
            from huggingface_hub import snapshot_download
            target_dir = Path("data/Tufts")
            print(f"Local Tufts folder not found. Downloading full dataset snapshot from {repo_id} to {target_dir}...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=str(target_dir),
                token=os.environ.get("HF_TOKEN"),
            )
            if target_dir.is_dir() and _is_valid_tufts_root(target_dir):
                return target_dir
        except Exception as e:
            print(f"Warning: snapshot_download from {repo_id} failed: {e}")

    return None


def _find_radiograph_dir(tufts_root: Path) -> Path | None:
    """Locate the raw radiograph images under the Tufts folder tree."""
    direct = tufts_root / "Radiographs"
    if direct.is_dir():
        image_files = list(direct.glob("*.jpg")) + list(direct.glob("*.JPG")) + list(direct.glob("*.png"))
        if len(image_files) > 0:
            return direct

    candidates = list(tufts_root.glob("**/Radiograph*/**")) + list(tufts_root.glob("**/radiograph*/**"))
    scored = []
    for c in candidates:
        if not c.is_dir():
            continue
        seen = set()
        for ext in ("*.jpg", "*.JPG", "*.png", "*.PNG"):
            for f in c.glob(ext):
                seen.add(f.resolve())
        if seen:
            scored.append((len(seen), c))
    if not scored:
        return None
    scored.sort(key=lambda t: -t[0])
    return scored[0][1]


def _find_annotation_file(tufts_root: Path, subfolder: str, filename: str) -> Path | None:
    """Locate one of the three annotation JSONs (teeth_bbox.json,
    expert.json, student.json), tolerant of case variation in the
    containing folder name (Segmentation/segmentation, Expert/expert)."""
    direct = tufts_root / subfolder / filename
    if direct.is_file():
        return direct
    matches = list(tufts_root.glob(f"**/{filename}"))
    return matches[0] if matches else None


def _stem_id(external_id: str) -> int:
    """Numeric image id from an 'External ID' like '797.jpg' or '245.JPG'
    or 'Radiographs/797.jpg'. Case-, path-, and extension-insensitive by
    construction -- teeth_bbox.json uses lowercase .jpg, expert.json/student.json
    use uppercase .JPG, for the exact same 1,000 underlying ids (verified)."""
    raw_str = str(external_id).strip()
    fname = raw_str.replace("\\", "/").split("/")[-1]
    stem = fname.rsplit(".", 1)[0] if "." in fname else fname
    return int(stem)


def _title_to_fdi(title: str, include_primary_teeth: bool) -> tuple[int, int, bool] | None:
    """One teeth_bbox.json tooth `title` -> (quadrant, tooth_position,
    is_primary) in 1-indexed FDI form, or None if it's a primary tooth and
    the caller didn't opt in (see module docstring re: quadrant 5-8 being
    out of range for existing prompts/rewards)."""
    if title.isdigit():
        n = int(title)
        if not (1 <= n <= 32):
            return None
        q, tp = _permanent_universal_to_fdi(n)
        return q, tp, False
    if title in _PRIMARY_LETTER_TO_FDI:
        if not include_primary_teeth:
            return None
        q, tp = _PRIMARY_LETTER_TO_FDI[title]
        return q, tp, True
    return None


def _bbox_yxyx_to_xywh(bb: list[float]) -> list[float]:
    """teeth_bbox.json stores [ymin, xmin, ymax, xmax] (verified against
    real coordinate ranges: axis bounded ~840 is y, ~1615 is x, matching
    the dataset's 1615x840 images). Converts to COCO-style [x, y, w, h]
    to match dentex.py's bbox convention."""
    ymin, xmin, ymax, xmax = bb
    return [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)]


def _is_degenerate_bbox(bb_yxyx: list[float], min_span: float = 3.0) -> bool:
    """Flags near-zero-area boxes (measured: 85/26,005 = 0.33% of all
    teeth_bbox.json boxes have a span <=2px on one axis, e.g.
    [371, 641, 372, 642] -- almost certainly digitization placeholders,
    not real tooth extents). Excluded from grounding/annotation output
    rather than silently trained on as if they were real boxes."""
    ymin, xmin, ymax, xmax = bb_yxyx
    return (ymax - ymin) < min_span or (xmax - xmin) < min_span


def _ring_bboxes(polygons: list[list[list[float]]]) -> list[tuple[float, float, float, float]]:
    """Per-ring bounding boxes (x0, y0, x1, y1) for one finding's polygon
    part(s) -- deliberately NOT unioned into a single box.

    Verified against real data: a finding-object's "polygons" list can
    hold several spatially DISJOINT rings describing physically separate
    lesions that happen to share one title/classification (e.g. image
    899's Periapical object has 7 rings spread across teeth 1, 13, 14 and
    16 -- exactly matching its own Description, "tooth number 1 ...
    14 16 and 13"). An earlier version of this function unioned every
    ring into one box, which for multi-ring findings produced a box
    spanning most of the image width and looked like it overlapped 20-30
    teeth at once -- an artifact of the union, not a real finding. Each
    ring's own bbox is used for the tooth-overlap join instead (see
    load_tufts_dataset), which also gives nudge_crop a tighter,
    tooth-specific target than one coarse union box would.

    Rings with under 3 points (measured: 55/721 = 7.6% of all rings) are
    dropped -- not enough vertices to enclose an area, almost certainly a
    stray digitization marker rather than a real lesion boundary.
    """
    boxes = []
    for ring in polygons:
        if len(ring) < 3:
            continue
        xs = [pt[0] for pt in ring]
        ys = [pt[1] for pt in ring]
        boxes.append((min(xs), min(ys), max(xs), max(ys)))
    return boxes


def _bbox_overlap_area(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection area of two (x0,y0,x1,y1) boxes; 0 if disjoint."""
    x0, y0 = max(a[0], b[0]), max(a[1], b[1])
    x1, y1 = min(a[2], b[2]), min(a[3], b[3])
    if x1 <= x0 or y1 <= y0:
        return 0.0
    return (x1 - x0) * (y1 - y0)


def _bbox_from_mask(mask_array: np.ndarray, instance_value: int) -> list[float] | None:
    """Compute a [x, y, w, h] bounding box for one labeled instance in a
    mask array. A standard, verifiable image-processing operation -- kept
    for future segmentation work (see load_tufts_segmentation) even
    though the primary tooth-grounding path above no longer needs it now
    that teeth_bbox.json gives verified per-tooth boxes directly."""
    ys, xs = np.where(mask_array == instance_value)
    if len(xs) == 0:
        return None
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def _load_tooth_index(
    teeth_bbox_path: Path, include_primary_teeth: bool
) -> tuple[dict[int, list[dict]], dict[str, int]]:
    """Parse teeth_bbox.json into {image_stem_id: [tooth dict, ...]}.

    Each tooth dict: {"title", "quadrant", "tooth_position", "is_primary",
    "bbox_xywh", "bbox_yxyx"}. Degenerate boxes and (unless opted in)
    primary teeth are excluded from the returned index; counts of both
    are returned in the stats dict so exclusions are visible, not silent.
    """
    import json

    with open(teeth_bbox_path) as f:
        records = json.load(f)

    index: dict[int, list[dict]] = {}
    stats = {"n_teeth_total": 0, "n_degenerate_excluded": 0, "n_primary_excluded": 0, "n_unrecognized_title": 0}
    for rec in records:
        if not isinstance(rec, dict) or not rec.get("External ID"):
            continue
        image_id = _stem_id(rec["External ID"])
        teeth = []
        objects = (rec.get("Label") or {}).get("objects", []) if isinstance(rec.get("Label"), dict) else []
        for obj in objects:
            stats["n_teeth_total"] += 1
            bb_yxyx = obj.get("bounding box")
            if not isinstance(bb_yxyx, list) or len(bb_yxyx) != 4:
                continue
            if _is_degenerate_bbox(bb_yxyx):
                stats["n_degenerate_excluded"] += 1
                continue
            title_str = str(obj.get("title", ""))
            fdi = _title_to_fdi(title_str, include_primary_teeth=include_primary_teeth)
            if fdi is None:
                if title_str in _PRIMARY_LETTER_TO_FDI:
                    stats["n_primary_excluded"] += 1
                else:
                    stats["n_unrecognized_title"] += 1
                continue
            quadrant, tooth_position, is_primary = fdi
            teeth.append({
                "title": obj.get("title"),
                "quadrant": quadrant,
                "tooth_position": tooth_position,
                "is_primary": is_primary,
                "bbox_xywh": _bbox_yxyx_to_xywh(bb_yxyx),
                "bbox_yxyx": bb_yxyx,
            })
        index[image_id] = teeth
    return index, stats


def _map_tufts_pathology_to_dentex_category(title: str) -> int | None:
    """Map one expert.json/student.json abnormality `title` to a DENTEX
    category_id_3, or None if it has no DENTEX analog. See module
    docstring point 2 -- this is a verified-complete lookup over the real
    5-value taxonomy (None/Periapical/Non-Odontogenic/Pericoronal/
    Inter-Radicular), not a partial keyword guess."""
    return _TUFTS_TITLE_TO_DENTEX_CATEGORY_ID.get(title)


def _load_findings(annotation_path: Path) -> list[dict]:
    """Parse expert.json or student.json into a flat list of per-image
    dicts: {"image_id", "description", "findings": [{"title",
    "dentex_category_id", "ring_bboxes_xyxy", "level_1".."level_4"}]}.
    ring_bboxes_xyxy is a list of one bbox per polygon ring (NOT unioned
    -- see _ring_bboxes' docstring for why that matters for multi-lesion
    findings). Handles the real schema quirk (verified): Level 3/4 classifications
    use a singular "answer" dict in some records and a plural "answers"
    list in others -- both are normalized to a single value here (first
    entry, if a list) rather than assuming one shape.
    """
    import json

    with open(annotation_path) as f:
        records = json.load(f)

    out = []
    for rec in records:
        if not isinstance(rec, dict) or not rec.get("External ID"):
            continue
        image_id = _stem_id(rec["External ID"])
        findings = []
        objects = (rec.get("Label") or {}).get("objects", []) if isinstance(rec.get("Label"), dict) else []
        for obj in objects:
            title = obj.get("title")
            if title in (None, "None"):
                continue
            dentex_cat = _map_tufts_pathology_to_dentex_category(title)
            polygons = obj.get("polygons")
            if not isinstance(polygons, list) or not polygons:
                continue  # can't geometrically locate this finding at all
            ring_boxes = _ring_bboxes(polygons)
            if not ring_boxes:
                continue  # every ring was degenerate (<3 points)
            levels: dict[str, Any] = {"level_1": None, "level_2": None, "level_3": None, "level_4": None}
            classifications = obj.get("classifications")
            if isinstance(classifications, list):
                for c in classifications:
                    lvl_key = {
                        "Level one": "level_1", "Level two": "level_2",
                        "Level three": "level_3", "Level four": "level_4",
                    }.get(c.get("title"))
                    if lvl_key is None:
                        continue
                    if "answer" in c:
                        ans = c["answer"]
                        levels[lvl_key] = ans.get("value") if isinstance(ans, dict) else ans
                    elif "answers" in c and c["answers"]:
                        first = c["answers"][0]
                        levels[lvl_key] = first.get("value") if isinstance(first, dict) else first
            findings.append({
                "title": title,
                "dentex_category_id": dentex_cat,
                "ring_bboxes_xyxy": ring_boxes,
                **levels,
            })
        out.append({"image_id": image_id, "description": rec.get("Description"), "findings": findings})
    return out


def _build_images_df(radiograph_dir: Path, max_images: int | None) -> pd.DataFrame:
    """Scan Radiographs/ into the standard images_df shape (id, file_name,
    local_path, width, height, source_dataset). Shared by load_tufts_dataset
    and load_tufts_tooth_boxes so both return images_df built the exact
    same way."""
    seen = set()
    unique_files = []
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"):
        for f in radiograph_dir.glob(ext):
            canon = f.resolve()
            if canon not in seen:
                seen.add(canon)
                unique_files.append(f)
    image_files = sorted(unique_files, key=lambda p: _stem_id(p.name))
    if max_images:
        image_files = image_files[:max_images]
    rows = []
    for f in image_files:
        image_id = _stem_id(f.name)
        with Image.open(f) as im:
            width, height = im.size
        rows.append({
            "id": image_id, "file_name": f.name, "local_path": str(f),
            "width": width, "height": height, "source_dataset": "tufts",
        })
    return pd.DataFrame(rows)


def load_tufts_tooth_boxes(
    data_dir: str | None = None,
    include_primary_teeth: bool = False,
    max_images: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """All (not just abnormal) tooth boxes from teeth_bbox.json, flat --
    for expanding locate_tooth's YOLO grounding corpus (~26,000 verified
    tooth boxes across 1,000 images once degenerate boxes are excluded;
    fewer if include_primary_teeth stays False).

    Returns (images_df, tooth_boxes_df, categories_df) -- the same 3-tuple
    shape every other dataset loader in this codebase returns (matches
    prepare_yolo_dataset.py's DATASET_LOADERS calling convention, which
    calls every non-DENTEX loader as load(data_dir=data_dir) expecting
    exactly this shape back). categories_df is DENTEX_CATEGORIES for
    interface consistency only -- tooth_boxes_df has no category_id_3
    column, this dataset has no diagnosis attached to it by construction.
    tooth_boxes_df columns: image_id, category_id_1 (quadrant, 1-indexed
    FDI), category_id_2 (tooth_position, 1-indexed FDI), bbox ([x,y,w,h]),
    is_primary, source_dataset.

    Separate from load_tufts_dataset() on purpose: this covers every
    annotated tooth (for grounding), not just teeth tied to a diagnosis
    finding (for the reward/trace-gen pipeline) -- the two have different
    consumers and there's no reason to couple them.
    """
    tufts_root = find_local_tufts_dir()
    if tufts_root is None or _find_radiograph_dir(tufts_root) is None:
        repo_id = os.environ.get("TUFTS_IMAGES_REPO", "Reza-Nadimi/tufts-train-images")
        if repo_id:
            try:
                from huggingface_hub import snapshot_download
                target_dir = Path("data/Tufts")
                print(f"Tufts radiographs not found locally. Auto-downloading full dataset from {repo_id} to {target_dir}...")
                snapshot_download(
                    repo_id=repo_id,
                    repo_type="dataset",
                    local_dir=str(target_dir),
                    token=os.environ.get("HF_TOKEN"),
                )
                tufts_root = target_dir
            except Exception as e:
                print(f"Warning: Failed to auto-download Tufts dataset from {repo_id}: {e}")

    if tufts_root is None:
        raise FileNotFoundError(
            "No local Tufts Dental Database directory found. Tufts is access-gated "
            "(request it at https://tdd.ece.tufts.edu/) -- download and extract it "
            "yourself, then set TUFTS_LOCAL_DIR in .env to the extracted folder."
        )
    radiograph_dir = _find_radiograph_dir(tufts_root)
    if radiograph_dir is None:
        raise FileNotFoundError(
            f"Found {tufts_root} but couldn't locate a Radiographs folder with images inside it."
        )
    bbox_path = _find_annotation_file(tufts_root, "Segmentation", "teeth_bbox.json")
    if bbox_path is None:
        raise FileNotFoundError(f"Found {tufts_root} but no teeth_bbox.json under it (expected under Segmentation/).")

    images_df = _build_images_df(radiograph_dir, max_images)
    available_ids = set(images_df["id"]) if len(images_df) else set()

    tooth_index, stats = _load_tooth_index(bbox_path, include_primary_teeth=include_primary_teeth)
    rows = []
    for image_id, teeth in tooth_index.items():
        if available_ids and image_id not in available_ids:
            continue  # image not present in this local copy
        for t in teeth:
            rows.append({
                "image_id": image_id,
                "category_id_1": t["quadrant"],
                "category_id_2": t["tooth_position"],
                "bbox": t["bbox_xywh"],
                "is_primary": t["is_primary"],
                "source_dataset": "tufts",
            })
    print(
        f"load_tufts_tooth_boxes: {len(rows)} tooth boxes from {images_df['id'].nunique() if len(images_df) else 0} images "
        f"({stats['n_degenerate_excluded']} degenerate boxes excluded, "
        f"{stats['n_primary_excluded']} primary-tooth boxes excluded "
        f"[include_primary_teeth={include_primary_teeth}], "
        f"{stats['n_unrecognized_title']} unrecognized titles skipped)"
    )
    return images_df, pd.DataFrame(rows), pd.DataFrame(DENTEX_CATEGORIES)


def load_tufts_dataset(
    data_dir: str | None = None,
    max_images: int | None = None,
    include_primary_teeth: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Tufts into the same (images_df, annots_df, categories_df) shape
    dentex.py's load_dentex_dataset returns.

    annots_df is diagnosis-bearing only (mirrors DENTEX: teeth without an
    abnormality finding don't get a row here -- use load_tufts_tooth_boxes
    separately for the full per-tooth grounding corpus). Each row's bbox is
    the AFFECTED TOOTH's box (from teeth_bbox.json), matching DENTEX's
    convention where bbox identifies the tooth region a diagnosis applies
    to; the lesion's own polygon-derived box is preserved in
    extra["pathology_bbox_xyxy"] for tools like nudge_crop that want the
    finer target. category_id_1/2 are already 1-indexed FDI (see module
    docstring point 1 -- do NOT pass these through dentex_row_to_fdi).

    A multi-tooth lesion produces one row per overlapping tooth (see
    module docstring point 3); extra["n_teeth_matched"] lets a caller
    filter to unambiguous single-tooth findings if they want a stricter
    subset. Tooth assignment is bbox-overlap-only and under-recalls true
    periapical findings for a clinical-definition reason -- see module
    docstring point 4 before treating the printed no-overlap count as a
    reliability figure.
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
            f"Found {tufts_root} but couldn't locate a Radiographs folder with images inside it. "
            "The extracted folder structure may not match what this loader expects -- "
            "check the actual layout and adjust _find_radiograph_dir."
        )
    bbox_path = _find_annotation_file(tufts_root, "Segmentation", "teeth_bbox.json")
    if bbox_path is None:
        raise FileNotFoundError(f"Found {tufts_root} but no teeth_bbox.json under it (expected under Segmentation/).")
    expert_path = _find_annotation_file(tufts_root, "Expert", "expert.json")
    if expert_path is None:
        raise FileNotFoundError(f"Found {tufts_root} but no expert.json under it (expected under Expert/).")

    images_df = _build_images_df(radiograph_dir, max_images)
    available_ids = set(images_df["id"]) if len(images_df) else set()

    tooth_index, tooth_stats = _load_tooth_index(bbox_path, include_primary_teeth=include_primary_teeth)
    findings_by_image = _load_findings(expert_path)

    annot_rows = []
    n_findings_unmapped_category = 0
    n_findings_no_tooth_overlap = 0
    n_findings_mapped = 0
    for entry in findings_by_image:
        image_id = entry["image_id"]
        if available_ids and image_id not in available_ids:
            continue  # image not present in this local copy
        teeth = tooth_index.get(image_id, [])
        for finding in entry["findings"]:
            if finding["dentex_category_id"] is None:
                n_findings_unmapped_category += 1
                continue
            # Join per ring (not per whole finding-object -- see
            # _ring_bboxes' docstring), then dedupe to one row per unique
            # tooth: a multi-ring finding whose separate lesions each
            # land on a different tooth (the common case) or whose
            # several rings all describe one lesion near one tooth (also
            # common) both collapse correctly here, keeping whichever
            # ring had the larger overlap with that tooth as the
            # tooth-specific pathology box.
            best_per_tooth: dict[tuple[int, int], tuple[dict, float, tuple]] = {}
            for ring_bbox in finding["ring_bboxes_xyxy"]:
                for t in teeth:
                    ymin, xmin, ymax, xmax = t["bbox_yxyx"]
                    tooth_bbox_xyxy = (xmin, ymin, xmax, ymax)
                    ov = _bbox_overlap_area(ring_bbox, tooth_bbox_xyxy)
                    if ov <= 0:
                        continue
                    key = (t["quadrant"], t["tooth_position"])
                    if key not in best_per_tooth or ov > best_per_tooth[key][1]:
                        best_per_tooth[key] = (t, ov, ring_bbox)
            if not best_per_tooth:
                n_findings_no_tooth_overlap += 1
                continue
            n_findings_mapped += 1
            n_teeth_matched = len(best_per_tooth)
            for t, _ov, ring_bbox in best_per_tooth.values():
                annot_rows.append({
                    "image_id": image_id,
                    "category_id_1": t["quadrant"],
                    "category_id_2": t["tooth_position"],
                    "category_id_3": finding["dentex_category_id"],
                    "bbox": t["bbox_xywh"],
                    "source_dataset": "tufts",
                    "extra": {
                        "tufts_title": finding["title"],
                        "pathology_bbox_xyxy": list(ring_bbox),
                        "n_teeth_matched": n_teeth_matched,
                        "level_1": finding["level_1"], "level_2": finding["level_2"],
                        "level_3": finding["level_3"], "level_4": finding["level_4"],
                        "is_primary_tooth": t["is_primary"],
                    },
                })
    annots_df = pd.DataFrame(annot_rows)
    categories_df = pd.DataFrame(DENTEX_CATEGORIES)

    print(
        f"load_tufts_dataset: {len(images_df)} images, {len(annots_df)} diagnosis-bearing "
        f"tooth annotations from {n_findings_mapped} findings "
        f"({n_findings_unmapped_category} findings excluded: no DENTEX category analog "
        f"[Non-Odontogenic/Pericoronal/Inter-Radicular]; "
        f"{n_findings_no_tooth_overlap} excluded: zero tooth-box overlap; "
        f"{tooth_stats['n_degenerate_excluded']} degenerate tooth boxes excluded; "
        f"{tooth_stats['n_primary_excluded']} primary-tooth boxes excluded "
        f"[include_primary_teeth={include_primary_teeth}])"
    )
    return images_df, annots_df, categories_df


def load_tufts_segmentation(*_args, **_kwargs):
    """NOT IMPLEMENTED -- pixel/instance-level Tufts segmentation.

    What's already verified and could be wired up immediately, without
    guessing anything: expert.json's/student.json's abnormality
    `polygons` field (per-finding lesion outline, used above only for its
    bounding box) -- that data is real, present for every non-None
    finding, and would support pathology polygon/mask training right now.
    If you want that specifically, ask for it -- it doesn't need this
    function, _polygon_union_bbox's sibling would just keep the full
    polygon instead of collapsing to a bbox.

    What's genuinely unverified and blocks a general segmentation loader:
      1. Segmentation/teeth_polygon.json (271-285MB per the two inventory
         docs, not yet uploaded/inspected) -- presumably per-tooth polygon
         vertices mirroring teeth_bbox.json's per-tooth boxes, but that's
         an assumption from the file name, not confirmed against real
         content the way teeth_bbox.json's schema was.
      2. Segmentation/teeth_mask/ and Segmentation/maxillomandibular/ --
         PNG/JPG masks where each tooth instance presumably gets a
         distinct pixel value. What that pixel value actually encodes
         (tooth position 1-32 directly? an arbitrary per-image instance
         index with no fixed meaning across images? something else?) is
         exactly the kind of thing this module's docstring warns against
         guessing -- _bbox_from_mask above is ready to consume whichever
         it turns out to be, but the value needs verifying against a real
         mask file (or a manifest, if one ships with the archive) first.
      3. Expert/mask/ and Student/mask/ (binary pathology masks) -- lower
         priority than the two above since expert.json's polygons already
         give equivalent pathology-region geometry without needing these.

    Point me at teeth_polygon.json and one real mask file (or their
    pixel-value/manifest documentation) and this gets filled in the same
    way _load_tooth_index was: verified against real content, not this
    docstring's assumptions.
    """
    raise NotImplementedError(
        "Tufts segmentation loading is not implemented -- see this function's "
        "docstring for exactly what's already usable (expert.json's pathology "
        "polygons) vs. what needs real files to verify against first "
        "(teeth_polygon.json, the *_mask/ image folders)."
    )


def download_tufts_slice(
    image_ids: list[int],
    repo_id: str | None = None,
    cache_dir: str | None = None,
    token: str | None = None,
    split_name: str | None = None,
    **kwargs,
) -> dict[int, Path | None]:
    """Download only the given image_ids from Tufts' HF images repo (once
    uploaded by scripts/upload_dataset_images_to_hf.py --dataset tufts --
    see hf_dataset_utils.py). image_ids here are Tufts' own native numeric
    ids (e.g. 149, 702, 1051 -- NOT a resequenced 0..999 index; see module
    docstring re: the real id set having gaps and going above 1000),
    matching images_df["id"] from load_tufts_dataset.

    filename_template mirrors Radiographs/{id}.JPG -- the real local Tufts
    folder layout, not a flattened images/{id}.jpg bundle.
    Accepts split_name (ignored since Tufts images reside under Radiographs/
    for all splits) and **kwargs for full polymorphism with download_dentex_slice.
    """
    if repo_id is None:
        repo_id = os.environ.get("TUFTS_IMAGES_REPO", "Reza-Nadimi/tufts-train-images")
    if token is None:
        token = os.environ.get("HF_TOKEN")
    from dental_agent.data.hf_dataset_utils import download_dataset_slice
    return download_dataset_slice(image_ids, repo_id=repo_id, filename_template="Radiographs/{id}.JPG", cache_dir=cache_dir, token=token)


def load_tufts_normal_dataset(
    data_dir: str | None = None,
    max_images: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the 660 clinician-verified NORMAL images from Tufts (Expert records with title: 'None').

    Returns (images_df, annots_df, categories_df) where annots_df is an empty DataFrame (0 findings).
    """
    tufts_root = find_local_tufts_dir()
    if tufts_root is None:
        raise FileNotFoundError("No Tufts dataset directory found.")
    radiograph_dir = _find_radiograph_dir(tufts_root)
    if radiograph_dir is None:
        raise FileNotFoundError(f"Found {tufts_root} but couldn't locate Radiographs folder.")
    expert_path = _find_annotation_file(tufts_root, "Expert", "expert.json")
    if expert_path is None:
        raise FileNotFoundError(f"Found {tufts_root} but no expert.json under it.")

    images_df = _build_images_df(radiograph_dir, max_images)
    available_ids = set(images_df["id"]) if len(images_df) else set()

    findings_by_image = _load_findings(expert_path)
    normal_ids = set()
    for entry in findings_by_image:
        image_id = entry["image_id"]
        # Strictly normal: must have either zero findings or only explicit 'None' annotations.
        # Images with unmapped pathologies (Non-Odontogenic cysts, Pericoronal impactions, Inter-Radicular lesions)
        # must NEVER be classified as normal/healthy negative controls.
        f_list = entry.get("findings", [])
        if not f_list or all(f.get("title") in ("None", None) for f in f_list):
            normal_ids.add(image_id)

    if available_ids:
        normal_ids = normal_ids.intersection(available_ids)

    normal_images_df = images_df[images_df["id"].isin(normal_ids)].copy()
    empty_annots_df = pd.DataFrame(columns=["image_id", "category_id_1", "category_id_2", "category_id_3", "bbox", "source_dataset"])
    print(f"load_tufts_normal_dataset: {len(normal_images_df)} clinician-verified normal images loaded.")
    return normal_images_df, empty_annots_df, pd.DataFrame(DENTEX_CATEGORIES)
