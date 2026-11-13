"""Official DENTEX test split (250 images): LabelMe -> versioned COCO, the label mapping, and the evaluation cohort.

The official test labels (``DENTEX/test_data.zip``: ``disease/label/test_N.json``) are LabelMe polygons whose label is
``<code>-<Turkish word>-<FDI>`` (the FDI tooth number is the LAST token). They are converted once into a COCO file with the same
top-level structure as the official ``validation_triple.json`` and published (``coco/official_test250_<version>.json`` and
``label_maps/official_test250_<version>.json`` in ``DENTEX_TEST_REPO``), so nobody converts them again.

The mapping file (``dental_agent/data/label_maps/official_test250_<version>.json``) states, for every raw code, a status
(translate | needs_confirmation | masked), the evidence for it, and a scoring ROLE:

* ``finding``  : becomes a ground-truth annotation. Its class is the English name of the Turkish word: the DENTEX name where
                 DENTEX has one (Caries, Impacted, Periapical Lesion), otherwise a class appended to ``categories_3``
                 (Root Canal, Curettage, Extraction, Fracture);
* ``negative`` : the healthy-tooth polygon (sağlam): not a finding, kept for audit; a disease predicted on that tooth is a false positive.

Nothing is dropped from scoring. Nothing here imports torch.
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
import re
import struct
import zipfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SPLIT = "official_test250"
EXPECTED_IMAGES = 250
DEFAULT_VERSION = "v1"
TEST_REPO_ENV, TEST_REPO_DEFAULT = "DENTEX_TEST_REPO", "Reza-Nadimi/dentex-official-test"
OFFICIAL_REPO = "ibrahimhamamci/DENTEX"
OFFICIAL_ZIP = "DENTEX/test_data.zip"
OFFICIAL_ZIP_SHA256 = "ed396d6daa133264ddb0ecdbed3f963868d2a31344ba0fab929a6a1773d82ce2"
MAPPING_DIR = Path(__file__).resolve().parent / "label_maps"
STATUSES = ("translate", "needs_confirmation", "masked")
ROLES = ("finding", "negative")
# Classes named in the prompts of this cohort: the four DENTEX diagnoses (spelled as the repository prompts spell them) and the English names
# of the other Turkish label words (kanal = root canal, küretaj = curettage, çekim = extraction, kırık = fracture).
PROMPT_CLASSES = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth", "Root Canal", "Curettage", "Extraction", "Fracture"]
_DENTEX_FOUR = ["Caries", "Deep Caries", "Periapical Lesion", "Impacted Tooth"]
LABEL_RE = re.compile(r"^(\d+)-([^-]+)-(\d+)$")
CATEGORY_KEYS = ("categories_1", "categories_2", "categories_3")


def coco_name(version: str = DEFAULT_VERSION) -> str:
    return f"coco/official_test250_{version}.json"


def map_name(version: str = DEFAULT_VERSION) -> str:
    return f"label_maps/official_test250_{version}.json"


def derive_prompt(base: str) -> str:
    """The cohort version of a repository prompt: the same text with the DENTEX class list replaced by ``PROMPT_CLASSES``. Raises if the
    base prompt no longer has the list in a form this knows, so a prompt edit can never silently leave the cohort on the old list."""
    union_old, union_new = "|".join(_DENTEX_FOUR), "|".join(PROMPT_CLASSES)
    quoted_old, quoted_new = ", ".join(f'"{c}"' for c in _DENTEX_FOUR), ", ".join(f'"{c}"' for c in PROMPT_CLASSES)
    plain_old, plain_new = "(" + ", ".join(_DENTEX_FOUR) + ")", "(" + ", ".join(PROMPT_CLASSES) + ")"
    out, hits = base, 0
    for old, new in ((union_old, union_new), (quoted_old, quoted_new), (plain_old, plain_new)):
        hits += out.count(old)
        out = out.replace(old, new)
    # after taking every new list out, no old list may be left (the new lists often start with the old text, so count what remains)
    pairs = ((union_old, union_new), (quoted_old, quoted_new), (plain_old, plain_new))
    if any(old in out.replace(new, "") for old, new in pairs) or hits < 2:
        raise ValueError("The base prompt's class list changed shape; update official_test250.derive_prompt")
    return out


def zero_shot_prompt() -> str:
    from dental_agent.agent.prompts import ZERO_SHOT_PROMPT
    return derive_prompt(ZERO_SHOT_PROMPT)


def no_tools_prompt() -> str:
    from dental_agent.agent.prompts import NO_TOOLS_SYSTEM_PROMPT
    return derive_prompt(NO_TOOLS_SYSTEM_PROMPT)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------------------ labels and geometry
def parse_label(label: str) -> tuple[int, str, int]:
    """``'1-çürük-26'`` -> (1, 'çürük', 26). The FDI number is the last token."""
    m = LABEL_RE.match(str(label))
    if not m:
        raise ValueError(f"label {label!r} is not of the form <code>-<word>-<FDI>")
    return int(m.group(1)), m.group(2), int(m.group(3))


def fdi_to_indices(fdi: int) -> tuple[int, int]:
    """FDI 26 -> (quadrant index 1, tooth index 5): the official categories_1 / categories_2 ids are 0-indexed (quadrant - 1, tooth - 1).
    Quadrants 1..4 and teeth 1..8 only; primary-tooth quadrants (5-8) and anything else are rejected."""
    quadrant, tooth = divmod(int(fdi), 10)
    if not (1 <= quadrant <= 4 and 1 <= tooth <= 8):
        raise ValueError(f"FDI {fdi} is outside quadrant 1..4 / tooth 1..8")
    return quadrant - 1, tooth - 1


def polygon_bbox(points: Sequence[Sequence[float]]) -> list[float]:
    xs, ys = [p[0] for p in points], [p[1] for p in points]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def polygon_area(points: Sequence[Sequence[float]]) -> float:
    """Shoelace formula (absolute value, halved)."""
    n = len(points)
    s = sum(points[i][0] * points[(i + 1) % n][1] - points[(i + 1) % n][0] * points[i][1] for i in range(n))
    return abs(s) / 2.0


def flatten(points: Sequence[Sequence[float]], decimals: int) -> list[float | int]:
    out: list[float | int] = []
    for x, y in points:
        for v in (x, y):
            r = round(float(v), decimals)
            out.append(int(r) if decimals == 0 else r)
    return out


def unflatten(flat: Sequence[float]) -> list[list[float]]:
    return [[flat[i], flat[i + 1]] for i in range(0, len(flat), 2)]


def decimals_of(values: Iterable[float], cap: int = 6) -> int:
    """Largest number of decimals found in ``values`` (0 for integers). Used to round like the official validation file."""
    best = 0
    for v in values:
        if isinstance(v, float) and v != int(v):
            best = max(best, min(cap, len(repr(v).split(".")[1])))
    return best


def check_area_convention(validation: Mapping[str, Any], tolerance: float = 1.0) -> dict[str, Any]:
    """Compare the shoelace area of each official validation segmentation with its ``area`` field. Returns the report; the caller stops on a mismatch."""
    rows = []
    for a in validation["annotations"]:
        for seg in a.get("segmentation", []):
            rows.append((a["id"], polygon_area(unflatten(seg)), float(a["area"])))
    diffs = [abs(s - f) for _, s, f in rows]
    return dict(n=len(rows), max_abs_diff=max(diffs) if diffs else None, exact=sum(d < 1e-9 for d in diffs),
                within_tolerance=sum(d <= tolerance for d in diffs), tolerance=tolerance,
                worst=sorted(((i, s, f) for i, s, f in rows), key=lambda r: -abs(r[1] - r[2]))[:5],
                matches=bool(rows) and all(d <= tolerance for d in diffs))


def png_size(data: bytes) -> tuple[int, int] | None:
    return struct.unpack(">II", data[16:24]) if data[:8] == b"\x89PNG\r\n\x1a\n" and len(data) >= 24 else None


# --------------------------------------------------------------------------------------------------------------------- mapping file
def load_mapping(path: str | Path) -> dict[str, Any]:
    mapping = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_mapping(mapping)
    return mapping


def validate_mapping(mapping: Mapping[str, Any], *, require_no_pending: bool = False) -> None:
    """Every code needs a status, a role and evidence; 'translate' also a DENTEX target. ``needs_confirmation`` blocks when asked to."""
    seen = set()
    extra_ids = [c["id"] for c in mapping["extra_categories_3"]]
    if len(set(extra_ids)) != len(extra_ids) or set(extra_ids) & {c["id"] for c in mapping["dentex_categories_3"]}:
        raise ValueError("extra_categories_3 ids must be unique and must not reuse a DENTEX id")
    for e in mapping["codes"]:
        code = e["code"]
        if code in seen:
            raise ValueError(f"code {code} listed twice")
        seen.add(code)
        if e["status"] not in STATUSES or e["role"] not in ROLES:
            raise ValueError(f"code {code}: status {e['status']!r} / role {e['role']!r} not allowed")
        if not e.get("evidence"):
            raise ValueError(f"code {code}: a mapping row without evidence is refused")
        if e["status"] == "translate":
            known = {c["id"]: c["name"] for c in mapping["dentex_categories_3"]} | {c["id"]: c["name"] for c in mapping["extra_categories_3"]}
            if e["role"] != "finding" or e.get("target_source") not in ("dentex", "cohort") or known.get(e.get("target_id")) != e.get("target_name"):
                raise ValueError(f"code {code}: 'translate' needs role finding and a target class that exists (id {e.get('target_id')}, name {e.get('target_name')!r})")
            if (e["target_source"] == "dentex") != (e["target_id"] in {c["id"] for c in mapping["dentex_categories_3"]}):
                raise ValueError(f"code {code}: target_source does not match the id")
        elif e["role"] == "finding" or e.get("target_id") is not None:
            raise ValueError(f"code {code}: only 'translate' rows may become findings or carry a target")
        if require_no_pending and e["status"] == "needs_confirmation":
            raise ValueError(f"code {code} still needs confirmation: the run is blocked")


# ------------------------------------------------------------------------------------------------------------------------- converter
def convert(labelme: Mapping[int, Mapping[str, Any]], mapping: Mapping[str, Any], validation: Mapping[str, Any],
            png_sizes: Mapping[int, tuple[int, int] | None], *, version: str, decimals: int, zip_sha256: str, mapping_sha256: str) -> dict[str, Any]:
    """LabelMe records (keyed by N of ``test_N``) -> COCO with the official validation structure. Image id = N + 1.

    Every polygon becomes an annotation (nothing is dropped); polygons whose code is not a finding carry ``masked: true``,
    ``category_id_3: None`` and their role."""
    validate_mapping(mapping, require_no_pending=True)
    by_code = {e["code"]: e for e in mapping["codes"]}
    images, annotations = [], []
    ann_id = 1
    for n in sorted(labelme):
        rec = labelme[n]
        width, height = int(rec["imageWidth"]), int(rec["imageHeight"])
        if png_sizes.get(n) != (width, height):
            raise ValueError(f"test_{n}: label says {width}x{height}, the PNG header says {png_sizes.get(n)}")
        images.append(dict(id=n + 1, file_name=f"test_{n}.png", height=height, width=width))
        for shape in rec["shapes"]:
            if shape.get("shape_type") != "polygon":
                raise ValueError(f"test_{n}: shape_type {shape.get('shape_type')!r} is not a polygon")
            code, word, fdi = parse_label(shape["label"])
            if code not in by_code:
                raise ValueError(f"test_{n}: code {code} ({shape['label']}) is not in the mapping file")
            try:
                q_idx, t_idx = fdi_to_indices(fdi)
            except ValueError as error:
                raise ValueError(f"test_{n}: {error} (label {shape['label']!r})") from error
            entry = by_code[code]
            flat = flatten(shape["points"], decimals)
            pts = unflatten(flat)
            bbox = [round(v, decimals) if decimals else int(round(v)) for v in polygon_bbox(pts)]
            area = round(polygon_area(pts), decimals) if decimals else int(round(polygon_area(pts)))
            is_finding = entry["role"] == "finding"
            annotations.append(dict(
                id=ann_id, image_id=n + 1, iscrowd=0, bbox=bbox, area=area, segmentation=[flat],
                category_id_1=q_idx, category_id_2=t_idx, category_id_3=int(entry["target_id"]) if is_finding else None,
                raw_code=code, raw_label=shape["label"], role=entry["role"], masked=not is_finding, polygon_points=shape["points"]))
            ann_id += 1
    coco = dict(
        info=dict(coco_version=version, mapping_version=mapping["version"], mapping_sha256=mapping_sha256, source_repo=OFFICIAL_REPO,
                  source_zip=OFFICIAL_ZIP, source_zip_sha256=zip_sha256, coordinate_decimals=decimals, n_images=len(images),
                  n_polygons=len(annotations), n_masked=sum(a["masked"] for a in annotations),
                  note="Image id = N + 1 for test_N.png. category_id_1/2 are 0-indexed (FDI quadrant - 1, FDI tooth - 1). Masked annotations "
                       "keep their polygon and carry category_id_3 = null; their role says how scoring treats the tooth."),
        images=images, annotations=annotations, **{k: copy.deepcopy(validation[k]) for k in CATEGORY_KEYS})
    official = {c["id"]: c["name"] for c in validation["categories_3"]}
    if official != {c["id"]: c["name"] for c in mapping["dentex_categories_3"]}:
        raise ValueError("The mapping's dentex_categories_3 differ from the validation file's categories_3")
    coco["categories_3"] = coco["categories_3"] + copy.deepcopy(mapping["extra_categories_3"])     # the four official classes stay first and identical
    problems = check_coco(coco, mapping)
    if problems:
        raise ValueError("converted COCO failed its checks: " + "; ".join(problems))
    return coco


def check_coco(coco: Mapping[str, Any], mapping: Mapping[str, Any]) -> list[str]:
    problems = []
    ids = [i["id"] for i in coco["images"]]
    if len(ids) != EXPECTED_IMAGES or len(set(ids)) != len(ids):
        problems.append(f"expected {EXPECTED_IMAGES} unique images, found {len(ids)} ({len(set(ids))} unique)")
    ann_ids = [a["id"] for a in coco["annotations"]]
    if len(set(ann_ids)) != len(ann_ids):
        problems.append("annotation ids are not unique")
    known = set(ids)
    valid = {k: {c["id"] for c in coco[k]} for k in CATEGORY_KEYS}
    findings = 0
    for a in coco["annotations"]:
        if a["image_id"] not in known:
            problems.append(f"annotation {a['id']} refers to missing image {a['image_id']}")
        if a["category_id_1"] not in valid["categories_1"] or a["category_id_2"] not in valid["categories_2"]:
            problems.append(f"annotation {a['id']}: quadrant/tooth id outside its category list")
        if a["masked"]:
            if a["category_id_3"] is not None or a["role"] == "finding":
                problems.append(f"annotation {a['id']}: masked annotation with a diagnosis or the finding role")
        else:
            findings += 1
            if a["category_id_3"] not in valid["categories_3"]:
                problems.append(f"annotation {a['id']}: category_id_3 {a['category_id_3']} outside categories_3")
    expected_findings = sum(1 for a in coco["annotations"] if next(e for e in mapping["codes"] if e["code"] == a["raw_code"])["role"] == "finding")
    if findings != expected_findings:
        problems.append(f"{findings} non-masked annotations but the mapping implies {expected_findings}")
    return problems[:20]


def eval_view(coco: Mapping[str, Any]) -> dict[str, Any]:
    """The evaluation-ready COCO: ``annotations`` holds the findings only (what ``load_dentex_dataset`` and ``extract_ground_truth`` read);
    the healthy-tooth polygons (role negative) move to ``masked_annotations``, so no existing consumer meets a null diagnosis."""
    out = copy.deepcopy(dict(coco))
    findings, others = [], []
    for a in out["annotations"]:
        a.pop("polygon_points", None)
        (others if a["masked"] else findings).append(a)
    out["annotations"] = findings
    out["masked_annotations"] = [dict(image_id=a["image_id"], category_id_1=a["category_id_1"], category_id_2=a["category_id_2"],
                                      raw_code=a["raw_code"], role=a["role"], bbox=a["bbox"]) for a in others]
    return out


# ----------------------------------------------------------------------------------------------------------------------- the cohort
def repo_type_of(api, repo: str) -> str:
    """'dataset' or 'model': whichever accessible repository exists under this id (never created by code)."""
    from huggingface_hub.errors import RepositoryNotFoundError
    found = []
    for kind in ("dataset", "model"):
        try:
            api.repo_info(repo, repo_type=kind)
        except RepositoryNotFoundError:
            continue
        found.append(kind)
    if not found:
        raise FileNotFoundError(f"{repo} not found as a dataset or model repository (or the token cannot see it)")
    return found[0]


SOURCE_ORDER = {"local": 0, "kaggle": 1, "huggingface": 2}


def origin(path: str | Path) -> str:
    """Where a file lives: ``local`` disk, an attached ``kaggle`` input (``/kaggle/input/<slug>`` or ``/kaggle/input/datasets/<owner>/<slug>``),
    or the ``huggingface`` cache. Files are READ in that order; a Hugging Face download is the last resort."""
    p = str(path).replace("\\", "/")
    if p.startswith("/kaggle/input"):
        return "kaggle"
    if "/huggingface/" in p or "/datasets--" in p or "/snapshots/" in p:
        return "huggingface"
    return "local"


def by_source(paths: Iterable[Path]) -> list[Path]:
    return sorted(paths, key=lambda p: (SOURCE_ORDER[origin(p)], str(p)))


def repo_mapping_path(version: str) -> Path:
    return MAPPING_DIR / f"official_test250_{version}.json"


def prepare_cohort(data_dir, api, download, token, files, *, image_ok, digest, fingerprint, write, version: str | None = None) -> dict[str, Any]:
    """Resolve the 250 images and the COCO before any model loads, like ``prepare_inputs`` does for the 50-image split. The helpers
    (``image_ok``, ``digest``, ``fingerprint``, ``write``) come from notebook_eval so both paths share one definition.

    Source of the COCO: ``official_test250_<version>.json`` from local disk, else from an attached Kaggle dataset, else from the Hugging Face
    cache, else downloaded from ``DENTEX_TEST_REPO``. It is never converted here. Images (``test_N.png``): local disk, attached Kaggle dataset,
    Hugging Face cache, a local official zip, then the Hub. When copies of one file differ between sources the first source in that order is
    used and the disagreement is printed. Exactly 250 images are required; otherwise FileNotFoundError lists the missing ids and nothing is launched."""
    version = version or os.environ.get("DENTEX_TEST_COCO_VERSION", DEFAULT_VERSION)
    data = Path(data_dir).expanduser().resolve()
    repo = os.environ.get(TEST_REPO_ENV, TEST_REPO_DEFAULT)
    name = f"official_test250_{version}.json"
    local = [p for p in files if p.name == name and p.suffix == ".json"]
    revision, rtype = None, None
    if local:
        local = by_source(local)
        coco_path = local[0]
        shas = {sha256_file(p) for p in local}
        if len(shas) > 1:
            print(f"[COHORT WARNING] {len(shas)} different {name} files found; using the {origin(coco_path)} one ({coco_path}); others: "
                  f"{[(origin(p), str(p)) for p in local[1:]]}")
    else:
        rtype = repo_type_of(api, repo)
        revision = api.repo_info(repo, repo_type=rtype).sha
        coco_path = Path(download(repo, coco_name(version), repo_type=rtype, revision=revision, token=token))
    full = json.loads(coco_path.read_text(encoding="utf-8"))
    info = full.get("info", {})
    if info.get("coco_version") != version or len(full["images"]) != EXPECTED_IMAGES:
        raise ValueError(f"{coco_path}: not the {version} COCO of {EXPECTED_IMAGES} images (info={info})")
    mapping_path = repo_mapping_path(version)
    mapping_sha = sha256_file(mapping_path)
    if info.get("mapping_sha256") != mapping_sha:
        raise ValueError(f"The COCO was converted with mapping sha256 {info.get('mapping_sha256')} but {mapping_path} is {mapping_sha}. The mapping "
                         "changed after conversion: publish a new version instead of editing a published one.")
    mapping = load_mapping(mapping_path)
    validate_mapping(mapping, require_no_pending=True)
    coco = eval_view(full)
    expected = {int(i["id"]) for i in coco["images"]}
    paths: dict[int, Path] = {}
    image_files = [p for p in files if p.suffix.lower() == ".png"]
    disagreements = 0
    for item in coco["images"]:
        iid, native = int(item["id"]), Path(str(item["file_name"])).name
        hits = by_source(p for p in image_files if p.name == native and image_ok(p, item))
        if hits:
            paths[iid] = hits[0].resolve()
            disagreements += len({digest(p) for p in hits}) > 1
    if disagreements:
        print(f"[COHORT WARNING] {disagreements} image(s) exist in several places with different content; the first of local, Kaggle, Hugging Face cache was used for each.")
    for archive in [p for p in files if p.name == "test_data.zip"]:
        if set(paths) == expected:
            break
        with zipfile.ZipFile(archive) as z:
            members = {n: n for n in z.namelist()}
            for item in coco["images"]:
                iid = int(item["id"])
                member = f"disease/input/{item['file_name']}"
                if iid in paths or member not in members:
                    continue
                out = data / "evaluation_inputs" / "extracted" / SPLIT / f"{iid}.png"
                out.parent.mkdir(parents=True, exist_ok=True)
                if not image_ok(out, item):
                    out.write_bytes(z.read(member))
                if image_ok(out, item):
                    paths[iid] = out.resolve()
    missing = expected - set(paths)
    if missing:
        rtype = rtype or repo_type_of(api, repo)
        revision = revision or api.repo_info(repo, repo_type=rtype).sha
        remote = set(api.list_repo_files(repo, repo_type=rtype, revision=revision))
        for item in coco["images"]:
            iid = int(item["id"])
            member = f"disease/input/{item['file_name']}"
            if iid in missing and member in remote:
                p = Path(download(repo, member, repo_type=rtype, revision=revision, token=token))
                if image_ok(p, item):
                    paths[iid] = p.resolve()
    if set(paths) != expected or len(expected) != EXPECTED_IMAGES:
        raise FileNotFoundError(f"Cohort incomplete ({len(paths)}/{EXPECTED_IMAGES}); no evaluation launched. Missing IDs: {sorted(expected - set(paths))}")
    hashes = {str(i): digest(paths[i]) for i in sorted(paths)}
    identity = dict(coco_version=version, coco_sha256=sha256_file(coco_path), mapping_version=mapping["version"], mapping_sha256=mapping_sha,
                    source_zip_sha256=info.get("source_zip_sha256"), n_masked=len(coco["masked_annotations"]))
    portable = copy.deepcopy(coco)
    for item in portable["images"]:
        item.pop("file_name", None)
    key = fingerprint(dict(coco=portable, images=hashes, split=SPLIT, identity=identity))
    pathkey = fingerprint({str(i): str(p) for i, p in paths.items()})[:12]
    out = data / "evaluation_inputs" / SPLIT / key[:16] / pathkey
    (out / "validation_data").mkdir(parents=True, exist_ok=True)
    for item in coco["images"]:
        item["file_name"] = str(paths[int(item["id"])])
    write(out / "validation_triple.json", coco)
    sources = dict(coco=origin(coco_path) if local else "huggingface", images=dict(collections.Counter(origin(p) for p in paths.values())))
    print(f"[COHORT] official_test250 {version}: COCO from {sources['coco']}; images by source {sources['images']} (read order: local disk, Kaggle, Hugging Face)")
    return dict(data_dir=str(out), ids=sorted(expected), hashes=hashes, fingerprint=key, split=SPLIT, identity=identity, sources=sources)


def load_cohort_split(args):
    """``load_eval_split`` for ``--split official_test250``: the loader that reads the 50-image folder layout, plus the masks and the
    prompt/scoring settings put on ``args`` for the evaluator, the zero-shot runner and the result records."""
    from dental_agent.data.dentex import load_dentex_dataset
    imgs_df, annots_df, cats_df = load_dentex_dataset(data_dir=args.data_dir, split_name="test")
    if len(imgs_df) != EXPECTED_IMAGES:
        raise SystemExit(f"--split {SPLIT} needs the {EXPECTED_IMAGES}-image cohort folder written by the notebook; {args.data_dir} has {len(imgs_df)} images.")
    coco = json.loads((Path(args.data_dir) / "validation_triple.json").read_text(encoding="utf-8"))
    info = coco.get("info", {})
    if info.get("coco_version") is None or "masked_annotations" not in coco:
        raise SystemExit(f"{args.data_dir}/validation_triple.json is not an {SPLIT} cohort file.")
    args.prompt_dataset = SPLIT               # selects the cohort prompt only; args.dataset still names the result paths, which must not change
    args.test250_settings = dict(coco_version=info["coco_version"], mapping_version=info["mapping_version"], mapping_sha256=info["mapping_sha256"],
                                 source_zip_sha256=info.get("source_zip_sha256"), scoring="test250_scoring_v2", prompt_classes=list(PROMPT_CLASSES))
    return imgs_df, annots_df, cats_df
