#!/usr/bin/env python3
"""Convert the official DENTEX test labels (LabelMe polygons, DENTEX/test_data.zip) into ONE versioned COCO file.

    python scripts/convert_official_test_to_coco.py [--version v1] [--out data/official_test250]

Needs the mapping file dental_agent/data/label_maps/official_test250_<version>.json (every row with a status, role and evidence; a
'needs_confirmation' row stops the run). Checks before it writes: the zip hash; the official validation file's area convention (the shoelace
area of its segmentation against its 'area' field, printed; a mismatch stops everything); image sizes against the PNG headers (read in memory,
no image is extracted); FDI ranges; 250 images; unique ids; category ids inside their lists; a dry run through the repository's own DENTEX
loader. Writes coco/official_test250_<version>.json and label_maps/official_test250_<version>.json under --out, prints their SHA256, and the
id-to-file table. Nothing is uploaded here (see publish_official_test_coco.py)."""
from __future__ import annotations

import argparse
import collections
import json
import shutil
import sys
import zipfile
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.data import official_test250 as o  # noqa: E402


def read_labelme(zip_path: Path):
    labelme, sizes = {}, {}
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        for n in sorted(names):
            if n.startswith("disease/label/") and n.endswith(".json"):
                idx = int(Path(n).stem.split("_")[1])
                labelme[idx] = json.loads(z.read(n).decode("utf-8"))
            elif n.startswith("disease/input/") and n.lower().endswith(".png"):
                sizes[int(Path(n).stem.split("_")[1])] = o.png_size(z.open(n).read(24))
    if set(labelme) != set(sizes):
        raise SystemExit(f"labels and images do not pair up: {sorted(set(labelme) ^ set(sizes))[:10]}")
    return labelme, sizes


def dry_run_loader(coco: dict) -> dict:
    """Run the evaluation view through the repository's own DENTEX dataframe builder and ground-truth extraction (no images needed)."""
    from dental_agent.data.dentex import build_dataframes
    from dental_agent.evaluation.ground_truth import extract_ground_truth
    view = o.eval_view(coco)
    images_df, annots_df, cats_df = build_dataframes(view, data_dir=None, use_cache=False, split_name="test")
    with_gt = [int(i) for i in images_df["id"] if len(annots_df[annots_df["image_id"] == i])][:1]
    sample = extract_ground_truth(with_gt[0], annots_df, cats_df) if with_gt else []
    return dict(images=len(images_df), findings=len(annots_df), masked=len(view["masked_annotations"]), sample_ground_truth=sample[:2])


def build(version: str, zip_path: Path, validation_path: Path, out_dir: Path, mapping_path: Path | None = None) -> dict:
    zip_sha = o.sha256_file(zip_path)
    print(f"zip sha256 {zip_sha} -> {'MATCH' if zip_sha == o.OFFICIAL_ZIP_SHA256 else 'MISMATCH'}")
    if zip_sha != o.OFFICIAL_ZIP_SHA256:
        raise SystemExit("This is not the official test_data.zip you verified; stop.")
    validation = json.loads(Path(validation_path).read_text(encoding="utf-8"))
    conv = o.check_area_convention(validation)
    print(f"AREA CONVENTION vs official validation file: {conv['n']} polygons | exact {conv['exact']} | within {conv['tolerance']} px^2: {conv['within_tolerance']} | "
          f"max |shoelace - area| = {conv['max_abs_diff']}")
    if not conv["matches"]:
        raise SystemExit(f"The shoelace area does NOT reproduce the official 'area' field; worst rows (id, shoelace, official): {conv['worst']}. "
                         "Stop and report: the area convention must be understood before converting.")
    decimals = o.decimals_of(v for a in validation["annotations"] for seg in a["segmentation"] for v in seg)
    print(f"coordinate precision of the official segmentation: {decimals} decimal(s); the converted file uses the same")
    mapping_path = mapping_path or o.repo_mapping_path(version)
    mapping = o.load_mapping(mapping_path)
    o.validate_mapping(mapping, require_no_pending=True)
    mapping_sha = o.sha256_file(mapping_path)
    labelme, sizes = read_labelme(zip_path)
    coco = o.convert(labelme, mapping, validation, sizes, version=version, decimals=decimals, zip_sha256=zip_sha, mapping_sha256=mapping_sha)
    codes = collections.Counter(a["raw_code"] for a in coco["annotations"])
    for e in mapping["codes"]:
        if codes[e["code"]] != e["count"]:
            raise SystemExit(f"code {e['code']}: converted {codes[e['code']]} polygons but the mapping file says {e['count']}")
    print(f"images {len(coco['images'])} | polygons {len(coco['annotations'])} | findings {sum(not a['masked'] for a in coco['annotations'])} | "
          f"masked {coco['info']['n_masked']} | per code {dict(sorted(codes.items()))}")
    print("id-to-file table: id = N + 1 for test_N.png (first, last):", [(i["id"], i["file_name"]) for i in (coco["images"][0], coco["images"][-1])],
          "| every id equals N + 1:", all(i["id"] == int(i["file_name"][5:-4]) + 1 for i in coco["images"]))
    print("dry run through the repository loader:", json.dumps(dry_run_loader(coco), default=str)[:600])
    out_coco, out_map = out_dir / o.coco_name(version), out_dir / o.map_name(version)
    out_coco.parent.mkdir(parents=True, exist_ok=True)
    out_map.parent.mkdir(parents=True, exist_ok=True)
    out_coco.write_text(json.dumps(coco, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    shutil.copyfile(mapping_path, out_map)
    table = {str(i["id"]): i["file_name"] for i in coco["images"]}
    (out_dir / f"id_to_file_{version}.json").write_text(json.dumps(table, indent=1), encoding="utf-8")
    result = dict(coco=str(out_coco), coco_sha256=o.sha256_file(out_coco), mapping=str(out_map), mapping_sha256=o.sha256_file(out_map))
    print(json.dumps(result, indent=1))
    return result


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default=o.DEFAULT_VERSION)
    p.add_argument("--out", default="data/official_test250")
    p.add_argument("--zip", default=None, help="local test_data.zip (default: downloaded from the official repository)")
    p.add_argument("--validation-json", default=None, help="local validation_triple.json (default: downloaded)")
    p.add_argument("--mapping", default=None)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    from huggingface_hub import hf_hub_download
    zip_path = Path(a.zip) if a.zip else Path(hf_hub_download(repo_id=o.OFFICIAL_REPO, filename=o.OFFICIAL_ZIP, repo_type="dataset"))
    val_path = Path(a.validation_json) if a.validation_json else Path(hf_hub_download(repo_id=o.OFFICIAL_REPO, filename="DENTEX/validation_triple.json", repo_type="dataset"))
    build(a.version, zip_path, val_path, Path(a.out), Path(a.mapping) if a.mapping else None)


if __name__ == "__main__":
    main()
