#!/usr/bin/env python3
"""Stage the converted (and possibly hand-corrected) 250-image cohort as a Kaggle dataset folder. Uploads nothing.

    python scripts/stage_official_test250_dataset.py --version v1 --src data/official_test250 --out data/official_test250/kaggle_dataset \
        --dataset-id rezanadimikj/dentex-official-test250

Layout (the same relative names as the Hugging Face repository, so one resolver reads either):
    coco/official_test250_<version>.json         the COCO exactly as it is in --src (checked here, never re-converted: a corrected file goes up as it is)
    label_maps/official_test250_<version>.json   the mapping it was converted with (its SHA256 must be the one inside the COCO)
    disease/input/test_N.png                     the 250 images, read from DENTEX/test_data.zip (hash checked)
    dataset-metadata.json                        for `kaggle datasets create|version`

On Kaggle the dataset mounts at /kaggle/input/datasets/<owner>/<slug> or /kaggle/input/<slug>; the cohort resolver finds both."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.data import official_test250 as o  # noqa: E402


def stage(version: str, src: Path, out: Path, dataset_id: str, zip_path: Path) -> dict:
    coco_src, map_src = src / o.coco_name(version), src / o.map_name(version)
    for path in (coco_src, map_src, zip_path):
        if not Path(path).is_file():
            raise SystemExit(f"{path} is missing (convert first: scripts/convert_official_test_to_coco.py --version {version}).")
    coco = json.loads(coco_src.read_text(encoding="utf-8"))
    mapping = o.load_mapping(map_src)
    if coco["info"]["mapping_sha256"] != o.sha256_file(map_src):
        raise SystemExit("The mapping next to the COCO is not the one it was converted with.")
    problems = o.check_coco(coco, mapping)
    if problems:
        raise SystemExit("The COCO fails its checks, nothing staged: " + "; ".join(problems))
    zip_sha = o.sha256_file(zip_path)
    if zip_sha != o.OFFICIAL_ZIP_SHA256:
        raise SystemExit(f"{zip_path} is not the official test_data.zip (sha256 {zip_sha}).")
    if out.exists():
        shutil.rmtree(out)
    (out / "coco").mkdir(parents=True)
    (out / "label_maps").mkdir()
    (out / "disease" / "input").mkdir(parents=True)
    shutil.copyfile(coco_src, out / o.coco_name(version))
    shutil.copyfile(map_src, out / o.map_name(version))
    wanted = {i["file_name"] for i in coco["images"]}
    with zipfile.ZipFile(zip_path) as z:
        members = {Path(n).name: n for n in z.namelist() if n.startswith("disease/input/") and n.lower().endswith(".png")}
        missing = sorted(wanted - set(members))
        if missing:
            raise SystemExit(f"{len(missing)} image(s) of the COCO are not in the zip, e.g. {missing[:5]}")
        for name in sorted(wanted):
            data = z.read(members[name])
            (out / "disease" / "input" / name).write_bytes(data)
    meta = dict(title="DENTEX official test (250 images): COCO + label mapping", id=dataset_id, licenses=[{"name": "other"}],
                description=(f"Official DENTEX test split ({o.OFFICIAL_ZIP}, sha256 {zip_sha}) converted to one versioned COCO file ({version}) with the label "
                             "mapping that defines its classes, plus the 250 images. Source: https://huggingface.co/datasets/ibrahimhamamci/DENTEX. "
                             "See the VLM-DENTAL repository, docs/OFFICIAL_TEST250.md."))
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    record = {str(p.relative_to(out)): o.sha256_file(p) for p in (out / o.coco_name(version), out / o.map_name(version))}
    print(f"staged {out}: {len(wanted)} images, COCO + mapping, dataset-metadata.json (id {dataset_id})")
    print(json.dumps(record, indent=1))
    return record


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default=o.DEFAULT_VERSION)
    p.add_argument("--src", default="data/official_test250")
    p.add_argument("--out", default="data/official_test250/kaggle_dataset")
    p.add_argument("--dataset-id", required=True, help="<kaggle owner>/<slug>, e.g. rezanadimikj/dentex-official-test250")
    p.add_argument("--zip", default=None, help="local test_data.zip (default: the Hugging Face cache / a download from the official repository)")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if "/" not in a.dataset_id:
        raise SystemExit("--dataset-id must look like owner/slug")
    if a.zip:
        zip_path = Path(a.zip)
    else:
        from huggingface_hub import hf_hub_download
        zip_path = Path(hf_hub_download(repo_id=o.OFFICIAL_REPO, filename=o.OFFICIAL_ZIP, repo_type="dataset"))
    stage(a.version, Path(a.src), Path(a.out), a.dataset_id, zip_path)


if __name__ == "__main__":
    main()
