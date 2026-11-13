#!/usr/bin/env python3
"""Publish the converted COCO and its mapping ONCE to the private DENTEX_TEST_REPO (default Reza-Nadimi/dentex-official-test).

    python scripts/publish_official_test_coco.py --version v1 [--dir data/official_test250]

Uploads exactly two files in one commit: coco/official_test250_<version>.json and label_maps/official_test250_<version>.json. It never touches
disease/label/ or disease/input/, never creates the repository, refuses a repository that is not private, and REFUSES to overwrite
unless you pass --overwrite (the notebook's commented one-time cell does, after you corrected the dataset): without it, an existing version stops
the run and tells you to publish the next version. Records the SHA256 of each uploaded file."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.data import official_test250 as o  # noqa: E402


def next_version(version: str) -> str:
    return f"v{int(version[1:]) + 1}" if version[1:].isdigit() else version + "-2"


def publish(api, repo: str, version: str, folder: Path, overwrite: bool = False) -> dict:
    local = {o.coco_name(version): folder / o.coco_name(version), o.map_name(version): folder / o.map_name(version)}
    for name, path in local.items():
        if not path.is_file():
            raise SystemExit(f"{path} is missing: run scripts/convert_official_test_to_coco.py --version {version} first.")
    coco = json.loads(local[o.coco_name(version)].read_text(encoding="utf-8"))
    if coco["info"]["mapping_sha256"] != o.sha256_file(local[o.map_name(version)]):
        raise SystemExit("The mapping file next to the COCO is not the one the COCO was converted with; convert again.")
    rtype = o.repo_type_of(api, repo)
    info = api.repo_info(repo, repo_type=rtype)
    if not getattr(info, "private", False):
        raise SystemExit(f"{repo} is not private; refusing to publish.")
    existing = set(api.list_repo_files(repo, repo_type=rtype))
    clash = [n for n in local if n in existing]
    if clash and overwrite:
        print(f"[OVERWRITE] replacing {clash} in {repo} (explicit --overwrite)")
    elif clash:
        nxt = next_version(version)
        raise SystemExit(f"{clash} already exist in {repo}; they are never overwritten without --overwrite. Create {nxt}: "
                         f"python scripts/convert_official_test_to_coco.py --version {nxt} && python scripts/publish_official_test_coco.py --version {nxt} "
                         f"(after adding dental_agent/data/label_maps/official_test250_{nxt}.json).")
    from huggingface_hub import CommitOperationAdd
    ops = [CommitOperationAdd(path_in_repo=n, path_or_fileobj=str(p)) for n, p in local.items()]
    api.create_commit(repo_id=repo, repo_type=rtype, operations=ops, commit_message=f"official_test250 {version}: COCO + label mapping" + (" (overwrite)" if clash and overwrite else ""))
    record = {n: o.sha256_file(p) for n, p in local.items()}
    (folder / f"publish_record_{version}.json").write_text(json.dumps(dict(repo=repo, repo_type=rtype, sha256=record), indent=1), encoding="utf-8")
    files = sorted(api.list_repo_files(repo, repo_type=rtype))
    print(f"uploaded to {repo} ({rtype}, private):")
    for n, sha in record.items():
        print(f"  {n}  sha256 {sha}")
    print(f"repo file list now ({len(files)} files): first {files[:3]} ... last {files[-3:]}; disease/label files untouched: "
          f"{sum(f.startswith('disease/label/') for f in files)}, disease/input files: {sum(f.startswith('disease/input/') for f in files)}")
    return dict(record=record, files=files)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default=o.DEFAULT_VERSION)
    p.add_argument("--dir", default="data/official_test250")
    p.add_argument("--overwrite", action="store_true", help="replace an existing version of the same two files (never touches disease/)")
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    from huggingface_hub import HfApi
    publish(HfApi(token=os.environ.get("HF_TOKEN")), os.environ.get(o.TEST_REPO_ENV, o.TEST_REPO_DEFAULT), a.version, Path(a.dir), a.overwrite)


if __name__ == "__main__":
    main()
