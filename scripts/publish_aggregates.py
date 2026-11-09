#!/usr/bin/env python3
"""Publish the aggregation tables of each flat area to the existing results repo, as separate files in separate folders.

Reads what ``scripts/flatten_results.py`` and the aggregators wrote under ``--flat-dir`` (default ``data/evaluations_flat``). Every folder
with a ``copied_files.json`` is an area::

    test, dev, <area>__harness_<mode>   copied_files.json + matrix_dentex_<split>/*      (scripts/aggregate_eval.py)
    k_sweep, k_sweep__harness_<mode>    copied_files.json + ksweep_dentex_dev_*/*        (scripts/aggregate_k_sweep.py, plot_k_sweep.py)
    rules_skills, speculative           copied_files.json + <cohort>/results.json, summary.json, ... (copied by flatten_results.py)

and uploads every file of an area, each as its own file, under::

    aggregates/dentex/<area>/<tag>/<stage>/<k<K> | sweep>/run_<id>/<relative path>

``run_<id>`` is a hash of the area, tag, stage, K and the sha256 of every file, so uploading the same files again changes nothing and a
changed table goes into a new folder: nothing already on the Hub is overwritten, and the per-cohort bundles under ``evaluations/`` (written
by ``NotebookEvaluations``) are never touched. A local ``published.json`` records the Hub paths.

    python scripts/publish_aggregates.py --areas test dev test__harness_rules_skills_index rules_skills --k 4 --tag workstation48_lora_bf16 --dry-run
    python scripts/publish_aggregates.py --areas test dev test__harness_rules_skills_index rules_skills --k 4 --tag workstation48_lora_bf16
    python scripts/publish_aggregates.py --areas k_sweep k_sweep__harness_rules_skills_index --tag workstation48_lora_bf16   # no --k

Without ``--areas`` every flattened area is published, so each of them must have been aggregated first. Re-drawn figures embed a creation time, so
publishing them again goes to a new ``run_<id>`` folder; the earlier one stays.

Needs ``HF_TOKEN`` with write access; the results repo must already exist (it is never created).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.harness.notebook_eval import fingerprint, result_repo_type  # noqa: E402

MANIFEST = "copied_files.json"
OUTPUT_FOLDERS = ("matrix_", "ksweep_")  # what aggregate_eval.py / aggregate_k_sweep.py create inside an area
RAW_AREAS = ("rules_skills", "speculative")  # copied result documents; no aggregator runs on them
SKIP = ("published.json",)


def find_areas(flat: Path) -> list[str]:
    return sorted(p.parent.name for p in flat.glob(f"*/{MANIFEST}"))


def is_sweep(area: str) -> bool:
    return area.split("__")[0] == "k_sweep"


def collect(flat: Path, area: str) -> dict[str, Path]:
    """``{relative path: file}`` of one area: its manifest and everything an aggregator wrote below it (raw areas: every copied file)."""
    area_dir = flat / area
    if not (area_dir / MANIFEST).is_file():
        raise SystemExit(f"{area_dir / MANIFEST} not found: run scripts/flatten_results.py first")
    files = {MANIFEST: area_dir / MANIFEST}
    if area.split("__")[0] in RAW_AREAS:
        for f in sorted(area_dir.rglob("*")):
            if f.is_file() and f.name not in (MANIFEST,) + SKIP:
                files[f.relative_to(area_dir).as_posix()] = f
        return files
    roots = sorted(p for p in area_dir.iterdir() if p.is_dir() and p.name.startswith(OUTPUT_FOLDERS))
    if not roots:
        raise SystemExit(f"no aggregation output in {area_dir}: run scripts/aggregate_eval.py (test/dev areas) or scripts/aggregate_k_sweep.py "
                         "(k_sweep areas) on it first")
    for root in roots:
        for f in sorted(root.rglob("*")):
            if f.is_file():
                files[f.relative_to(area_dir).as_posix()] = f
    return files


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def hub_prefix(area: str, tag: str, stage: str, k: int | None, digests: dict[str, str]) -> str:
    group = "sweep" if is_sweep(area) else f"k{k}" if k is not None else "k-unset"
    run = fingerprint(dict(area=area, tag=tag, stage=stage, k=k, files=digests))[:16]
    return f"aggregates/dentex/{area}/{tag}/{stage}/{group}/run_{run}"


def plan_area(flat: Path, area: str, tag: str, stage: str, k: int | None) -> tuple[str, dict[str, Path]]:
    files = collect(flat, area)
    manifest = json.loads(files[MANIFEST].read_text(encoding="utf-8"))
    if manifest.get("tag") != tag or manifest.get("stage") != stage:
        raise SystemExit(f"{area}: flattened for tag/stage {manifest.get('tag')}/{manifest.get('stage')}, not {tag}/{stage}")
    if not is_sweep(area) and manifest.get("k") != k:
        raise SystemExit(f"{area}: flattened with --k {manifest.get('k')}, publishing with --k {k}: rerun flatten and the aggregation with the same K")
    digests = {rel: sha256(path.read_bytes()) for rel, path in files.items()}
    return hub_prefix(area, tag, stage, k, digests), files


def publish(flat: Path, areas: list[str], tag: str, stage: str, k: int | None, repo: str, repo_type: str | None, api=None,
            dry_run: bool = False) -> list[dict]:
    plans = {area: plan_area(flat, area, tag, stage, k) for area in areas}
    results = []
    if dry_run:
        for area, (prefix, files) in plans.items():
            print(f"[DRY RUN] {area}: {len(files)} file(s) -> {repo}:{prefix}/")
            for rel in files:
                print(f"    {rel}")
            results.append(dict(area=area, hub_path=prefix, files=sorted(files), status="dry_run"))
        return results
    from huggingface_hub import CommitOperationAdd, HfApi
    api = api or HfApi(token=os.environ.get("HF_TOKEN"))
    repo_type = result_repo_type(api, repo, repo_type)
    for area, (prefix, files) in plans.items():
        for attempt in range(3):
            info = api.repo_info(repo, repo_type=repo_type)
            existing = set(api.list_repo_files(repo, repo_type=repo_type, revision=info.sha))
            todo = {rel: path for rel, path in files.items() if f"{prefix}/{rel}" not in existing}
            if not todo:
                status = "unchanged"
                break
            try:
                api.create_commit(repo, repo_type=repo_type, parent_commit=info.sha,
                                  operations=[CommitOperationAdd(path_in_repo=f"{prefix}/{rel}", path_or_fileobj=path.read_bytes())
                                              for rel, path in todo.items()],
                                  commit_message=f"aggregates: {area} {tag} {stage} ({len(todo)} files)")
                status = "uploaded"
                break
            except Exception as error:  # retry only a confirmed rejected parent commit, never an upload with unknown outcome
                if getattr(getattr(error, "response", None), "status_code", None) == 409 and attempt < 2:
                    continue
                raise
        print(f"[HUB {status.upper()}] {area}: {repo}:{prefix}/ ({len(files)} file(s))", flush=True)
        record = dict(area=area, hub_path=prefix, results_repo=repo, results_repo_type=repo_type, files=sorted(files), status=status)
        (flat / area / "published.json").write_text(json.dumps(record, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        results.append(record)
    return results


def main(argv: list[str] | None = None, api=None) -> list[dict]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--flat-dir", default="data/evaluations_flat")
    ap.add_argument("--areas", nargs="+", default=None, help="area folder names under --flat-dir (default: every area that has a flatten manifest)")
    ap.add_argument("--tag", required=True, help="profile tag, e.g. workstation48_lora_bf16")
    ap.add_argument("--stage", default="dentex_alone")
    ap.add_argument("--k", type=int, default=None, help="the chosen K (required for test and dev; the K sweep has none)")
    ap.add_argument("--repo", default=os.environ.get("HF_EVAL_REPO", "Reza-Nadimi/vlm-dental-evaluations"))
    ap.add_argument("--repo-type", choices=["model", "dataset"], default=None, help="default: detect the existing repository")
    ap.add_argument("--dry-run", action="store_true", help="list what would be uploaded; no network, no token")
    args = ap.parse_args(argv)
    flat = Path(args.flat_dir)
    areas = args.areas or find_areas(flat)
    if not areas:
        raise SystemExit(f"no flattened area under {flat}: run scripts/flatten_results.py first")
    if args.k is None and any(not is_sweep(a) for a in areas):
        raise SystemExit("--k is required for every area except the K sweep ones (k_sweep, k_sweep__harness_*)")
    return publish(flat, areas, args.tag, args.stage, args.k, args.repo, args.repo_type, api=api, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
