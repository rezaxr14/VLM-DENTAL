#!/usr/bin/env python3
"""Copy notebook evaluation results into separate flat folders that the aggregation scripts read (no model needed).

``NotebookEvaluations`` stores every cohort in its own run folder::

    data/evaluations/dentex/<split>/<tag>/<stage>/<kind>/<condition>[/harness_<mode>][/k<K>]/<size>/run_<id>/

``scripts/aggregate_eval.py`` and ``scripts/aggregate_k_sweep.py`` read one flat folder (``eval_*.jsonl`` directly inside). A GRPO policy
trained with the rules/skills harness has the same file name as the one trained without it, so the two can never share a folder. This
script copies (never moves, never edits) the files into separate areas under ``--out`` (default ``data/evaluations_flat``)::

    test/                          baseline + SFT + GRPO (harness off) at the chosen --k, test split
    dev/                           the same on the dev split, plus the tool/grounding probes (ablation suffix kept in the file name)
    k_sweep/k<K>/                  the unablated dev grpo_with_tools result of every K (harness off)
    <area>__harness_<mode>/        one folder per harness mode that has a trained GRPO cohort: that GRPO + the baseline and SFT
                                   controls (harness independent; copied so the folder is self-contained for aggregate_eval.py)
    rules_skills/<cohort>/         results.json, summary.json, identity.json, runtime.json of every rules/skills cohort (dev)
    speculative/<cohort>/          the same for every speculative-decoding cohort (dev)

Each area gets a ``copied_files.json`` (source path, sha256 and rows of every copied file, tag, stage, K). Files an earlier run copied
into an area are removed first, so a changed K or a finished cohort never leaves a stale file behind; nothing else in ``--out`` is touched.
Skipped on purpose: other profile tags and stages, and GRPO / rules-skills / speculative cohorts of a K other than ``--k`` (outside the K
sweep). Without ``--k`` no per-K file is copied into the test, dev, rules_skills or speculative areas (K sweep only). Two different files that
map to one flat name stop the script before anything is copied: resolve that by hand, never by guessing.

    python scripts/flatten_results.py --k 4
    python scripts/aggregate_eval.py --output-dir data/evaluations_flat/test --dataset dentex --split test
    python scripts/aggregate_k_sweep.py --sweep-dir data/evaluations_flat/k_sweep --reference-dir data/evaluations_flat/dev
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

MANIFEST = "copied_files.json"  # list of the files copied into an area: source path, sha256, rows, tag, stage, K
RAW_KINDS = ("rules_skills", "speculative")
RAW_FILES = ("results.json", "summary.json", "identity.json", "runtime.json")
MATRIX_KINDS = ("baseline", "sft", "grpo")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def n_rows(path: Path) -> int:
    if path.suffix != ".jsonl":
        return 1
    with path.open("rb") as stream:
        return sum(1 for line in stream if line.strip())


def area_name(base: str, mode: str) -> str:
    return base if mode == "off" else f"{base}__harness_{mode}"


def parse(src: Path, f: Path) -> dict | None:
    """Split a result path into its parts, or None when it is not a result file of the expected shape."""
    parts = f.relative_to(src).parts  # split, tag, stage, kind, condition, [harness_x], [kN], size, run_x, file
    if len(parts) < 8:
        return None
    middle = parts[5:-3]
    harness = [m[len("harness_"):] for m in middle if m.startswith("harness_")]
    kdirs = [m for m in middle if re.fullmatch(r"k\d+", m)]
    return dict(split=parts[0], tag=parts[1], stage=parts[2], kind=parts[3], condition=parts[4], mode=harness[0] if harness else "off",
                k=int(kdirs[0][1:]) if kdirs else None, size=parts[-3], name=parts[-1])


def plan_copies(src: Path, out: Path, tag: str, stage: str, k: int | None) -> tuple[dict[Path, Path], list[tuple[Path, Path, Path]]]:
    """``({destination: source}, conflicts)`` for every result file under ``src`` that belongs in a flat area."""
    plan: dict[Path, Path] = {}
    conflicts: list[tuple[Path, Path, Path]] = []

    def put(dest: Path, source: Path) -> None:
        if dest in plan and plan[dest] != source:
            conflicts.append((dest, plan[dest], source))
        plan[dest] = source

    controls: list[tuple[str, Path, str]] = []  # (split, file, name) of harness-independent baseline/SFT cells
    grpo_areas: set[str] = set()  # matrix areas that received a GRPO cohort (the harness ones also need the controls)
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        p = parse(src, f)
        if p is None or p["tag"] != tag or p["stage"] != stage or p["split"] not in ("test", "dev"):
            continue
        kind, split, mode, this_k = p["kind"], p["split"], p["mode"], p["k"]
        if kind in MATRIX_KINDS and p["name"].startswith("eval_dentex_") and p["name"].endswith(".jsonl"):
            if kind == "grpo" and this_k is not None:
                if split == "dev" and p["condition"] == "grpo_with_tools" and "__" not in p["name"]:
                    put(out / area_name("k_sweep", mode) / f"k{this_k}" / p["name"], f)
                if k is None or this_k != k:
                    continue
                grpo_areas.add(area_name(split, mode))
                put(out / area_name(split, mode) / p["name"], f)
            elif kind == "grpo":
                continue
            else:
                controls.append((split, f, p["name"]))
        elif kind in RAW_KINDS and p["name"] in RAW_FILES:
            if this_k is not None and (k is None or this_k != k):
                continue
            cohort = "__".join(x for x in (p["condition"], f"harness_{mode}" if mode != "off" else "", f"k{this_k}" if this_k is not None else "", p["size"]) if x)
            put(out / area_name(kind, "off") / cohort / p["name"], f)
    for split, f, name in controls:  # baseline and SFT do not depend on the harness: the plain area and every harness area get them
        for area in sorted({split} | {a for a in grpo_areas if a.split("__")[0] == split}):
            put(out / area / name, f)
    return plan, conflicts


def remove_previous(out: Path) -> None:
    """Delete only what an earlier run recorded in the area manifests."""
    for manifest in sorted(out.glob(f"*/{MANIFEST}")):
        area_dir = manifest.parent
        for rel in json.loads(manifest.read_text(encoding="utf-8")).get("files", {}):
            target = (area_dir / rel).resolve()
            if area_dir.resolve() in target.parents and target.is_file():
                target.unlink()
                if not any(target.parent.iterdir()) and target.parent != area_dir.resolve():
                    target.parent.rmdir()
        manifest.unlink()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--k", type=int, default=None, help="the chosen K, fixed on dev before any test result is read (omit for the K sweep only)")
    ap.add_argument("--src", default="data/evaluations/dentex")
    ap.add_argument("--out", default="data/evaluations_flat")
    ap.add_argument("--tag", default="workstation48_lora_bf16", help="profile tag folder (workstation48_lora_bf16 for the 48 GB GPU)")
    ap.add_argument("--stage", default="dentex_alone")
    args = ap.parse_args(argv)

    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"{src} not found: run from the repo root after at least one evaluation has finished")
    plan, conflicts = plan_copies(src, out, args.tag, args.stage, args.k)
    if conflicts:
        for dest, a, b in conflicts:
            print(f"CONFLICT {dest}\n   {a}\n   {b}")
        raise SystemExit("stopped before copying: the same flat name comes from two run folders")
    if not plan:
        raise SystemExit(f"no result files for tag {args.tag!r} and stage {args.stage!r} under {src}")

    remove_previous(out)
    files: dict[str, dict[str, dict]] = {}
    for dest in sorted(plan):
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan[dest], dest)
        area = dest.relative_to(out).parts[0]
        files.setdefault(area, {})[dest.relative_to(out / area).as_posix()] = dict(source=plan[dest].as_posix(), sha256=sha256(dest), rows=n_rows(dest))
    for area, entries in sorted(files.items()):
        doc = dict(area=area, tag=args.tag, stage=args.stage, k=args.k, files=entries)
        (out / area / MANIFEST).write_text(json.dumps(doc, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        print(f"[FLATTEN] {area}: {len(entries)} file(s) -> {out / area}")
        for rel in entries:
            print(f"    {rel}")


if __name__ == "__main__":
    main(sys.argv[1:])
