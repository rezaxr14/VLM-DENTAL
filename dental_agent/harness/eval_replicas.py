"""Data-parallel evaluation: one independent single-GPU process per GPU, each holding a FULL model replica.

Why: a model that does not fit one GPU (Qwen3.5-9B in 16-bit on a 15 GiB T4) has to be split layer-wise over both T4s and every token then
crosses the PCIe link twice. Qwen3.5-4B fits one T4, so each GPU runs its own replica on its own half of the images; nothing is
communicated while the replicas run. The replicas are separate processes pinned with ``CUDA_VISIBLE_DEVICES`` (the loaders'
``device_map="auto"`` then puts the whole model on the one visible GPU), they write separate shard files, and this module merges the shard files into the one
result file the rest of the notebook runner already knows how to validate, summarise and upload.

Pure Python, no torch import.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

SHARDABLE_KINDS = ("zeroshot", "baseline", "sft", "grpo")
_PROGRESS = re.compile(r"^\[\d+/\d+\]")


def eval_replicas(profile_config: dict, devices: list[dict] | None, model_size: str) -> int:
    """Replicas (one per GPU) for an evaluation job: the profile's ``replicas`` key, only when the model is small enough to fit one GPU."""
    if model_size != "4b":
        return 1
    n = int(profile_config.get("replicas", 1))
    if devices is not None:
        n = min(n, len([d for d in devices if not d.get("tpu")]))
    return max(n, 1)


def _replace_value(command: list[str], flag: str, value: str) -> list[str]:
    out = list(command)
    i = out.index(flag)
    out[i + 1] = value
    return out


def replica_commands(command: list[str], kind: str, folder: Path, result: Path, n: int) -> list[dict[str, Any]]:
    """The ``n`` single-GPU commands that together do what ``command`` does, with their own output folders and GPU."""
    if kind not in SHARDABLE_KINDS:
        raise ValueError(f"{kind} evaluations cannot be sharded across replicas")
    jobs = []
    for i in range(n):
        shard_dir = Path(folder) / "_replicas" / f"gpu{i}"
        cmd = _replace_value(command, "--output-dir", str(shard_dir))
        if kind == "zeroshot":
            shard_result = shard_dir / "results.jsonl"
            cmd = _replace_value(cmd, "--output", str(shard_result)) + ["--total-slices", str(n), "--slice-index", str(i + 1)]
        else:
            shard_result = shard_dir / Path(result).name
            cmd += ["--shard-index", str(i), "--shard-count", str(n)]
        jobs.append(dict(command=cmd, result=shard_result, dir=shard_dir, env={"CUDA_VISIBLE_DEVICES": str(i)}, tag=f"gpu{i}"))
    return jobs


def _complete_lines(path: Path) -> list[bytes]:
    if not Path(path).is_file():
        return []
    return [l for l in Path(path).read_bytes().splitlines(keepends=True) if l.endswith(b"\n") and l.strip()]   # never a torn final record


def seed_replicas(result: Path, jobs: list[dict]) -> None:
    """A resume starts from the merged result file (it is what the Hub restores): give each replica the finished records it would
    otherwise redo. Records of images that belong to another replica are carried along harmlessly and come back in the merge."""
    lines = _complete_lines(result)
    if not lines:
        return
    for job in jobs:
        job["dir"].mkdir(parents=True, exist_ok=True)
        have = {int(json.loads(l)["image_id"]) for l in _complete_lines(job["result"])}
        extra = [l for l in lines if int(json.loads(l)["image_id"]) not in have]
        if extra:
            with open(job["result"], "ab") as f:
                f.write(b"".join(extra))


def merge_replicas(result: Path, shard_results: list[Path], order: list[int]) -> int:
    """Union of the merged file and every replica file, one record per image (the newest wins), in cohort order. Returns the count."""
    records: dict[int, bytes] = {}
    for path in [Path(result), *map(Path, shard_results)]:
        for line in _complete_lines(path):
            records[int(json.loads(line)["image_id"])] = line
    if not records:
        return 0                                    # nothing finished yet: do not create an empty result file
    position = {int(i): k for k, i in enumerate(order)}
    extra = [i for i in records if i not in position]
    if extra:
        raise ValueError(f"replica results contain images outside the cohort: {sorted(extra)[:5]}")
    ordered = sorted(records, key=position.__getitem__)
    tmp = Path(result).with_suffix(Path(result).suffix + ".merge")
    tmp.write_bytes(b"".join(records[i] for i in ordered))
    tmp.replace(result)
    return len(ordered)


def start_replicas(jobs: list[dict], base_env: dict, cwd: Path, *, stagger: bool = True, wait_seconds: float = 1800.0) -> list[subprocess.Popen]:
    """Launch the replicas. Replica k+1 starts once replica k prints its first image line (the weights are loaded and the host RAM is free again):
    two simultaneous 4B loads would double the peak host memory. The wait is announced and bounded; it never skips a replica."""
    procs: list[subprocess.Popen] = []

    def pump(proc, tag, ready):
        for raw in proc.stdout:
            line = raw.rstrip("\n")
            if _PROGRESS.match(line):
                ready.set()
            print(f"[{tag}] {line}", flush=True)
        ready.set()

    for i, job in enumerate(jobs):
        env = {**base_env, **job["env"]}
        proc = subprocess.Popen(job["command"], cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        ready = threading.Event()
        threading.Thread(target=pump, args=(proc, job["tag"], ready), daemon=True).start()
        procs.append(proc)
        if stagger and i + 1 < len(jobs):
            print(f"[REPLICAS] {job['tag']} started; starting the next replica after it has loaded its model (first image line, at most {wait_seconds / 60:.0f} min).", flush=True)
            if not ready.wait(wait_seconds):
                print(f"[REPLICAS] {job['tag']} printed no image line within {wait_seconds / 60:.0f} min; starting the next replica anyway.", flush=True)
    return procs
