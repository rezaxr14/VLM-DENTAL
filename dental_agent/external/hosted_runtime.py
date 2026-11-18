"""What runs INSIDE a hosted Kaggle task: one shard of one condition for the one model the task was scheduled for.

The pushed task source (``sharding.render_task_source``) clones this repository, installs what the shard needs and calls
``run_shard``. The container has no access to this machine and no annotations, so it only produces TRANSPORT records (answers and
usage); scoring happens later, locally. Everything it learned about itself goes to ``shard_manifest.json`` (written even on
failure) so the driver can say exactly what went wrong without anyone reading logs.

Outputs are FLAT files in the working directory (``transport.jsonl``, ``usage.jsonl``, ``journal.jsonl``, ``meta.json``,
``shard_manifest.json``): that is what Kaggle returns for a run. When the attached ``.env`` dataset supplies ``HF_TOKEN`` and the
spec names a results repository, the finished records are also pushed there after every chunk of images, so a killed or capped
run loses at most one chunk and the work survives the end of this session.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

from dental_agent.config import apply_env_values, env_file_in, parse_env_file
from dental_agent.external import preflight as PF
from dental_agent.external.conditions import get_condition
from dental_agent.external.coordinator import run_cell
from dental_agent.external.identity import fingerprint, run_settings
from dental_agent.external.importer import append_jsonl, cell_paths
from dental_agent.external.kaggle_provider import OpenAICompatBackend
from dental_agent.external.model_profiles import HOSTED_DECODING
from dental_agent.external.persist import ExternalHub, result_rows
from dental_agent.external.plugins import load_plugins
from dental_agent.external.sharding import ShardSpec

KAGGLE_INPUT = "/kaggle/input"
FLAT = {"transport": "transport.jsonl", "usage": "usage.jsonl", "journal": "journal.jsonl", "meta": "meta.json"}


class ShardFailed(RuntimeError):
    def __init__(self, stage: str, message: str):
        super().__init__(f"{stage}: {message}")
        self.stage = stage


def dataset_dir(dataset_id: str, roots=(KAGGLE_INPUT,)) -> Path | None:
    """Where an attached ``owner/name`` dataset is mounted: ``<root>/<name>`` (hosted) or ``<root>/datasets/<owner>/<name>``."""
    owner, _, name = dataset_id.partition("/")
    for root in roots:
        for cand in (Path(root) / name, Path(root) / "datasets" / owner / name):
            if cand.is_dir():
                return cand
    return None


def load_env_dataset(dataset_id: str | None, roots=(KAGGLE_INPUT,), env=os.environ) -> list[str]:
    """The explicit form of ``config.load_env_from_datasets``: the spec NAMES the env dataset, so a missing one fails the shard with a
    clear stage instead of silently running without tokens. Returns the key NAMES loaded (values are never printed or stored)."""
    if not dataset_id:
        return []
    folder = dataset_dir(dataset_id, roots)
    if folder is None:
        raise ShardFailed("env", f"dataset {dataset_id} is not attached to this run")
    path = env_file_in(folder)
    if path is None:
        raise ShardFailed("env", f"no .env (or env / dot_env / env.txt) file in {dataset_id}")
    return apply_env_values(parse_env_file(path), env)


def images_table(spec: ShardSpec, roots=(KAGGLE_INPUT,)):
    import pandas as pd
    folders = [dataset_dir(d, roots) for d in spec.image_datasets]
    folders = [f for f in folders if f is not None]
    if not folders:
        raise ShardFailed("images", f"none of the image datasets {spec.image_datasets} is attached")
    rows = []
    for sid in spec.image_ids:
        rel = spec.rel_paths[str(sid)]
        path = next((f / rel for f in folders if (f / rel).is_file()), folders[0] / rel)
        rows.append({"id": int(sid), "local_path": str(path)})
    return pd.DataFrame(rows)


def make_backend(llm_model: str, env=os.environ) -> OpenAICompatBackend:
    if not llm_model:
        raise ShardFailed("model", "the task did not receive a model slug")
    return OpenAICompatBackend(llm_model, base_url=env.get("MODEL_PROXY_URL"), api_key=env.get("MODEL_PROXY_API_KEY"), adaptive=True)


def preflight(spec: ShardSpec, images_df, env=os.environ) -> dict[str, Any]:
    out: dict[str, Any] = {"prompts": PF.check_prompts_render(spec.split, spec.dataset), "images": PF.check_images(images_df, spec.image_ids)}
    if spec.needs_grounder:
        try:
            from dental_agent.utils.inputs import resolve_yolo_checkpoint
            found = resolve_yolo_checkpoint(hub="download")
            env["GROUNDING_MODEL_PATH"] = str(found.path)
        except Exception as exc:
            out["grounder"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
        else:
            out["grounder"] = PF.check_grounder()
    try:
        PF.require(out)
    except PF.PreflightFailed as exc:
        raise ShardFailed("preflight", str(exc)[:600]) from exc
    return out


def _chunks(ids: list[int], n: int):
    for i in range(0, len(ids), n):
        yield ids[i:i + n]


def run_shard(spec_in: dict[str, Any] | ShardSpec, llm_model: str, *, bootstrap_info: dict[str, Any] | None = None,
              working: str | Path = "/kaggle/working", roots=(KAGGLE_INPUT,), env=os.environ, backend=None,
              hub: ExternalHub | None | str = "auto", run_image: Callable[..., Any] | None = None,
              chunk_images: int | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    spec = spec_in if isinstance(spec_in, ShardSpec) else ShardSpec(**spec_in)
    working = Path(working)
    working.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"task_name": spec.task_name, "condition": spec.condition, "model_raw_slug": llm_model,
                                "image_ids": list(spec.image_ids), "cohort_sha256": spec.cohort_sha256, "split": spec.split,
                                "started": time.time(), "python": sys.version.split()[0], "bootstrap": bootstrap_info or {},
                                "status": "running", "spec_sha": spec.sha()}
    out_dir = Path("/tmp/vlm_shard_out")
    shutil.rmtree(out_dir, ignore_errors=True)
    try:
        manifest["env_keys_loaded"] = load_env_dataset(spec.env_dataset, roots, env)
        try:
            manifest["plugins"] = load_plugins(spec.plugins)
        except Exception as exc:
            raise ShardFailed("plugins", f"{type(exc).__name__}: {exc}"[:400]) from exc
        images = images_table(spec, roots)
        manifest["preflight"] = preflight(spec, images, env)
        gsha = (manifest["preflight"].get("grounder") or {}).get("sha256")
        settings = run_settings(get_condition(spec.condition), llm_model, cohort=spec.cohort_sha256, split=spec.split, dataset=spec.dataset,
                                grounder_sha256=gsha, overrides=spec.overrides, decoding=HOSTED_DECODING,
                                cohort_settings=spec.cohort_settings)
        manifest["grounder_sha256"] = gsha
        manifest["settings_fingerprint"] = fingerprint(settings)
        paths = cell_paths(out_dir, llm_model, spec.condition)

        if hub == "auto":
            hub = ExternalHub.from_env(spec.hf_results_repo, env, spec.hf_repo_type) if spec.hf_results_repo else None
        manifest["hub"] = "off" if hub is None else "on"
        if hub is not None:                                    # resume from what an earlier run of this shard already pushed
            try:
                done_rows = hub.restore(working / "_hub_restore", settings, spec.task_name)
                for r in done_rows:
                    if int(r["image_id"]) in set(spec.image_ids):
                        append_jsonl(paths["transport"], r)
                manifest["restored_from_hub"] = len(done_rows)
            except Exception as exc:
                manifest["hub"] = f"restore failed: {type(exc).__name__}: {str(exc)[:200]}"
                hub = None

        be = backend or make_backend(llm_model, env)
        n_chunk = int(chunk_images or max(4, spec.workers * 2))
        total = {"ran": 0, "errors": 0, "replayed_turns": 0, "stopped": None, "paused": None}
        kw = {"run_image": run_image} if run_image is not None else {}
        for chunk in _chunks(list(spec.image_ids), n_chunk):
            stats = run_cell(backend=be, condition=spec.condition, images_df=images, image_ids=chunk, out_dir=out_dir, settings=settings,
                             overrides=spec.overrides, workers=spec.workers, log=log, **kw)
            for k in ("ran", "errors", "replayed_turns"):
                total[k] += stats[k]
            total["stopped"], total["paused"] = stats["stopped"], stats["paused"]
            if hub is not None:
                try:
                    rows = result_rows(paths["transport"])
                    hub.save(working / "_hub_save", settings, rows, complete=len(rows) == len(spec.image_ids), shard=spec.task_name)
                except Exception as exc:                        # the outputs are still returned by Kaggle; do not fail the shard
                    manifest["hub_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            if stats["stopped"]:
                break
        done = {int(r["image_id"]) for r in result_rows(paths["transport"])}
        missing = sorted(set(spec.image_ids) - done)
        manifest.update(stats=total, completed_images=len(done), missing_images=missing, adaptations=list(getattr(be, "adaptations", [])))
        manifest["status"] = "complete" if not missing else ("paused" if total["paused"] else "incomplete")
    except ShardFailed as exc:
        manifest.update(status="failed", stage=exc.stage, error=str(exc)[:800])
    except Exception as exc:                                    # nothing here may stop the manifest from being written
        manifest.update(status="failed", stage="runtime", error=f"{type(exc).__name__}: {exc}"[:800], traceback=traceback.format_exc()[-1500:])
    finally:
        manifest["finished"] = time.time()
        try:
            paths = cell_paths(out_dir, llm_model, spec.condition)
            for key, flat in FLAT.items():
                if paths[key].exists():
                    shutil.copyfile(paths[key], working / flat)
        except Exception:
            pass
        (working / "shard_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        log("SHARD_MANIFEST:" + json.dumps({k: manifest.get(k) for k in ("task_name", "status", "stage", "error", "completed_images")}))
    return manifest
