"""TPU inference bootstrap shared by every evaluator (evaluate_models, rules/skills dev runs, zero-shot).

The evaluators already reach text generation through ``local_runtime.generate_local``. On a TPU that call is served by the static
shape generator of ``tpu_generation`` once the model carries its config. This module does the three things the evaluators need
around it, in one place so they cannot drift apart:

1. ``enable_spmd`` (before the XLA device is touched), 2. ``finalize_tpu_inference_model`` (shard the merged 9B weights over the
mesh, install the generator, warm every reachable tier) and 3. refusal handling: an episode whose prompt does not fit the static
cache or the vision slots is recorded as a refused episode (no answer, flagged), never truncated and never a crash.

Nothing here runs on GPU/CPU: every function is reached only when ``tpu_generation_chunk_tokens > 0``.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import os
from pathlib import Path
from typing import Any

from dental_agent.model.tpu_generation import (
    GenerationTooLong,
    SlotOverflow,
    TPUGenerationConfig,
    find_attr,
    install_tpu_generation,
    note_dropped_rollouts,
    validate_tpu_generation_args,
    warmup_tpu_generation,
    ATTR,
)

GENERATION_REFUSED = (GenerationTooLong, SlotOverflow)
REPORT_NAME = "tpu_generation_report.json"


def tpu_generation_requested(options: Any) -> bool:
    return int(getattr(options, "tpu_generation_chunk_tokens", 0) or 0) > 0


def tpu_runtime_present() -> bool:
    """True when this process can see a TPU, WITHOUT initialising the XLA runtime (a notebook parent must leave the TPU free
    for the evaluation child process)."""
    if importlib.util.find_spec("torch_xla") is None:
        return False
    if any(os.environ.get(k) for k in ("TPU_NAME", "TPU_WORKER_ID", "COLAB_TPU_ADDR", "TPU_ACCELERATOR_TYPE", "CLOUD_TPU_TASK_ID")):
        return True
    if os.environ.get("PJRT_DEVICE", "").upper() == "TPU":
        return True
    return bool(glob.glob("/dev/accel*") or glob.glob("/dev/vfio/[0-9]*"))


def require_static_views(*, canonical_resize: bool, letterbox: bool, what: str) -> None:
    """Static slots exist only for the fixed canonical canvases, so TPU evaluation always uses canonical letterbox views."""
    if not (canonical_resize and letterbox):
        raise ValueError(
            f"{what}: TPU evaluation needs --canonical-resize --canonical-letterbox (the static vision slots exist only for the "
            "fixed canonical canvases). Original-size evaluation is a GPU-profile cohort; TPU results are canonical-view results.")


def enable_spmd() -> None:
    """Must run before any tensor touches the XLA device. Idempotent."""
    import torch_xla.runtime as xr
    xr.use_spmd()
    os.environ["XLA_USE_SPMD"] = "1"


def finalize_tpu_inference_model(model: Any, processor: Any, options: Any, *, max_new_tokens: int,
                                 repetition_penalty: float | None = None, num_cores: int = 8) -> Any:
    """Shard an already merged (or adapter-free) model over the mesh and install static generation. Call once per process.

    ``model`` is on the host (that is how ``load_model`` returns it on TPU). The weights are never placed on a single chip:
    FSDPv2 shards them while moving them. Returns the wrapped model; generation reaches the config through its wrappers."""
    from dental_agent.training.sft import setup_spmd_mesh, wrap_spmd_model

    chunk = int(options.tpu_generation_chunk_tokens)
    tiers = tuple(int(t) for t in options.tpu_generation_tiers)
    validate_tpu_generation_args(chunk_tokens=chunk, tiers=tiers, is_tpu=True, spmd=True, fsdp=True,
                                 pad_vision_to_slots=True, canonical_resize=True)
    if not any(t >= chunk + int(max_new_tokens) for t in tiers):
        raise ValueError(f"max_new_tokens={max_new_tokens} does not fit any --tpu-generation-tiers {list(tiers)} "
                         f"(a tier must hold one chunk of prompt plus the full generation budget). Budgets are never lowered: "
                         "add a larger tier.")
    from dental_agent.utils.canonical import parse_slot_budget
    slots = parse_slot_budget(*options.tpu_vision_slots)
    mesh = setup_spmd_mesh(num_cores=num_cores)
    model.eval()
    model.requires_grad_(False)          # inference: no trainable-leaf FSDP wrapping, no autograd bookkeeping
    install_tpu_generation(model, TPUGenerationConfig(chunk_tokens=chunk, cache_tiers=tiers, slot_budget=slots,
                                                      batch_multiple=num_cores, mesh=mesh))
    model = wrap_spmd_model(model, mesh=mesh, is_master=True)
    model.eval()
    if options.tpu_generation_warmup:
        report = warmup_tpu_generation(model, processor, batches=(1,), max_new_tokens=int(max_new_tokens),
                                       repetition_penalty=repetition_penalty)
        print(f"[TPU EVAL] generation warm-up: {report}", flush=True)
    return model


def refused_trajectory(image_id: int, reason: str):
    """A trajectory with no answer, for an episode whose generation was refused. It scores as a failed episode (no findings,
    format not ok) and is flagged by the caller (``generation_refused``) so it is counted, never hidden."""
    from dental_agent.agent.loop import AgentTrajectory
    return AgentTrajectory(image_id=int(image_id), turns=[], tool_calls=0, final_answer=None, format_ok=False, messages=[])


def note_refusal(model: Any, image_id: int, exc: BaseException) -> str:
    """Count and log a refused episode; returns the reason stored in its record."""
    reason = f"{type(exc).__name__}: {exc}"
    config = find_attr(model, ATTR)
    if config is not None:
        note_dropped_rollouts(model, 1)
        config.stats.setdefault("refused_images", []).append({"image_id": int(image_id), "reason": reason[:400]})
    print(f"[TPU EVAL] image {image_id} REFUSED by static generation (recorded as no answer, flagged): {reason}", flush=True)
    return reason


def generation_report(model: Any) -> dict:
    config = find_attr(model, ATTR)
    if config is None:
        return {}
    return {"chunk_tokens": config.chunk_tokens, "cache_tiers": list(config.cache_tiers),
            "slot_budget": dict(config.slot_budget), "stats": json.loads(json.dumps(config.stats, default=str))}


def write_generation_report(model: Any, path: str | Path) -> dict:
    """Sidecar file next to the results: how many episodes the static generator refused, which, and tier/compile usage.
    A refusal raised mid-episode discards that episode's partial transcript (the episode scores as no answer)."""
    report = generation_report(model)
    if report:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2))
        n = report["stats"].get("rollouts_dropped", 0)
        print(f"[TPU EVAL] static generation: {report['stats'].get('calls', 0)} calls, tiers used {report['stats'].get('tiers')}, "
              f"{n} episode(s) refused (details: {path})", flush=True)
    return report
