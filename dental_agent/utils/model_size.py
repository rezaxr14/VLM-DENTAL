"""The one place that knows which Qwen3.5 size a run uses.

``DENTAL_MODEL_SIZE`` = ``9b`` (default, unchanged behaviour) or ``4b``. Everything that used to hardcode "Qwen3.5-9B" asks this module:
the Hub id, the local folder name, the local-folder env var, the checkpoint / Hub artifact slug, and the size-dependent memory figures.

A run uses exactly one size. The size is part of every artifact name (``qwen3_5_9b_*`` / ``qwen3_5_4b_*``), and each size has its own
local-folder env var (``QWEN35_9B_LOCAL_DIR`` / ``QWEN35_4B_LOCAL_DIR``), so a 4B run can never load, resume or evaluate a 9B artifact
by accident (and the reverse). Nothing here imports torch or reads the network.

Architecture numbers are from the two published config.json files (text hidden size, MLP width, layers, vision depth). The weight
footprints are ESTIMATES from those shapes (parameters x 2 bytes), not measurements; the doctor and the preflight print the real peak.
"""
from __future__ import annotations

import os
from typing import Any

SIZE_ENV = "DENTAL_MODEL_SIZE"
DEFAULT_SIZE = "9b"

SPECS: dict[str, dict[str, Any]] = {
    "9b": dict(
        size="9b", hub_id="Qwen/Qwen3.5-9B", folder="Qwen3.5-9B", slug="qwen3_5_9b", local_env="QWEN35_9B_LOCAL_DIR",
        kaggle_dirs=("/kaggle/input/datasets/rezanadimikj/qwen3-5-9b", "/kaggle/input/qwen3-5-9b", "/kaggle/input/qwen3_5_9b",
                     "/kaggle/input/qwen-3-5-9b", "/kaggle/input/qwen3-5-9b-base"),
        norm_names=("qwen359b",), norm_pair=("qwen35", "9b"),
        text_hidden=4096, text_layers=32, text_intermediate=12288, vision_depth=27, vision_hidden=1152, tied_embeddings=False,
        weights_gib_16bit=18.0, download_gb=19.0,
    ),
    "4b": dict(
        size="4b", hub_id="Qwen/Qwen3.5-4B", folder="Qwen3.5-4B", slug="qwen3_5_4b", local_env="QWEN35_4B_LOCAL_DIR",
        kaggle_dirs=("/kaggle/input/datasets/rezanadimikj/qwen3-5-4b", "/kaggle/input/qwen3-5-4b", "/kaggle/input/qwen3_5_4b",
                     "/kaggle/input/qwen-3-5-4b", "/kaggle/input/qwen3-5-4b-base"),
        norm_names=("qwen354b",), norm_pair=("qwen35", "4b"),
        text_hidden=2560, text_layers=32, text_intermediate=9216, vision_depth=24, vision_hidden=1024, tied_embeddings=True,
        weights_gib_16bit=9.5, download_gb=10.0,       # estimates from the config shapes (about 4.6B parameters), not measured
    ),
}


def active_size(environ: dict | None = None) -> str:
    """``9b`` or ``4b`` from ``DENTAL_MODEL_SIZE`` (unset / empty = 9b). Anything else raises: no silent fallback to another size."""
    raw = ((os.environ if environ is None else environ).get(SIZE_ENV) or "").strip().lower()
    if not raw:
        return DEFAULT_SIZE
    raw = raw.replace("qwen3.5-", "").replace("qwen3_5_", "")
    if raw not in SPECS:
        raise ValueError(f"{SIZE_ENV}={raw!r} is not a supported model size; use one of {sorted(SPECS)}")
    return raw


def spec(size: str | None = None, environ: dict | None = None) -> dict[str, Any]:
    return SPECS[size if size is not None else active_size(environ)]


def set_active_size(size: str, environ: dict | None = None) -> str:
    """Validate and export ``DENTAL_MODEL_SIZE`` so later cells, imports and child processes all follow it."""
    size = str(size).strip().lower()
    if size not in SPECS:
        raise ValueError(f"MODEL_SIZE={size!r} is not supported; use one of {sorted(SPECS)}")
    (os.environ if environ is None else environ)[SIZE_ENV] = size
    return size


def hub_id(size: str | None = None) -> str:
    return spec(size)["hub_id"]


def default_model_id(size: str | None = None) -> str:
    """Default for every ``--model-id`` / ``MODEL_NAME`` argument."""
    return spec(size)["hub_id"]


def slug(size: str | None = None) -> str:
    """Artifact-name prefix: ``qwen3_5_9b`` / ``qwen3_5_4b``."""
    return spec(size)["slug"]


def label(size: str | None = None) -> str:
    """Human name for messages: ``Qwen3.5-9B`` / ``Qwen3.5-4B``."""
    return spec(size)["folder"]


def local_env_name(size: str | None = None) -> str:
    return spec(size)["local_env"]


def slug_regex() -> str:
    """Regex alternation matching an artifact slug of ANY supported size (used when scanning a Hub listing)."""
    return "|".join(sorted(s["slug"] for s in SPECS.values()))


def size_of_slug(name: str) -> str | None:
    for s in SPECS.values():
        if name.startswith(s["slug"] + "_"):
            return s["size"]
    return None


def config_problem(cfg: dict, size: str | None = None) -> str | None:
    """Why a config.json dict is NOT the active size (None = consistent or not comparable). Catches a 4B run pointed at a 9B folder
    and the reverse before any weight is read."""
    sp = spec(size)
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    hidden = text.get("hidden_size")
    if not isinstance(hidden, int):
        return None
    if hidden != sp["text_hidden"]:
        other = next((s["size"] for s in SPECS.values() if s["text_hidden"] == hidden), None)
        return (f"config.json has text hidden_size={hidden}" + (f" (that is the {other} model)" if other else "") +
                f", but {SIZE_ENV}={sp['size']} expects {sp['text_hidden']} ({sp['folder']})")
    return None


# ---- profile names carry the size -------------------------------------------------------------------------------------------------
# The notebooks select ONE hardware profile and everything else follows from it: the model size, the data-parallel / weight layout and the
# runtime knobs. A profile name is `<hardware profile>` (Qwen3.5-9B, exactly as before), `<hardware profile>_4b` (the same hardware recipe
# with Qwen3.5-4B), or one of the named variants below. Nothing else in a notebook has to agree with it.
PROFILE_VARIANTS: dict[str, tuple[str, str, str]] = {
    # name: (hardware profile, model size, weight layout)
    "tpu_4b_dp": ("tpu", "4b", "replicated"),                       # a full bf16 copy of the 4B model on every v5e core, batch split over the 8 cores
    "dual_t4_ddp_fp16": ("dual_t4_ddp_fp16", "4b", "replicated"),   # one process per T4 with a full 4B fp16 replica (torchrun)
}


def profile_parts(name: str) -> tuple[str, str, str]:
    """``(hardware profile, model size, weight layout)`` of a profile name. ``layout`` is ``sharded`` (FSDPv2 / layers split) or ``replicated``."""
    if name in PROFILE_VARIANTS:
        return PROFILE_VARIANTS[name]
    for size in SPECS:
        if size != DEFAULT_SIZE and name.endswith("_" + size) and len(name) > len(size) + 1:
            return name[: -(len(size) + 1)], size, "sharded"
    return name, DEFAULT_SIZE, "sharded"


def select_profile(name: str, environ: dict | None = None) -> tuple[str, str, str]:
    """Resolve a profile name and make its model size the active one (exports ``DENTAL_MODEL_SIZE``). Returns ``profile_parts(name)``."""
    base, size, layout = profile_parts(name)
    set_active_size(size, environ)
    return base, size, layout
