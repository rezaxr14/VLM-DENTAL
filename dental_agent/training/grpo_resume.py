"""GRPO policy initialisation and resume helpers.

Two things GRPO needs that were missing:

1. The trainable ``grpo_policy`` adapter must START from the SFT weights. ``PeftModel.add_adapter`` creates a fresh
   zero-initialised LoRA, so without a copy the policy began as the bare base model while the frozen ``reference``
   adapter was the SFT model. Step 1 then measured (and the KL term pulled against) the whole SFT gain, and the rollouts
   came from an un-fine-tuned model.
2. A run killed on Kaggle/Colab must be able to continue. A step checkpoint stores the adapters plus
   ``checkpoint_meta.json`` (step, EMA baseline, group size, pool fingerprint). Resuming reloads the policy adapter and
   skips the images already trained on. Optimizer moments are NOT saved (they would double the upload size and cannot be
   gathered reliably from sharded XLA tensors), so Adam restarts cold at the resumed step.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from typing import Any, Iterable

import torch

REFERENCE = "reference"
POLICY = "grpo_policy"

_CONFIG_FIELDS = ("r", "lora_alpha", "use_rslora", "use_dora", "rank_pattern", "alpha_pattern")


def init_policy_from_reference(model: Any, reference: str = REFERENCE, policy: str = POLICY) -> int:
    """Copy every LoRA tensor of adapter ``reference`` into adapter ``policy``. Returns the number of tensors copied.

    Raises if the two adapters are not shape-compatible (different rank, alpha, scaling mode or target modules), since a
    partial copy would silently start the policy somewhere other than the SFT model.
    """
    cfg = model.peft_config
    diffs = [f for f in _CONFIG_FIELDS if getattr(cfg[reference], f, None) != getattr(cfg[policy], f, None)]
    if diffs:
        raise ValueError(f"Adapter '{policy}' differs from '{reference}' in {diffs}; the SFT weights cannot be copied.")

    params = dict(model.named_parameters())
    ref_marker, pol_marker = f".{reference}.", f".{policy}."
    n = 0
    with torch.no_grad():
        for name, src in params.items():
            if ref_marker not in name:
                continue
            dst_name = name.replace(ref_marker, pol_marker)
            dst = params.get(dst_name)
            if dst is None or dst.shape != src.shape:
                raise ValueError(f"No matching '{policy}' tensor for '{name}' (looked for '{dst_name}').")
            dst.copy_(src.to(dst.dtype))
            n += 1
    n_policy = sum(1 for name in params if pol_marker in name)
    if n == 0 or n != n_policy:
        raise ValueError(f"Copied {n} LoRA tensors from '{reference}' but '{policy}' has {n_policy}.")
    return n


def pool_fingerprint(image_ids: Iterable[Any]) -> str:
    """Order-sensitive hash of the training pool, so a resume cannot continue on a different image order."""
    text = ",".join(str(int(i)) for i in image_ids)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


_STEP_DIR = re.compile(r"^grpo-(?P<track>[a-z_]+)-step-(?P<step>\d+)$")


def find_latest_step_checkpoint(checkpoint_dir: str | Path, track: str) -> tuple[Path, dict[str, Any]] | None:
    """Newest ``grpo-<track>-step-N`` folder (with a readable checkpoint_meta.json and a policy adapter), or None."""
    root = Path(checkpoint_dir)
    best: tuple[int, Path, dict[str, Any]] | None = None
    if not root.is_dir():
        return None
    for child in root.iterdir():
        m = _STEP_DIR.match(child.name)
        if not m or m.group("track") != track:
            continue
        meta_path = child / "checkpoint_meta.json"
        if not meta_path.is_file() or not (child / POLICY / "adapter_config.json").is_file():
            continue
        try:
            extra = json.loads(meta_path.read_text()).get("extra", {})
        except Exception:
            continue
        step = int(m.group("step"))
        if best is None or step > best[0]:
            best = (step, child, extra)
    return (best[1], best[2]) if best else None


def load_policy_state(model: Any, checkpoint_path: str | Path, policy: str = POLICY) -> None:
    """Load the trained policy adapter of a GRPO checkpoint into adapter ``policy`` of ``model``."""
    from peft.utils import set_peft_model_state_dict

    folder = Path(checkpoint_path) / policy
    st = folder / "adapter_model.safetensors"
    if st.is_file():
        from safetensors.torch import load_file
        state = load_file(str(st))
    else:
        state = torch.load(folder / "adapter_model.bin", map_location="cpu")
    set_peft_model_state_dict(model, state, adapter_name=policy)


def fetch_latest_checkpoint_from_hub(
    hf_repo: str, path_in_repo_prefix: str, track: str, checkpoint_dir: str | Path, scratch_dir: str | Path,
) -> str | None:
    """Download only the newest ``grpo-<track>-step-N`` folder of ``<hf_repo>/<prefix>`` into ``checkpoint_dir/<tag>``.

    (The old code downloaded EVERY step checkpoint into ``<output>/<prefix>/...``, a folder training never reads.)
    Returns the tag, or None when the repo holds no step checkpoint.
    """
    from huggingface_hub import HfApi, snapshot_download

    prefix = path_in_repo_prefix.strip("/") + "/"
    steps: dict[int, str] = {}
    for f in HfApi().list_repo_files(repo_id=hf_repo):
        if not f.startswith(prefix):
            continue
        m = _STEP_DIR.match(f[len(prefix):].split("/", 1)[0])
        if m and m.group("track") == track:
            steps[int(m.group("step"))] = f[len(prefix):].split("/", 1)[0]
    if not steps:
        return None
    tag = steps[max(steps)]
    scratch = Path(scratch_dir)
    snapshot_download(repo_id=hf_repo, allow_patterns=[f"{prefix}{tag}/*"], local_dir=str(scratch))
    dest = Path(checkpoint_dir) / tag
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(scratch / prefix / tag, dest)
    shutil.rmtree(scratch, ignore_errors=True)
    return tag
