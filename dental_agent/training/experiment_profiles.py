"""Shared explicit experiment identity; no hardware detection or precision changes."""
import re
from pathlib import Path

from dental_agent.utils.model_size import slug as model_slug

PROFILE_TAGS = {
    "dual_t4": "dual_t4_fp16",
    "rtx4090": "rtx4090_bf16",
    "rtx4090_qlora": "rtx4090_bf16",
    "workstation48_lora": "workstation48_lora_bf16",
    "quadro_rtx8000": "quadro_rtx8000_fp16",
    "tpu": "tpu_bf16",
    "dual_t4_fp16": "dual_t4_unquantized_fp16",
    # Two T4s, one full replica of the model on each (python -m torch.distributed.run, --data-parallel). A separate tag, so a DDP run is its own
    # cohort: one GRPO update averages two images, which is not the same recipe as the single-process dual_t4_fp16 profile.
    "dual_t4_ddp_fp16_4b": "dual_t4_ddp_unquantized_fp16",
}

def grpo_name(track, k, stage, precision, tag=None, harness_mode="off"):
    from dental_agent.harness.configuration import validate_harness_mode
    validate_harness_mode(harness_mode, track)
    name = f"{model_slug()}_grpo_{track}_k{k}_{stage}"
    if tag:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", tag):
            raise ValueError("Invalid experiment tag")
        name += f"_{precision}__{tag}"
    if harness_mode != "off": name += "--harness_"+harness_mode
    return name

def fetch_sft(repo, track, stage, precision, tag):
    from huggingface_hub import snapshot_download
    name = f"{model_slug()}_sft_{track}_{stage}_{precision}__{tag}"
    # Cache snapshot is immutable; no fallback to another experiment or partial local copy.
    root = Path(snapshot_download(repo_id=repo, repo_type="model",
                                 allow_patterns=[f"sft/{name}/*"]))
    folder = root / "sft" / name
    if not (folder / "adapter_config.json").is_file():
        raise FileNotFoundError(f"Missing exact SFT reference: {repo}/sft/{name}")
    return str(folder)

def can_resume(folder, track):
    from dental_agent.training.grpo_resume import find_latest_step_checkpoint
    folder = Path(folder)
    return ((folder / f"grpo-{track}-final" / "checkpoint_meta.json").is_file()
            or find_latest_step_checkpoint(folder, track) is not None)
