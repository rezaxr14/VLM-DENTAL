#!/usr/bin/env python3
"""
Production CLI to run Stage 2 Group Relative Policy Optimization (GRPO) (§17).

Supports:
- Strict Track Segregation: --track {with_tools, no_tools}
- Flexible Group Size: --group-size K in {1, 2, 4, 8, 16}
- [G2] Batched Rollouts & [G3] Dual-LoRA Reference/Policy Toggle
- Multi-Finding Complete Ground Truth Evaluation (Rule 13)
- Lightweight Hugging Face Checkpoint Sync (~760 MB) & Cross-Account Resume
- SIGTERM Preemption Handler for Kaggle 9-Hour Session Limits
"""

import argparse
import json
import os
import signal
import sys
from pathlib import Path

# Clear conflicting legacy TPU multi-host cluster address variables on Kaggle/Colab
for _var in ["TPU_PROCESS_ADDRESSES", "TPU_PROCESS_COUNT", "CLOUD_TPU_TASK_ID"]:
    os.environ.pop(_var, None)

# NOTE: PJRT_DEVICE is set lazily inside run_worker(), NOT here.
# Setting it before `import torch` causes torch_xla's torch plugin to auto-initialize
# the PJRT TPU client at import time, leading to double-initialization crashes.

from dotenv import load_dotenv
load_dotenv()

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.config import load_config
from dental_agent.data.dentex import load_dentex_dataset
from dental_agent.data.tufts import load_tufts_dataset
from dental_agent.training.grpo import train_grpo


def upload_checkpoint_to_hf(checkpoint_dir: Path, hf_repo: str, step: int):
    """Upload lightweight LoRA checkpoint (~760 MB) to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        commit_msg = f"VLM-DENTAL GRPO Checkpoint: Step {step}"
        print(f"[HF-HUB] Uploading GRPO checkpoint from {checkpoint_dir} to {hf_repo}...")
        api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=hf_repo,
            commit_message=commit_msg,
            ignore_patterns=["*.tmp", "*.lock"],
        )
        print(f"[HF-HUB] GRPO checkpoint successfully uploaded to {hf_repo}.")
    except Exception as e:
        print(f"[HF-HUB WARNING] Failed to upload checkpoint to {hf_repo}: {e}")


def resolve_sft_reference(
    sft_dir_arg: str | None,
    track: str,
    sft_stage: str,
    hf_repo: str | None = None,
) -> str:
    """Resolve local path to SFT reference model or auto-download from Hugging Face Models repo."""
    if sft_dir_arg:
        chosen_path = Path(sft_dir_arg)
        if chosen_path.exists() and any(chosen_path.iterdir()):
            print(f"[SFT-REF] Using explicitly provided SFT reference at {chosen_path}")
            return str(chosen_path)

    # Standard candidate directories
    candidate_paths = [
        Path(f"data/models/qwen3_5_9b_sft_{track}_{sft_stage}"),
        Path(f"checkpoints/sft_{track}_{sft_stage}"),
        Path(f"data/models/qwen3_5_9b_sft_{track}"),
    ]

    for cand in candidate_paths:
        if cand.exists() and any(cand.iterdir()):
            print(f"[SFT-REF] Resolved local SFT reference model at {cand}")
            return str(cand)

    # If not found locally, attempt to download from HF Hub
    if hf_repo:
        path_in_repo = f"sft/qwen3_5_9b_sft_{track}_{sft_stage}"
        dest_parent = Path("data/models")
        print(f"[SFT-REF] Local SFT adapter not found. Fetching reference adapter from {hf_repo}/{path_in_repo}...")
        try:
            from huggingface_hub import snapshot_download
            dest_parent.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=hf_repo,
                allow_patterns=[f"{path_in_repo}/*"],
                local_dir=str(dest_parent),
            )
            downloaded = dest_parent / path_in_repo
            if downloaded.exists() and any(downloaded.iterdir()):
                print(f"[SFT-REF] Successfully downloaded reference SFT adapter to {downloaded}")
                return str(downloaded)
        except Exception as e:
            print(f"[SFT-REF WARNING] Could not download SFT reference from {hf_repo}: {e}")

    default_fallback = f"data/models/qwen3_5_9b_sft_{track}_{sft_stage}"
    print(f"[SFT-REF WARNING] Reference model directory not found locally or remotely: {default_fallback}")
    return default_fallback


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 GRPO Policy Trainer")
    parser.add_argument(
        "--track",
        type=str,
        default="with_tools",
        choices=["with_tools", "no_tools"],
        help="Strict training track: 'with_tools' (multi-turn agent) or 'no_tools' (direct radiologist)",
    )
    parser.add_argument(
        "--sft-stage",
        type=str,
        default="dentex_alone",
        choices=["dentex_alone", "dentex_tufts_overlap", "multicohort_all"],
        help="Curriculum SFT reference stage: 'dentex_alone', 'dentex_tufts_overlap', or 'multicohort_all'",
    )
    parser.add_argument("--config", "-c", default=None, help="Path to config YAML")
    parser.add_argument("--dataset", default="dentex", help="Dataset name ('dentex' or 'tufts')")
    parser.add_argument("--group-size", "-g", type=int, default=4, help="GRPO group size K (1, 2, 4, 8, 16)")
    parser.add_argument("--epochs", "-e", type=int, default=2, help="Optimization epochs per rollout batch")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for policy gradients")
    parser.add_argument("--kl-beta", type=float, default=0.04, help="Schulman k3 KL divergence penalty weight")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument(
        "--sft-model-dir",
        type=str,
        default=None,
        help="Path to SFT adapter for reference policy (auto-defaults per track & sft-stage)",
    )
    parser.add_argument("--output-dir", type=str, default="data/models", help="Directory to save RL checkpoints")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=os.environ.get("HF_ARTIFACT_REPO", "Reza-Nadimi/vlm-dental-models"),
        help="Hugging Face Hub repository for checkpoint sync (default: Reza-Nadimi/vlm-dental-models)",
    )
    parser.add_argument("--push-every-steps", type=int, default=25, help="Frequency of HF checkpoint upload in steps")
    parser.add_argument("--resume-hf", type=str, default=None, help="Hugging Face repo to resume latest checkpoint from")
    parser.add_argument(
        "--num-cores",
        type=int,
        default=1,
        help="Number of TPU cores for distributed data-parallel execution (1 for single core/GPU, 8 for Kaggle TPU v5e-8)",
    )
    parser.add_argument(
        "--fsdp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable PyTorch/XLA FSDP parameter sharding across TPU cores to fit 9B BF16 model within 16 GB HBM (default: True on multi-core TPU)",
    )
    args = parser.parse_args()

    is_tpu = False
    try:
        import torch_xla  # noqa: F401 — lightweight check; do NOT import xla_model here
        is_tpu = True     # (importing xla_model initializes libtpu and claims /dev/vfio exclusively)
    except ImportError:
        pass

    if is_tpu and args.num_cores > 1:
        if "PJRT_DEVICE" not in os.environ:
            os.environ["PJRT_DEVICE"] = "TPU"
        try:
            try:
                import torch_xla.distributed.xla_multiprocessing as xmp
            except ImportError:
                import torch_xla.distributed.xmp as xmp
            print(f"[LAUNCH] Spawning multi-core Cloud TPU v5e-8 GRPO across available TPU cores via xmp.spawn(nprocs=None)...")
            xmp.spawn(run_worker, args=(args,), nprocs=None)
        except Exception as e:
            print(f"\n[FATAL TPU ERROR] Could not spawn multi-core GRPO via xmp: {e}")
            print("[DIAGNOSTIC] On Cloud TPU VMs, hardware access to /dev/vfio/* is exclusive.")
            print("[DIAGNOSTIC] If running from Jupyter/Colab/Kaggle, ensure the notebook kernel did not call xm.xla_device() before launching this script.")
            print("[DIAGNOSTIC] Please restart the notebook session (Run -> Restart Session) and re-run to release /dev/vfio/*.\n")
            raise RuntimeError(f"Multi-core Cloud TPU GRPO launch failed: {e}") from e
    else:
        run_worker(0, args)


def run_worker(index: int, args: argparse.Namespace):
    """Per-device worker routine for GRPO training."""
    cfg = load_config(args.config)

    is_tpu = False
    try:
        if "PJRT_DEVICE" not in os.environ:
            os.environ["PJRT_DEVICE"] = "TPU"
        import torch_xla.core.xla_model as xm
        is_tpu = True
        device = xm.xla_device()
        is_master = xm.is_master_ordinal()
    except Exception:
        is_master = True

    # Auto-resolve SFT reference directory per track and sft-stage
    resolved_sft_dir = resolve_sft_reference(
        sft_dir_arg=args.sft_model_dir,
        track=args.track,
        sft_stage=args.sft_stage,
        hf_repo=args.hf_repo,
    )

    # Checkpoint directory naming: qwen3_5_9b_grpo_{track}_k{group_size}_{sft_stage}
    target_name = f"qwen3_5_9b_grpo_{args.track}_k{args.group_size}_{args.sft_stage}"
    out_dir = Path(args.output_dir) / target_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path_in_repo_prefix = f"grpo/{target_name}"

    # Handle resume from HF Hub across Kaggle accounts
    if args.resume_hf and is_master:
        print(f"[RESUME] Checking HF Hub for latest checkpoint in {args.resume_hf}/{path_in_repo_prefix}...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id=args.resume_hf,
                allow_patterns=[f"{path_in_repo_prefix}/*"],
                local_dir=str(Path(args.output_dir)),
            )
            print(f"[RESUME] Restored checkpoint from {args.resume_hf} under {path_in_repo_prefix}.")
        except Exception as e:
            print(f"[RESUME WARNING] Could not restore from HF Hub: {e}")

    dataset_name = args.dataset.strip().lower()
    if dataset_name == "tufts":
        images_df, annots_df, categories_df = load_tufts_dataset(cfg.data.data_dir)
    else:
        images_df, annots_df, categories_df = load_dentex_dataset(cfg.data.data_dir)

    # If multi-core TPU, partition dataset among replicas
    if is_tpu and args.num_cores > 1:
        images_df = images_df.iloc[index::args.num_cores].reset_index(drop=True)

    if is_master:
        print("======================================================================")
        print(f"VLM-DENTAL: STAGE 2 GRPO RL ({args.track.upper()})")
        print(f"* SFT Stage   : {args.sft_stage}")
        print(f"* Group Size K: {args.group_size}")
        print(f"* SFT Ref Dir : {resolved_sft_dir}")
        print(f"* Target Ckpt : {out_dir}")
        print(f"* HF Repo Sync: {args.hf_repo} ({path_in_repo_prefix})")
        print(f"* Dataset     : {args.dataset}")
        print(f"* KL Beta     : {args.kl_beta}")
        print(f"* Learning Rate: {args.lr}")
        print(f"* Replicas    : {args.num_cores} device cores")
        print("======================================================================")

    train_grpo(
        images_df=images_df,
        annots_df=annots_df,
        categories_df=categories_df,
        config=cfg,
        sft_model_dir=resolved_sft_dir,
        checkpoint_dir=out_dir,
        group_size=args.group_size,
        epochs_per_batch=args.epochs,
        kl_beta=args.kl_beta,
        clip_eps=args.clip_eps,
        learning_rate=args.lr,
        track=args.track,
        hf_repo=args.hf_repo if is_master else None,
        push_every_steps=args.push_every_steps,
        path_in_repo_prefix=path_in_repo_prefix,
        num_cores=args.num_cores,
        use_fsdp=args.fsdp,
    )


if __name__ == "__main__":
    main()
