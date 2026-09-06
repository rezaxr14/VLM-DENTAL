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


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 GRPO Policy Trainer")
    parser.add_argument(
        "--track",
        type=str,
        default="with_tools",
        choices=["with_tools", "no_tools"],
        help="Strict training track: 'with_tools' (multi-turn agent) or 'no_tools' (direct radiologist)",
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
        help="Path to SFT adapter for reference policy (auto-defaults per track)",
    )
    parser.add_argument("--output-dir", type=str, default="data/models", help="Directory to save RL checkpoints")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face Hub repository for checkpoint sync")
    parser.add_argument("--push-every-steps", type=int, default=25, help="Frequency of HF checkpoint upload in steps")
    parser.add_argument("--resume-hf", type=str, default=None, help="Hugging Face repo to resume latest checkpoint from")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Auto-resolve SFT reference directory per track
    if args.sft_model_dir is None:
        args.sft_model_dir = f"data/models/qwen3_5_9b_sft_{args.track}"

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Handle resume from HF Hub across Kaggle accounts
    if args.resume_hf:
        print(f"[RESUME] Checking HF Hub for latest checkpoint in {args.resume_hf}...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=args.resume_hf, local_dir=str(out_dir))
            print(f"[RESUME] Restored checkpoint from {args.resume_hf}.")
        except Exception as e:
            print(f"[RESUME WARNING] Could not restore from HF Hub: {e}")

    dataset_name = args.dataset.strip().lower()
    if dataset_name == "tufts":
        images_df, annots_df, categories_df = load_tufts_dataset(cfg.data.data_dir)
    else:
        images_df, annots_df, categories_df = load_dentex_dataset(cfg.data.data_dir)

    print("======================================================================")
    print(f"VLM-DENTAL: STAGE 2 GRPO RL ({args.track.upper()})")
    print(f"* Group Size K: {args.group_size}")
    print(f"* SFT Ref Dir : {args.sft_model_dir}")
    print(f"* Dataset     : {args.dataset}")
    print(f"* KL Beta     : {args.kl_beta}")
    print(f"* Learning Rate: {args.lr}")
    print("======================================================================")

    train_grpo(
        images_df=images_df,
        annots_df=annots_df,
        categories_df=categories_df,
        config=cfg,
        sft_model_dir=args.sft_model_dir,
        checkpoint_dir=out_dir,
        group_size=args.group_size,
        epochs_per_batch=args.epochs,
        kl_beta=args.kl_beta,
        clip_eps=args.clip_eps,
        learning_rate=args.lr,
        track=args.track,
        hf_repo=args.hf_repo,
        push_every_steps=args.push_every_steps,
    )


if __name__ == "__main__":
    main()
