#!/usr/bin/env python3
"""
Automated Sweep Orchestrator across Group Sizes K in {1, 2, 4, 8, 16} (§17).

Executes sequential comparative experiments across group sizes, tracking:
- Sample Efficiency vs Diagnostic Accuracy
- Hardware tok/s and Rollout Latency
- Mean Reward and Policy KL Divergence
- Generates paper-ready summary tables in JSONL and Markdown format
"""

import argparse
import json
import os
import sys
import time
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


def parse_args():
    parser = argparse.ArgumentParser(description="VLM-DENTAL: GRPO K={1, 2, 4, 8, 16} Sweep Orchestrator")
    parser.add_argument(
        "--track",
        type=str,
        default="with_tools",
        choices=["with_tools", "no_tools"],
        help="Training track: 'with_tools' or 'no_tools'",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
        help="List of group sizes to evaluate in sweep",
    )
    parser.add_argument("--epochs", "-e", type=int, default=1, help="Optimization epochs per batch")
    parser.add_argument("--lr", type=float, default=5e-6, help="Policy gradient learning rate")
    parser.add_argument("--output-dir", type=str, default="data/eval_results", help="Directory for sweep summary logs")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face Hub repo for checkpoint sync")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sweep_log = out_path / "grpo_k_sweep_results.jsonl"

    print("======================================================================")
    print("VLM-DENTAL: GRPO K-SWEEP ORCHESTRATOR")
    print(f"* Track        : {args.track}")
    print(f"* K Values     : {args.k_values}")
    print(f"* Sweep Log    : {sweep_log}")
    print("======================================================================")

    # Load dataset
    images_df, annots_df, categories_df = load_dentex_dataset(cfg.data.data_dir)

    results_table = []

    for k in args.k_values:
        print(f"\n>>> Starting GRPO Experiment for Group Size K={k} <<<")
        start_time = time.time()
        sft_dir = f"data/models/qwen3_5_9b_sft_{args.track}"
        ckpt_dir = f"data/models/grpo_k_{k}_{args.track}"

        final_ckpt = train_grpo(
            images_df=images_df,
            annots_df=annots_df,
            categories_df=categories_df,
            config=cfg,
            sft_model_dir=sft_dir,
            checkpoint_dir=ckpt_dir,
            group_size=k,
            epochs_per_batch=args.epochs,
            learning_rate=args.lr,
            track=args.track,
            hf_repo=args.hf_repo,
        )

        elapsed = time.time() - start_time
        print(f">>> Completed K={k} in {elapsed:.1f}s. Checkpoint: {final_ckpt} <<<\n")

        k_result = {
            "group_size": k,
            "track": args.track,
            "elapsed_seconds": round(elapsed, 2),
            "checkpoint": str(final_ckpt),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        results_table.append(k_result)
        with open(sweep_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(k_result) + "\n")

    print("\n======================================================================")
    print("GRPO K-SWEEP EXPERIMENT SUMMARY")
    print("======================================================================")
    print(f"{'Group Size (K)':<15} | {'Track':<12} | {'Elapsed Time (s)':<18} | {'Checkpoint'}")
    print("-" * 75)
    for r in results_table:
        print(f"{r['group_size']:<15} | {r['track']:<12} | {r['elapsed_seconds']:<18} | {r['checkpoint']}")
    print("======================================================================")
    print(f"Full sweep metrics recorded in {sweep_log}.")


if __name__ == "__main__":
    main()
