"""
Standalone script to run Stage 2 GRPO training.
"""

import argparse
from dental_agent.config import load_config
from dental_agent.data.dentex import load_dentex_dataset
from dental_agent.training.grpo import train_grpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 GRPO Policy Trainer")
    parser.add_argument("--config", "-c", default=None, help="Path to config YAML")
    parser.add_argument("--group-size", "-g", type=int, default=4, help="GRPO group size")
    parser.add_argument("--epochs", "-e", type=int, default=2, help="Epochs per batch")
    parser.add_argument("--sft-model-dir", type=str, default="data/models/qwen3_5_9b_sft", help="Path to SFT adapter for reference policy")
    args = parser.parse_args()

    cfg = load_config(args.config)
    images_df, annots_df, categories_df = load_dentex_dataset(cfg.data.data_dir)

    train_grpo(
        images_df=images_df,
        annots_df=annots_df,
        categories_df=categories_df,
        config=cfg,
        sft_model_dir=args.sft_model_dir,
        group_size=args.group_size,
        epochs_per_batch=args.epochs,
    )


if __name__ == "__main__":
    main()
