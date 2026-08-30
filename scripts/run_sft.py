"""
Standalone script to run Stage 1 Supervised Fine-Tuning.
"""

import argparse
from dental_agent.config import load_config
from dental_agent.training.sft import train_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 1 SFT Trainer")
    parser.add_argument("--config", "-c", default=None, help="Path to config YAML")
    parser.add_argument("--data", "-d", default="data/traces/train_cot_traces.jsonl", help="Trace dataset JSONL")
    parser.add_argument("--epochs", "-e", type=int, default=3, help="Epochs")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_sft(
        data_path=args.data,
        config=cfg,
        epochs=args.epochs,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
