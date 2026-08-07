"""
Standalone script to train the Stage 0 specialist tooth detector.
"""

import argparse
from dental_agent.config import load_config
from dental_agent.data.dentex import load_dentex_dataset
from dental_agent.training.detector import train_stage0_detector


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 0 Tooth Detector Trainer")
    parser.add_argument("--config", "-c", default=None, help="Path to config YAML")
    parser.add_argument("--epochs", "-e", type=int, default=5, help="Epochs")
    parser.add_argument("--output", "-o", default="checkpoints/stage0_detector.pt", help="Output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    images_df, annots_df, _ = load_dentex_dataset(cfg.data.data_dir)

    train_stage0_detector(
        images_df=images_df,
        annots_df=annots_df,
        output_path=args.output,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
