"""
Standalone script to run evaluation and report generation.
"""

import argparse
from dental_agent.config import load_config
from dental_agent.data.dentex import load_dentex_dataset
from dental_agent.evaluation.batch_runner import evaluate_dataset
from dental_agent.evaluation.reporting import generate_markdown_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Dental-Agent Evaluation Harness")
    parser.add_argument("--config", "-c", default=None, help="Path to config YAML")
    parser.add_argument("--checkpoint-tag", default=None, help="Checkpoint tag to evaluate")
    parser.add_argument("--limit", type=int, default=None, help="Sample limit")
    parser.add_argument("--output", "-o", default="experiments/evaluation_report.md", help="Report output path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    images_df, annots_df, _ = load_dentex_dataset(cfg.data.data_dir)

    metrics = evaluate_dataset(
        images_df=images_df,
        annots_df=annots_df,
        checkpoint_tag=args.checkpoint_tag,
        sample_limit=args.limit,
    )

    tag = args.checkpoint_tag or "Dental-Agent (Default)"
    generate_markdown_report({tag: metrics}, output_path=args.output)


if __name__ == "__main__":
    main()
