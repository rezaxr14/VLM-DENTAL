"""
Standalone script to run Aim 1 expert trace generation.
"""

import argparse
from dental_agent.config import load_config
from dental_agent.data.dentex import load_dentex_dataset
from dental_agent.training.trace_generation import generate_trace_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Aim 1 Synthetic Trace Generator")
    parser.add_argument("--config", "-c", default=None, help="Path to config YAML")
    parser.add_argument("--n-samples", "-n", type=int, default=20, help="Number of traces")
    parser.add_argument("--output", "-o", default="data/synthetic_traces.jsonl", help="Output JSONL")
    args = parser.parse_args()

    cfg = load_config(args.config)
    images_df, annots_df, categories_df = load_dentex_dataset(cfg.data.data_dir)

    generate_trace_dataset(
        images_df=images_df,
        annots_df=annots_df,
        categories_df=categories_df,
        output_jsonl=args.output,
        n_samples=args.n_samples,
    )


if __name__ == "__main__":
    main()
