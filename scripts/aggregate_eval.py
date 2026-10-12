#!/usr/bin/env python3
"""Recompute the evaluation-matrix tables from existing per-image result files (no model needed).

    python scripts/aggregate_eval.py --output-dir data/evaluations --dataset dentex --split test

Writes matrix_summary.json / matrix_pooled.csv / matrix_per_class.csv / matrix_comparisons.csv / matrix_summary.md to
``<output-dir>/matrix_<dataset>_<split>/``. See dental_agent/evaluation/matrix.py for the definitions.
"""

import argparse
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.evaluation import matrix  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Aggregate per-image evaluation results into the matrix tables.")
    p.add_argument("--output-dir", type=str, default="data/evaluations", help="Folder with the eval_*.jsonl files.")
    p.add_argument("--dataset", type=str, default="dentex")
    p.add_argument("--split", type=str, default="test", choices=["test", "dev"])
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    cells = matrix.load_matrix(args.output_dir, args.dataset, args.split)
    if not cells:
        raise SystemExit(f"No eval_{args.dataset}_{args.split}_*.jsonl files in {args.output_dir}.")
    summary = matrix.build_summary(cells, n_boot=args.n_boot, seed=args.seed)
    out_dir = Path(args.output_dir) / f"matrix_{args.dataset}_{args.split}"
    paths = matrix.write_matrix_outputs(summary, out_dir)
    print(f"{len(cells)} cell(s) -> {out_dir}")
    print(paths["markdown"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
