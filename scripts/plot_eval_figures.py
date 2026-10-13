#!/usr/bin/env python3
"""Draw the paper figures of the evaluation matrix from the tables ``aggregate_eval.py`` wrote (no model needed).

    python scripts/aggregate_eval.py --output-dir data/evaluations --dataset dentex --split test
    python scripts/plot_eval_figures.py --output-dir data/evaluations --dataset dentex --split test

Reads ``<output-dir>/matrix_<dataset>_<split>/`` (matrix_pooled.csv, matrix_comparisons.csv, matrix_ablations.csv,
matrix_localization.csv, matrix_summary.json) and writes the figures to ``<that folder>/figures/`` unless
``--figures-dir`` is given. A figure whose table has no rows (no ablation was run yet) is skipped and said so.
See dental_agent/evaluation/figures.py for what each figure shows.
"""

import argparse
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.evaluation import figures  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Draw the evaluation-matrix figures from the aggregate tables.")
    p.add_argument("--output-dir", type=str, default="data/evaluations", help="Folder with the eval_*.jsonl files and matrix_<dataset>_<split>/.")
    p.add_argument("--dataset", type=str, default="dentex")
    p.add_argument("--split", type=str, default="test", choices=["test", "dev"])
    p.add_argument("--figures-dir", type=str, default=None, help="Where to write the figures (default: <matrix folder>/figures).")
    p.add_argument("--formats", type=str, nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"], help="File formats to write.")
    p.add_argument("--levels", type=str, nargs="+", default=list(figures.LEVELS), choices=list(figures.LEVELS),
                   help="exact = right tooth and diagnosis, fdi = right tooth.")
    p.add_argument("--dpi", type=int, default=300, help="Resolution of the png files.")
    args = p.parse_args()

    matrix_dir = Path(args.output_dir) / f"matrix_{args.dataset}_{args.split}"
    if not (matrix_dir / figures.POOLED_CSV).is_file():
        raise SystemExit(f"{matrix_dir / figures.POOLED_CSV} not found. Run scripts/aggregate_eval.py for this dataset and split first.")
    written = figures.make_all(matrix_dir, args.figures_dir, formats=args.formats, levels=args.levels, dpi=args.dpi)
    if not written:
        raise SystemExit(f"No figure had data in {matrix_dir}.")
    for name, paths in written.items():
        print(f"{name}: " + ", ".join(str(x) for x in paths))
    skipped = [n for n in figures.figure_names(args.levels) if n not in written]
    if skipped:
        print("skipped (no rows in the table): " + ", ".join(skipped))


if __name__ == "__main__":
    main()
