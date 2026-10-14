#!/usr/bin/env python3
"""Draw the figures of the GRPO group-size (K) sweep from the tables ``aggregate_k_sweep.py`` wrote (no model needed).

    python scripts/aggregate_k_sweep.py --sweep-dir data/evaluations/k_sweep --reference-dir data/evaluations --log-template "..."
    python scripts/plot_k_sweep.py --sweep-dir data/evaluations/k_sweep

Reads ``<sweep-dir>/ksweep_<dataset>_<split>_<condition>_<size>/`` and writes ``k_sweep_exact``, ``k_sweep_fdi`` and ``k_sweep_training`` to its
``figures/`` folder unless ``--figures-dir`` is given (the training figure needs ``--log-template`` at aggregation time).
See dental_agent/evaluation/figures.py.
"""

import argparse
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.evaluation import figures, k_sweep, matrix  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Draw the GRPO group-size sweep figures from the aggregate tables.")
    p.add_argument("--sweep-dir", type=str, required=True, help="The folder given to aggregate_k_sweep.py.")
    p.add_argument("--dataset", type=str, default="dentex")
    p.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    p.add_argument("--condition", type=str, default="grpo_with_tools", choices=list(matrix.CONDITIONS), help="As given to aggregate_k_sweep.py.")
    p.add_argument("--size", type=str, default="canonical", choices=list(matrix.SIZES), help="As given to aggregate_k_sweep.py.")
    p.add_argument("--k-dir", type=str, default=None, help="The table folder when aggregate_k_sweep.py got --out-dir.")
    p.add_argument("--figures-dir", type=str, default=None, help="Where to write the figures (default: <table folder>/figures).")
    p.add_argument("--formats", type=str, nargs="+", default=["png", "pdf"], choices=["png", "pdf", "svg"])
    p.add_argument("--levels", type=str, nargs="+", default=list(figures.LEVELS), choices=list(figures.LEVELS),
                   help="exact = right tooth and diagnosis, fdi = right tooth.")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args(argv)

    k_dir = Path(args.k_dir) if args.k_dir else Path(args.sweep_dir) / k_sweep.out_dir_name(args.dataset, args.split, args.condition, args.size)
    if not (k_dir / figures.K_TABLE_CSV).is_file():
        raise SystemExit(f"{k_dir / figures.K_TABLE_CSV} not found. Run scripts/aggregate_k_sweep.py first.")
    written = figures.make_k_sweep_figures(k_dir, args.figures_dir, formats=args.formats, levels=args.levels, dpi=args.dpi)
    for name, paths in written.items():
        print(f"{name}: " + ", ".join(str(x) for x in paths))
    skipped = [n for n in figures.k_sweep_figure_names(args.levels) if n not in written]
    if skipped:
        print("skipped (no rows in the table): " + ", ".join(skipped))


if __name__ == "__main__":
    main()
