#!/usr/bin/env python3
"""Tables of the GRPO group-size (K) sweep on the dev split (no model needed).

    # 1. one evaluation per K (the notebook loops this), each into its own folder:
    python scripts/run_eval_matrix.py --split dev --dev-split data/splits/dentex_dev.json --conditions grpo_with_tools \
        --sizes canonical --grpo-adapter data/models/qwen3_5_9b_grpo_with_tools_k4_dentex_alone/grpo-with_tools-final \
        --precision qlora --output-dir data/evaluations/k_sweep/k4
    # 2. the SFT reference comes from the ordinary dev matrix in data/evaluations (sft_with_tools, canonical)
    # 3. this script:
    python scripts/aggregate_k_sweep.py --sweep-dir data/evaluations/k_sweep --reference-dir data/evaluations \
        --log-template "data/models/qwen3_5_9b_grpo_with_tools_k{k}_dentex_alone/grpo_training_log.jsonl"

Every K folder ``<sweep-dir>/k<K>`` holds the per-image results ``run_eval_matrix.py`` wrote for the K-th GRPO checkpoint. The
tables go to ``<sweep-dir>/ksweep_<dataset>_<split>_<condition>_<size>/`` (k_sweep_summary.md, k_sweep_table.csv, k_sweep_adjacent.csv,
k_sweep_training.csv, k_sweep_summary.json). Nothing selects a K for you: the table reports which K are not distinguishable
from the best and which of them is cheapest, and the choice stays yours. See dental_agent/evaluation/k_sweep.py.
"""

import argparse
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.evaluation import k_sweep, matrix  # noqa: E402


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Aggregate the GRPO group-size sweep (one result folder per K) into tables.")
    p.add_argument("--sweep-dir", type=str, required=True, help="Folder with one sub-folder k<K> per group size.")
    p.add_argument("--reference-dir", type=str, default=None,
                   help="Folder with the SFT reference results (the ordinary dev matrix). Without it no paired comparison to SFT is made.")
    p.add_argument("--dataset", type=str, default="dentex")
    p.add_argument("--split", type=str, default="dev", choices=["dev", "test"],
                   help="dev for the sweep; test only for the one final run of the chosen K.")
    p.add_argument("--condition", type=str, default="grpo_with_tools", choices=list(matrix.CONDITIONS), help="Condition of every K folder.")
    p.add_argument("--reference-condition", type=str, default="sft_with_tools", choices=list(matrix.CONDITIONS),
                   help="Condition of the reference folder.")
    p.add_argument("--size", type=str, default="canonical", choices=list(matrix.SIZES), help="Image size of both (canonical is primary).")
    p.add_argument("--k-values", type=int, nargs="+", default=None, help="Only these K (default: every k<K> folder).")
    p.add_argument("--log-template", type=str, default=None,
                   help="GRPO step log of each K with {k} for the group size, e.g. data/models/qwen3_5_9b_grpo_with_tools_k{k}_dentex_alone/"
                        "grpo_training_log.jsonl. Adds the training-cost table; a missing log is reported and skipped.")
    p.add_argument("--out-dir", type=str, default=None, help="Where the tables go (default: <sweep-dir>/ksweep_<dataset>_<split>_<condition>_<size>).")
    p.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for the intervals.")
    p.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    args = p.parse_args(argv)

    cells = k_sweep.load_k_cells(args.sweep_dir, args.dataset, args.split, args.condition, args.size, args.k_values)
    if not cells:
        raise SystemExit(f"No eval_{args.dataset}_{args.split}_{args.condition}* result file in any k<K> folder of {args.sweep_dir}.")
    reference = None
    if args.reference_dir:
        reference = k_sweep.load_reference(args.reference_dir, args.dataset, args.split, args.reference_condition, args.size)
    logs = {}
    if args.log_template:
        for k in cells:
            path = Path(args.log_template.format(k=k))
            if path.is_file():
                logs[k] = k_sweep.load_training_log(path)
            else:
                print(f"[K-SWEEP] no training log for K={k}: {path}")
    summary = k_sweep.build_k_sweep(cells, reference, logs, n_boot=args.n_boot, seed=args.seed)
    out_dir = Path(args.out_dir) if args.out_dir else Path(args.sweep_dir) / k_sweep.out_dir_name(args.dataset, args.split, args.condition, args.size)
    paths = k_sweep.write_k_sweep_outputs(summary, out_dir, args.condition, args.reference_condition)
    print(f"K = {summary['ks']} -> {out_dir}")
    print(paths["markdown"].read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
