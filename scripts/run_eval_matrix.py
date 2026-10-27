#!/usr/bin/env python3
"""Run the evaluation matrix: base, SFT and GRPO checkpoints, each with and without tools, at canonical and original
image size, on one split. Every checkpoint is loaded ONCE and all of its cells run on it.

    python scripts/run_eval_matrix.py --split test --sft-adapter data/models/qwen3_5_9b_sft_with_tools_dentex_alone_qlora \
        --grpo-adapter data/models/qwen3_5_9b_grpo_with_tools_k4_dentex_alone/grpo-with_tools-final \
        --precision qlora --existing-results resume

Cells are the product of --conditions and --sizes; the plan is printed before anything runs. Results go to
``<output-dir>/eval_<dataset>_<split>_<condition>[_canonical].jsonl`` (one record per image, with its transcript) and the
pooled tables to ``<output-dir>/matrix_<dataset>_<split>/`` (see dental_agent/evaluation/matrix.py).

``scripts/evaluate_models.py`` runs ONE cell (one condition at one size); this script runs the grid.
Nothing is chosen for you: every checkpoint, size, budget and precision is a flag, and a requested condition whose
adapter is missing stops the run before the first image.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from scripts import evaluate_models as ev  # noqa: E402
from dental_agent.evaluation import matrix  # noqa: E402
from dental_agent.evaluation.tool_ablation import spec_from_args  # noqa: E402
from dental_agent.utils.canonical import add_letterbox_arg, set_letterbox  # noqa: E402

CHECKPOINTS = ("base", "sft", "grpo")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the evaluation matrix (checkpoints x tools x image size) on one split.")
    p.add_argument("--conditions", nargs="+", default=list(matrix.CONDITIONS), choices=list(matrix.CONDITIONS),
                   help="Conditions to run (default: all six).")
    p.add_argument("--sizes", nargs="+", default=list(matrix.SIZES), choices=list(matrix.SIZES),
                   help="Image sizes: canonical (primary; what the models were trained on) and/or original.")
    p.add_argument("--sft-adapter", type=str, default=None, help="SFT LoRA adapter folder (needed by sft_* conditions).")
    p.add_argument("--grpo-adapter", type=str, default=None,
                   help="GRPO checkpoint folder, e.g. .../grpo-with_tools-final (needed by grpo_* conditions; its trained "
                        "'grpo_policy' adapter is used).")
    p.add_argument("--model-id", type=str, default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B"), help="Base VLM.")
    p.add_argument("--dataset", type=str, default="dentex")
    p.add_argument("--split", type=str, default="test", choices=["test", "dev"],
                   help="test = official DENTEX validation set (final numbers only); dev = held-out train images (needs --dev-split).")
    p.add_argument("--dev-split", type=str, default=None, help="Dev split JSON (scripts/make_dev_split.py).")
    p.add_argument("--data-dir", type=str, default=os.environ.get("DENTAL_AGENT_DATA_DIR", "data"))
    p.add_argument("--output-dir", type=str, default=ev.DEFAULT_OUTPUT_DIR)
    p.add_argument("--limit", type=int, default=None, help="First N images only (smoke test).")
    p.add_argument("--max-turns", type=int, default=None, help="Assistant turns cap (default: --max-tool-calls + 15).")
    p.add_argument("--max-tool-calls", type=int, default=50, help="Tool-call budget per image (the GRPO budget).")
    p.add_argument("--max-new-tokens", type=int, default=2048, help="Token cap of ONE assistant turn (the GRPO value).")
    p.add_argument("--malformed-retries", type=int, default=0)
    p.add_argument("--remove-tools", nargs="+", default=[], metavar="TOOL",
                   help="Ablation: withhold these tools (e.g. nudge_crop). Only with_tools cells run; results go to files "
                        "tagged with the ablation, so the unablated results are untouched.")
    p.add_argument("--locate-shift-frac", type=float, default=0.0,
                   help="Ablation: move every locate_tooth box by this fraction of its own size (fixed pseudo-random direction).")
    p.add_argument("--locate-scale", type=float, default=1.0, help="Ablation: rescale every locate_tooth box about its centre.")
    p.add_argument("--ablation-seed", type=int, default=0, help="Seed of the perturbation direction.")
    p.add_argument("--iou-threshold", type=float, default=0.5, help="IoU at which a box counts as hitting the annotated tooth.")
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32", "qlora"])
    p.add_argument("--existing-results", type=str, default="error", choices=["error", "resume", "overwrite"],
                   help="When a cell's result file already has records: error (default), resume (finished images are "
                        "kept, settings must match) or overwrite.")
    p.add_argument("--n-boot", type=int, default=2000, help="Bootstrap resamples for the intervals.")
    p.add_argument("--seed", type=int, default=0, help="Bootstrap seed.")
    from dental_agent.model.local_runtime import add_runtime_args
    add_runtime_args(p)
    add_letterbox_arg(p)   # applies to the 'canonical' size cells only; 'original' cells show the full image
    return p.parse_args(argv)


def checkpoint_of(condition: str) -> str:
    return condition.split("_", 1)[0]


def adapter_for(checkpoint: str, args: argparse.Namespace) -> str | None:
    return {"base": None, "sft": args.sft_adapter, "grpo": args.grpo_adapter}[checkpoint]


def plan_cells(args: argparse.Namespace) -> list[tuple[str, list[tuple[str, str]]]]:
    """[(checkpoint, [(condition, size), ...]), ...] in run order; a missing adapter stops here, before any work."""
    groups = []
    for ck in CHECKPOINTS:
        conds = [c for c in matrix.CONDITIONS if c in args.conditions and checkpoint_of(c) == ck]
        if spec_from_args(args).active:
            conds = [c for c in conds if c.endswith("with_tools")]  # a tool ablation has nothing to change without tools
        if not conds:
            continue
        if ck != "base" and not adapter_for(ck, args):
            raise SystemExit(f"{', '.join(conds)} requested but --{ck}-adapter was not given.")
        groups.append((ck, [(c, size) for size in args.sizes for c in conds]))
    return groups


def cell_args(args: argparse.Namespace, condition: str, size: str) -> argparse.Namespace:
    ns = argparse.Namespace(**vars(args))
    ns.condition = condition
    ns.canonical_resize = size == "canonical"
    ns.adapter_path = adapter_for(checkpoint_of(condition), args)
    ns.reuse_cached = False  # the matrix is live runs under one prompt; the cached zero-shot file used another
    return ns


def run_matrix(args: argparse.Namespace, imgs_df, annots_df, cats_df) -> dict[str, list]:
    groups = plan_cells(args)
    tag = spec_from_args(args).tag()
    n_eligible = len(ev.eligible_images(imgs_df, args.limit))
    print(f"[MATRIX] {sum(len(c) for _, c in groups)} cell(s) on {n_eligible} {args.split} image(s):")
    for _, cells in groups:
        for cond, size in cells:
            path = matrix.cell_path(args.output_dir, args.dataset, args.split, cond, size, tag)
            print(f"  - {matrix.cell_name(cond, size):<32} {path}")

    if args.existing_results == "error":
        # Refuse before the first model load, not after the first cells have already run.
        existing = [
            str(matrix.cell_path(args.output_dir, args.dataset, args.split, c, z, tag))
            for _, cells in groups for c, z in cells
            if matrix.cell_path(args.output_dir, args.dataset, args.split, c, z, tag).is_file()
            and matrix.cell_path(args.output_dir, args.dataset, args.split, c, z, tag).stat().st_size > 0
        ]
        if existing:
            raise SystemExit("These result files already have records:\n  " + "\n  ".join(existing) +
                             "\nAdd --existing-results resume to continue them, or --existing-results overwrite to start again.")

    results: dict[str, list] = {}
    for ck, cells in groups:
        adapter = adapter_for(ck, args)
        loaded = None
        for cond, size in cells:
            ns = cell_args(args, cond, size)
            path = matrix.cell_path(args.output_dir, args.dataset, args.split, cond, size, tag)
            if args.existing_results == "resume":
                # a finished cell must not cost a model load; its settings are still verified
                done = ev.prepare_output_file(path, "resume", ev.eval_settings(ns, cond, adapter))
                if n_eligible and len(done) >= n_eligible:
                    print(f"[MATRIX] {matrix.cell_name(cond, size)} is complete ({len(done)} images); skipped.")
                    results[matrix.cell_name(cond, size)] = list(done.values())
                    continue
            if loaded is None:
                device, dtype, _ = ev.setup_eval_hardware(args.precision, getattr(args,"qlora_compute_dtype","auto"))
                from dental_agent.model.local_runtime import runtime_from_args
                loaded = ev.load_eval_model(args.model_id, adapter, device, dtype, cond, quantize_4bit=args.precision == "qlora",
                                            runtime_options=runtime_from_args(args))
            results[matrix.cell_name(cond, size)] = ev.evaluate_condition(cond, ns, imgs_df, annots_df, cats_df, loaded=loaded)
        if loaded is not None:
            del loaded
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
    return results


def write_tables(args: argparse.Namespace) -> Path:
    summary = matrix.build_full_summary(args.output_dir, args.dataset, args.split, n_boot=args.n_boot, seed=args.seed,
                                        iou_threshold=args.iou_threshold)
    out_dir = Path(args.output_dir) / f"matrix_{args.dataset}_{args.split}"
    paths = matrix.write_matrix_outputs(summary, out_dir)
    print(f"[MATRIX] Tables written to {out_dir} ({paths['markdown'].name}, matrix_pooled.csv, matrix_per_class.csv, matrix_comparisons.csv).")
    return out_dir


def main() -> None:
    args = parse_args()
    ev.load_env()
    set_letterbox(args.canonical_letterbox)
    plan_cells(args)  # fail on a missing adapter before the dataset is touched
    imgs_df, annots_df, cats_df = ev.load_eval_split(args)
    print(f"[DATA] {len(imgs_df)} {args.split} images, {len(annots_df)} annotations.")
    run_matrix(args, imgs_df, annots_df, cats_df)
    write_tables(args)


if __name__ == "__main__":
    main()
