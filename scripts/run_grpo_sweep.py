#!/usr/bin/env python3
"""
Automated Sweep Orchestrator across Group Sizes K in {1, 2, 4, 8, 16} (§17).

Executes sequential comparative experiments across group sizes, tracking:
- Sample Efficiency vs Diagnostic Accuracy
- Hardware tok/s and Rollout Latency
- Mean Reward and Policy KL Divergence
- Generates paper-ready summary tables in JSONL and Markdown format
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.config import load_config
from dental_agent.data.dentex import load_dentex_dataset
from dental_agent.data.dev_split import load_dev_keys, select_training_pool
from dental_agent.data.tufts import load_tufts_dataset
from dental_agent.training.grpo import train_grpo
from dental_agent.utils.canonical import SLOT_BUDGET, add_letterbox_arg, check_gpu_image_policy, parse_slot_budget, set_letterbox
from scripts.run_grpo import ensure_pool_images, resolve_grpo_inputs, resolve_sft_reference, sft_reference_candidates


from dental_agent.training.experiment_profiles import grpo_name, can_resume

def parse_args():
    parser = argparse.ArgumentParser(description="VLM-DENTAL: GRPO K={1, 2, 4, 8, 16} Sweep Orchestrator")
    parser.add_argument(
        "--track",
        type=str,
        default="with_tools",
        choices=["with_tools", "no_tools"],
        help="Training track: 'with_tools' or 'no_tools'",
    )
    parser.add_argument(
        "--sft-stage",
        type=str,
        default="dentex_alone",
        choices=["dentex_alone", "dentex_tufts_overlap", "multicohort_all"],
        help="Curriculum SFT reference stage: 'dentex_alone', 'dentex_tufts_overlap', or 'multicohort_all'",
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 16],
        help="List of group sizes to evaluate in sweep",
    )
    parser.add_argument("--epochs", "-e", type=int, default=1,
                        help="Optimization epochs per batch (run_grpo.py defaults to 2: pass the same value to both so a K of the "
                             "sweep equals a single run of that K).")
    parser.add_argument("--lr", type=float, default=5e-6, help="Policy gradient learning rate")
    parser.add_argument("--kl-beta", type=float, default=0.04, help="Schulman k3 KL penalty weight (same as run_grpo.py).")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clipping epsilon (same as run_grpo.py).")
    parser.add_argument("--push-every-steps", type=int, default=25, help="Checkpoint upload/save frequency in steps.")
    parser.add_argument("--sft-model-dir", type=str, default=None,
                        help="SFT adapter folder for the frozen reference (same as run_grpo.py; every K uses it). Without it the "
                             "folder is searched like run_grpo.py does, including the _<precision> suffix of --precision.")
    parser.add_argument("--checkpoint-dir", type=str, default="data/models",
                        help="Parent folder of the per-K checkpoint folders qwen3_5_9b_grpo_<track>_k<K>_<stage> "
                             "(run_grpo.py calls this --output-dir).")
    parser.add_argument("--hf-run-tag", default=None, help="Separate upload/resume experiment namespace")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False,
                        help="Continue an interrupted sweep: a K whose final checkpoint exists is skipped without loading the model; "
                             "an interrupted K continues from its newest step checkpoint (same pool, seed and K needed).")
    parser.add_argument("--resume-hf", type=str, default=None,
                        help="Hub repo to download each K's newest step checkpoint from first (fresh session). Needs --resume.")
    parser.add_argument("--preflight", action="store_true",
                        help="Run the run_grpo.py preflight for every K of this exact command and exit without loading the model.")
    parser.add_argument("--output-dir", type=str, default="data/eval_results", help="Directory for sweep summary logs")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=os.environ.get("HF_ARTIFACT_REPO", "Reza-Nadimi/vlm-dental-models"),
        help="Hugging Face Hub repo for checkpoint sync (default: Reza-Nadimi/vlm-dental-models)",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=1,
        help="Number of TPU cores for distributed data-parallel execution (1 for single core/GPU, 8 for Kaggle TPU v5e-8)",
    )
    parser.add_argument(
        "--fsdp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable PyTorch/XLA FSDP parameter sharding across TPU cores to fit 9B BF16 model within 16 GB HBM (default: True on multi-core TPU)",
    )
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature of the GRPO rollouts (default 0.7; must be > 0).")
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "qlora"],
                        help="bf16 = LoRA on bf16 weights; qlora = LoRA on 4-bit weights (CUDA only). Never chosen for you.")
    parser.add_argument("--model-id", type=str, default=None,
                        help="Base model path/repo (same as run_grpo.py; the notebook passes it in sweep mode).")
    parser.add_argument("--spmd", action=argparse.BooleanOptionalAction, default=True,
                        help="Single-process SPMD (default on); --no-spmd = legacy path.")
    parser.add_argument("--pad-vision-to-slots", action=argparse.BooleanOptionalAction, default=False,
                        help="Static-shape policy update (see run_grpo.py).")
    parser.add_argument("--max-seq-len", type=int, default=16384,
                        help="Maximum update sequence length (default: 16384; use the SFT value).")
    parser.add_argument("--vision-slots", type=int, nargs=3, metavar=("FULL", "CROP", "COMPARE"),
                        default=[SLOT_BUDGET["FULL"], SLOT_BUDGET["CROP"], SLOT_BUDGET["COMPARE"]],
                        help="Static slot budget (must equal the SFT run's).")
    parser.add_argument("--triangular-shim", action=argparse.BooleanOptionalAction, default=True,
                        help="Matmul replacement for solve_triangular on XLA.")
    parser.add_argument(
        "--canonical-resize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Roll out with canonical views; must match the SFT stage's --canonical-resize setting.",
    )
    from dental_agent.model.local_runtime import add_runtime_args
    add_runtime_args(parser,scoring=True)
    parser.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    add_letterbox_arg(parser)
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Token cap of ONE assistant turn during rollouts (same as run_grpo.py).")
    parser.add_argument("--dev-split", type=str, default=None,
                        help="Dev split JSON (scripts/make_dev_split.py); its images are held out of every K run.")
    parser.add_argument("--max-images", type=int, default=None,
                        help="Train every K on the same seeded subset of this many images (default: whole pool).")
    parser.add_argument("--seed", type=int, default=42, help="Seed of the image-pool shuffle.")
    parser.add_argument("--include-healthy", action=argparse.BooleanOptionalAction, default=True,
                        help="Keep healthy images (ground truth []) in the pool.")
    from dental_agent.harness.configuration import HARNESS_MODES
    parser.add_argument("--harness-mode", choices=HARNESS_MODES, default="off",
                        help="Rules/skills during GRPO rollouts; off preserves the original control")
    return parser.parse_args()


def _preflight_all(args) -> int:
    """The run_grpo.py preflight for every K of this command (nothing is loaded or changed). Returns the number of FAILs."""
    import argparse as _ap
    from dental_agent.training.preflight import run_preflight
    candidates = sft_reference_candidates(args.track, args.sft_stage, args.precision)
    fails = 0
    for k in args.k_values:
        ns = _ap.Namespace(**vars(args))
        ns.group_size, ns.dataset, ns.config, ns.output_dir = k, "dentex", None, args.checkpoint_dir
        fails += run_preflight(ns, candidates)
    print(f"K-sweep preflight: {fails} FAIL over K = {args.k_values}.")
    return fails


def main():
    args = parse_args()
    from dental_agent.harness.configuration import validate_harness_mode
    validate_harness_mode(args.harness_mode, args.track)
    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0 (greedy rollouts give identical samples and zero advantages).")
    if args.pad_vision_to_slots and not args.canonical_resize:
        raise SystemExit("--pad-vision-to-slots requires --canonical-resize.")
    if args.pad_vision_to_slots and not args.canonical_letterbox:
        raise SystemExit("--pad-vision-to-slots requires --canonical-letterbox (static slots assume the fixed canonical canvas).")
    set_letterbox(args.canonical_letterbox)
    if not args.preflight:   # the doctor reports the same rule as a FAIL line instead of stopping
        import importlib.util
        import torch
        check_gpu_image_policy(args.canonical_resize, args.canonical_letterbox, cuda=torch.cuda.is_available(),
                               tpu=importlib.util.find_spec("torch_xla") is not None, what="the GRPO K-sweep")
    if args.resume_hf and not args.resume:
        raise SystemExit("--resume-hf only downloads checkpoints; add --resume to continue the sweep from them.")
    if args.preflight:
        sys.exit(1 if _preflight_all(args) else 0)
    if args.spmd:
        try:
            import torch_xla.runtime as xr
            if hasattr(xr, "use_spmd"):
                xr.use_spmd()  # before any device is initialised
            os.environ["XLA_USE_SPMD"] = "1"
        except ImportError:
            pass
    if args.triangular_shim:
        from dental_agent.training.xla_patches import install_xla_solve_triangular_shim
        install_xla_solve_triangular_shim()
    cfg = load_config()
    if args.model_id:
        cfg.model.name = args.model_id
    cfg.model.load_in_4bit = args.precision == "qlora"

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    sweep_log = out_path / "grpo_k_sweep_results.jsonl"

    print("======================================================================")
    print("VLM-DENTAL: GRPO K-SWEEP ORCHESTRATOR")
    print(f"* Track        : {args.track}")
    print(f"* SFT Stage    : {args.sft_stage}")
    print(f"* K Values     : {args.k_values}")
    print(f"* HF Repo Sync : {args.hf_repo}")
    print(f"* Sweep Log    : {sweep_log}")
    print("======================================================================")

    # Load dataset
    # TRAIN images only (the default split of load_dentex_dataset is the official validation = test set).
    # Every input is found (or fetched) once, before the first K loads the 9B model.
    sft_dir = resolve_grpo_inputs(args, cfg.data_dir)
    images_df, annots_df, categories_df = load_dentex_dataset(cfg.data_dir, split_name="train")
    images_df = ensure_pool_images(images_df, "dentex", cfg.data_dir)
    images_df = select_training_pool(
        images_df, annots_df, dataset="dentex", dev_keys=load_dev_keys(args.dev_split),
        include_healthy=args.include_healthy, max_images=args.max_images, seed=args.seed,
    )
    print(f"* Training pool: {len(images_df)} images (dev split: {args.dev_split or 'none'})")

    results_table = []

    # sft_dir was resolved above with the same order as run_grpo.py: --sft-model-dir, local folders (_<precision> first), Kaggle, hub.

    for k in args.k_values:
        print(f"\n>>> Starting GRPO Experiment for Group Size K={k} <<<")
        start_time = time.time()
        target_name = grpo_name(args.track, k, args.sft_stage, args.precision, args.hf_run_tag, harness_mode=args.harness_mode)
        ckpt_dir = f"{args.checkpoint_dir}/{target_name}"
        path_in_repo_prefix = f"grpo/{target_name}"
        final_meta = Path(ckpt_dir) / f"grpo-{args.track}-final" / "checkpoint_meta.json"
        if args.resume and final_meta.is_file():
            from dental_agent.harness.configuration import assert_harness_identity
            assert_harness_identity(json.loads(final_meta.read_text()).get("extra", {}).get("harness"), args.harness_mode)
            print(f">>> K={k} is already complete ({final_meta.parent}); skipped. <<<")
            continue
        if args.resume_hf:
            try:
                from dental_agent.training.grpo_resume import fetch_latest_checkpoint_from_hub
                tag = fetch_latest_checkpoint_from_hub(
                    args.resume_hf, path_in_repo_prefix, args.track, Path(ckpt_dir), Path(args.checkpoint_dir) / "_hf_resume")
                print(f"[RESUME] K={k}: downloaded {tag}." if tag else f"[RESUME] K={k}: the Hub holds no step checkpoint.")
            except Exception as e:
                print(f"[RESUME WARNING] K={k}: could not restore from the Hub: {e}")

        from dental_agent.model.local_runtime import runtime_from_args
        final_ckpt = train_grpo(
            images_df=images_df,
            annots_df=annots_df,
            categories_df=categories_df,
            config=cfg,
            sft_model_dir=sft_dir,
            checkpoint_dir=ckpt_dir,
            group_size=k,
            epochs_per_batch=args.epochs,
            kl_beta=args.kl_beta,
            clip_eps=args.clip_eps,
            learning_rate=args.lr,
            track=args.track,
            hf_repo=args.hf_repo,
            push_every_steps=args.push_every_steps,
            resume=args.resume and can_resume(ckpt_dir, args.track),
            path_in_repo_prefix=path_in_repo_prefix,
            num_cores=args.num_cores,
            use_fsdp=args.fsdp,
            canonical_resize=args.canonical_resize,
            temperature=args.temperature,
            use_spmd=args.spmd,
            max_seq_len=args.max_seq_len,
            pad_vision_to_slots=args.pad_vision_to_slots,
            slot_budget=parse_slot_budget(*args.vision_slots),
            max_new_tokens=args.max_new_tokens,
            log_path=f"{ckpt_dir}/grpo_training_log.jsonl",  # one log per K
            runtime_options=runtime_from_args(args),
            gradient_checkpointing=args.gradient_checkpointing,
            harness_mode=args.harness_mode, seed=args.seed,
        )

        elapsed = time.time() - start_time
        print(f">>> Completed K={k} in {elapsed:.1f}s. Checkpoint: {final_ckpt} <<<\n")

        k_result = {
            "group_size": k,
            "track": args.track,
            "sft_stage": args.sft_stage,
            "elapsed_seconds": round(elapsed, 2),
            "checkpoint": str(final_ckpt),
            "log_path": f"{ckpt_dir}/grpo_training_log.jsonl",
            "kl_beta": args.kl_beta,
            "clip_eps": args.clip_eps,
            "epochs": args.epochs,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        results_table.append(k_result)
        with open(sweep_log, "a", encoding="utf-8") as f:
            f.write(json.dumps(k_result) + "\n")

    print("\n======================================================================")
    print("GRPO K-SWEEP EXPERIMENT SUMMARY")
    print("======================================================================")
    print(f"{'Group Size (K)':<15} | {'Track':<12} | {'Elapsed Time (s)':<18} | {'Checkpoint'}")
    print("-" * 75)
    for r in results_table:
        print(f"{r['group_size']:<15} | {r['track']:<12} | {r['elapsed_seconds']:<18} | {r['checkpoint']}")
    print("======================================================================")
    print(f"Full sweep metrics recorded in {sweep_log}.")


if __name__ == "__main__":
    main()
