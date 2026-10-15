#!/usr/bin/env python3
"""
Production CLI to run Stage 2 Group Relative Policy Optimization (GRPO) (§17).

Supports:
- Strict Track Segregation: --track {with_tools, no_tools}
- Flexible Group Size: --group-size K in {1, 2, 4, 8, 16}
- [G2] Batched Rollouts & [G3] Dual-LoRA Reference/Policy Toggle
- Multi-Finding Complete Ground Truth Evaluation (Rule 13)
- Lightweight Hugging Face Checkpoint Sync (~760 MB) & Cross-Account Resume
- SIGTERM Preemption Handler for Kaggle 9-Hour Session Limits
"""

import argparse
import json
import os
import signal
import sys
from pathlib import Path

# Clear conflicting legacy TPU variables and PJRT_DEVICE on Kaggle/Colab
for _var in ["TPU_PROCESS_ADDRESSES", "TPU_PROCESS_COUNT", "CLOUD_TPU_TASK_ID", "PJRT_DEVICE"]:
    os.environ.pop(_var, None)

# NOTE: PJRT_DEVICE is set lazily inside run_worker(), NOT here.
# Setting it before `import torch` causes torch_xla's torch plugin to auto-initialize
# the PJRT TPU client at import time, leading to double-initialization crashes.

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
from dental_agent.utils.canonical import SLOT_BUDGET, parse_slot_budget, slot_totals


def upload_checkpoint_to_hf(checkpoint_dir: Path, hf_repo: str, step: int):
    """Upload lightweight LoRA checkpoint (~760 MB) to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        commit_msg = f"VLM-DENTAL GRPO Checkpoint: Step {step}"
        print(f"[HF-HUB] Uploading GRPO checkpoint from {checkpoint_dir} to {hf_repo}...")
        api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=hf_repo,
            commit_message=commit_msg,
            ignore_patterns=["*.tmp", "*.lock"],
        )
        print(f"[HF-HUB] GRPO checkpoint successfully uploaded to {hf_repo}.")
    except Exception as e:
        print(f"[HF-HUB WARNING] Failed to upload checkpoint to {hf_repo}: {e}")


def sft_reference_candidates(track: str, sft_stage: str, precision: str | None = None) -> list[Path]:
    """Local folders searched for the SFT adapter when --sft-model-dir is not given (also used by --preflight).

    ``train_sft.py`` writes ``data/models/qwen3_5_9b_sft_<track>_<stage>_<precision>``. With ``precision`` (the run's own
    explicit ``--precision``, never guessed) that folder is searched first, so a QLoRA SFT adapter is found by a QLoRA GRPO
    run and a bf16 one by a bf16 run; the other precision's folder is not a candidate. One table for every caller:
    dental_agent/utils/inputs.py."""
    from dental_agent.utils.inputs import sft_adapter_candidates
    return sft_adapter_candidates(track, sft_stage, precision)


def resolve_sft_reference(
    sft_dir_arg: str | None,
    track: str,
    sft_stage: str,
    hf_repo: str | None = None,
    precision: str | None = None,
) -> str:
    """Path of the SFT reference adapter: --sft-model-dir, else the local folders (``_<precision>`` first), Kaggle inputs,
    then the Hub. Raises ``InputNotFound`` (what was searched + the fix) instead of returning a path that does not exist:
    the old fallback let the run load the 9B model and then train a FRESH LoRA, i.e. GRPO without the SFT policy."""
    from dental_agent.utils.inputs import resolve_sft_adapter

    r = resolve_sft_adapter(sft_dir_arg, track, sft_stage, precision, hf_repo)
    print(f"[SFT-REF] Using SFT reference adapter {r.path} ({r.source})")
    return str(r.path)


def resolve_grpo_inputs(args: argparse.Namespace, data_dir: str) -> str:
    """Find every file this GRPO command needs BEFORE the model is loaded; returns the SFT adapter folder.

    Paths only, no setting changes. Stops with ``[INPUT NOT FOUND]`` naming what was searched and the fix."""
    from dental_agent.utils import inputs as inp

    try:
        sft_dir = resolve_sft_reference(args.sft_model_dir, args.track, args.sft_stage, args.hf_repo, args.precision)
        inp.assert_adapter_loadable(sft_dir)
        inp.resolve_base_model(args.model_id)
        if args.track == "with_tools":
            yolo = inp.resolve_yolo_checkpoint(repo=args.hf_repo)
            os.environ["GROUNDING_MODEL_PATH"] = str(Path(yolo.path).resolve())  # a path, so locate_tooth finds it from any cwd
            print(f"[INPUTS] YOLO grounding checkpoint {yolo.path} ({yolo.source})")
        dev = inp.resolve_dev_split(args.dev_split, data_dir)
        if dev is not None:
            args.dev_split = str(dev)
    except inp.InputNotFound as e:
        raise SystemExit(f"[INPUT NOT FOUND] {e}") from e
    return sft_dir


def ensure_pool_images(images_df, dataset: str, data_dir: str):
    """Give every training row a file (search, then fetch missing images from the Hub in one call per dataset). Rows without
    a file used to be dropped from the pool silently; now any that stay missing stop the run before the model loads."""
    from dental_agent.utils import inputs as inp

    images_df, chk = inp.ensure_dataset_images(images_df, dataset, data_dir, split="train")
    if chk.fetched:
        print(f"[INPUTS] {chk.fetched} images fetched from the Hub.")
    if chk.n_missing:
        raise SystemExit("[INPUT NOT FOUND] " + str(inp.InputNotFound(
            f"Training images ({chk.describe()})", [f"data roots under {data_dir}", "attached Kaggle inputs", "local Hugging Face cache",
                                                    f"Hub dataset {inp.images_repo(dataset)}"],
            "python scripts/fetch_hf_assets.py --images   (HF_TOKEN set), or attach the Kaggle image dataset")))
    return images_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Stage 2 GRPO Policy Trainer")
    parser.add_argument(
        "--track",
        type=str,
        default="with_tools",
        choices=["with_tools", "no_tools"],
        help="Strict training track: 'with_tools' (multi-turn agent) or 'no_tools' (direct radiologist)",
    )
    parser.add_argument(
        "--sft-stage",
        type=str,
        default="dentex_alone",
        choices=["dentex_alone", "dentex_tufts_overlap", "multicohort_all"],
        help="Curriculum SFT reference stage: 'dentex_alone', 'dentex_tufts_overlap', or 'multicohort_all'",
    )
    parser.add_argument("--config", "-c", default=None, help="Path to config YAML")
    parser.add_argument("--dataset", default="dentex", help="Dataset name ('dentex' or 'tufts')")
    parser.add_argument("--group-size", "-g", type=int, default=4, help="GRPO group size K (1, 2, 4, 8, 16)")
    parser.add_argument("--epochs", "-e", type=int, default=2, help="Optimization epochs per rollout batch")
    parser.add_argument("--lr", type=float, default=5e-6, help="Learning rate for policy gradients")
    parser.add_argument("--kl-beta", type=float, default=0.04, help="Schulman k3 KL divergence penalty weight")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument(
        "--sft-model-dir",
        type=str,
        default=None,
        help="Path to SFT adapter for reference policy (auto-defaults per track & sft-stage)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B"),
        help="Base VLM model identifier or local directory (e.g. /kaggle/input/qwen3-5-9b)",
    )
    parser.add_argument("--output-dir", type=str, default="data/models", help="Directory to save RL checkpoints")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=os.environ.get("HF_ARTIFACT_REPO", "Reza-Nadimi/vlm-dental-models"),
        help="Hugging Face Hub repository for checkpoint sync (default: Reza-Nadimi/vlm-dental-models)",
    )
    parser.add_argument("--push-every-steps", type=int, default=25, help="Frequency of HF checkpoint upload in steps")
    parser.add_argument(
        "--resume-hf",
        type=str,
        default=None,
        help="Hugging Face repo to download the NEWEST step checkpoint of this run from (a fresh Kaggle/Colab session). "
             "Only downloads; add --resume to continue from it.",
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Continue from the newest grpo-<track>-step-N checkpoint under the output folder: policy adapter, step and EMA "
             "baseline are restored and the images already trained on are skipped. Needs the same --dev-split, --seed, "
             "--max-images, --include-healthy and --group-size. Adam moments restart cold.",
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
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature of the GRPO rollouts (default 0.7). Must be > 0: greedy rollouts of the same image are "
             "identical, so every advantage would be zero.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "qlora"],
        help="bf16 = LoRA on bf16 weights (A100-class GPUs). qlora = LoRA on 4-bit NF4 weights (24 GB GPUs such as "
             "RTX 4090). CUDA only; nothing selects it for you. (Previously 4-bit came implicitly from "
             "configs/default.yaml model.load_in_4bit.)",
    )
    parser.add_argument(
        "--spmd",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Single-process SPMD (FSDPv2 over one device mesh). Default: on. --no-spmd selects legacy xmp.spawn, "
             "which loads one full model copy per process and exhausts host RAM on v5e-8.",
    )
    parser.add_argument(
        "--pad-vision-to-slots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Run every policy-gradient forward at the SFT static shapes (padded to --max-seq-len and --vision-slots; "
             "one rollout per core) so XLA compiles ONE update graph. Requires --canonical-resize and --max-seq-len.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=16384,
        help="Maximum sequence length of the policy update (default: 16384; use the SFT value). With "
             "--pad-vision-to-slots every update forward is padded to it. Longer rollouts are excluded and reported.",
    )
    parser.add_argument(
        "--vision-slots",
        type=int,
        nargs=3,
        metavar=("FULL", "CROP", "COMPARE"),
        default=[SLOT_BUDGET["FULL"], SLOT_BUDGET["CROP"], SLOT_BUDGET["COMPARE"]],
        help="Static slot budget (must equal the SFT run's --vision-slots).",
    )
    parser.add_argument(
        "--triangular-shim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Matmul replacement for torch.linalg.solve_triangular on XLA (--no-triangular-shim = native solver).",
    )
    parser.add_argument(
        "--canonical-resize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Roll out with aspect-preserving canonical views (FULL 1536x768 / CROP 256x384 / COMPARE 512x384). MUST "
             "match the SFT run that produced --sft-stage (train_sft.py --canonical-resize), otherwise the policy is "
             "optimised on a different image distribution than it was fine-tuned on.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Token cap of ONE assistant turn during rollouts (the SFT traces contain turns up to ~2,700 tokens; the old "
             "implicit 512 truncated about 2% of them). Use the same value as evaluation.",
    )
    parser.add_argument(
        "--dev-split",
        type=str,
        default=None,
        help="Dev split JSON (scripts/make_dev_split.py). Its images are held out of GRPO training (same file as SFT).",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Train on a seeded random subset of this many images (default: the whole pool). One GRPO step = one image.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed of the image-pool shuffle (also fixes which --max-images are drawn).")
    parser.add_argument(
        "--include-healthy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep healthy images (no annotations, ground truth []) in the pool so the policy learns to answer 'no findings'.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="data/grpo_training_log.jsonl",
        help="JSONL file that receives one record per GRPO step (reward, KL, rollout and update seconds, turns, tool "
             "calls). Give every run its own file (a probe run, each K of a sweep) so logs do not mix.",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Check this exact command (SFT adapter, dev split, pool, image files, resume state, hub token, hardware) "
             "and exit without loading the model. Run it first; every FAIL would have stopped or corrupted the run.",
    )
    args = parser.parse_args()

    if args.preflight:
        from dental_agent.training.preflight import run_preflight
        sys.exit(1 if run_preflight(args, sft_reference_candidates(args.track, args.sft_stage, args.precision)) else 0)

    is_tpu = False
    try:
        import torch_xla  # noqa: F401 — lightweight check; do NOT import xla_model here
        is_tpu = True     # (importing xla_model initializes libtpu and claims /dev/vfio exclusively)
    except ImportError:
        pass

    if is_tpu and args.num_cores > 1 and not args.spmd:
        try:
            try:
                import torch_xla.distributed.xla_multiprocessing as xmp
            except ImportError:
                import torch_xla.distributed.xmp as xmp
            print(f"[LAUNCH] Spawning multi-core Cloud TPU v5e-8 GRPO across available TPU cores via xmp.spawn(nprocs=None)...")
            xmp.spawn(run_worker, args=(args,), nprocs=None)
        except Exception as e:
            print(f"\n[FATAL TPU ERROR] Could not spawn multi-core GRPO via xmp: {e}")
            print("[DIAGNOSTIC] On Cloud TPU VMs, hardware access to /dev/vfio/* is exclusive.")
            print("[DIAGNOSTIC] If running from Jupyter/Colab/Kaggle, ensure the notebook kernel did not call xm.xla_device() before launching this script.")
            print("[DIAGNOSTIC] Please restart the notebook session (Run -> Restart Session) and re-run to release /dev/vfio/*.\n")
            raise RuntimeError(f"Multi-core Cloud TPU GRPO launch failed: {e}") from e
    else:
        run_worker(0, args)


def run_worker(index: int, args: argparse.Namespace):
    """Per-device worker routine for GRPO training."""
    cfg = load_config(args.config)
    if args.model_id:
        cfg.model.name = args.model_id
    cfg.model.load_in_4bit = args.precision == "qlora"  # explicit; the config file value is overridden

    is_tpu = False
    try:
        if "PJRT_DEVICE" not in os.environ:
            os.environ["PJRT_DEVICE"] = "TPU"
        if args.spmd:
            import torch_xla.runtime as xr
            if hasattr(xr, "use_spmd"):
                xr.use_spmd()  # must precede the first device initialisation
            os.environ["XLA_USE_SPMD"] = "1"
        import torch_xla.core.xla_model as xm
        is_tpu = True
        device = xm.xla_device()
        is_master = xm.is_master_ordinal()
    except Exception:
        is_master = True

    if args.temperature <= 0:
        raise SystemExit("--temperature must be > 0 (greedy rollouts give identical samples and zero advantages).")
    if args.resume_hf and not args.resume:
        raise SystemExit("--resume-hf only downloads the newest checkpoint; add --resume to continue training from it.")
    if args.precision == "qlora" and is_tpu:
        raise SystemExit("--precision qlora is not supported on TPU/XLA (bitsandbytes is CUDA-only). Use --precision bf16.")
    if args.pad_vision_to_slots and not is_tpu:
        raise SystemExit("--pad-vision-to-slots is TPU-only (static shapes avoid XLA recompilation). GPU/CPU runs pad dynamically; remove the flag.")
    if args.pad_vision_to_slots and not args.canonical_resize:
        raise SystemExit("--pad-vision-to-slots requires --canonical-resize.")
    slot_budget = parse_slot_budget(*args.vision_slots)
    if args.pad_vision_to_slots and args.max_seq_len <= slot_totals(slot_budget)["tokens"]:
        raise SystemExit(f"--max-seq-len {args.max_seq_len} cannot hold the static vision tokens of --vision-slots.")
    if args.triangular_shim:
        from dental_agent.training.xla_patches import install_xla_solve_triangular_shim
        install_xla_solve_triangular_shim()

    # Every input is found (or fetched) before the 9B model is touched: SFT adapter (incl. the _<precision> folder, Kaggle
    # inputs, Hub), base model, YOLO checkpoint, dev split. A missing one stops here, never as a fresh-LoRA run.
    resolved_sft_dir = resolve_grpo_inputs(args, cfg.data_dir)

    # Checkpoint directory naming: qwen3_5_9b_grpo_{track}_k{group_size}_{sft_stage}
    target_name = f"qwen3_5_9b_grpo_{args.track}_k{args.group_size}_{args.sft_stage}"
    out_dir = Path(args.output_dir) / target_name
    out_dir.mkdir(parents=True, exist_ok=True)
    path_in_repo_prefix = f"grpo/{target_name}"

    # Download the newest step checkpoint of this run from the Hub (Kaggle/Colab sessions do not persist).
    if args.resume_hf and is_master:
        print(f"[RESUME] Checking HF Hub for the newest checkpoint in {args.resume_hf}/{path_in_repo_prefix}...")
        try:
            from dental_agent.training.grpo_resume import fetch_latest_checkpoint_from_hub
            tag = fetch_latest_checkpoint_from_hub(
                args.resume_hf, path_in_repo_prefix, args.track, out_dir, Path(args.output_dir) / "_hf_resume",
            )
            print(f"[RESUME] Downloaded {tag} to {out_dir}." if tag else "[RESUME] The Hub holds no step checkpoint for this run.")
        except Exception as e:
            print(f"[RESUME WARNING] Could not restore from HF Hub: {e}")

    dataset_name = args.dataset.strip().lower()
    if dataset_name == "tufts":
        images_df, annots_df, categories_df = load_tufts_dataset(cfg.data_dir)
    else:
        # TRAIN images only. The default split of load_dentex_dataset is the official validation set = the test set.
        images_df, annots_df, categories_df = load_dentex_dataset(cfg.data_dir, split_name="train")
    images_df = ensure_pool_images(images_df, dataset_name, cfg.data_dir)
    images_df = select_training_pool(
        images_df,
        annots_df,
        dataset=dataset_name,
        dev_keys=load_dev_keys(args.dev_split),
        include_healthy=args.include_healthy,
        max_images=args.max_images,
        seed=args.seed,
    )

    # Legacy xmp.spawn only: each process owns a shard of the images. SPMD is one process driving every core.
    if is_tpu and args.num_cores > 1 and not args.spmd:
        images_df = images_df.iloc[index::args.num_cores].reset_index(drop=True)

    if is_master:
        print("======================================================================")
        print(f"VLM-DENTAL: STAGE 2 GRPO RL ({args.track.upper()})")
        print(f"* SFT Stage   : {args.sft_stage}")
        print(f"* Group Size K: {args.group_size}")
        print(f"* SFT Ref Dir : {resolved_sft_dir}")
        print(f"* Target Ckpt : {out_dir}")
        print(f"* HF Repo Sync: {args.hf_repo} ({path_in_repo_prefix})")
        print(f"* Dataset     : {args.dataset} ({len(images_df)} training images; dev split: {args.dev_split or 'none'})")
        print(f"* KL Beta     : {args.kl_beta}")
        print(f"* Learning Rate: {args.lr}")
        print(f"* Replicas    : {args.num_cores} device cores ({'SPMD single process' if args.spmd else 'xmp.spawn'})")
        print("======================================================================")

    train_grpo(
        images_df=images_df,
        annots_df=annots_df,
        categories_df=categories_df,
        config=cfg,
        sft_model_dir=resolved_sft_dir,
        checkpoint_dir=out_dir,
        group_size=args.group_size,
        epochs_per_batch=args.epochs,
        kl_beta=args.kl_beta,
        clip_eps=args.clip_eps,
        learning_rate=args.lr,
        track=args.track,
        hf_repo=args.hf_repo if is_master else None,
        push_every_steps=args.push_every_steps,
        path_in_repo_prefix=path_in_repo_prefix,
        num_cores=args.num_cores,
        use_fsdp=args.fsdp,
        canonical_resize=args.canonical_resize,
        temperature=args.temperature,
        use_spmd=args.spmd,
        max_seq_len=args.max_seq_len,
        pad_vision_to_slots=args.pad_vision_to_slots,
        slot_budget=slot_budget,
        resume=args.resume,
        max_new_tokens=args.max_new_tokens,
        log_path=args.log_path,
    )


if __name__ == "__main__":
    main()
