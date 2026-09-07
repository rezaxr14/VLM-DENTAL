#!/usr/bin/env python3
"""
Production Training CLI for Stage 1 Supervised Fine-Tuning (SFT) (§16, §17).

Supports:
- Multi-Stage SFT Curriculum: --stage {dentex_alone, dentex_tufts_overlap, multicohort_all}
- Negative Controls Calibration: Healthy control traces included across all curriculum stages
- LoRA on Multimodal Vision Projector: --lora-target-vision {projector, none} (adapts merger.mlp)
- Native Image Resolutions: Zero pixel clamping / downsampling, preserving dental panoramic details
- Hardware Optimization: Cloud TPU v5e-8 (PyTorch/XLA FSDPv2) and Multi-GPU (BF16/FP16 LoRA)
- Sequence Length Bucketing & Right-Padding via BucketedQwenVLCollator
- Conversational Assistant-Only Loss Masking via build_conversational_labels
- Cosine Annealing with Linear Warmup and Gradient Clipping
- 5% Validation Split & best_adapter Checkpoint Tracking
- Lightweight Hugging Face Hub Checkpoint Sync (~760 MB LoRA + optimizer) & Cross-Account Resume
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.training.sft import (
    DentalSFTDataset,
    BucketedQwenVLCollator,
)
from dental_agent.model.backbone import get_model_classes


def parse_args():
    parser = argparse.ArgumentParser(description="VLM-DENTAL: Stage 1 Supervised Fine-Tuning (SFT)")
    parser.add_argument(
        "--track",
        type=str,
        required=True,
        choices=["with_tools", "no_tools"],
        help="Strict training track: 'with_tools' (multi-turn agent) or 'no_tools' (direct radiologist)",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="dentex_alone",
        choices=["dentex_alone", "dentex_tufts_overlap", "multicohort_all"],
        help="Curriculum stage: dentex_alone, dentex_tufts_overlap, or multicohort_all",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B"),
        help="Base VLM model identifier or local directory",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("DENTAL_AGENT_DATA_DIR", "data"),
        help="Base directory for datasets, images, and traces",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path(s) to verified traces JSONL (defaults automatically to canonical files per stage and track)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save fine-tuned LoRA checkpoint",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "qlora"],
        help="Numerical precision: bf16 (TPU/Ampere+), fp16, or qlora (4-bit NF4, GPU only)",
    )
    parser.add_argument(
        "--lora-target-vision",
        type=str,
        default="projector",
        choices=["projector", "none"],
        help="Attach LoRA to multimodal patch projector ('projector': merger.mlp.0, merger.mlp.2) or 'none'",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Peak learning rate for AdamW")
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Linear warmup ratio for learning rate scheduler")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank dimension")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument("--eval-every-steps", type=int, default=25, help="Frequency of validation evaluation in steps")
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=os.environ.get("HF_ARTIFACT_REPO", "Reza-Nadimi/vlm-dental-models"),
        help="Hugging Face Hub repository for checkpoint sync (default: Reza-Nadimi/vlm-dental-models)",
    )
    parser.add_argument("--push-every-steps", type=int, default=25, help="Frequency of HF checkpoint upload in steps")
    parser.add_argument("--resume-hf", type=str, default=None, help="Hugging Face repo to resume latest checkpoint from")
    return parser.parse_args()


def resolve_stage_traces(stage: str, track: str, data_dir: str | Path) -> List[str]:
    """Resolve exact trace paths (disease + negative controls) for curriculum stage and track."""
    traces_dir = Path(data_dir) / "traces"
    if not traces_dir.exists():
        traces_dir = Path("data/traces")

    resolved: List[str] = []
    is_tools = track == "with_tools"

    if stage == "dentex_alone":
        # Stage 1a: DENTEX Alone + DENTEX Healthy Controls
        candidates = [
            traces_dir / ("train_cot_traces_dentex.jsonl" if is_tools else "train_cot_traces_dentex_no_tools.jsonl"),
            traces_dir / ("train_cot_traces_healthy_dentex.jsonl" if is_tools else "train_cot_traces_healthy_dentex_no_tools.jsonl"),
        ]
    elif stage == "dentex_tufts_overlap":
        # Stage 1b: DENTEX + Tufts Overlap + Negative Controls
        candidates = [
            traces_dir / ("train_cot_traces_dentex.jsonl" if is_tools else "train_cot_traces_dentex_no_tools.jsonl"),
            traces_dir / ("train_cot_traces_tufts.jsonl" if is_tools else "train_cot_traces_tufts_no_tools.jsonl"),
            traces_dir / ("train_cot_traces_healthy_dentex.jsonl" if is_tools else "train_cot_traces_healthy_dentex_no_tools.jsonl"),
            traces_dir / ("train_cot_traces_healthy_tufts.jsonl" if is_tools else "train_cot_traces_healthy_tufts_no_tools.jsonl"),
        ]
    else:  # multicohort_all
        # Stage 1c: Full Multi-Cohort: DENTEX + Tufts All 4 Findings + Full Negative Controls
        candidates = [
            traces_dir / ("train_cot_traces_dentex.jsonl" if is_tools else "train_cot_traces_dentex_no_tools.jsonl"),
            traces_dir / ("train_cot_traces_tufts_all.jsonl" if is_tools else "train_cot_traces_tufts_all_no_tools.jsonl"),
            traces_dir / ("train_cot_traces_healthy_dentex.jsonl" if is_tools else "train_cot_traces_healthy_dentex_no_tools.jsonl"),
            traces_dir / ("train_cot_traces_healthy_tufts.jsonl" if is_tools else "train_cot_traces_healthy_tufts_no_tools.jsonl"),
        ]

    for p in candidates:
        if p.is_file():
            resolved.append(str(p))
        else:
            print(f"[STAGE TRACES WARNING] Trace file not found: {p}")

    if not resolved:
        # Fallback to canonical train_cot_traces if specific stage split files are not yet generated
        fallback = traces_dir / ("train_cot_traces.jsonl" if is_tools else "train_cot_traces_no_tools.jsonl")
        if fallback.is_file():
            resolved.append(str(fallback))

    return resolved


def setup_hardware(precision: str):
    """Detect hardware backend: Cloud TPU v5e-8 vs CUDA GPU vs CPU."""
    is_tpu = False
    device = None
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        is_tpu = True
        print(f"[HARDWARE] Initialized Cloud TPU device: {device} ({xm.xla_device_hw(device)})")
    except Exception:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"[HARDWARE] Initialized CUDA GPU: {torch.cuda.get_device_name(0)} (Count: {torch.cuda.device_count()})")
        else:
            device = torch.device("cpu")
            print("[HARDWARE] Running on CPU.")

    if is_tpu and precision == "qlora":
        print("[WARNING] 4-bit QLoRA is not supported on TPU/XLA devices. Switching to native BF16.")
        precision = "bf16"

    dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)
    return is_tpu, device, dtype, precision


def upload_checkpoint_to_hf(checkpoint_dir: Path, hf_repo: str, step: int, epoch: int, path_in_repo: Optional[str] = None):
    """Upload lightweight LoRA checkpoint (~760 MB) to Hugging Face Hub under structured path_in_repo."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        commit_msg = f"VLM-DENTAL SFT Checkpoint: Step {step} (Epoch {epoch})"
        target_path_str = f" to {hf_repo}/{path_in_repo}" if path_in_repo else f" to {hf_repo}"
        print(f"[HF-HUB] Uploading checkpoint from {checkpoint_dir}{target_path_str}...")
        kwargs: Dict[str, Any] = {
            "folder_path": str(checkpoint_dir),
            "repo_id": hf_repo,
            "commit_message": commit_msg,
            "ignore_patterns": ["*.tmp", "*.lock"],
        }
        if path_in_repo:
            kwargs["path_in_repo"] = path_in_repo
        api.upload_folder(**kwargs)
        print(f"[HF-HUB] Checkpoint successfully uploaded{target_path_str}.")
    except Exception as e:
        print(f"[HF-HUB WARNING] Failed to upload checkpoint to {hf_repo}: {e}")


def evaluate_loss(model: torch.nn.Module, val_loader: DataLoader, device: torch.device, is_tpu: bool) -> float:
    """Compute validation loss on held-out 5% validation set."""
    model.eval()
    total_val_loss = 0.0
    val_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            total_val_loss += outputs.loss.item()
            val_batches += 1
            if val_batches >= 20:  # Fast validation cap
                break
    model.train()
    return total_val_loss / max(val_batches, 1)


def main():
    args = parse_args()

    # Resolve stage-specific traces and default output directory
    resolved_traces: List[str] = []
    if args.dataset_path:
        resolved_traces = [p.strip() for p in args.dataset_path.split(",") if p.strip()]
    else:
        resolved_traces = resolve_stage_traces(args.stage, args.track, args.data_dir)

    if not resolved_traces:
        raise FileNotFoundError(f"No valid trace files found for stage='{args.stage}', track='{args.track}'")

    if args.output_dir is None:
        args.output_dir = f"data/models/qwen3_5_9b_sft_{args.track}_{args.stage}"

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_adapter_path = out_path / "best_adapter"
    path_in_repo = f"sft/{out_path.name}"

    print("======================================================================")
    print(f"VLM-DENTAL: STAGE 1 SFT TRAINING ({args.track.upper()} - {args.stage.upper()})")
    print(f"* Stage       : {args.stage}")
    print(f"* Model ID    : {args.model_id}")
    print(f"* Data Dir    : {args.data_dir}")
    print(f"* Traces ({len(resolved_traces)} files):")
    for t in resolved_traces:
        print(f"    - {t}")
    print(f"* Output Dir  : {args.output_dir}")
    print(f"* Path in HF  : {path_in_repo}")
    print(f"* Precision   : {args.precision}")
    print(f"* Vision LoRA : {args.lora_target_vision}")
    print(f"* LoRA Config : r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"* Effective BS: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"* Warmup Ratio: {args.warmup_ratio} | Max Grad Norm: {args.max_grad_norm}")
    print("======================================================================")

    is_tpu, device, compute_dtype, active_precision = setup_hardware(args.precision)

    # Load processor and model
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ModelClass = get_model_classes()

    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": compute_dtype,
    }

    if active_precision == "qlora":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        load_kwargs["device_map"] = "auto"
    elif not is_tpu and torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    print(f"[MODEL] Loading {args.model_id}...")
    model = ModelClass.from_pretrained(args.model_id, **load_kwargs)

    # Define LoRA Target Modules (Language Model + optional Vision Projector)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.lora_target_vision == "projector":
        # Multimodal patch projector linear projections
        target_modules.extend(["merger.mlp.0", "merger.mlp.2"])
        print("[LORA] Enabled LoRA on Multimodal Vision Projector ('merger.mlp.0', 'merger.mlp.2')")

    from peft import LoraConfig, get_peft_model
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if is_tpu:
        model = model.to(device)

    # Dataset & Bucketed Collator
    full_dataset = DentalSFTDataset(resolved_traces, processor=processor, track=args.track, data_dir=args.data_dir)
    val_size = max(int(len(full_dataset) * 0.05), 1) if len(full_dataset) >= 20 else 0
    train_size = len(full_dataset) - val_size

    if val_size > 0:
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )
    else:
        train_dataset = full_dataset
        val_dataset = None

    collator = BucketedQwenVLCollator(processor=processor, track=args.track)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator) if val_dataset else None

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_update_steps = (len(train_dataloader) // args.gradient_accumulation_steps) * args.epochs
    num_warmup_steps = max(int(total_update_steps * args.warmup_ratio), 1)

    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max(total_update_steps, 1),
    )

    total_steps = 0
    start_epoch = 1
    best_val_loss = float("inf")

    # Resume capability from HF Hub or local checkpoint
    state_file = out_path / "training_state.json"
    if args.resume_hf:
        print(f"[RESUME] Checking HF Hub for latest checkpoint in {args.resume_hf} (subfolder={path_in_repo})...")
        try:
            from huggingface_hub import snapshot_download
            import shutil
            download_dir = out_path.parent / ".hf_cache"
            snapshot_download(
                repo_id=args.resume_hf,
                allow_patterns=[f"{path_in_repo}/*"],
                local_dir=str(download_dir),
            )
            sub_dir = download_dir / path_in_repo
            if sub_dir.exists():
                for item in sub_dir.iterdir():
                    dst = out_path / item.name
                    if item.is_file():
                        shutil.copy2(item, dst)
                    elif item.is_dir():
                        shutil.copytree(item, dst, dirs_exist_ok=True)
            elif (download_dir / "training_state.json").exists():
                for item in download_dir.iterdir():
                    if item.name != ".hf_cache":
                        dst = out_path / item.name
                        if item.is_file():
                            shutil.copy2(item, dst)
                        elif item.is_dir():
                            shutil.copytree(item, dst, dirs_exist_ok=True)

            if state_file.is_file():
                with open(state_file, "r") as f:
                    saved_state = json.load(f)
                    total_steps = saved_state.get("step", 0)
                    start_epoch = saved_state.get("epoch", 1)
                    best_val_loss = saved_state.get("best_val_loss", float("inf"))
                opt_path = out_path / "optimizer.pt"
                if opt_path.is_file():
                    optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
                sched_path = out_path / "scheduler.pt"
                if sched_path.is_file():
                    scheduler.load_state_dict(torch.load(sched_path, map_location="cpu"))
                print(f"[RESUME] Resumed from HF at Epoch {start_epoch}, Step {total_steps} (best_val_loss={best_val_loss:.4f}).")
        except Exception as e:
            print(f"[RESUME WARNING] Could not resume from HF: {e}")

    # Emergency SIGTERM handler for Kaggle 9-hour session preemption
    def sigterm_handler(signum, frame):
        print("\n[PREEMPTION] Caught SIGTERM signal! Flushing emergency checkpoint...")
        model.save_pretrained(str(out_path))
        torch.save(optimizer.state_dict(), out_path / "optimizer.pt")
        torch.save(scheduler.state_dict(), out_path / "scheduler.pt")
        with open(state_file, "w") as f:
            json.dump({
                "step": total_steps,
                "epoch": start_epoch,
                "track": args.track,
                "stage": args.stage,
                "best_val_loss": best_val_loss,
            }, f)
        if args.hf_repo:
            upload_checkpoint_to_hf(out_path, args.hf_repo, total_steps, start_epoch, path_in_repo=path_in_repo)
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    log_file = out_path / "training_loss.jsonl"
    model.train()

    print(f"\n[TRAIN] Beginning training: {len(train_dataset)} train samples, {val_size} val samples across {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs + 1):
        pbar = tqdm(train_dataloader, desc=f"SFT Epoch {epoch}/{args.epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if is_tpu:
                    import torch_xla.core.xla_model as xm
                    xm.optimizer_step(optimizer)
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                total_steps += 1

                step_loss = loss.item() * args.gradient_accumulation_steps
                lr_current = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else args.learning_rate
                pbar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{lr_current:.2e}"})

                log_entry = {
                    "epoch": epoch,
                    "step": total_steps,
                    "loss": step_loss,
                    "lr": lr_current,
                }

                # Periodic evaluation on held-out validation set
                if val_dataloader and total_steps % args.eval_every_steps == 0:
                    val_loss = evaluate_loss(model, val_dataloader, device, is_tpu)
                    log_entry["val_loss"] = val_loss
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_adapter_path.mkdir(parents=True, exist_ok=True)
                        model.save_pretrained(str(best_adapter_path))
                        print(f"\n[VALIDATION] New best adapter saved! Step {total_steps}: val_loss = {val_loss:.4f}")

                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(log_entry) + "\n")

                # Periodic checkpoint push to HF Hub
                if args.hf_repo and total_steps % args.push_every_steps == 0:
                    model.save_pretrained(str(out_path))
                    torch.save(optimizer.state_dict(), out_path / "optimizer.pt")
                    torch.save(scheduler.state_dict(), out_path / "scheduler.pt")
                    with open(state_file, "w") as f:
                        json.dump({
                            "step": total_steps,
                            "epoch": epoch,
                            "track": args.track,
                            "stage": args.stage,
                            "best_val_loss": best_val_loss,
                        }, f)
                    upload_checkpoint_to_hf(out_path, args.hf_repo, total_steps, epoch, path_in_repo=path_in_repo)

        # Epoch end checkpoint
        model.save_pretrained(str(out_path))
        torch.save(optimizer.state_dict(), out_path / "optimizer.pt")
        torch.save(scheduler.state_dict(), out_path / "scheduler.pt")
        with open(state_file, "w") as f:
            json.dump({
                "step": total_steps,
                "epoch": epoch + 1,
                "track": args.track,
                "stage": args.stage,
                "best_val_loss": best_val_loss,
            }, f)

        if args.hf_repo:
            upload_checkpoint_to_hf(out_path, args.hf_repo, total_steps, epoch, path_in_repo=path_in_repo)

    print(f"\n[COMPLETE] Stage 1 SFT ({args.stage}) finished! Final checkpoint saved to {out_path}.")
    if best_adapter_path.exists():
        print(f"[COMPLETE] Best adapter preserved at {best_adapter_path} (best_val_loss={best_val_loss:.4f}).")


if __name__ == "__main__":
    main()
