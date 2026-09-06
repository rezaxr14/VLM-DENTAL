#!/usr/bin/env python3
"""
Production Training CLI for Stage 1 Supervised Fine-Tuning (SFT) (§16, §17).

Supports:
- Strict Track Segregation: --track {with_tools, no_tools}
- Hardware Optimization: Cloud TPU v5e-8 (PyTorch/XLA FSDPv2) and Multi-GPU (BF16/FP16 LoRA)
- Sequence Length Bucketing & Right-Padding via BucketedQwenVLCollator
- Conversational Assistant-Only Loss Masking via build_conversational_labels
- Lightweight Hugging Face Hub Checkpoint Sync (~760 MB LoRA + optimizer) & Cross-Account Resume
"""

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
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
        "--model-id",
        type=str,
        default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B"),
        help="Base VLM model identifier or local directory",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to verified traces JSONL (defaults automatically to canonical file per track)",
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
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16, help="Gradient accumulation steps")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Peak learning rate for AdamW")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank dimension")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument("--hf-repo", type=str, default=None, help="Hugging Face Hub repository for checkpoint sync")
    parser.add_argument("--push-every-steps", type=int, default=25, help="Frequency of HF checkpoint upload in steps")
    parser.add_argument("--resume-hf", type=str, default=None, help="Hugging Face repo to resume latest checkpoint from")
    return parser.parse_args()


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


def upload_checkpoint_to_hf(checkpoint_dir: Path, hf_repo: str, step: int, epoch: int):
    """Upload lightweight LoRA checkpoint (~760 MB) to Hugging Face Hub."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        commit_msg = f"VLM-DENTAL SFT Checkpoint: Step {step} (Epoch {epoch})"
        print(f"[HF-HUB] Uploading checkpoint from {checkpoint_dir} to {hf_repo}...")
        api.upload_folder(
            folder_path=str(checkpoint_dir),
            repo_id=hf_repo,
            commit_message=commit_msg,
            ignore_patterns=["*.tmp", "*.lock"],
        )
        print(f"[HF-HUB] Checkpoint successfully uploaded to {hf_repo}.")
    except Exception as e:
        print(f"[HF-HUB WARNING] Failed to upload checkpoint to {hf_repo}: {e}")


def main():
    args = parse_args()

    # Resolve track-specific canonical paths
    if args.dataset_path is None:
        if args.track == "with_tools":
            args.dataset_path = "data/traces/train_cot_traces.jsonl"
        else:
            args.dataset_path = "data/traces/train_cot_traces_no_tools.jsonl"

    if args.output_dir is None:
        args.output_dir = f"data/models/qwen3_5_9b_sft_{args.track}"

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("======================================================================")
    print(f"VLM-DENTAL: STAGE 1 SFT TRAINING ({args.track.upper()})")
    print(f"* Model ID    : {args.model_id}")
    print(f"* Dataset     : {args.dataset_path}")
    print(f"* Output Dir  : {args.output_dir}")
    print(f"* Precision   : {args.precision}")
    print(f"* LoRA Config : r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"* Effective BS: {args.batch_size * args.gradient_accumulation_steps}")
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

    # Attach LoRA adapter
    from peft import LoraConfig, get_peft_model
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    if is_tpu:
        model = model.to(device)

    # Dataset & Bucketed Collator
    dataset = DentalSFTDataset(args.dataset_path, processor=processor, track=args.track)
    collator = BucketedQwenVLCollator(processor=processor, track=args.track)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.learning_rate,
        weight_decay=0.01,
    )

    total_steps = 0
    start_epoch = 1

    # Resume capability from HF Hub or local checkpoint
    state_file = out_path / "training_state.json"
    if args.resume_hf:
        print(f"[RESUME] Checking HF Hub for latest checkpoint in {args.resume_hf}...")
        try:
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=args.resume_hf, local_dir=str(out_path))
            if state_file.is_file():
                with open(state_file, "r") as f:
                    saved_state = json.load(f)
                    total_steps = saved_state.get("step", 0)
                    start_epoch = saved_state.get("epoch", 1)
                opt_path = out_path / "optimizer.pt"
                if opt_path.is_file():
                    optimizer.load_state_dict(torch.load(opt_path, map_location="cpu"))
                print(f"[RESUME] Resumed from HF at Epoch {start_epoch}, Step {total_steps}.")
        except Exception as e:
            print(f"[RESUME WARNING] Could not resume from HF: {e}")

    # Emergency SIGTERM handler for Kaggle 9-hour session preemption
    def sigterm_handler(signum, frame):
        print("\n[PREEMPTION] Caught SIGTERM signal! Flushing emergency checkpoint...")
        model.save_pretrained(str(out_path))
        torch.save(optimizer.state_dict(), out_path / "optimizer.pt")
        with open(state_file, "w") as f:
            json.dump({"step": total_steps, "epoch": start_epoch, "track": args.track}, f)
        if args.hf_repo:
            upload_checkpoint_to_hf(out_path, args.hf_repo, total_steps, start_epoch)
        sys.exit(0)

    signal.signal(signal.SIGTERM, sigterm_handler)

    log_file = out_path / "training_loss.jsonl"
    model.train()

    print(f"\n[TRAIN] Beginning training for {args.epochs} epochs...")
    for epoch in range(start_epoch, args.epochs + 1):
        pbar = tqdm(dataloader, desc=f"SFT Epoch {epoch}/{args.epochs}")
        optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(dataloader):
                if is_tpu:
                    import torch_xla.core.xla_model as xm
                    xm.optimizer_step(optimizer)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                total_steps += 1

                step_loss = loss.item() * args.gradient_accumulation_steps
                pbar.set_postfix({"loss": f"{step_loss:.4f}"})

                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"epoch": epoch, "step": total_steps, "loss": step_loss}) + "\n")

                # Periodic checkpoint push to HF Hub
                if args.hf_repo and total_steps % args.push_every_steps == 0:
                    model.save_pretrained(str(out_path))
                    torch.save(optimizer.state_dict(), out_path / "optimizer.pt")
                    with open(state_file, "w") as f:
                        json.dump({"step": total_steps, "epoch": epoch, "track": args.track}, f)
                    upload_checkpoint_to_hf(out_path, args.hf_repo, total_steps, epoch)

        # Epoch end checkpoint
        model.save_pretrained(str(out_path))
        torch.save(optimizer.state_dict(), out_path / "optimizer.pt")
        with open(state_file, "w") as f:
            json.dump({"step": total_steps, "epoch": epoch + 1, "track": args.track}, f)

        if args.hf_repo:
            upload_checkpoint_to_hf(out_path, args.hf_repo, total_steps, epoch)

    print(f"\n[COMPLETE] Stage 1 SFT finished! Final checkpoint saved to {out_path}.")


if __name__ == "__main__":
    main()
