#!/usr/bin/env python3
"""
VLM-DENTAL: Ahead-Of-Time (AOT) XLA Persistent Cache Warmup Runner

Compiles static sequence-length bucket computation graphs sequentially across
Cloud TPU v5e-8 cores, persisting compiled TPU binaries to disk via
`torch_xla.runtime.initialize_cache`.

Eliminates runtime JIT compilation pauses (15+ minutes) and host CPU memory
exhaustion (330 GB swap thrash) during training.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dental_agent.config import load_env
load_env(_repo_root / ".env")

from dental_agent.model.backbone import get_model_classes
from dental_agent.training.sft import (
    BucketedQwenVLCollator,
    wrap_distributed_model,
)

DEFAULT_BUCKETS_WITH_TOOLS = [8192, 16384, 32768]
DEFAULT_BUCKETS_NO_TOOLS = [1536, 2048, 2560, 3072, 8192]


def setup_hardware(precision: str, rank: int = 0):
    """Detect hardware and initialize device for worker rank."""
    is_tpu = False
    device = torch.device("cpu")
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        is_tpu = True

        import torch_xla
        if not hasattr(torch, "xla"):
            torch._register_device_module("xla", torch_xla)

        if xm.is_master_ordinal():
            print(f"[HARDWARE] Cloud TPU initialized: {device} ({xm.xla_device_hw(device)})")
    except Exception as e:
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
            print(f"[HARDWARE] CUDA GPU initialized: {device}")
        else:
            print(f"[HARDWARE] Running on CPU ({e})")

    compute_dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)
    return device, compute_dtype, is_tpu


def build_dummy_multimodal_batch(
    bucket_len: int,
    processor: Any,
    device: torch.device,
    include_image: bool = True,
) -> Dict[str, torch.Tensor]:
    """Construct a synthetic batch matching real multimodal conversations snapped to bucket_len."""
    tokenizer = processor.tokenizer
    pad_id = getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", 0)

    # 1. Base tokens with assistant supervision
    # We construct a short conversation text with image token so the vision projector is activated
    if include_image:
        text = (
            "<|im_start|>system\nYou are a dental radiologist.<|im_end|>\n"
            "<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>Examine this panoramic X-ray.<|im_end|>\n"
            "<|im_start|>assistant\nI observe findings.<|im_end|>\n"
        )
        dummy_img = Image.new("RGB", (512, 256), color=(128, 128, 128))
        inputs = processor(text=[text], images=[dummy_img], return_tensors="pt")
    else:
        text = (
            "<|im_start|>system\nYou are a dental radiologist.<|im_end|>\n"
            "<|im_start|>user\nExamine this dental patient case.<|im_end|>\n"
            "<|im_start|>assistant\nDiagnostic reasoning steps.<|im_end|>\n"
        )
        inputs = processor(text=[text], return_tensors="pt")

    cur_len = inputs["input_ids"].shape[1]
    if cur_len > bucket_len:
        # Truncate if base text unexpectedly exceeded target bucket
        input_ids = inputs["input_ids"][:, :bucket_len]
        attention_mask = inputs["attention_mask"][:, :bucket_len]
    else:
        pad_size = bucket_len - cur_len
        pad_tokens = torch.full((1, pad_size), pad_id, dtype=torch.long)
        pad_mask = torch.zeros((1, pad_size), dtype=torch.long)
        input_ids = torch.cat([inputs["input_ids"], pad_tokens], dim=1)
        attention_mask = torch.cat([inputs["attention_mask"], pad_mask], dim=1)

    # Labels: mask everything except the assistant turn with -100
    labels = input_ids.clone()
    # Mask padding
    labels[attention_mask == 0] = -100

    batch: Dict[str, torch.Tensor] = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }

    # Add multimodal tokens if present
    if include_image and "pixel_values" in inputs:
        batch["pixel_values"] = inputs["pixel_values"].to(device)
    if include_image and "image_grid_thw" in inputs:
        batch["image_grid_thw"] = inputs["image_grid_thw"].to(device)

    # 3D M-RoPE token types if supported by processor
    image_token_id = getattr(processor, "image_token_id", None)
    if not isinstance(image_token_id, int):
        try:
            image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        except Exception:
            image_token_id = None

    if isinstance(image_token_id, int) and image_token_id > 0:
        mm_types = torch.zeros_like(batch["input_ids"])
        mm_types[batch["input_ids"] == image_token_id] = 1
        batch["mm_token_type_ids"] = mm_types

    return batch


def run_warmup_worker(index: int, args: argparse.Namespace):
    """Worker process initializing persistent cache and pre-compiling target bucket graphs."""
    device, compute_dtype, is_tpu = setup_hardware(args.precision, rank=index)

    is_master = True
    if is_tpu:
        import torch_xla.core.xla_model as xm
        is_master = xm.is_master_ordinal()

    # Initialize persistent XLA compilation cache BEFORE any model execution
    cache_base = Path(args.cache_dir).resolve()
    rank_cache_dir = cache_base / f"rank_{index}"
    rank_cache_dir.mkdir(parents=True, exist_ok=True)

    if is_tpu:
        try:
            import torch_xla.runtime as xr
            xr.initialize_cache(str(rank_cache_dir), readonly=False)
            if is_master:
                print(f"[XLA CACHE] Initialized persistent compilation cache at: {cache_base}")
        except Exception as e:
            if is_master:
                print(f"[XLA CACHE WARNING] Could not initialize XLA persistent cache: {e}")

    # Load processor
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ModelClass = get_model_classes()

    load_dtype = torch.float32 if (is_tpu and args.fsdp and args.num_cores > 1) else compute_dtype
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": load_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": args.attn_implementation,
    }

    if is_master:
        print(f"[MODEL] Loading '{args.model_id}' for AOT graph compilation...")
    model = ModelClass.from_pretrained(args.model_id, **load_kwargs)

    # Attach LoRA adapters to match training architecture
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.lora_target_vision == "projector":
        target_modules.extend(["merger.mlp.0", "merger.mlp.2"])

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

    # Enable reentrant checkpointing
    try:
        model.config.use_cache = False
    except AttributeError:
        pass
    try:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": True, "preserve_rng_state": False}
        )
    except Exception as e:
        if is_master:
            print(f"[CHECKPOINT WARNING] Could not enable checkpointing: {e}")

    # Wrap model with FSDP
    model = wrap_distributed_model(
        model,
        is_tpu=is_tpu,
        num_cores=args.num_cores,
        use_fsdp=args.fsdp,
        is_master=is_master,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # Resolve target buckets to compile
    if args.buckets:
        target_buckets = sorted(args.buckets)
    elif args.track == "with_tools":
        target_buckets = DEFAULT_BUCKETS_WITH_TOOLS
    elif args.track == "no_tools":
        target_buckets = DEFAULT_BUCKETS_NO_TOOLS
    else:  # "both"
        target_buckets = sorted(set(DEFAULT_BUCKETS_WITH_TOOLS + DEFAULT_BUCKETS_NO_TOOLS))

    if is_master:
        print("=" * 70)
        print("VLM-DENTAL: AHEAD-OF-TIME (AOT) BUCKET COMPILATION MANIFEST")
        print(f"* Target Buckets : {target_buckets}")
        print(f"* Attention Impl : {args.attn_implementation}")
        print(f"* TPU Cores      : {args.num_cores}")
        print(f"* Cache Output   : {cache_base}")
        print("=" * 70)

    # Compile buckets sequentially with memory flushing
    results: List[Dict[str, Any]] = []
    include_images = (args.track != "no_tools") or True  # Dental models utilize images across both tracks

    for idx, b_len in enumerate(target_buckets, 1):
        if is_master:
            print(f"\n[AOT COMPILATION {idx}/{len(target_buckets)}] Starting warmup for bucket {b_len} tokens...")

        t0 = time.time()
        dummy_batch = build_dummy_multimodal_batch(
            bucket_len=b_len,
            processor=processor,
            device=device,
            include_image=include_images,
        )

        model.train()
        outputs = model(**dummy_batch)
        loss = outputs.loss
        loss.backward()

        if is_tpu:
            import torch_xla.core.xla_model as xm
            xm.optimizer_step(optimizer)
            xm.mark_step()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)
        del dummy_batch, outputs, loss

        # Reclaim memory between bucket compilations
        gc.collect()
        if is_tpu:
            import torch_xla.core.xla_model as xm
            xm.rendezvous(f"bucket_{b_len}_done")

        elapsed = time.time() - t0
        if is_master:
            print(f"[AOT COMPILATION {idx}/{len(target_buckets)}] Bucket {b_len} compiled and cached in {elapsed:.1f}s.")
            results.append({"bucket": b_len, "compile_seconds": round(elapsed, 1)})

    # Master ordinal creates Kaggle dataset metadata and summary
    if is_master:
        print("\n" + "=" * 70)
        print("ALL TARGET BUCKETS COMPILED SUCCESSFULLY!")
        print("=" * 70)
        for r in results:
            print(f"  - Bucket {r['bucket']:5d} tokens: {r['compile_seconds']}s")

        # Generate dataset-metadata.json for Kaggle upload
        meta = {
            "title": "VLM-DENTAL SFT XLA Persistent Cache",
            "id": args.kaggle_dataset_id,
            "licenses": [{"name": "apache-2.0"}],
            "description": (
                f"Pre-compiled XLA persistent cache for VLM-DENTAL SFT training on Cloud TPU v5e-8. "
                f"Compiled buckets: {target_buckets}."
            ),
        }
        meta_file = cache_base / "dataset-metadata.json"
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"\n[METADATA] Generated Kaggle dataset metadata at: {meta_file}")
        print(f"[KAGGLE CLI] To upload to Kaggle Datasets, run:")
        print(f"  kaggle datasets create -p {cache_base} -u --dir-mode tar")
        print(f"  (or `kaggle datasets version -p {cache_base} -m 'Update cache' --dir-mode tar`)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM-DENTAL: Ahead-Of-Time (AOT) XLA Persistent Cache Warmup",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B"),
        help="Base VLM model identifier or local directory",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="/kaggle/working/xla_cache",
        help="Directory to save persistent XLA compilation cache",
    )
    parser.add_argument(
        "--track",
        type=str,
        default="both",
        choices=["with_tools", "no_tools", "both"],
        help="Training track for default bucket selection: 'with_tools' [8192, 16384, 32768], 'no_tools' [1536, 2048, 2560, 3072, 8192], or 'both'",
    )
    parser.add_argument(
        "--buckets",
        type=int,
        nargs="+",
        default=None,
        help="Custom sequence length buckets to pre-compile (overrides track defaults)",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "eager"],
        help="Attention implementation to compile: 'sdpa' (fused) or 'eager'",
    )
    parser.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--lora-target-vision", type=str, default="projector", choices=["projector", "none"])
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--num-cores", type=int, default=8, help="Number of TPU cores (8 for Cloud TPU v5e-8)")
    parser.add_argument(
        "--fsdp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable PyTorch/XLA FSDP parameter sharding across TPU cores",
    )
    parser.add_argument(
        "--kaggle-dataset-id",
        type=str,
        default=os.environ.get("KAGGLE_XLA_CACHE_REPO", "rezanadimikj/vlm-dental-xla-cache"),
        help="Kaggle Dataset slug for auto-generated dataset-metadata.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    is_tpu = False
    if os.path.exists("/dev/vfio") or os.environ.get("PJRT_DEVICE") == "TPU" or "COLAB_TPU_ADDR" in os.environ:
        is_tpu = True

    if is_tpu and args.num_cores > 1:
        import torch_xla.distributed.xla_multiprocessing as xmp
        print(f"[LAUNCH] Spawning AOT compilation warmup across available TPU cores via xmp.spawn(nprocs=None)...")
        xmp.spawn(run_warmup_worker, args=(args,), nprocs=None)
    else:
        run_warmup_worker(0, args)


if __name__ == "__main__":
    main()
