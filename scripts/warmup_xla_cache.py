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
import ctypes
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List

# Clear conflicting legacy TPU variables and PJRT_DEVICE on Kaggle/Colab
for _var in ["TPU_PROCESS_ADDRESSES", "TPU_PROCESS_COUNT", "CLOUD_TPU_TASK_ID", "PJRT_DEVICE"]:
    os.environ.pop(_var, None)

import torch
from PIL import Image

# Ensure repository root is on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dental_agent.config import load_env
load_env(_repo_root / ".env")

from dental_agent.model.backbone import get_model_classes, safe_process_vision_info
from dental_agent.training.sft import (
    BucketedQwenVLCollator,
    build_conversational_labels,
    wrap_distributed_model,
)

DEFAULT_BUCKETS_WITH_TOOLS = [8192, 16384, 32768]
DEFAULT_BUCKETS_NO_TOOLS = [1536, 2048, 2560, 3072, 8192]


def setup_hardware(precision: str, rank: int = 0):
    """Detect hardware and initialize device for worker rank."""
    is_tpu = False
    device = torch.device("cpu")
    try:
        # Set PJRT_DEVICE right before the first xla_model import to ensure
        # a single, controlled initialization of the PJRT TPU client.
        if "PJRT_DEVICE" not in os.environ:
            os.environ["PJRT_DEVICE"] = "TPU"
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
    """Construct a synthetic batch matching real multimodal conversations snapped to bucket_len.

    Uses authentic `apply_chat_template` and `safe_process_vision_info` so that the
    `<|image_pad|>` tokens generated in text match the visual patches in `pixel_values`
    and `image_grid_thw` 1:1, preventing tensor dimension mismatch crashes in Qwen VL.
    """
    if include_image:
        # Use a neutral 512x512 dummy image (matching standard dental dummy image)
        dummy_img = Image.new("RGB", (512, 512), color=(128, 128, 128))
        messages = [
            {"role": "system", "content": "You are an expert dental radiologist AI."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": dummy_img},
                    {"type": "text", "text": "Analyze this panoramic X-ray. Identify any abnormal teeth and determine the diagnosis."},
                ],
            },
            {"role": "assistant", "content": '{"diagnosis": "caries", "fdi": 16}'},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, video_inputs = safe_process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=False,
            return_tensors="pt",
        )
    else:
        messages = [
            {"role": "system", "content": "You are an expert dental radiologist AI."},
            {"role": "user", "content": "Analyze this dental case."},
            {"role": "assistant", "content": '{"diagnosis": "caries", "fdi": 16}'},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        inputs = processor(
            text=[text],
            padding=False,
            return_tensors="pt",
        )

    # Supervise the assistant turns using standard conversational label builder
    labels = build_conversational_labels(inputs["input_ids"], processor.tokenizer)
    sample = {k: v for k, v in inputs.items()}
    sample["labels"] = labels

    # Snap cleanly to bucket_len using the production BucketedQwenVLCollator
    collator = BucketedQwenVLCollator(processor=processor, custom_buckets=[bucket_len])
    collated = collator([sample])

    # Move tensors to the target hardware device
    batch = {k: v.to(device) for k, v in collated.items()}
    return batch


def run_warmup_worker(index: int, args: argparse.Namespace):
    """Worker process initializing persistent cache and pre-compiling target bucket graphs."""
    try:
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

        # Stagger worker process initialization to prevent simultaneous 8-way CPU RAM / disk surge
        if is_tpu and args.num_cores > 1:
            time.sleep(index * 2.0)

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
        try:
            model = ModelClass.from_pretrained(args.model_id, **load_kwargs)
        except TypeError:
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            model = ModelClass.from_pretrained(args.model_id, **load_kwargs)

        # Attach LoRA adapters to match training architecture exactly
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

        # Host RAM reclamation: force glibc to release unmapped model-loading heap memory back to OS
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            if is_master:
                print("[MEMORY] glibc malloc_trim(0) invoked: released unmapped host CPU loading buffers back to OS.")
        except Exception:
            pass

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

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
            print(f"* Trainable Params: {len(trainable_params)} tensors (~80M LoRA parameters)")
            print("=" * 70)

        # Compile buckets sequentially with memory flushing
        results: List[Dict[str, Any]] = []
        include_images = True  # Both tracks process dental panoramic images

        for idx, b_len in enumerate(target_buckets, 1):
            if is_master:
                t_str = time.strftime("%H:%M:%S")
                print(f"\n[{t_str}] [AOT COMPILATION {idx}/{len(target_buckets)}] Preparing bucket {b_len} tokens...")

            t0 = time.time()
            dummy_batch = build_dummy_multimodal_batch(
                bucket_len=b_len,
                processor=processor,
                device=device,
                include_image=include_images,
            )

            if is_master:
                t_str = time.strftime("%H:%M:%S")
                print(f"[{t_str}] [AOT {idx}/{len(target_buckets)}] Tracing forward & backward pass...")

            model.train()
            outputs = model(**dummy_batch)
            loss = outputs.loss
            loss.backward()

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            if is_master:
                t_str = time.strftime("%H:%M:%S")
                print(f"[{t_str}] [AOT {idx}/{len(target_buckets)}] Graph traced. Triggering XLA compilation & execution (xm.mark_step)...")

            # Align optimizer step with FSDP execution pattern
            if is_tpu:
                import torch_xla.core.xla_model as xm
                if args.fsdp and args.num_cores > 1:
                    optimizer.step()
                    xm.mark_step()
                else:
                    xm.optimizer_step(optimizer)
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            del dummy_batch, outputs, loss

            # Reclaim host memory between bucket compilations
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

            if is_tpu:
                import torch_xla.core.xla_model as xm
                xm.rendezvous(f"bucket_{b_len}_done")

            elapsed = time.time() - t0
            if is_master:
                t_str = time.strftime("%H:%M:%S")
                print(f"[{t_str}] [AOT COMPILATION {idx}/{len(target_buckets)}] Bucket {b_len} compiled and cached in {elapsed:.1f}s.")
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

    except Exception as e:
        import sys, traceback
        print(f"\n[WORKER {index} FATAL ERROR] {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise


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
