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
import warnings
from pathlib import Path
from typing import Any, Dict, List

# Permanently suppress torch_xla / tensorflow conflict and deprecation warnings
warnings.filterwarnings("ignore", category=UserWarning, message=".*tensorflow.*can conflict with.*torch-xla.*")
warnings.filterwarnings("ignore", category=UserWarning, module="torch_xla.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch_xla.*")
os.environ.setdefault("PYTHONWARNINGS", "ignore::UserWarning,ignore::DeprecationWarning")

# Clear conflicting legacy TPU variables and PJRT_DEVICE on Kaggle/Colab
for _var in ["TPU_PROCESS_ADDRESSES", "TPU_PROCESS_COUNT", "CLOUD_TPU_TASK_ID", "PJRT_DEVICE"]:
    os.environ.pop(_var, None)

# Defensive scrubbing: strip any invalid flags inherited from parent Jupyter notebook environments
for _flag_var in ["XLA_FLAGS", "LIBTPU_INIT_ARGS"]:
    if _flag_var in os.environ:
        cleaned = " ".join([
            f for f in os.environ[_flag_var].split()
            if not f.startswith("--xla_tpu_enable_flash_attention")
            and not f.startswith("--xla_tpu_flash_attention_max_seq_len")
        ]).strip()
        if cleaned:
            os.environ[_flag_var] = cleaned
        else:
            os.environ.pop(_flag_var, None)

# Cap compiler and runtime thread concurrency to prevent multi-process heap explosion across 8 workers
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

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
    setup_spmd_mesh,
    wrap_spmd_model,
    log_xla_memory,
    freeze_and_guard_vision_tower,
    apply_spmd_input_sharding,
)

DEFAULT_BUCKETS_WITH_TOOLS = [10240]
DEFAULT_BUCKETS_NO_TOOLS = [1536, 2048, 2560, 3072, 8192]


def setup_hardware(precision: str, rank: int = 0, xla_pallas: bool = True, xla_spmd: bool = False):
    """Detect hardware and initialize device for worker rank."""
    is_tpu = False
    device = torch.device("cpu")
    try:
        # If SPMD is requested, enable SPMD in the runtime before PJRT device initialization
        if xla_spmd:
            import torch_xla.runtime as xr
            if hasattr(xr, "use_spmd"):
                xr.use_spmd()
            os.environ["XLA_USE_SPMD"] = "1"

        # Set PJRT_DEVICE right before the first xla_model import to ensure
        # a single, controlled initialization of the PJRT TPU client.
        if "PJRT_DEVICE" not in os.environ:
            os.environ["PJRT_DEVICE"] = "TPU"

        # Attention optimization on TPU is driven natively via PyTorch SDPA (--attn-implementation sdpa),
        # without injecting unparsed C++ flags into XLA_FLAGS which cause parse_flags_from_env.cc fatal crashes.
        if xla_pallas and rank == 0:
            print("[ATTENTION] TPU attention acceleration active via PyTorch SDPA lowering.")

        import torch_xla
        import torch_xla.core.xla_model as xm
        try:
            device = torch_xla.device()
        except (AttributeError, Exception):
            device = xm.xla_device()
        is_tpu = True

        if not hasattr(torch, "xla"):
            torch._register_device_module("xla", torch_xla)

        if xm.is_master_ordinal():
            print(f"[HARDWARE] Cloud TPU initialized: {device} ({xm.xla_device_hw(device)})")
            if xla_pallas:
                print("[HARDWARE] XLA TPU Pallas FlashAttention compilation flags injected.")
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
    batch_size: int = 1,
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
    collated = collator([sample] * batch_size)

    # Move tensors to the target hardware device
    batch = {k: v.to(device) for k, v in collated.items()}
    return batch


def run_warmup_worker(index: int, args: argparse.Namespace):
    """Worker process initializing persistent cache and pre-compiling target bucket graphs."""
    try:
        device, compute_dtype, is_tpu = setup_hardware(
            args.precision,
            rank=index,
            xla_pallas=getattr(args, "xla_pallas", True),
            xla_spmd=getattr(args, "xla_spmd", False),
        )

        is_master = True
        if is_tpu:
            import torch_xla.core.xla_model as xm
            is_master = xm.is_master_ordinal()

        # Initialize persistent XLA compilation cache BEFORE any model execution
        cache_base = Path(args.cache_dir).resolve()
        cache_base.mkdir(parents=True, exist_ok=True)

        if is_tpu:
            try:
                import torch_xla.runtime as xr
                xr.initialize_cache(str(cache_base), readonly=False)
                if is_master:
                    print(f"[XLA CACHE] Initialized persistent compilation cache at: {cache_base}")
            except Exception as e:
                if is_master:
                    print(f"[XLA CACHE WARNING] Could not initialize XLA persistent cache: {e}")

        # In SPMD mode, only 1 process runs, so rank locks are not needed.
        # PyTorch/XLA FSDP strictly requires master parameters in torch.float32 for sharding.
        # Under SPMD, model stays in compute_dtype (bfloat16, ~18 GB, avoiding 36 GB float32 allocation).
        use_spmd = getattr(args, "xla_spmd", False)
        if is_tpu and args.num_cores > 1 and not use_spmd:
            lock_dir = Path("/tmp/xla_warmup_locks")
            lock_dir.mkdir(parents=True, exist_ok=True)
            my_turn_file = lock_dir / f"rank_{index}.done"
            prev_turn_file = lock_dir / f"rank_{index - 1}.done"
            
            if index > 0:
                # Wait until the previous rank finishes loading and sharding
                while not prev_turn_file.exists():
                    time.sleep(1.0)

        # Load processor
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

        ModelClass = get_model_classes()

        # PyTorch/XLA FSDP strictly requires master parameters in torch.float32 for sharding
        load_dtype = torch.float32 if (is_tpu and args.fsdp and args.num_cores > 1 and not use_spmd) else compute_dtype
        load_kwargs: Dict[str, Any] = {
            "trust_remote_code": True,
            "dtype": load_dtype,
            "low_cpu_mem_usage": True,
            "attn_implementation": args.attn_implementation,
        }

        if is_master:
            prec_name = "float32 (FSDP master weights)" if load_dtype == torch.float32 else "bfloat16"
            print(f"[MODEL] Loading '{args.model_id}' ({prec_name}, serialized rank-by-rank loading active)...")
        try:
            model = ModelClass.from_pretrained(args.model_id, **load_kwargs)
        except TypeError:
            load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
            model = ModelClass.from_pretrained(args.model_id, **load_kwargs)

        # Attach LoRA adapters to match training architecture exactly
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        if args.lora_target_vision == "projector":
            target_modules.extend(["merger.mlp.0", "merger.mlp.2"])

        # Defensive guard: peft raises an unhandled ImportError if torchao < 0.16.0 is installed
        # (common on Kaggle default images), even when torchao is completely unused.
        try:
            import peft.import_utils
            orig_is_torchao = getattr(peft.import_utils, "is_torchao_available", None)
            if orig_is_torchao is not None:
                def _safe_is_torchao():
                    try:
                        return orig_is_torchao()
                    except ImportError:
                        return False
                peft.import_utils.is_torchao_available = _safe_is_torchao
        except Exception:
            pass

        from peft import LoraConfig, get_peft_model
        try:
            import peft.tuners.lora.torchao as _lora_torchao
            _lora_torchao.is_torchao_available = lambda: False
        except Exception:
            pass

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

        # Vision Guard: Freeze visual blocks and isolate forward under torch.no_grad()
        train_merger = (args.lora_target_vision == "projector")
        freeze_and_guard_vision_tower(model, train_merger=train_merger, is_master=is_master)

        # Wrap model with FSDP or SPMD (matching training pipeline)
        use_spmd = getattr(args, "xla_spmd", False)
        spmd_mesh = None
        if use_spmd and args.fsdp:
            spmd_mesh = setup_spmd_mesh(num_cores=args.num_cores)
            model = wrap_spmd_model(model, mesh=spmd_mesh, is_master=is_master)
        else:
            model = wrap_distributed_model(
                model,
                is_tpu=is_tpu,
                num_cores=args.num_cores,
                use_fsdp=args.fsdp,
                is_master=is_master,
            )

        log_xla_memory("Post-Model Initialization & Sharding", is_master=is_master)

        # Host RAM reclamation: force glibc to release unmapped model-loading heap memory back to OS
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            if is_master:
                print("[MEMORY] glibc malloc_trim(0) invoked: released unmapped host CPU loading buffers back to OS.")
        except Exception:
            pass

        # Signal that this rank has finished loading & sharding, allowing the next worker to start loading
        if is_tpu and args.num_cores > 1 and not use_spmd:
            try:
                my_turn_file.touch()
            except Exception:
                pass

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=args.learning_rate)

        # Resolve target sequence lengths to compile (Single static length by default, or explicit custom buckets)
        if args.buckets:
            target_buckets = sorted(args.buckets)
        elif getattr(args, "max_seq_len", None):
            target_buckets = [args.max_seq_len]
        elif args.track == "with_tools":
            target_buckets = [10240]
        elif args.track == "no_tools":
            target_buckets = [8192]
        else:  # "both"
            target_buckets = [10240]

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

        warmup_batch_size = args.num_cores if use_spmd else 1
        for idx, b_len in enumerate(target_buckets, 1):
            if is_master:
                t_str = time.strftime("%H:%M:%S")
                print(f"\n[{t_str}] [AOT COMPILATION {idx}/{len(target_buckets)}] Preparing bucket {b_len} tokens (batch_size={warmup_batch_size})...")

            t0 = time.time()
            dummy_batch = build_dummy_multimodal_batch(
                bucket_len=b_len,
                processor=processor,
                device=device,
                include_image=include_images,
                batch_size=warmup_batch_size,
            )
            if use_spmd and spmd_mesh is not None:
                apply_spmd_input_sharding(dummy_batch, spmd_mesh, num_cores=args.num_cores)

            log_xla_memory(f"AOT Bucket {b_len} Pre-Forward", is_master=is_master)

            if is_master:
                t_str = time.strftime("%H:%M:%S")
                print(f"[{t_str}] [AOT {idx}/{len(target_buckets)}] Tracing forward & backward pass...")

            model.train()
            outputs = model(**dummy_batch)
            loss = outputs.loss
            # Free logits lazy tensor handle and input tensors immediately before backward
            del outputs
            del dummy_batch

            log_xla_memory(f"AOT Bucket {b_len} Post-Forward", is_master=is_master)

            loss.backward()

            log_xla_memory(f"AOT Bucket {b_len} Post-Backward", is_master=is_master)

            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)

            # Flush host CPU memory right before xm.mark_step triggers compilation
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

            if is_master:
                t_str = time.strftime("%H:%M:%S")
                print(f"[{t_str}] [AOT {idx}/{len(target_buckets)}] Graph traced. Triggering XLA compilation & execution (xm.mark_step)...")

            # Align optimizer step with FSDP or SPMD execution pattern
            if is_tpu:
                import torch_xla.core.xla_model as xm
                if args.fsdp and args.num_cores > 1 and not use_spmd:
                    optimizer.step()
                    xm.mark_step()
                else:
                    xm.optimizer_step(optimizer)
            else:
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            del loss

            # Reclaim host memory between bucket compilations
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

            if is_tpu and args.num_cores > 1 and not use_spmd:
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

            # Create rank symlinks for full backward compatibility with any script expecting rank_{r}
            for r in range(args.num_cores):
                r_dir = cache_base / f"rank_{r}"
                if not r_dir.exists():
                    try:
                        r_dir.symlink_to(".", target_is_directory=True)
                    except Exception:
                        pass

            print(f"[KAGGLE CLI] To upload to Kaggle Datasets, run:")
            print(f"  kaggle datasets version -p {cache_base} -m 'Update SFT XLA persistent cache' --dir-mode tar || kaggle datasets create -p {cache_base} --dir-mode tar")

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
        "--max-seq-len",
        type=int,
        default=10240,
        help="Single static sequence length to pre-compile (e.g. 10240)",
    )
    parser.add_argument(
        "--xla-pallas",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable XLA TPU attention compilation and kernel optimizations (SDPA lowering).",
    )
    parser.add_argument(
        "--xla-spmd",
        "--spmd",
        dest="xla_spmd",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compile using SPMD-based FSDPv2 (single process, shared mesh) instead of legacy xmp.spawn",
    )
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

    use_spmd = getattr(args, "xla_spmd", False)
    if is_tpu and args.num_cores > 1 and not use_spmd:
        import torch_xla.distributed.xla_multiprocessing as xmp
        print(f"[LAUNCH] Spawning AOT compilation warmup across available TPU cores via xmp.spawn(nprocs=None)...")
        xmp.spawn(run_warmup_worker, args=(args,), nprocs=None)
    else:
        if use_spmd:
            print(f"[LAUNCH] Running PyTorch/XLA GSPMD AOT compilation in single-process mode across {args.num_cores} TPU cores...")
        run_warmup_worker(0, args)


if __name__ == "__main__":
    main()
