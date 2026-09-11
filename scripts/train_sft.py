#!/usr/bin/env python3
"""
Production Training CLI for Stage 1 Supervised Fine-Tuning (SFT) (§16, §17).

Supports:
- Multi-Stage SFT Curriculum: --stage {dentex_alone, dentex_tufts_overlap, multicohort_all}
- Negative Controls Calibration: Healthy control traces included across all curriculum stages
- LoRA on Multimodal Vision Projector: --lora-target-vision {projector, none} (adapts merger.mlp)
- Native Image Resolutions: Zero pixel clamping / downsampling, preserving dental panoramic details
- Hardware Optimization: Multi-Core Cloud TPU v5e-8 distributed execution via torch_xla.distributed.xmp.spawn (8-way cross-replica gradient synchronization) and Multi-GPU (BF16/FP16 LoRA)
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

# Clear conflicting legacy TPU variables and PJRT_DEVICE on Kaggle/Colab
for _var in ["TPU_PROCESS_ADDRESSES", "TPU_PROCESS_COUNT", "CLOUD_TPU_TASK_ID", "PJRT_DEVICE"]:
    os.environ.pop(_var, None)

# Cap OpenXLA compiler thread concurrency to prevent multi-process heap explosion on 96-vCPU hosts
os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

# NOTE: PJRT_DEVICE is set lazily inside setup_hardware(), NOT here.
# Setting it before `import torch` causes torch_xla's torch plugin to auto-initialize
# the PJRT TPU client at import time. Then when setup_hardware() explicitly calls
# xm.xla_device(), libtpu's RuntimeMetricAggregator crashes with
# "Check failed: reporting_closure_ == nullptr" because it's initialized twice.

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
    wrap_distributed_model,
    unwrap_peft_model,
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
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps (default: 1 on multi-core TPU for effective batch size 8, 16 on single device)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-5,
        help="Peak learning rate for AdamW (default: 5e-5 for multimodal LoRA)",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.05, help="Linear warmup ratio for learning rate scheduler")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm for gradient clipping")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--lora-r", type=int, default=32, help="LoRA rank dimension")
    parser.add_argument("--lora-alpha", type=int, default=64, help="LoRA alpha scaling factor")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout probability")
    parser.add_argument("--eval-every-steps", type=int, default=0, help="Frequency of validation evaluation in steps (0 = epoch-boundary only; only active when --eval-strategy includes 'steps')")
    parser.add_argument(
        "--eval-strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "both"],
        help="Validation evaluation strategy: 'epoch' (at end of each epoch), 'steps' (every --eval-every-steps), or 'both' (default: epoch)",
    )
    parser.add_argument(
        "--save-strategy",
        type=str,
        default="epoch",
        choices=["epoch", "steps", "both"],
        help="Checkpoint save & HF push strategy: 'epoch' (at end of each epoch), 'steps' (every --push-every-steps), or 'both' (default: epoch)",
    )
    parser.add_argument(
        "--hf-repo",
        type=str,
        default=os.environ.get("HF_ARTIFACT_REPO", "Reza-Nadimi/vlm-dental-models"),
        help="Hugging Face Hub repository for checkpoint sync (default: Reza-Nadimi/vlm-dental-models)",
    )
    parser.add_argument("--push-every-steps", type=int, default=0, help="Frequency of HF checkpoint upload in steps (0 = epoch-boundary only; only active when --save-strategy includes 'steps')")
    parser.add_argument("--resume-hf", type=str, default=None, help="Hugging Face repo to resume latest checkpoint from")
    parser.add_argument(
        "--num-cores",
        type=int,
        default=1,
        help="Number of TPU cores for distributed data-parallel execution (1 for single core/GPU, 8 for Kaggle TPU v5e-8)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Maximum sequence length bucket ceiling (e.g. 40960, 49152) for collator",
    )
    parser.add_argument(
        "--custom-buckets",
        type=int,
        nargs="+",
        default=None,
        help="Custom sequence length buckets for collator",
    )
    parser.add_argument(
        "--fsdp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable PyTorch/XLA FSDP parameter sharding across TPU cores to fit 9B BF16 model within 16 GB HBM (default: True on multi-core TPU)",
    )
    parser.add_argument(
        "--xla-cache-dir",
        type=str,
        default=None,
        help="Directory for XLA persistent compilation cache (auto-detected if attached as Kaggle input or in data/xla_cache)",
    )
    args = parser.parse_args()
    if args.gradient_accumulation_steps is None:
        args.gradient_accumulation_steps = 1 if args.num_cores > 1 else 16
    return args


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


def get_xla_world_size(is_tpu: bool = False) -> int:
    """Retrieve world size across modern torch_xla.runtime, legacy xrt, or distributed."""
    if is_tpu:
        try:
            import torch_xla.runtime as xr
            return xr.world_size()
        except Exception:
            pass
        try:
            import torch_xla.core.xla_model as xm
            if hasattr(xm, "xrt_world_size"):
                return xm.xrt_world_size()
        except Exception:
            pass
    try:
        import torch.distributed as dist
        if dist.is_initialized():
            return dist.get_world_size()
    except Exception:
        pass
    return 1


def setup_hardware(precision: str, rank: int = 0):
    """Detect hardware backend: Cloud TPU v5e-8 vs CUDA GPU vs CPU."""
    is_tpu = False
    device = None
    try:
        # Set PJRT_DEVICE right before the first xla_model import to ensure
        # a single, controlled initialization of the PJRT TPU client.
        if "PJRT_DEVICE" not in os.environ:
            os.environ["PJRT_DEVICE"] = "TPU"
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        is_tpu = True
        
        # torch.utils.checkpoint's fork_rng() calls torch.get_device_module("xla")
        # unconditionally -- before it even checks preserve_rng_state -- and that
        # lookup is just getattr(torch, "xla", None). XLA has no torch.xla submodule,
        # so it raises. Registering torch_xla here satisfies that existence check;
        # preserve_rng_state=False means its actual methods never get called.
        import torch_xla
        if not hasattr(torch, "xla"):
            torch._register_device_module("xla", torch_xla)
            
        if xm.is_master_ordinal():
            print(f"[HARDWARE] Initialized Cloud TPU device: {device} ({xm.xla_device_hw(device)}) | World Size: {get_xla_world_size(is_tpu)}")
    except Exception as e:
        print(f"[HARDWARE WARNING] TPU backend initialization failed ({e}); checking CUDA/CPU.")
        if torch.cuda.is_available():
            device = torch.device(f"cuda:{rank}" if torch.cuda.device_count() > rank else "cuda:0")
            print(f"[HARDWARE] Initialized CUDA GPU: {torch.cuda.get_device_name(device)} (Count: {torch.cuda.device_count()})")
        else:
            device = torch.device("cpu")
            print("[HARDWARE] Running on CPU.")

    if is_tpu and precision == "qlora":
        try:
            import torch_xla.core.xla_model as xm
            if xm.is_master_ordinal():
                print("[WARNING] 4-bit QLoRA is not supported on TPU/XLA devices. Switching to native BF16.")
        except Exception:
            pass
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


def resolve_xla_cache_dir(cache_arg: str | None) -> tuple[Path | None, bool]:
    """Resolve persistent XLA compilation cache directory and whether it is read-only.

    Checks in order:
    1. Explicit CLI argument --xla-cache-dir
    2. Environment variable XLA_PERSISTENT_CACHE_PATH or XLA_CACHE_DIR
    3. Attached Kaggle input dataset directories (/kaggle/input/*xla*cache*)
    4. Local working directories (/kaggle/working/xla_cache, data/xla_cache)

    Returns (resolved_path, readonly). If no cache is found or valid, returns (None, False).
    """
    candidates: list[Path] = []
    if cache_arg:
        candidates.append(Path(cache_arg))
    for env_var in ["XLA_PERSISTENT_CACHE_PATH", "XLA_CACHE_DIR"]:
        val = os.environ.get(env_var)
        if val:
            candidates.append(Path(val))

    # Standard Kaggle input dataset paths
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for item in sorted(kaggle_input.glob("*xla*cache*")):
            if item.is_dir():
                candidates.append(item)
        for item in sorted(kaggle_input.glob("*cache*")):
            if item.is_dir():
                candidates.append(item)

    # Writable scratch / local fallbacks
    candidates.extend([
        Path("/kaggle/working/xla_cache"),
        Path("data/xla_cache"),
    ])

    for cand in candidates:
        if cand.is_dir():
            # Check if directory is under /kaggle/input (read-only mount)
            try:
                is_readonly = str(cand.resolve()).startswith(str(kaggle_input.resolve()))
            except Exception:
                is_readonly = False
            return cand, is_readonly

    return None, False


def run_training(index: int, args: argparse.Namespace):
    """Worker function executed per core/device."""
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

    is_tpu, device, compute_dtype, active_precision = setup_hardware(args.precision, rank=index)

    is_master = True
    if is_tpu:
        import torch_xla.core.xla_model as xm
        is_master = xm.is_master_ordinal()

    if is_master:
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
        prec_str = f"{args.precision} (master weights: float32, compute: bfloat16 via FSDP)" if (is_tpu and args.fsdp and args.num_cores > 1) else args.precision
        print(f"* Precision   : {prec_str}")
        print(f"* Vision LoRA : {args.lora_target_vision}")
        print(f"* LoRA Config : r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
        world_size = get_xla_world_size(is_tpu)
        print(f"* Effective BS: {args.batch_size * args.gradient_accumulation_steps * world_size} ({world_size} device replicas)")
        print(f"* Warmup Ratio: {args.warmup_ratio} | Max Grad Norm: {args.max_grad_norm}")
        print("======================================================================")

    # Initialize persistent XLA compilation cache if available (or continue gracefully without crashing)
    cache_path, is_readonly = resolve_xla_cache_dir(args.xla_cache_dir)
    if is_tpu and cache_path:
        try:
            import torch_xla.runtime as xr
            rank_cache = cache_path / f"rank_{index}"
            target_cache = rank_cache if (rank_cache.is_dir() and any(rank_cache.iterdir())) else cache_path
            xr.initialize_cache(str(target_cache), readonly=is_readonly)
            if is_master:
                mode_str = "read-only" if is_readonly else "writable"
                print(f"[XLA CACHE] Initialized persistent compilation cache ({mode_str}): {target_cache}")
        except Exception as e:
            if is_master:
                print(f"[XLA CACHE WARNING] Could not initialize XLA persistent cache ({e}); continuing with normal on-the-fly compilation.")
    elif is_master and is_tpu:
        print("[XLA CACHE] No persistent compilation cache specified or detected; continuing with normal on-the-fly compilation.")

    # Stagger worker process initialization to prevent simultaneous 8-way CPU RAM surge
    if is_tpu and args.num_cores > 1:
        time.sleep(index * 1.5)

    # Load processor and model
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ModelClass = get_model_classes()

    # For FSDP on multi-core TPU, load directly in float32 to avoid duplicate RAM spike from .float()
    load_dtype = torch.float32 if (is_tpu and args.fsdp and args.num_cores > 1) else compute_dtype
    load_kwargs: Dict[str, Any] = {
        "trust_remote_code": True,
        "dtype": load_dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
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

    if is_master:
        if is_tpu and args.fsdp and args.num_cores > 1:
            print(f"[MODEL] Loading {args.model_id} (master weights: float32 for FSDP sharding across {args.num_cores} cores, compute: bfloat16)...")
        else:
            print(f"[MODEL] Loading {args.model_id}...")
    try:
        model = ModelClass.from_pretrained(args.model_id, **load_kwargs)
    except TypeError:
        # Fallback for older transformers (<4.48) that only accept torch_dtype
        load_kwargs["torch_dtype"] = load_kwargs.pop("dtype")
        model = ModelClass.from_pretrained(args.model_id, **load_kwargs)

    # Define LoRA Target Modules (Language Model + optional Vision Projector)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.lora_target_vision == "projector":
        # Multimodal patch projector linear projections
        target_modules.extend(["merger.mlp.0", "merger.mlp.2"])
        if is_master:
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
    if is_master:
        model.print_trainable_parameters()

    # Enable reentrant gradient checkpointing before FSDP wrapping to bound activation
    # memory for large sequence buckets and protect TPU v5e-8's 16 GB per-core HBM
    try:
        model.config.use_cache = False
    except AttributeError:
        pass
    try:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True, "preserve_rng_state": False})
        if is_master:
            print("[MEMORY] Gradient checkpointing enabled (use_reentrant=True, preserve_rng_state=False) to bound activation memory for large sequence-length buckets.")
    except Exception as e:
        if is_master:
            print(f"[MEMORY WARNING] Could not enable gradient checkpointing: {e}")

    model = wrap_distributed_model(
        model,
        is_tpu=is_tpu,
        num_cores=args.num_cores,
        use_fsdp=args.fsdp,
        is_master=is_master,
    )

    # Host RAM reclamation: force glibc to release unmapped model-loading heap memory back to the Linux kernel
    import gc, ctypes
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
        if is_master:
            print("[MEMORY] glibc malloc_trim(0) invoked: released unmapped host CPU loading buffers back to OS.")
    except Exception:
        pass

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

    custom_buckets = args.custom_buckets
    if not custom_buckets and args.max_seq_len:
        base_buckets = (
            BucketedQwenVLCollator.BUCKETS_WITH_TOOLS
            if args.track == "with_tools"
            else BucketedQwenVLCollator.BUCKETS_NO_TOOLS
        )
        extended = [b for b in base_buckets if b <= args.max_seq_len]
        if not extended or extended[-1] < args.max_seq_len:
            extended.append(args.max_seq_len)
        custom_buckets = extended

    collator = BucketedQwenVLCollator(processor=processor, track=args.track, custom_buckets=custom_buckets)
    if is_master:
        print(f"[COLLATOR] Active sequence length buckets: {collator.buckets} (max headroom: {collator.buckets[-1]})")

    train_sampler = None
    ws = get_xla_world_size(is_tpu)
    if is_tpu and ws > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset,
            num_replicas=ws,
            rank=index,
            shuffle=True,
        )
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=collator)
    else:
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collator)

    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator) if (val_dataset and is_master) else None

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
    if args.resume_hf and is_master:
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

    def save_sft_checkpoint(epoch_num: int, is_preemption: bool = False):
        state_data = {
            "step": total_steps,
            "epoch": epoch_num,
            "track": args.track,
            "stage": args.stage,
            "best_val_loss": best_val_loss,
            "preempted": is_preemption,
        }
        if is_tpu:
            import torch_xla.core.xla_model as xm
            if xm.is_master_ordinal():
                unwrap_peft_model(model).save_pretrained(str(out_path))
                with open(state_file, "w") as f:
                    json.dump(state_data, f)
            xm.save(optimizer.state_dict(), out_path / "optimizer.pt")
            xm.save(scheduler.state_dict(), out_path / "scheduler.pt")
            if args.hf_repo and xm.is_master_ordinal():
                upload_checkpoint_to_hf(out_path, args.hf_repo, total_steps, epoch_num, path_in_repo=path_in_repo)
        else:
            unwrap_peft_model(model).save_pretrained(str(out_path))
            torch.save(optimizer.state_dict(), out_path / "optimizer.pt")
            torch.save(scheduler.state_dict(), out_path / "scheduler.pt")
            with open(state_file, "w") as f:
                json.dump(state_data, f)
            if args.hf_repo:
                upload_checkpoint_to_hf(out_path, args.hf_repo, total_steps, epoch_num, path_in_repo=path_in_repo)

    # Emergency SIGTERM handler for Kaggle 9-hour session preemption
    def sigterm_handler(signum, frame):
        if is_master:
            print("\n[PREEMPTION] Caught SIGTERM signal! Flushing emergency checkpoint...")
        save_sft_checkpoint(start_epoch, is_preemption=True)
        sys.exit(0)

    try:
        import threading
        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGTERM, sigterm_handler)
    except (ValueError, AttributeError, RuntimeError):
        pass

    log_file = out_path / "training_loss.jsonl"
    model.train()

    if is_master:
        print(f"\n[TRAIN] Beginning training: {len(train_dataset)} train samples, {val_size} val samples across {args.epochs} epochs...")

    for epoch in range(start_epoch, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        pbar = tqdm(train_dataloader, desc=f"SFT Epoch {epoch}/{args.epochs}") if is_master else train_dataloader
        optimizer.zero_grad()
        accum_loss_sum = 0.0
        accum_valid_tokens = 0

        for step, batch in enumerate(pbar):
            # Count valid tokens directly on CPU tensor before moving to device.
            # This completely eliminates premature TPU-CPU synchronization barriers
            # before backward, keeping lazy tensor graph execution clean.
            num_valid = int((batch["labels"] != -100).sum().item())
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)

            if num_valid > 0:
                batch_loss = outputs.loss
                del outputs  # Free logits lazy tensor handle immediately
                batch_loss_sum = batch_loss * num_valid
                batch_loss_sum.backward()
                accum_loss_sum += batch_loss_sum.item()
                accum_valid_tokens += num_valid
            else:
                del outputs

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                # Normalize accumulated gradients by total valid tokens across the window
                if accum_valid_tokens > 0:
                    if is_tpu and get_xla_world_size(is_tpu) > 1:
                        token_t = torch.tensor([accum_valid_tokens], dtype=torch.float32, device=device)
                        reduce_op = getattr(xm, "REDUCE_SUM", "sum")
                        global_tokens = xm.all_reduce(reduce_op, token_t).item()
                        scale = 1.0 / max(global_tokens, 1.0)
                    else:
                        scale = 1.0 / max(float(accum_valid_tokens), 1.0)

                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(scale)

                if hasattr(model, "clip_grad_norm_"):
                    model.clip_grad_norm_(args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if is_tpu:
                    if args.fsdp and args.num_cores > 1:
                        optimizer.step()
                        xm.mark_step()
                    else:
                        xm.optimizer_step(optimizer)
                else:
                    optimizer.step()

                scheduler.step()
                optimizer.zero_grad()
                total_steps += 1

                step_loss = accum_loss_sum / max(accum_valid_tokens, 1)
                accum_loss_sum = 0.0
                accum_valid_tokens = 0

                if is_master:
                    lr_current = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else args.learning_rate
                    if hasattr(pbar, "set_postfix"):
                        pbar.set_postfix({"loss": f"{step_loss:.4f}", "lr": f"{lr_current:.2e}"})

                    log_entry = {
                        "epoch": epoch,
                        "step": total_steps,
                        "loss": step_loss,
                        "lr": lr_current,
                    }

                    # Step-interval evaluation on held-out validation set
                    eval_at_steps = args.eval_strategy in ("steps", "both")
                    if eval_at_steps and val_dataloader and args.eval_every_steps > 0 and total_steps % args.eval_every_steps == 0:
                        val_loss = evaluate_loss(model, val_dataloader, device, is_tpu)
                        log_entry["val_loss"] = val_loss
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            best_adapter_path.mkdir(parents=True, exist_ok=True)
                            unwrap_peft_model(model).save_pretrained(str(best_adapter_path))
                            print(f"\n[VALIDATION] New best adapter saved! Step {total_steps}: val_loss = {val_loss:.4f}")

                    with open(log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(log_entry) + "\n")

                    # Step-interval checkpoint push to HF Hub
                    save_at_steps = args.save_strategy in ("steps", "both")
                    if save_at_steps and args.hf_repo and args.push_every_steps > 0 and total_steps % args.push_every_steps == 0:
                        save_sft_checkpoint(epoch)

        # ── Epoch-End: Guaranteed Validation & Checkpoint ──
        # Evaluate validation loss at every epoch boundary regardless of step-interval,
        # ensuring best_adapter is always compared at the true epoch completion point.
        if is_master and val_dataloader and args.eval_strategy in ("epoch", "both"):
            val_loss = evaluate_loss(model, val_dataloader, device, is_tpu)
            print(f"\n[EPOCH {epoch} EVALUATION] val_loss = {val_loss:.4f} (previous best = {best_val_loss:.4f})")
            epoch_log = {"epoch": epoch, "step": total_steps, "epoch_end_val_loss": val_loss}
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_adapter_path.mkdir(parents=True, exist_ok=True)
                unwrap_peft_model(model).save_pretrained(str(best_adapter_path))
                print(f"[VALIDATION] New best adapter saved at Epoch {epoch} boundary! val_loss = {val_loss:.4f}")
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(epoch_log) + "\n")

        # Epoch-end checkpoint save
        if is_master and args.save_strategy in ("epoch", "both"):
            save_sft_checkpoint(epoch + 1)

        # Multi-core TPU synchronization barrier: ensure master finishes
        # epoch-end validation and checkpoint saving before workers proceed.
        if is_tpu and args.num_cores > 1:
            import torch_xla.core.xla_model as xm
            xm.rendezvous(f"epoch_end_{epoch}")

    if is_master:
        print(f"\n[COMPLETE] Stage 1 SFT ({args.stage}) finished! Final checkpoint saved to {out_path}.")
        if best_adapter_path.exists():
            print(f"[COMPLETE] Best adapter preserved at {best_adapter_path} (best_val_loss={best_val_loss:.4f}).")


def main():
    args = parse_args()
    is_tpu = False
    try:
        import torch_xla  # noqa: F401 — lightweight check; do NOT import xla_model here
        is_tpu = True     # (importing xla_model initializes libtpu and claims /dev/vfio exclusively)
    except ImportError:
        pass

    # Pre-sync clinical traces to local disk if missing, preventing FileNotFoundError in workers
    resolved_traces = resolve_stage_traces(args.stage, args.track, args.data_dir)
    if not resolved_traces:
        traces_repo = os.environ.get("HF_TRACES_REPO", "Reza-Nadimi/vlm-dental-traces")
        print(f"\n[AUTO-SYNC] No local trace files found for stage='{args.stage}'.")
        print(f"[AUTO-SYNC] Synchronizing verified clinical traces from Hugging Face ({traces_repo})...")
        try:
            from huggingface_hub import snapshot_download
            traces_dir = Path(args.data_dir) / "traces"
            traces_dir.mkdir(parents=True, exist_ok=True)
            snapshot_download(
                repo_id=traces_repo,
                repo_type="dataset",
                local_dir=str(traces_dir),
                token=os.environ.get("HF_TOKEN"),
            )
            resolved_traces = resolve_stage_traces(args.stage, args.track, args.data_dir)
            print(f"[AUTO-SYNC] Traces successfully synchronized: {resolved_traces}\n")
        except Exception as e:
            print(f"[AUTO-SYNC WARNING] Could not auto-download traces from {traces_repo}: {e}")

    if is_tpu and args.num_cores > 1:
        # Pre-cache base model to local disk once before spawning 8 worker processes,
        # preventing 8-way concurrent Hugging Face download lock contention.
        if not Path(args.model_id).exists():
            try:
                from huggingface_hub import snapshot_download
                print(f"[PRE-FETCH] Verifying {args.model_id} in local cache before spawning workers...")
                snapshot_download(repo_id=args.model_id)
                print(f"[PRE-FETCH] Base model {args.model_id} ready in local cache.")
            except Exception as e:
                print(f"[PRE-FETCH WARNING] Pre-download check skipped ({e}); workers will load directly.")

        try:
            try:
                import torch_xla.distributed.xla_multiprocessing as xmp
            except ImportError:
                import torch_xla.distributed.xmp as xmp
            print(f"[LAUNCH] Spawning multi-core Cloud TPU v5e-8 training across available TPU cores via xmp.spawn(nprocs=None)...")
            xmp.spawn(run_training, args=(args,), nprocs=None)
        except Exception as e:
            print(f"\n[FATAL TPU ERROR] Could not spawn multi-core training via xmp: {e}")
            print("[DIAGNOSTIC] On Cloud TPU VMs, hardware access to /dev/vfio/* is exclusive.")
            print("[DIAGNOSTIC] If running from Jupyter/Colab/Kaggle, ensure the notebook kernel did not call xm.xla_device() before launching this script.")
            print("[DIAGNOSTIC] Please restart the notebook session (Run -> Restart Session) and re-run to release /dev/vfio/*.\n")
            raise RuntimeError(f"Multi-core Cloud TPU launch failed: {e}") from e
    else:
        run_training(0, args)


if __name__ == "__main__":
    main()
