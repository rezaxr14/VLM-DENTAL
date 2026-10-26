"""Explicit CUDA precision, scoped T4 workarounds, and reproducibility metadata.

No CUDA context or monkey patch is installed at import time. The workarounds
are active only inside the training process/context, never on TPU by default.
"""
from __future__ import annotations

import contextlib
import faulthandler
import importlib.metadata
import torch


def native_bf16() -> bool:
    return torch.cuda.is_available() and all(
        torch.cuda.get_device_capability(i)[0] >= 8
        for i in range(torch.cuda.device_count())
    )


def resolve_compute_dtype(precision: str, qlora_compute_dtype: str = "auto"):
    """Explicit fp16 always means fp16. QLoRA compute is independent of NF4 storage."""
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16":
        return torch.float16
    if precision == "qlora":
        if qlora_compute_dtype == "auto":
            return torch.bfloat16 if native_bf16() else torch.float16
        if qlora_compute_dtype not in {"bf16", "fp16"}:
            raise ValueError("qlora_compute_dtype must be auto, bf16 or fp16")
        return torch.bfloat16 if qlora_compute_dtype == "bf16" else torch.float16
    raise ValueError(f"Unsupported precision: {precision}")


def select_attention_backend(requested: str, is_tpu: bool = False) -> str:
    if is_tpu:
        if requested not in {"auto", "sdpa"}:
            raise ValueError("TPU requires --attn-implementation sdpa (or auto)")
        return "sdpa"
    if requested == "sdpa":
        return requested
    if requested not in {"auto", "flash_attention_2"}:
        raise ValueError(f"Unsupported attention backend: {requested}")
    compatible = torch.cuda.is_available() and all(
        torch.cuda.get_device_capability(i)[0] >= 8
        for i in range(torch.cuda.device_count())
    )
    if not compatible:
        if requested == "flash_attention_2":
            raise ValueError("This CUDA FlashAttention-2 path requires SM80+; use SDPA on T4")
        return "sdpa"
    try:
        import flash_attn  # noqa: F401
    except Exception as exc:
        if requested == "flash_attention_2":
            raise RuntimeError(f"Requested FlashAttention-2 cannot import: {exc}") from exc
        print(f"[ATTN] FlashAttention-2 unavailable ({exc!r}); selecting SDPA. "
              "Run scripts/probe_sft_attention.py to verify fused forward/backward.", flush=True)
        return "sdpa"
    return "flash_attention_2"


@contextlib.contextmanager
def sdpa_gqa_policy(mode: str = "auto"):
    """Repeat KV via HF's existing branch on pre-Ampere CUDA; restore on exit.

    'on' forces repetition (also useful for a CPU equivalence test); 'off'
    leaves the installed Transformers policy untouched. On modern CUDA auto
    preserves native GQA. This does not enable the quadratic math backend.
    """
    if mode not in {"auto", "on", "off"}:
        raise ValueError("sdpa_repeat_kv must be auto, on or off")
    needed = mode == "on" or (mode == "auto" and torch.cuda.is_available() and any(
        torch.cuda.get_device_capability(i)[0] < 8 for i in range(torch.cuda.device_count())
    ))
    if not needed:
        yield
        return
    import transformers.integrations.sdpa_attention as sdpa
    original = getattr(sdpa, "use_gqa_in_sdpa", None)
    if not callable(original):
        raise RuntimeError("Transformers SDPA helper changed; use --sdpa-repeat-kv off and inspect before adapting")

    def choose(attention_mask, key, value):
        if mode == "on" or (key.device.type == "cuda" and torch.cuda.get_device_capability(key.device)[0] < 8):
            return False
        return original(attention_mask, key, value)

    sdpa.use_gqa_in_sdpa = choose
    print(f"[SDPA] KV repetition policy={mode}; built-in repeat_kv branch; math fallback policy unchanged", flush=True)
    try:
        yield
    finally:
        sdpa.use_gqa_in_sdpa = original


def _staged_materialize(tensor, device=None, dtype=None):
    tensor = tensor[...]  # safetensors slice -> real tensor
    if tensor.device.type == "cpu" and dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device=device, dtype=dtype, non_blocking=False)


@contextlib.contextmanager
def model_loading_context(mode: str = "standard", cpu_threads: int = 0, traceback_seconds: int = 0):
    """Scope CPU thread limit and separate CPU cast/device transfer to loading only.

    We observed the combined BF16->FP16 copy stall on Kaggle T4. This context
    was the successful workaround; it does not prove a universal root cause.
    """
    if mode not in {"standard", "staged"} or cpu_threads < 0 or traceback_seconds < 0:
        raise ValueError("Invalid model loading options")
    previous_threads = torch.get_num_threads()
    cml = original = None
    timer_started = False
    fault_was_enabled = faulthandler.is_enabled()
    try:
        if cpu_threads:
            torch.set_num_threads(cpu_threads)
        if mode == "staged":
            import transformers.core_model_loading as cml
            original = getattr(cml, "_materialize_copy", None)
            if not callable(original):
                raise RuntimeError("Transformers materializer changed; staged loading needs adaptation")
            cml._materialize_copy = _staged_materialize
        if traceback_seconds:
            faulthandler.enable()
            faulthandler.dump_traceback_later(traceback_seconds, repeat=True)
            timer_started = True
        print(f"[LOAD] mode={mode}; CPU threads during load={torch.get_num_threads()}; "
              f"traceback interval={traceback_seconds}s", flush=True)
        yield
    finally:
        if timer_started:
            faulthandler.cancel_dump_traceback_later()
            if not fault_was_enabled:
                faulthandler.disable()
        if original is not None:
            cml._materialize_copy = original
        if cpu_threads:
            torch.set_num_threads(previous_threads)


def runtime_metadata() -> dict:
    versions = {}
    for name in ("torch", "transformers", "peft", "accelerate", "bitsandbytes", "flash-attn", "flash-linear-attention", "causal-conv1d"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    gpus = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({"index": i, "name": props.name,
                     "compute_capability": list(torch.cuda.get_device_capability(i)),
                     "total_gib": round(props.total_memory / 2**30, 3)})
    return {"versions": versions, "torch_cuda": torch.version.cuda, "gpus": gpus}
