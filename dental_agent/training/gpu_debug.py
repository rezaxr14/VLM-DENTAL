"""CUDA SFT diagnostics. No hardware initialization at import time."""
from __future__ import annotations

import contextlib
import warnings
import torch


def gpu_memory(phase: str) -> None:
    for i in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(i)
        print(f"[GPU {i}] {phase}: allocated={torch.cuda.memory_allocated(i)/2**30:.2f} "
              f"reserved={torch.cuda.memory_reserved(i)/2**30:.2f} "
              f"peak={torch.cuda.max_memory_allocated(i)/2**30:.2f} "
              f"free={free/2**30:.2f}/{total/2**30:.2f} GiB", flush=True)


@contextlib.contextmanager
def gpu_attention_context(enabled: bool, require_efficient_sdpa: bool = False):
    with contextlib.ExitStack() as stack:
        if enabled and require_efficient_sdpa:
            stack.enter_context(warnings.catch_warnings())
            warnings.simplefilter("always", UserWarning)
            from torch.nn.attention import SDPBackend, sdpa_kernel
            stack.enter_context(sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]))
        yield


@contextlib.contextmanager
def gpu_forward_context(enabled: bool, dtype, require_efficient_sdpa: bool = False):
    with contextlib.ExitStack() as stack:
        if enabled:
            stack.enter_context(torch.autocast("cuda", dtype=dtype))
        stack.enter_context(gpu_attention_context(enabled, require_efficient_sdpa))
        yield


def check_adapter_gradients(model):
    """Validate actual gradients; keep one small nonzero LoRA tensor for update comparison."""
    seen = nonzero = 0
    probe = None
    for name, p in model.named_parameters():
        if not p.requires_grad or p.grad is None:
            continue
        seen += 1
        if not torch.isfinite(p.grad).all().item():
            raise RuntimeError(f"[DEBUG] nonfinite gradient in {name}")
        if torch.count_nonzero(p.grad).item():
            nonzero += 1
            if probe is None and "lora_" in name:
                probe = (name, p, p.detach().clone())
    if probe is None:
        raise RuntimeError(f"[DEBUG] no nonzero LoRA gradient ({seen} gradient tensors)")
    print(f"[GRAD] {seen} finite gradient tensors, {nonzero} nonzero; probe={probe[0]}", flush=True)
    return probe


def check_adapter_update(probe, allow_unchanged: bool = False):
    name, p, before = probe
    if not torch.isfinite(p).all().item():
        raise RuntimeError(f"[DEBUG] nonfinite adapter weights after optimizer: {name}")
    delta = (p.detach() - before).abs().max().item()
    if delta == 0 and not allow_unchanged:
        raise RuntimeError(f"[DEBUG] optimizer did not update {name}")
    print(f"[UPDATE] {name}: max_abs_delta={delta:.8g}" +
          (" (zero learning rate during warmup)" if delta == 0 and allow_unchanged else ""), flush=True)
