"""
Environment and hardware detection utilities (§17).

Includes:
- Kaggle / Colab / Local runtime detection (`detect_environment`)
- System & GPU capabilities inspection (`get_system_summary`)
- GRPO VRAM memory consumption estimator (`estimate_grpo_memory_gb`)
"""

from __future__ import annotations

import os
import platform
import sys
from typing import Any


def detect_environment() -> str:
    """Detect whether code is running in Kaggle, Colab, or a local environment."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    return "local"


def get_system_summary() -> dict[str, Any]:
    """Inspect and return system, Python, PyTorch, and CUDA specifications."""
    summary: dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
        "environment": detect_environment(),
    }

    try:
        import torch
        summary["torch_version"] = torch.__version__
        summary["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            summary["gpu_count"] = torch.cuda.device_count()
            summary["gpu_name"] = torch.cuda.get_device_name(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            summary["vram_gb"] = round(vram_bytes / (1024**3), 2)
            summary["bf16_supported"] = torch.cuda.is_bf16_supported()
        else:
            summary["gpu_name"] = "None"
            summary["vram_gb"] = 0.0
            summary["bf16_supported"] = False
    except ImportError:
        summary["torch_version"] = "not_installed"

    return summary


def estimate_grpo_memory_gb(
    group_size: int = 4,
    max_seq_len: int = 2048,
    is_4bit: bool = True,
    lora_rank: int = 16,
) -> dict[str, Any]:
    """Rule-of-thumb VRAM estimate for GRPO with QLoRA + activation checkpointing on Qwen2.5-VL-7B."""
    base_weights = 4.5 if is_4bit else 15.0  # ~4.5 GB for 4-bit 7B
    lora_overhead = 0.3  # trainable adapter weights + optimizer states (AdamW)
    act_per_seq = 0.6 * (max_seq_len / 2048)
    peak_act = act_per_seq * group_size
    vision_features = 1.0  # cached vision encoder output per image
    cuda_context = 1.2
    total = base_weights + lora_overhead + peak_act + vision_features + cuda_context
    return {
        "base_weights_gb": base_weights,
        "lora_and_opt_gb": lora_overhead,
        "peak_activations_gb": round(peak_act, 2),
        "vision_cache_gb": vision_features,
        "cuda_context_gb": cuda_context,
        "estimated_peak_gb": round(total, 2),
        "fits_in_24gb_rtx4090": total <= 23.0,
        "fits_in_16gb_t4": total <= 15.0,
    }
