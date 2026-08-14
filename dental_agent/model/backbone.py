"""
Model backbone initialization and LoRA wrapping.

Loads Qwen/Qwen3.5-9B (or configured multimodal backbone) with 4-bit NF4 quantization and QLoRA adapters.
"""

from __future__ import annotations

from typing import Any, Tuple
from dental_agent.config import ProjectConfig, ModelConfig


def get_model_classes():
    """Dynamically import vision-language / multimodal model classes with graceful fallback."""
    try:
        from transformers import AutoModelForImageTextToText as ModelClass
    except ImportError:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as ModelClass
            except ImportError:
                from transformers import AutoModelForCausalLM as ModelClass
    return ModelClass


def load_model(
    config: ProjectConfig | ModelConfig | None = None,
    device_map: str = "auto",
) -> Tuple[Any, Any]:
    """Load multimodal backbone and processor with 4-bit NF4 quantization.

    Returns (model, processor).
    """
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    model_cfg = config.model if isinstance(config, ProjectConfig) else (config or ModelConfig())
    ModelClass = get_model_classes()

    bnb_config = None
    if model_cfg.load_in_4bit and torch.cuda.is_available():
        compute_dtype = (
            torch.bfloat16
            if model_cfg.bnb_compute_dtype == "bfloat16"
            else torch.float16
        )
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.bnb_quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_cfg.bnb_double_quant,
        )

    processor = AutoProcessor.from_pretrained(model_cfg.name, trust_remote_code=True)
    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model_kwargs: dict[str, Any] = {
        "device_map": device_map if torch.cuda.is_available() else None,
        "trust_remote_code": True,
    }
    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config
    elif not torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.float32

    model = ModelClass.from_pretrained(model_cfg.name, **model_kwargs)
    return model, processor


def apply_lora(
    model: Any,
    config: ProjectConfig | None = None,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    """Prepare quantized model and attach LoRA adapter."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if config:
        lora_cfg = config.training.lora
        r, alpha, dropout = lora_cfg.r, lora_cfg.alpha, lora_cfg.dropout
        target_modules = lora_cfg.target_modules

    target_modules = target_modules or [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ]

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    try:
        model = prepare_model_for_kbit_training(model)
    except Exception:
        pass

    model = get_peft_model(model, peft_config)
    return model


def estimate_grpo_memory(
    param_count_billions: float = 3.0,
    quant_bits: int = 4,
    group_size: int = 4,
    seq_len: int = 2048,
    lora_fraction: float = 0.01,
) -> dict[str, float]:
    """Analytical GRPO VRAM budget estimator."""
    base_model_gb = param_count_billions * 1e9 * (quant_bits / 8) / (1024**3)
    lora_params = param_count_billions * 1e9 * lora_fraction
    lora_optimizer_gb = lora_params * 16 / (1024**3)
    lora_weights_gb = lora_params * 2 / (1024**3)

    # Approximated activation & KV memory per group rollout
    activations_gb = (group_size * seq_len * 4096 * 4 * 2) / (1024**3)
    ref_policy_pass_gb = (seq_len * 4096 * 2) / (1024**3)
    cuda_overhead_gb = 1.2

    total_gb = (
        base_model_gb
        + lora_optimizer_gb
        + lora_weights_gb
        + activations_gb
        + ref_policy_pass_gb
        + cuda_overhead_gb
    )

    return {
        "base_model_gb": round(base_model_gb, 2),
        "lora_weights_gb": round(lora_weights_gb, 2),
        "lora_optimizer_gb": round(lora_optimizer_gb, 2),
        "activations_gb": round(activations_gb, 2),
        "cuda_overhead_gb": round(cuda_overhead_gb, 2),
        "total_estimated_vram_gb": round(total_gb, 2),
    }
