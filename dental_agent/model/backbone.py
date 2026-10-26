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
        from transformers import AutoModelForMultimodalLM as ModelClass
    except ImportError:
        try:
            from transformers import Qwen3_5ForConditionalGeneration as ModelClass
        except ImportError:
            try:
                from transformers import AutoModelForImageTextToText as ModelClass
            except ImportError:
                try:
                    from transformers import AutoModelForVision2Seq as ModelClass
                except ImportError:
                    try:
                        from transformers import AutoModelForCausalLM as ModelClass
                    except ImportError:
                        from transformers import Qwen2_5_VLForConditionalGeneration as ModelClass
    return ModelClass


def load_model(
    config: ProjectConfig | ModelConfig | None = None,
    device_map: str = "auto",
) -> Tuple[Any, Any]:
    """Load multimodal backbone and processor with 4-bit NF4 quantization.

    Returns (model, processor).
    """
    import torch
    from transformers import AutoProcessor

    model_cfg = config.model if isinstance(config, ProjectConfig) else (config or ModelConfig())
    ModelClass = get_model_classes()

    bnb_config = None
    if model_cfg.load_in_4bit and torch.cuda.is_available():
        compute_dtype = (
            torch.bfloat16
            if model_cfg.bnb_compute_dtype == "bfloat16"
            else torch.float16
        )
        bnb_config = qlora_bnb_config(compute_dtype, quant_type=model_cfg.bnb_quant_type, double_quant=model_cfg.bnb_double_quant)

    try:
        processor = AutoProcessor.from_pretrained(model_cfg.name, trust_remote_code=True)
    except Exception as e:
        if "does not recognize this architecture" in str(e) or "qwen3_5" in str(e):
            raise RuntimeError(
                f"Your installed Transformers version is too old to load '{model_cfg.name}' (architecture 'qwen3_5'). "
                f"Please run `pip install --upgrade transformers>=5.0.0` and restart the session. Original error: {e}"
            ) from e
        raise

    if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
        if processor.tokenizer.pad_token_id is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    if torch.cuda.is_available():
        # On Turing sm_75 (Tesla T4) and above, enable PyTorch native SDPA Memory-Efficient Attention
        # (Cutlass backend) to eliminate the O(N^2) dense attention allocation on high-resolution images.
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        major_cap = torch.cuda.get_device_capability()[0]
        if major_cap < 8 and hasattr(torch.backends.cuda, "enable_flash_sdp"):
            torch.backends.cuda.enable_flash_sdp(False)

    model_kwargs: dict[str, Any] = {
        "device_map": device_map if torch.cuda.is_available() else None,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        model_kwargs["attn_implementation"] = "sdpa"
        if bnb_config is not None:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        model_kwargs["torch_dtype"] = torch.float32

    try:
        model = ModelClass.from_pretrained(model_cfg.name, **model_kwargs)
    except Exception as e:
        if "does not recognize this architecture" in str(e) or "qwen3_5" in str(e):
            raise RuntimeError(
                f"Your installed Transformers version is too old to load '{model_cfg.name}' (architecture 'qwen3_5'). "
                f"Please run `pip install --upgrade transformers>=5.0.0` and restart the session. Original error: {e}"
            ) from e
        raise
    return model, processor


def qlora_bnb_config(compute_dtype: Any, quant_type: str = "nf4", double_quant: bool = True) -> Any:
    """The one 4-bit config for SFT, GRPO, evaluation and inference: NF4 (default), double quantization, ``compute_dtype`` compute,
    and ``QLORA_SKIP_MODULES`` left in 16-bit so only the language model is quantized (the vision tower, projector included, and
    lm_head are not). Quantization settings are the caller's; only the skip list is fixed here."""
    from transformers import BitsAndBytesConfig
    from dental_agent.config import QLORA_SKIP_MODULES

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=quant_type,
        bnb_4bit_use_double_quant=double_quant,
        bnb_4bit_compute_dtype=compute_dtype,
        llm_int8_skip_modules=list(QLORA_SKIP_MODULES),
    )


def is_vision_param_name(name: str) -> bool:
    """True for a parameter or module name inside the vision tower (``model.visual.*`` or ``visual.*``)."""
    return ".visual." in f".{name}."


def prepare_qlora_model(model: Any, **kwargs: Any) -> Any:
    """``peft.prepare_model_for_kbit_training`` that keeps frozen 16-bit weights in the dtype they were loaded in.

    peft upcasts every non-4-bit 16-bit parameter to fp32. For this model that is wasteful in two places, both frozen:
      * the vision tower (about +0.9 GB and an fp32 vision forward);
      * the token embedding and lm_head (248,320 x 4,096 each: 3.8 GiB apiece in fp32, 1.9 GiB in bf16). An fp32 embedding also
        makes the whole residual stream fp32 (transformers then logs "Casting fp32 inputs back to torch.bfloat16 for flash-attn"),
        which doubles every activation and every checkpointed layer input; the bf16 path and the TPU path run a bf16 stream.
    All three are put back to the dtype they were loaded in (the bf16 values are exact; only the arithmetic precision of the
    residual adds and the head matmul differs, as in the bf16 path). Norm weights stay fp32, which is what peft's upcast is for.
    No precision is chosen here."""
    from peft import prepare_model_for_kbit_training

    heavy = {id(m.weight) for m in (model.get_input_embeddings(), model.get_output_embeddings()) if m is not None and hasattr(m, "weight")}
    before = {n: p.dtype for n, p in model.named_parameters() if is_vision_param_name(n)}
    before_heavy = {n: p.dtype for n, p in model.named_parameters() if id(p) in heavy}
    model = prepare_model_for_kbit_training(model, **kwargs)
    restored = restored_heavy = 0
    for n, p in model.named_parameters():
        want = before.get(n) or before_heavy.get(n)
        if want is not None and p.dtype != want:
            p.data = p.data.to(want)
            restored += n in before
            restored_heavy += n in before_heavy
    if restored:
        print(f"[QLORA] Vision tower kept in {next(iter(set(before.values())))} ({restored} tensors not upcast to fp32).")
    if restored_heavy:
        print(f"[QLORA] Token embedding and lm_head kept in {next(iter(set(before_heavy.values())))} ({restored_heavy} tensors not upcast to fp32; norms stay fp32).")
    return model


def check_qlora_model(
    model: Any,
    target_modules: list[str] | None = None,
    linear4bit_cls: Any = None,
) -> dict[str, Any]:
    """Assert a QLoRA-loaded model is quantized where it must be and only there; returns a summary, raises AssertionError.

    (a) no 4-bit Linear anywhere under the vision tower (the merger is plain Linear, so projector LoRA attaches to it);
    (b) every Linear of the language model, lm_head excepted, is a 4-bit Linear;
    (c) every LoRA target (``target_modules``, suffix names such as ``q_proj`` or full names) matches at least one module.
    ``linear4bit_cls`` defaults to ``bitsandbytes.nn.Linear4bit`` (needs bitsandbytes); tests pass a stand-in class."""
    import torch

    if linear4bit_cls is None:
        try:
            from bitsandbytes.nn import Linear4bit as linear4bit_cls
        except Exception as e:  # pragma: no cover - environment
            raise RuntimeError(f"check_qlora_model needs bitsandbytes (CUDA) to recognise 4-bit layers: {e}") from e

    # Works before and after peft wraps the model: LoRA's own A/B Linears are not base layers, and a wrapped layer's real
    # weight sits in ``<name>.base_layer``.
    linears = [(n[: -len(".base_layer")] if n.endswith(".base_layer") else n, m)
               for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and ".lora_" not in n]
    vision = [(n, m) for n, m in linears if is_vision_param_name(n)]
    head = [(n, m) for n, m in linears if n == "lm_head" or n.endswith(".lm_head")]
    head_names = {n for n, _ in head}
    language = [(n, m) for n, m in linears if not is_vision_param_name(n) and n not in head_names]
    q = lambda m: isinstance(m, linear4bit_cls)  # noqa: E731

    bad_vision = [n for n, m in vision if q(m)]
    bad_language = [n for n, m in language if not q(m)]
    missing_targets: list[str] = []
    for t in (target_modules or []):
        if not any(n == t or n.endswith("." + t) for n, _ in linears):
            missing_targets.append(t)
    merger = [n for n, _ in vision if "merger" in n]

    summary = {
        "vision_linears": len(vision), "vision_linear4bit": len(bad_vision),
        "vision_dtypes": sorted({str(m.weight.dtype).replace("torch.", "") for _, m in vision}),
        "merger_linears": merger,
        "language_linears": len(language), "language_linear4bit": len(language) - len(bad_language),
        "lm_head_quantized": any(q(m) for _, m in head),
        "lora_targets": len(target_modules or []), "lora_targets_missing": missing_targets,
    }
    problems = []
    if bad_vision:
        problems.append(f"(a) {len(bad_vision)} vision-tower layers are 4-bit, e.g. {bad_vision[:3]} - the skip list did not apply")
    if not vision:
        problems.append("(a) no Linear layers found under the vision tower - module names changed? check QLORA_SKIP_MODULES")
    if bad_language:
        problems.append(f"(b) {len(bad_language)} language-model layers are NOT 4-bit, e.g. {bad_language[:3]} - this is not a QLoRA load")
    if not language:
        problems.append("(b) no language-model Linear layers found")
    if summary["lm_head_quantized"]:
        problems.append("(b) lm_head is 4-bit; it is meant to stay 16-bit (QLORA_SKIP_MODULES lists it)")
    if missing_targets:
        problems.append(f"(c) LoRA targets matching no module: {missing_targets}")
    summary["ok"] = not problems
    summary["problems"] = problems
    if problems:
        raise AssertionError("QLoRA load check failed:\n  " + "\n  ".join(problems) + f"\n  summary: {summary}")
    return summary


def projector_module_names(model: Any) -> list[str]:
    """Full names of the Linear layers of the vision-to-text projector (the ``merger`` modules under ``visual``).

    Discovered from the loaded model because the module names differ across transformers versions
    (``merger.linear_fc1/2`` today, ``merger.mlp.0/2`` before); a hardcoded name silently matches nothing.
    """
    import torch

    return [
        name for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and ".visual." in f".{name}." and "merger" in name
    ]


def lora_target_modules(model: Any = None, linear_attention: bool = True, vision_projector: bool = False) -> list[str]:
    """LoRA target projections. ``linear_attention`` adds the Gated-DeltaNet layers' in/out projections;
    ``vision_projector`` adds the projector's Linear layers (needs ``model``; raises if none are found)."""
    from dental_agent.config import LORA_FULL_ATTENTION_TARGETS, LORA_LINEAR_ATTENTION_TARGETS, LORA_MLP_TARGETS

    targets = [*LORA_FULL_ATTENTION_TARGETS, *LORA_MLP_TARGETS]
    if linear_attention:
        targets += LORA_LINEAR_ATTENTION_TARGETS
    if vision_projector:
        if model is None:
            raise ValueError("lora_target_modules(vision_projector=True) needs the loaded model to find the projector.")
        names = projector_module_names(model)
        if not names:
            raise ValueError("Projector LoRA requested (--lora-target-vision projector) but no vision merger Linear "
                             "layers were found in this model. Use --lora-target-vision none.")
        targets += names
    return targets


def lora_coverage(model: Any) -> dict[str, int]:
    """Count LoRA-adapted modules by kind (full-attention, linear-attention, mlp, projector) for logging."""
    counts = {"full_attention": 0, "linear_attention": 0, "mlp": 0, "projector": 0}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A"):
            continue
        if ".linear_attn." in name:
            counts["linear_attention"] += 1
        elif ".self_attn." in name:
            counts["full_attention"] += 1
        elif ".mlp." in name and "merger" not in name:
            counts["mlp"] += 1
        elif "merger" in name:
            counts["projector"] += 1
    return counts


def apply_lora(
    model: Any,
    config: ProjectConfig | None = None,
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules: list[str] | None = None,
) -> Any:
    """Prepare quantized model and attach LoRA adapter."""
    from peft import LoraConfig, get_peft_model

    if config:
        lora_cfg = config.training.lora
        r, alpha, dropout = lora_cfg.r, lora_cfg.alpha, lora_cfg.dropout
        target_modules = lora_cfg.target_modules

    target_modules = target_modules or lora_target_modules()

    peft_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    try:
        model = prepare_qlora_model(model)
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


def _extract_pil_images(messages: list[dict[str, Any]]) -> list[Any]:
    """Collect PIL images from message content in document order (no resizing)."""
    from PIL import Image

    images: list[Any] = []
    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "image":
                img = item.get("image", item.get("image_url"))
                if isinstance(img, Image.Image):
                    images.append(img if img.mode == "RGB" else img.convert("RGB"))
                else:
                    return []  # non-PIL source (path/url/base64): defer to qwen_vl_utils
    return images


def safe_process_vision_info(messages: list[dict[str, Any]]) -> Tuple[Any, Any]:
    """Return a clean ``(image_inputs, video_inputs)`` 2-tuple for ``processor(...)``.

    Fast path (all images are already PIL, which is always true in SFT/GRPO here):
    return them untouched. This deliberately bypasses ``qwen_vl_utils``, whose
    ``process_vision_info`` defaults to ``image_patch_size=14`` (resize factor 28) and would
    resample our canonical 32-aligned images (e.g. 1536x768 -> 1540x756) before the HF
    processor resamples them a second time.

    Fallback (paths/URLs/base64): call ``qwen_vl_utils`` with ``image_patch_size=16`` when the
    installed version supports it. Failures are surfaced (warning) instead of silently
    returning ``(None, None)`` for messages that do contain images.
    """
    pil_images = _extract_pil_images(messages)
    if pil_images:
        return pil_images, None

    has_image = any(
        isinstance(m, dict)
        and isinstance(m.get("content"), list)
        and any(isinstance(i, dict) and i.get("type") == "image" for i in m["content"])
        for m in messages
    )
    if not has_image:
        return None, None

    try:
        from qwen_vl_utils import process_vision_info

        try:
            vision_res = process_vision_info(messages, image_patch_size=16)
        except TypeError:  # older qwen_vl_utils without image_patch_size
            vision_res = process_vision_info(messages)
        if isinstance(vision_res, (tuple, list)):
            image_inputs = vision_res[0] if len(vision_res) > 0 else None
            video_inputs = vision_res[1] if len(vision_res) > 1 else None
            return image_inputs, video_inputs
    except Exception as exc:  # noqa: BLE001
        import warnings

        warnings.warn(
            f"safe_process_vision_info: could not load images via qwen_vl_utils ({exc!r}); "
            "returning (None, None) although the messages contain images.",
            RuntimeWarning,
            stacklevel=2,
        )
    return None, None
