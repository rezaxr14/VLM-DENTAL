#!/usr/bin/env python3
"""Check that QLoRA quantizes the language model only (no training, no data).

    python scripts/check_qlora_load.py --model-id D:/models/Qwen3.5-9B            # RTX 4090: loads 4-bit, asserts, prints a summary
    python scripts/check_qlora_load.py --model-id D:/models/Qwen3.5-9B --names-only   # any machine: module names + what 4-bit would touch

Full check (CUDA + bitsandbytes), loading exactly as ``train_sft.py --precision qlora`` does (same ``qlora_bnb_config``, same
``prepare_qlora_model``, same LoRA targets), asserts:
  (a) no 4-bit Linear anywhere under the vision tower (the projector is a plain 16-bit Linear),
  (b) every language-model Linear (lm_head excepted) is a 4-bit Linear,
  (c) every LoRA target module is found, and the projector LoRA attaches to the plain merger layers.
``--names-only`` builds the model on the meta device (config only, no weights, no GPU) and prints the vision tower / merger Linear
names and which layers the skip list leaves unquantized. Exit code 0 = pass, 1 = a check failed, 2 = cannot run here.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model-id", required=True, help="Local folder (config.json + weights) or Hub id; same value as train_sft.py --model-id")
    ap.add_argument("--lora-linear-attn", action=argparse.BooleanOptionalAction, default=True, help="Same meaning as train_sft.py")
    ap.add_argument("--lora-target-vision", choices=["projector", "none"], default="projector", help="Same meaning as train_sft.py")
    ap.add_argument("--names-only", action="store_true", help="Meta-device name listing only (no weights, no GPU, no bitsandbytes)")
    return ap.parse_args(argv)


def list_names(model_id: str) -> int:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig
    from transformers.quantizers.base import HfQuantizer
    from transformers.quantizers.quantizers_utils import should_convert_module

    from dental_agent.config import QLORA_SKIP_MODULES
    from dental_agent.model.backbone import get_model_classes, is_vision_param_name, projector_module_names, qlora_bnb_config

    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    with init_empty_weights():
        model = get_model_classes().from_config(cfg, trust_remote_code=True)
    print(f"[NAMES] {type(model).__name__} (model_type {cfg.model_type}) built on the meta device")
    for n, c in model.named_modules():
        if n.count(".") <= 2 and is_vision_param_name(n + ".x") and n.split(".")[-1] in ("visual", "patch_embed", "pos_embed", "rotary_pos_emb", "blocks", "merger"):
            print(f"[NAMES]   {n}  ({type(c).__name__})")
    linears = [n for n, c in model.named_modules() if type(c) is torch.nn.Linear]
    vision = [n for n in linears if is_vision_param_name(n)]
    print(f"[NAMES] vision-tower Linear layers: {len(vision)}; first: {vision[:3]}")
    print(f"[NAMES] projector (merger) Linear layers (LoRA targets with --lora-target-vision projector): {projector_module_names(model)}")
    patterns = HfQuantizer.get_modules_to_not_convert(model, qlora_bnb_config(torch.bfloat16).llm_int8_skip_modules, getattr(model, "_keep_in_fp32_modules", None))
    converted = [n for n in linears if should_convert_module(n, patterns)]
    kept = [n for n in linears if n not in set(converted)]
    print(f"[NAMES] skip list {list(QLORA_SKIP_MODULES)} -> {len(converted)} Linear layers would be 4-bit, {len(kept)} stay 16-bit "
          f"({len([n for n in kept if is_vision_param_name(n)])} vision tower, others: {[n for n in kept if not is_vision_param_name(n)]})")
    leaked = [n for n in converted if is_vision_param_name(n)]
    if leaked:
        print(f"[FAIL] {len(leaked)} vision-tower layers would still be quantized: {leaked[:3]}")
        return 1
    print("[PASS] no vision-tower layer would be quantized")
    return 0


def full_check(args) -> int:
    import torch

    from dental_agent.model.backbone import (
        check_qlora_model, get_model_classes, lora_coverage, lora_target_modules, prepare_qlora_model, qlora_bnb_config,
    )

    try:
        import bitsandbytes  # noqa: F401
    except Exception as e:
        print(f"[CANNOT RUN] bitsandbytes is not importable here ({e}). Run this on the CUDA machine, or use --names-only.")
        return 2
    if not torch.cuda.is_available():
        print("[CANNOT RUN] no CUDA device. Run this on the RTX 4090, or use --names-only.")
        return 2

    from transformers import AutoProcessor
    AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)          # the processor/tokenizer must load from this id too
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"[LOAD] {args.model_id}: 4-bit NF4 + double quant, compute {dtype}, skip list from dental_agent.config.QLORA_SKIP_MODULES")
    model = get_model_classes().from_pretrained(
        args.model_id, trust_remote_code=True, dtype=dtype, low_cpu_mem_usage=True, device_map="auto",
        quantization_config=qlora_bnb_config(dtype),
    )
    targets = lora_target_modules(model, linear_attention=args.lora_linear_attn, vision_projector=args.lora_target_vision == "projector")
    s = check_qlora_model(model, targets)
    print(f"[PASS] after load: vision Linears {s['vision_linears']} (4-bit: {s['vision_linear4bit']}, dtype {s['vision_dtypes']}), "
          f"language Linears {s['language_linear4bit']}/{s['language_linears']} 4-bit, lm_head 4-bit: {s['lm_head_quantized']}")

    model = prepare_qlora_model(model, use_gradient_checkpointing=False)
    from peft import LoraConfig, get_peft_model
    wrapped = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=targets, bias="none", task_type="CAUSAL_LM"))
    s = check_qlora_model(wrapped, targets)
    cov = lora_coverage(wrapped)
    print(f"[PASS] after prepare + LoRA: vision dtype {s['vision_dtypes']}, LoRA targets found {s['lora_targets']}/{s['lora_targets']}, adapted modules {cov}")
    if args.lora_target_vision == "projector":
        merger_lora = [n for n, _ in wrapped.named_parameters() if "lora_A" in n and "merger" in n]
        if len(merger_lora) != len(s["merger_linears"]) or not merger_lora:
            print(f"[FAIL] projector LoRA expected on {s['merger_linears']} but found {merger_lora}")
            return 1
        print(f"[PASS] projector LoRA attached to the plain Linear merger layers: {s['merger_linears']}")
    free, total = torch.cuda.mem_get_info()
    print(f"[MEMORY] CUDA allocated {torch.cuda.memory_allocated() / 2**30:.1f} GiB, reserved {torch.cuda.memory_reserved() / 2**30:.1f} GiB "
          f"of {total / 2**30:.1f} GiB (weights + LoRA only, no activations)")
    print("[RESULT] QLoRA quantizes the language model only. Summary:", {k: v for k, v in s.items() if k not in ("problems", "merger_linears")})
    return 0


def main(argv=None) -> int:
    args = parse_args(argv)
    from dental_agent.utils.inputs import InputNotFound, resolve_base_model

    try:
        resolved = resolve_base_model(args.model_id, hub="off")
    except InputNotFound as e:
        print(e)
        return 2
    print(f"[MODEL] {args.model_id} -> {resolved.path} ({resolved.source})")
    if args.names_only:
        return list_names(args.model_id)
    try:
        return full_check(args)
    except AssertionError as e:
        print(f"[FAIL] {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
