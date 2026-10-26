#!/usr/bin/env python3
"""Small real-CUDA attention forward/backward probe; loads config, never model weights.

This checks kernel compatibility, not 16k activation memory or complete VLM
training. Run separately before loading 9B weights. See docs/SFT_GPU_RUNBOOK.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from dental_agent.training.gpu_debug import gpu_forward_context, gpu_attention_context
from dental_agent.training.gpu_runtime import (
    native_bf16, resolve_compute_dtype, runtime_metadata, sdpa_gqa_policy, select_attention_backend,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", required=True, help="Existing local model directory; only config.json is read")
    p.add_argument("--precision", choices=["qlora", "bf16", "fp16"], default="qlora")
    p.add_argument("--qlora-compute-dtype", choices=["auto", "bf16", "fp16"], default="auto")
    p.add_argument("--attn-implementation", choices=["auto", "sdpa", "flash_attention_2"], default="auto")
    p.add_argument("--sdpa-repeat-kv", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--probe-tokens", type=int, default=128, help="Synthetic probe length; does not change training max-seq-len")
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("This probe requires real CUDA GPUs; CPU results cannot establish CUDA compatibility")
    if args.probe_tokens < 2:
        p.error("--probe-tokens must be at least 2")
    cfg_path = Path(args.model_id).expanduser() / "config.json"
    config = json.loads(cfg_path.read_text())
    text = config.get("text_config", config)
    heads = int(text["num_attention_heads"])
    kv_heads = int(text["num_key_value_heads"])
    dim = int(text.get("head_dim") or int(text["hidden_size"]) // heads)
    dropout = float(text.get("attention_dropout", 0.0))
    if heads % kv_heads:
        raise SystemExit("Config has incompatible GQA head counts")
    dtype = resolve_compute_dtype(args.precision, args.qlora_compute_dtype)
    if dtype == torch.bfloat16 and not native_bf16():
        raise SystemExit("Native BF16 unavailable; select QLoRA compute fp16 on T4")
    backend = select_attention_backend(args.attn_implementation)
    print(json.dumps(runtime_metadata(), indent=2), flush=True)
    print(f"[PROBE] backend={backend} dtype={dtype} Q/KV heads={heads}/{kv_heads} "
          f"head_dim={dim} dropout={dropout} tokens={args.probe_tokens}", flush=True)
    with sdpa_gqa_policy(args.sdpa_repeat_kv):
        for i in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(i)
            shape = (1, heads, args.probe_tokens, dim)
            kv_shape = (1, kv_heads, args.probe_tokens, dim)
            q = torch.randn(shape, device=f"cuda:{i}", dtype=dtype, requires_grad=True)
            k = torch.randn(kv_shape, device=f"cuda:{i}", dtype=dtype, requires_grad=True)
            v = torch.randn(kv_shape, device=f"cuda:{i}", dtype=dtype, requires_grad=True)
            with gpu_forward_context(True, dtype, require_efficient_sdpa=True):
                if backend == "sdpa":
                    from transformers.integrations.sdpa_attention import sdpa_attention_forward
                    module = SimpleNamespace(num_key_value_groups=heads // kv_heads, is_causal=True)
                    out, _ = sdpa_attention_forward(module, q, k, v, None, dropout=dropout, scaling=dim ** -0.5)
                else:
                    from flash_attn import flash_attn_func
                    out = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                          dropout_p=dropout, softmax_scale=dim ** -0.5, causal=True)
                loss = out.float().square().mean()
            with gpu_attention_context(True, require_efficient_sdpa=True):
                loss.backward()
            torch.cuda.synchronize(i)
            if not torch.isfinite(out).all().item() or not all(t.grad is not None and torch.isfinite(t.grad).all().item() for t in (q, k, v)):
                raise RuntimeError(f"Nonfinite/missing attention outputs or gradients on GPU {i}")
            print(f"[PASS GPU {i}] {torch.cuda.get_device_name(i)} {backend} forward/backward; "
                  f"peak allocated={torch.cuda.max_memory_allocated(i)/2**30:.3f} GiB", flush=True)
            del q, k, v, out, loss
    print("[PROBE COMPLETE] Small kernel probe passed. Full VLM integration and long-context VRAM still require training.")


if __name__ == "__main__":
    main()
