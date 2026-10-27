"""Opt-in, fixed-shape TPU linear cross-entropy with a recomputed backward.

This module is not imported by the CUDA path. No boolean compaction, .item(),
host synchronization, full-sequence logits, or Python-unrolled XLA chunk loop.
Only a frozen plain Linear head is supported (the normal Qwen LoRA setup).
CPU execution is a mathematical reference, not an XLA simulator.
"""
from __future__ import annotations

import functools
import types
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


class TPULossOutput(NamedTuple):
    """Attribute access for the trainer, tuple output for FSDPv2 backward hooks."""
    loss: torch.Tensor
    logits: None = None
    past_key_values: None = None
    hidden_states: Any = None
    attentions: Any = None


def validate_tpu_loss_args(args, *, is_tpu: bool):
    n = getattr(args, "tpu_loss_chunk_tokens", 0)
    if n < 0:
        raise ValueError("--tpu-loss-chunk-tokens must be nonnegative")
    if not n:
        return
    if not is_tpu:
        raise ValueError("--tpu-loss-chunk-tokens requires a real TPU; GPU/CPU path is unchanged")
    if getattr(args, "loss_chunk_tokens", 0):
        raise ValueError("TPU requires --loss-chunk-tokens 0; use --tpu-loss-chunk-tokens instead")
    if not (getattr(args, "xla_spmd", False) and getattr(args, "fsdp", False)
            and getattr(args, "pad_vision_to_slots", False)):
        raise ValueError("TPU chunked SFT requires --spmd --fsdp --pad-vision-to-slots (fixed shapes)")


def require_xla_loop():
    """Capability check before loading weights; do not silently unroll on failure.

    Use scan's non-autograd core: its lowering builds one XLA while-loop body.
    Public autograd scan would save per-iteration activations; this outer
    autograd Function instead supplies an explicitly recomputed backward.
    The core API is experimental/private, so fail clearly if it changes.
    """
    try:
        from torch_xla.experimental.scan import _scan_impl_pytree
    except (ImportError, AttributeError) as exc:
        raise RuntimeError(
            "TPU chunked loss needs torch_xla.experimental.scan._scan_impl_pytree. "
            "Run scripts/probe_tpu_chunked_loss.py on the installed torch/torch_xla pair "
            "before loading 9B weights. There is no Python-unroll fallback."
        ) from exc
    import inspect
    if not callable(_scan_impl_pytree) or not {"fn", "init", "xs"}.issubset(inspect.signature(_scan_impl_pytree).parameters):
        raise RuntimeError("Installed XLA scan core API changed; probe/adapt before loading weights")
    return _scan_impl_pytree


def _loop(body, state, *, chunks, weights):
    if state[0].device.type == "xla":
        count = len(state) - 1
        # Weights are explicit read-only carry, never closure-captured tensors.
        def step(carry, i):
            updated = body(i, *carry[:count], *carry[count:])
            return (*updated[1:], *carry[count:]), None
        carry, _ = require_xla_loop()(
            step, init=(*state[1:], *weights),
            xs=torch.arange(chunks, device=state[0].device, dtype=torch.long))
        return (state[0] + chunks, *carry[:count])
    if state[0].device.type != "cpu":
        raise ValueError("TPU loss supports XLA, or CPU for numerical reference only")
    # An explicit CPU oracle. This branch is never traced on XLA or used on CUDA.
    for _ in range(chunks):
        state = body(*state, *weights)
    return state


def _batch_shard(value, mesh):
    if mesh is not None:
        import torch_xla.distributed.spmd as xs
        xs.mark_sharding(value, mesh, ("fsdp",) + (None,) * (value.ndim - 1))
    return value


def _chunk(h, targets, i, size):
    positions = torch.arange(size, device=h.device, dtype=torch.long) + i * size
    return h.index_select(1, positions), targets.index_select(1, positions), positions


def _logits(h, weight, bias, mesh):
    logits = F.linear(h, weight, bias if bias.numel() else None).float()
    return _batch_shard(logits, mesh)


class _FrozenHeadCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, targets, size, mesh):
        chunks = hidden.shape[1] // size
        zero = torch.zeros((), device=hidden.device, dtype=torch.long)
        total = torch.zeros((), device=hidden.device, dtype=torch.float32)
        denominator = (targets != -100).sum().to(torch.float32).clamp_min(1)

        def body(i, total, h, t, w, b):
            hc, tc, _ = _chunk(h, t, i, size)
            z = _logits(hc, w, b, mesh)
            # Keep batch explicit for SPMD. No reshape that mixes batch/sequence.
            selected = z.gather(-1, tc.masked_fill(tc == -100, 0).unsqueeze(-1)).squeeze(-1)
            nll = torch.where(tc != -100, torch.logsumexp(z, -1) - selected, 0.0)
            return i + 1, total + nll.sum(), h, t

        _, total, _, _ = _loop(body, (zero, total, hidden, targets), chunks=chunks, weights=(weight, bias))
        ctx.save_for_backward(hidden, weight, bias, targets, denominator)
        ctx.size, ctx.mesh = size, mesh
        return total / denominator

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        hidden, weight, bias, targets, denominator = ctx.saved_tensors
        size, mesh = ctx.size, ctx.mesh
        chunks = hidden.shape[1] // size
        zero = torch.zeros((), device=hidden.device, dtype=torch.long)
        dh = _batch_shard(torch.zeros_like(hidden), mesh)
        scale = grad_output / denominator

        def body(i, dh, h, t, scale, w, b):
            hc, tc, positions = _chunk(h, t, i, size)
            probabilities = _logits(hc, w, b, mesh).softmax(-1)
            corrections = -torch.ones_like(tc, dtype=torch.float32).unsqueeze(-1)
            dz = probabilities.scatter_add(-1, tc.masked_fill(tc == -100, 0).unsqueeze(-1), corrections)
            dz = dz * (tc != -100).unsqueeze(-1) * scale
            # Match Linear's chain rule: gradient into its 16-bit output is
            # rounded before matmul, as in stock FP32 CE over BF16 logits.
            dhc = F.linear(dz.to(h.dtype), w.transpose(0, 1))
            dhc = _batch_shard(dhc, mesh)
            return i + 1, dh.index_copy(1, positions, dhc), h, t, scale

        _, dh, _, _, _ = _loop(
            body, (zero, dh, hidden, targets, scale), chunks=chunks, weights=(weight, bias))
        return dh, None, None, None, None, None


def frozen_head_causal_loss(hidden, head, labels, chunk_tokens: int, *, mesh=None):
    """Exact shifted supervised-token mean; ignored/all-masked positions are safe.

    The sequence tail is padded, never truncated. `chunk_tokens` counts positions
    per batch row; with batch sharding, a core normally processes one row.
    """
    if chunk_tokens <= 0:
        raise ValueError("TPU chunk size must be positive")
    if type(head) is not torch.nn.Linear or any(p.requires_grad for p in head.parameters()):
        raise ValueError("TPU chunked loss requires a frozen plain nn.Linear lm_head (no head LoRA/quantization)")
    if hidden.device.type not in {"cpu", "xla"}:
        raise ValueError("TPU chunked loss cannot run on CUDA; keep the existing GPU loss")
    if hidden.ndim != 3 or tuple(labels.shape) != tuple(hidden.shape[:2]):
        raise ValueError("Expected hidden [batch, sequence, hidden] and matching labels [batch, sequence]")
    if hidden.shape[1] == 0:
        raise ValueError("Empty sequence")
    if mesh is not None and hidden.device.type != "xla":
        raise ValueError("A sharding mesh requires XLA tensors")
    if head.weight.device != hidden.device:
        raise ValueError("TPU head and hidden states must reside on the same XLA device")
    if labels.device != hidden.device or labels.dtype != torch.long:
        raise ValueError("Labels must be int64 on the same device as hidden states")
    hidden = hidden.to(head.weight.dtype)
    # Predict label t+1 from hidden t. Use the full static T, masking its last slot.
    targets = torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], -100)), dim=1)
    padding = (-hidden.shape[1]) % chunk_tokens
    if padding:
        hidden = F.pad(hidden, (0, 0, 0, padding))
        targets = F.pad(targets, (0, padding), value=-100)
    hidden, targets = _batch_shard(hidden, mesh), _batch_shard(targets, mesh)
    bias = head.bias if head.bias is not None else hidden.new_empty((0,))
    return _FrozenHeadCE.apply(hidden, head.weight, bias, targets, chunk_tokens, mesh)


def install_tpu_chunked_loss(model: Any, chunk_tokens: int, *, mesh=None):
    """Install BEFORE FSDP wrapping; keep the normal model(**inputs) invocation.

    Only labeled forwards change. Unlabeled forwards retain the original method.
    The CUDA trainer never calls this installer or imports this module.
    """
    base = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    if getattr(base, "_dental_tpu_loss_chunk_tokens", None) is not None:
        raise ValueError("TPU chunked loss is already installed; use a fresh model")
    head = getattr(base, "lm_head", None)
    if type(head) is not torch.nn.Linear or any(p.requires_grad for p in head.parameters()):
        raise ValueError("TPU chunked loss requires a frozen plain nn.Linear lm_head")
    # FSDPv2 shards CPU parameters while transferring them; moving a complete
    # 9B model to XLA before wrapping could itself exhaust one chip's HBM.
    if head.weight.device.type not in {"cpu", "xla"} or (head.weight.device.type == "cpu" and mesh is None):
        raise ValueError("Install TPU loss on XLA, or before CPU-to-XLA FSDPv2 wrapping with a real mesh")
    if not hasattr(base, "model") or chunk_tokens <= 0:
        raise ValueError("Expected a Qwen-style .model body and positive TPU chunk size")
    require_xla_loop()
    base._dental_tpu_original_forward = base.forward

    @functools.wraps(type(base).forward)
    def forward(self, *args, **kwargs):
        if kwargs.get("labels") is None:
            return self._dental_tpu_original_forward(*args, **kwargs)
        if args:
            raise ValueError("TPU chunked training expects keyword inputs, as used by train_sft.py")
        if kwargs.get("num_items_in_batch") is not None:
            raise ValueError("TPU loss normalizes its global supervised-token count; external denominator unsupported")
        if kwargs.get("logits_to_keep", 0) not in (0, None):
            raise ValueError("TPU training loss must not use last-token-only decode logits")
        body_inputs = {k: v for k, v in kwargs.items()
                       if k not in {"labels", "num_items_in_batch", "logits_to_keep", "return_dict"}}
        body_inputs.update(use_cache=False, return_dict=True)
        outputs = self.model(**body_inputs)
        loss = frozen_head_causal_loss(outputs[0], self.lm_head, kwargs["labels"], chunk_tokens, mesh=mesh)
        if kwargs.get("return_dict", True) is False:
            return (loss,)
        return TPULossOutput(loss=loss, hidden_states=getattr(outputs, "hidden_states", None),
                             attentions=getattr(outputs, "attentions", None))

    base.forward = types.MethodType(forward, base)
    base._dental_tpu_loss_chunk_tokens = chunk_tokens
    print(f"[TPU LOSS] fixed chunk={chunk_tokens} per row; XLA while-loop; frozen-head recomputed backward; no logits output", flush=True)
    return model
