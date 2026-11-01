"""Opt-in, fixed-shape TPU selected-token log-probabilities with a recomputed backward (GRPO scoring).

Sibling of ``tpu_chunked_loss`` (which returns one scalar for SFT). GRPO needs a log-probability PER TOKEN, with a gradient
that depends on a per-token weight (PPO ratio, advantage, KL), so the scalar-loss function cannot be reused. This module
reuses its XLA ``scan`` loop and helpers and changes only what each chunk computes:

    forward   lp[b, t] = z[b, t, y] - logsumexp(z[b, t, :])       z = frozen_head(hidden) in FP32, one chunk at a time
    backward  dz = (onehot(y) - softmax(z)) * g[b, t]              g = incoming per-token gradient; dh = dz @ W

The full ``[B, T, V]`` logits never exist (a 256-position FP32 chunk at V=248,320 is 0.237 GiB per row; the full buffer at
T=32,768 would be ~30 GiB). Positions are fixed (no boolean compaction, no ``.item()``, no Python unrolling), so the graph
is compiled once per static shape. The result equals ``-F.cross_entropy(reduction="none")`` of the stock FP32 path.

Only a frozen plain ``nn.Linear`` head is supported (Qwen's normal LoRA setup). Not imported by the CUDA path.
"""
from __future__ import annotations

import functools
import types
from typing import Any, NamedTuple

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable

from dental_agent.training.tpu_chunked_loss import _batch_shard, _chunk, _logits, _loop, require_xla_loop

SCORE_KWARG = "score_labels"  # keyword that asks the installed forward for log-probs instead of logits


class TPULogProbOutput(NamedTuple):
    """Attribute access for the scorer, tuple output for FSDPv2's ``shard_output`` (``logits`` stays None)."""
    log_probs: torch.Tensor
    mask: torch.Tensor
    logits: None = None
    past_key_values: None = None
    hidden_states: Any = None
    attentions: Any = None


def validate_tpu_logprob_args(*, chunk_tokens: int, is_tpu: bool, spmd: bool, fsdp: bool, pad_vision_to_slots: bool,
                              logprob_chunk_tokens: int = 0) -> None:
    """Fail before the 9B weights load. ``chunk_tokens == 0`` means the feature is off (stock full-logit scoring)."""
    if chunk_tokens < 0:
        raise ValueError("--tpu-logprob-chunk-tokens must be nonnegative")
    if not chunk_tokens:
        return
    if not is_tpu:
        raise ValueError("--tpu-logprob-chunk-tokens requires a real TPU; the GPU/CPU path (--logprob-chunk-tokens) is unchanged")
    if logprob_chunk_tokens:
        raise ValueError("TPU requires --logprob-chunk-tokens 0; use --tpu-logprob-chunk-tokens instead")
    if not (spmd and fsdp and pad_vision_to_slots):
        raise ValueError("TPU chunked scoring requires --spmd --fsdp --pad-vision-to-slots (fixed shapes)")


def _targets(labels: torch.Tensor) -> torch.Tensor:
    """Predict label t+1 from hidden t over the full static T; the last slot has no target."""
    return torch.cat((labels[:, 1:], torch.full_like(labels[:, :1], -100)), dim=1)


class _FrozenHeadLogProb(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, bias, targets, size, mesh):
        chunks = hidden.shape[1] // size
        zero = torch.zeros((), device=hidden.device, dtype=torch.long)
        out = _batch_shard(torch.zeros(targets.shape, device=hidden.device, dtype=torch.float32), mesh)

        def body(i, out, h, t, w, b):
            hc, tc, positions = _chunk(h, t, i, size)
            z = _logits(hc, w, b, mesh)
            selected = z.gather(-1, tc.masked_fill(tc == -100, 0).unsqueeze(-1)).squeeze(-1)
            lp = torch.where(tc != -100, selected - torch.logsumexp(z, -1), 0.0)
            return i + 1, out.index_copy(1, positions, lp), h, t

        _, out, _, _ = _loop(body, (zero, out, hidden, targets), chunks=chunks, weights=(weight, bias))
        ctx.save_for_backward(hidden, weight, bias, targets)
        ctx.size, ctx.mesh = size, mesh
        return out

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        hidden, weight, bias, targets = ctx.saved_tensors
        size, mesh = ctx.size, ctx.mesh
        chunks = hidden.shape[1] // size
        zero = torch.zeros((), device=hidden.device, dtype=torch.long)
        dh = _batch_shard(torch.zeros_like(hidden), mesh)
        grad = grad_output.to(torch.float32)

        def body(i, dh, h, t, g, w, b):
            hc, tc, positions = _chunk(h, t, i, size)
            gc = g.index_select(1, positions)
            probabilities = _logits(hc, w, b, mesh).softmax(-1)
            corrections = -torch.ones_like(tc, dtype=torch.float32).unsqueeze(-1)
            dz = probabilities.scatter_add(-1, tc.masked_fill(tc == -100, 0).unsqueeze(-1), corrections)
            # d lp / d z = onehot - softmax = -(softmax - onehot); masked positions contribute nothing.
            dz = -dz * (tc != -100).unsqueeze(-1) * gc.unsqueeze(-1)
            # Same rounding as the scalar loss: the gradient into the 16-bit head output is rounded before the matmul.
            dhc = _batch_shard(F.linear(dz.to(h.dtype), w.transpose(0, 1)), mesh)
            return i + 1, dh.index_copy(1, positions, dhc), h, t, g

        _, dh, _, _, _ = _loop(body, (zero, dh, hidden, targets, grad), chunks=chunks, weights=(weight, bias))
        return dh, None, None, None, None, None


def frozen_head_token_log_probs(hidden, head, labels, chunk_tokens: int, *, mesh=None):
    """Selected-token log-probs ``[B, T-1]`` (FP32, 0 where ``labels[:, 1:] == -100``) and the ``[B, T-1]`` float mask.

    Same contract as ``chunked_log_probs.token_log_probs`` (position i scores ``labels[:, i+1]``). The sequence tail is
    padded to a multiple of ``chunk_tokens``, never truncated. ``chunk_tokens`` counts positions per batch row."""
    if chunk_tokens <= 0:
        raise ValueError("TPU chunk size must be positive")
    if type(head) is not torch.nn.Linear or any(p.requires_grad for p in head.parameters()):
        raise ValueError("TPU chunked scoring requires a frozen plain nn.Linear lm_head (no head LoRA/quantization)")
    if hidden.device.type not in {"cpu", "xla"}:
        raise ValueError("TPU chunked scoring cannot run on CUDA; keep the existing GPU scorer")
    if hidden.ndim != 3 or tuple(labels.shape) != tuple(hidden.shape[:2]):
        raise ValueError("Expected hidden [batch, sequence, hidden] and matching labels [batch, sequence]")
    if hidden.shape[1] < 2:
        raise ValueError("Need at least two positions to score a token")
    if mesh is not None and hidden.device.type != "xla":
        raise ValueError("A sharding mesh requires XLA tensors")
    if head.weight.device != hidden.device:
        raise ValueError("TPU head and hidden states must reside on the same XLA device")
    if labels.device != hidden.device or labels.dtype != torch.long:
        raise ValueError("Labels must be int64 on the same device as hidden states")
    length = hidden.shape[1]
    hidden = hidden.to(head.weight.dtype)
    targets = _targets(labels)
    padding = (-length) % chunk_tokens
    if padding:
        hidden = F.pad(hidden, (0, 0, 0, padding))
        targets = F.pad(targets, (0, padding), value=-100)
    hidden, targets = _batch_shard(hidden, mesh), _batch_shard(targets, mesh)
    bias = head.bias if head.bias is not None else hidden.new_empty((0,))
    lp = _FrozenHeadLogProb.apply(hidden, head.weight, bias, targets, chunk_tokens, mesh)
    lp = lp[:, : length - 1]
    return lp, (labels[:, 1:] != -100).to(torch.float32)


def install_tpu_chunked_logprobs(model: Any, chunk_tokens: int, *, mesh=None):
    """Install BEFORE FSDP wrapping (like ``install_tpu_chunked_loss``). Only forwards that pass ``score_labels`` change:
    they return ``TPULogProbOutput`` instead of logits. Generation and every other forward keep the original method."""
    base = model.get_base_model() if callable(getattr(model, "get_base_model", None)) else model
    if getattr(base, "_dental_tpu_logprob_chunk_tokens", None) is not None:
        raise ValueError("TPU chunked scoring is already installed; use a fresh model")
    if getattr(base, "_dental_tpu_loss_chunk_tokens", None) is not None:
        raise ValueError("TPU chunked SFT loss is installed on this model; GRPO scoring needs a fresh model")
    head = getattr(base, "lm_head", None)
    if type(head) is not torch.nn.Linear or any(p.requires_grad for p in head.parameters()):
        raise ValueError("TPU chunked scoring requires a frozen plain nn.Linear lm_head")
    if head.weight.device.type not in {"cpu", "xla"} or (head.weight.device.type == "cpu" and mesh is None):
        raise ValueError("Install TPU scoring on XLA, or before CPU-to-XLA FSDPv2 wrapping with a real mesh")
    if not hasattr(base, "model") or chunk_tokens <= 0:
        raise ValueError("Expected a Qwen-style .model body and positive TPU chunk size")
    require_xla_loop()
    base._dental_tpu_logprob_original_forward = base.forward

    @functools.wraps(type(base).forward)
    def forward(self, *args, **kwargs):
        labels = kwargs.pop(SCORE_KWARG, None)
        if labels is None:
            return self._dental_tpu_logprob_original_forward(*args, **kwargs)
        if args:
            raise ValueError("TPU chunked scoring expects keyword inputs")
        if kwargs.get("labels") is not None:
            raise ValueError(f"Pass the targets as {SCORE_KWARG}, not labels")
        if kwargs.get("logits_to_keep", 0) not in (0, None):
            raise ValueError("TPU scoring must not use last-token-only decode logits")
        body_inputs = {k: v for k, v in kwargs.items() if k not in {"logits_to_keep", "num_items_in_batch", "return_dict"}}
        body_inputs.update(use_cache=False, return_dict=True)
        outputs = self.model(**body_inputs)
        lp, mask = frozen_head_token_log_probs(outputs[0], self.lm_head, labels, chunk_tokens, mesh=mesh)
        return TPULogProbOutput(log_probs=lp, mask=mask)

    base.forward = types.MethodType(forward, base)
    base._dental_tpu_logprob_chunk_tokens = chunk_tokens
    print(f"[TPU SCORE] fixed chunk={chunk_tokens} per row; XLA while-loop; frozen-head recomputed backward; no logits", flush=True)
    return model
