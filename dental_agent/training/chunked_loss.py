"""Memory-bounded causal-LM loss for GPU training (the TPU path keeps the HuggingFace loss).

Why: a long sequence times a large vocabulary creates multi-GiB logits before
cross-entropy allocates its work buffers. Chunking bounds that output-layer work;
it does not bound transformer/vision activations or attention workspace.

What this does instead (same value, same gradients):
  1. run only the transformer body (`base.model`) to get the final hidden states [B, T, H];
  2. keep only the positions whose NEXT token is supervised (label != -100): assistant tokens, typically a small
     share of the sample, so the system prompt, images and tool results never reach the output layer;
  3. push those hidden states through `lm_head` + cross-entropy `chunk_tokens` at a time, each chunk under
     torch.utils.checkpoint, so only one chunk of logits exists at any moment (forward and backward).
The result is the mean over supervised tokens, exactly what `outputs.loss` is for labels with -100 masking.

Boolean-mask selection has a data-dependent shape, which would recompile on XLA, so this is GPU/CPU only.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

IGNORE_INDEX = -100


def _chunk_nll(head: torch.nn.Module, hidden: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Summed negative log-likelihood of one chunk; logits live only inside this call."""
    weight = getattr(head, "weight", None)
    # The body and output head may be on DIFFERENT GPUs. Dispatch hooks move
    # hidden states, but do not move CE targets outside the head's forward.
    if weight is not None:
        if weight.device.type == "meta":
            raise RuntimeError("Chunked training does not support CPU/disk-offloaded lm_head.")
        hidden = hidden.to(device=weight.device, dtype=weight.dtype)
        target = target.to(weight.device)
    logits = head(hidden)
    return F.cross_entropy(logits.float(), target.to(logits.device), ignore_index=IGNORE_INDEX, reduction="sum")


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """The Qwen3_5ForConditionalGeneration under a PEFT wrapper (LoRA layers are injected in place, so they still run)."""
    get_base = getattr(model, "get_base_model", None)
    return get_base() if callable(get_base) else model


def causal_lm_loss(model: torch.nn.Module, inputs: dict[str, Any], chunk_tokens: int = 0) -> torch.Tensor:
    """Mean cross-entropy over supervised tokens. chunk_tokens <= 0: the stock `model(**inputs).loss` (TPU, tiny models)."""
    if chunk_tokens <= 0:
        return model(**inputs).loss

    labels = inputs["labels"]
    body_inputs = {k: v for k, v in inputs.items() if k not in {"labels", "logits_to_keep", "num_items_in_batch"}}
    body_inputs.update(use_cache=False, return_dict=True)
    base = _unwrap(model)
    body, head = getattr(base, "model", None), getattr(base, "lm_head", None)
    if body is None or head is None:
        raise RuntimeError("--loss-chunk-tokens needs a model with .model and .lm_head (Qwen3.5); use --loss-chunk-tokens 0 for this model.")

    hidden = body(**body_inputs)[0]                              # [B, T, H]
    target = labels[:, 1:].to(hidden.device)                     # position t predicts token t+1
    keep = target != IGNORE_INDEX
    selected, target = hidden[:, :-1][keep], target[keep]        # [K, H], [K]
    n = int(target.numel())
    if n == 0:
        return hidden.sum() * 0.0                                # keeps the graph; the caller skips such batches anyway

    run = torch.is_grad_enabled() and (selected.requires_grad or any(p.requires_grad for p in head.parameters()))
    total = selected.new_zeros((), dtype=torch.float32)
    for start in range(0, n, chunk_tokens):
        h, t = selected[start:start + chunk_tokens], target[start:start + chunk_tokens]
        nll = checkpoint(_chunk_nll, head, h, t, use_reentrant=False) if run else _chunk_nll(head, h, t)
        total = total + nll.to(total.device)
    return total / n
