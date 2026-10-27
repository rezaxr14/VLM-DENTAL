"""Exact full-vocabulary selected-token log probabilities without full [B,T,V] logits.

Returns [B,T-1] FP32 log probabilities, zero at ignored labels, plus a mask.
There is no vocabulary sampling, context truncation or PPO objective change.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def _selected_lp(head, hidden, target):
    weight = getattr(head, 'weight', None)
    if weight is not None:
        if weight.device.type == 'meta':
            raise RuntimeError('Chunked GRPO scoring requires GPU/CPU resident lm_head, not disk offload')
        hidden = hidden.to(device=weight.device, dtype=weight.dtype)
        target = target.to(weight.device)
    logits = head(hidden).float()
    # Negative per-token CE equals target log-softmax, including the full vocab.
    return -F.cross_entropy(logits, target.to(logits.device), reduction='none')


def token_log_probs(model, enc, chunk_tokens=0):
    labels = enc['labels']
    body_inputs = {k:v for k,v in enc.items() if k not in {'labels','logits_to_keep','num_items_in_batch','return_dict'}}
    body_inputs['use_cache'] = False
    if not chunk_tokens:
        outputs = model(**body_inputs)
        logits = outputs.logits[:, :-1].float()
        targets = labels[:, 1:].to(logits.device)
        mask = targets != -100
        lp = logits.log_softmax(-1).gather(-1, targets.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        return lp.masked_fill(~mask, 0.0), mask.float()
    if chunk_tokens < 0:
        raise ValueError('logprob_chunk_tokens must be nonnegative')
    if labels.device.type == 'xla':
        raise ValueError('Dynamic selected-token scoring is not supported on TPU; use chunk size 0')
    base = model.get_base_model() if callable(getattr(model,'get_base_model',None)) else model
    body, head = getattr(base,'model',None), getattr(base,'lm_head',None)
    if body is None or head is None:
        raise ValueError('Chunked scoring needs a Qwen-style .model/.lm_head; choose 0 for unsupported models')
    hidden = body(**body_inputs, return_dict=True)[0]
    targets = labels[:, 1:].to(hidden.device)
    keep = targets != -100
    chosen, target = hidden[:, :-1][keep], targets[keep]
    n = target.numel()
    if not n:
        return hidden.sum().float() * 0.0 + torch.zeros_like(targets,dtype=torch.float32), keep.float()
    use_checkpoint = torch.is_grad_enabled() and (chosen.requires_grad or any(p.requires_grad for p in head.parameters()))
    pieces = []
    for start in range(0,n,chunk_tokens):
        h,t = chosen[start:start+chunk_tokens],target[start:start+chunk_tokens]
        lp = checkpoint(_selected_lp,head,h,t,use_reentrant=False) if use_checkpoint else _selected_lp(head,h,t)
        pieces.append(lp.to(hidden.device))
    result = hidden.new_zeros(targets.shape, dtype=torch.float32).masked_scatter(keep,torch.cat(pieces))
    return result, keep.float()
