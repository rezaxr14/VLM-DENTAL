"""
Unit test verifying exact mathematical token-weighted gradient accumulation vs mean-of-means distortion.

Tests:
1. Conventional `outputs.loss / grad_accum` produces mean-of-means distortion when token counts differ.
2. Exact token-weighted accumulation (`outputs.loss * valid_tokens` summed and divided by total tokens)
   produces gradients identical to computing loss over all tokens in a single unaccumulated batch.
"""

import pytest
import torch
import torch.nn as nn


def test_token_weighted_gradient_accumulation_matches_full_batch():
    torch.manual_seed(42)

    # Simple linear model with vocab projection
    d_model = 16
    vocab_size = 32
    linear = nn.Linear(d_model, vocab_size, bias=False)

    # Microbatch 1: 5 active tokens (e.g. short turn)
    # Microbatch 2: 25 active tokens (e.g. long reasoning trace)
    n1 = 5
    n2 = 25
    total_tokens = n1 + n2

    x1 = torch.randn(n1, d_model)
    targets1 = torch.randint(0, vocab_size, (n1,))

    x2 = torch.randn(n2, d_model)
    targets2 = torch.randint(0, vocab_size, (n2,))

    loss_fn = nn.CrossEntropyLoss(reduction="mean")

    # -----------------------------------------------------------------------
    # Ground Truth: Single Full Batch over all (n1 + n2) tokens
    # -----------------------------------------------------------------------
    linear_gt = nn.Linear(d_model, vocab_size, bias=False)
    linear_gt.load_state_dict(linear.state_dict())

    x_full = torch.cat([x1, x2], dim=0)
    targets_full = torch.cat([targets1, targets2], dim=0)

    logits_full = linear_gt(x_full)
    loss_gt = loss_fn(logits_full, targets_full)
    loss_gt.backward()
    grad_gt = linear_gt.weight.grad.clone()

    # -----------------------------------------------------------------------
    # Flawed Conventional Accumulation: loss = outputs.loss / 2
    # -----------------------------------------------------------------------
    linear_flawed = nn.Linear(d_model, vocab_size, bias=False)
    linear_flawed.load_state_dict(linear.state_dict())

    logits1 = linear_flawed(x1)
    loss1 = loss_fn(logits1, targets1)
    (loss1 / 2.0).backward()

    logits2 = linear_flawed(x2)
    loss2 = loss_fn(logits2, targets2)
    (loss2 / 2.0).backward()

    grad_flawed = linear_flawed.weight.grad.clone()

    # Flawed accumulation diverges from ground truth!
    assert not torch.allclose(grad_flawed, grad_gt, atol=1e-5), \
        "Conventional accumulation should diverge from ground truth due to mean-of-means distortion"

    # -----------------------------------------------------------------------
    # Corrected Mathematical Accumulation (Claude Point 1)
    # -----------------------------------------------------------------------
    linear_exact = nn.Linear(d_model, vocab_size, bias=False)
    linear_exact.load_state_dict(linear.state_dict())

    accum_loss_sum = 0.0
    accum_tokens = 0

    # Step 1: Microbatch 1
    logits1_exact = linear_exact(x1)
    loss1_mean = loss_fn(logits1_exact, targets1)
    batch1_sum = loss1_mean * n1
    batch1_sum.backward()
    accum_loss_sum += batch1_sum.item()
    accum_tokens += n1

    # Step 2: Microbatch 2
    logits2_exact = linear_exact(x2)
    loss2_mean = loss_fn(logits2_exact, targets2)
    batch2_sum = loss2_mean * n2
    batch2_sum.backward()
    accum_loss_sum += batch2_sum.item()
    accum_tokens += n2

    # Step boundary normalization
    for p in linear_exact.parameters():
        if p.grad is not None:
            p.grad.mul_(1.0 / accum_tokens)

    grad_exact = linear_exact.weight.grad.clone()
    true_mean_loss = accum_loss_sum / accum_tokens

    # Mathematical identity: exact accumulation MUST match ground truth exactly!
    assert torch.allclose(grad_exact, grad_gt, atol=1e-6), \
        f"Corrected accumulation gradients must match ground truth. Max diff: {(grad_exact - grad_gt).abs().max()}"
    assert abs(true_mean_loss - loss_gt.item()) < 1e-6, \
        f"True mean loss must match ground truth loss: {true_mean_loss} vs {loss_gt.item()}"
