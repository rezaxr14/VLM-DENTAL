"""
Unit test for Group Relative Advantage Normalization across K in {1, 2, 4, 8, 16}.

Verifies:
1. K = 1: Smooth fallback to REINFORCE with EMA baseline; zero division-by-zero.
2. K = 2: Variance stabilization with tie-breaker clamping (if std < 1e-6, A = 0; no NaNs).
3. K >= 4: Standard group-relative advantage normalization (zero-centered, unit variance).
"""

import pytest
import torch

from dental_agent.training.grpo import compute_group_advantages


def test_grpo_advantage_k1_ema_fallback():
    # K = 1
    rewards = [0.85]
    initial_baseline = 0.50
    adv, new_baseline = compute_group_advantages(rewards, running_ema_baseline=initial_baseline, beta=0.9)

    assert not torch.isnan(adv).any(), "K=1 must not produce NaN"
    assert adv.shape == (1,)
    # Advantage = reward - baseline = 0.85 - 0.50 = 0.35
    assert pytest.approx(adv.item(), rel=1e-3) == 0.35
    # New baseline = 0.9 * 0.50 + 0.1 * 0.85 = 0.45 + 0.085 = 0.535
    assert pytest.approx(new_baseline, rel=1e-3) == 0.535


def test_grpo_advantage_k2_tie_breaking():
    # K = 2 with identical rewards (std == 0)
    rewards = [0.75, 0.75]
    adv, _ = compute_group_advantages(rewards)

    assert not torch.isnan(adv).any(), "K=2 tie must not produce NaN"
    assert (adv == 0.0).all(), "Uninformative ties must clamp advantage to 0.0"

    # K = 2 with distinct rewards
    rewards_distinct = [0.50, 0.90]
    adv_distinct, _ = compute_group_advantages(rewards_distinct)
    assert not torch.isnan(adv_distinct).any()
    assert adv_distinct[0] < 0.0  # lower reward gets negative advantage
    assert adv_distinct[1] > 0.0  # higher reward gets positive advantage
    assert pytest.approx(float(adv_distinct.sum()), abs=1e-4) == 0.0


def test_grpo_advantage_k4_k8_k16():
    for k in [4, 8, 16]:
        rewards = [float(i) * 0.1 for i in range(k)]
        adv, _ = compute_group_advantages(rewards)

        assert not torch.isnan(adv).any(), f"K={k} produced NaN"
        assert adv.shape == (k,)
        # Mean must be 0
        assert pytest.approx(float(adv.mean()), abs=1e-4) == 0.0
        # Standard deviation must be ~1.0
        assert pytest.approx(float(adv.std(unbiased=False)), rel=1e-2) == 1.0
