"""
[G3] Dual-LoRA Adapter Reference/Policy Verification Unit Test.

Mathematically proves:
1. Toggling active parameters between "reference" and "grpo_policy".
2. Forward log-probs diverge when "grpo_policy" weights are updated.
3. Switching back to "reference" reproduces original baseline log-probs to 6 decimal places.
4. Schulman k3 KL divergence is strictly non-negative (D_KL >= 0.0).
"""

import pytest
import torch
import torch.nn as nn


class PurePyTorchDualAdapter(nn.Module):
    """Pure PyTorch implementation of the dual-adapter toggle pattern.
    Allows local unit testing of the exact adapter switching math even without peft installed.
    """
    def __init__(self, hidden_dim=64, rank=8):
        super().__init__()
        self.base_weight = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.ref_lora_A = nn.Parameter(torch.randn(rank, hidden_dim) * 0.1)
        self.ref_lora_B = nn.Parameter(torch.zeros(hidden_dim, rank))

        self.policy_lora_A = nn.Parameter(self.ref_lora_A.clone())
        self.policy_lora_B = nn.Parameter(torch.zeros(hidden_dim, rank))

        self.active_adapter = "reference"

    def set_adapter(self, name: str):
        if name not in ["reference", "grpo_policy"]:
            raise ValueError(f"Unknown adapter: {name}")
        self.active_adapter = name

    def forward(self, x):
        if self.active_adapter == "reference":
            delta = self.ref_lora_B @ self.ref_lora_A
        else:
            delta = self.policy_lora_B @ self.policy_lora_A
        effective_w = self.base_weight + delta
        return x @ effective_w.T


class DummyDualAdapterCausalLM(nn.Module):
    def __init__(self, vocab_size=100, hidden_dim=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.adapter_layer = PurePyTorchDualAdapter(hidden_dim=hidden_dim)
        self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

    def set_adapter(self, name: str):
        self.adapter_layer.set_adapter(name)

    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        x = self.adapter_layer(x)
        logits = self.head(x)
        return type("Output", (), {"logits": logits})()


def test_dual_adapter_switching_and_k3_kl():
    """Verify dual adapter mechanics and Schulman k3 KL penalty."""
    torch.manual_seed(42)
    model = DummyDualAdapterCausalLM(vocab_size=100, hidden_dim=64)

    input_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    target_ids = input_ids[:, 1:]

    # 1. Compute baseline log-probs under "reference"
    model.set_adapter("reference")
    model.eval()
    with torch.no_grad():
        out_ref = model(input_ids)
        logits_ref = out_ref.logits[:, :-1, :]
        log_probs_ref = torch.log_softmax(logits_ref, dim=-1)
        ref_lp = torch.gather(log_probs_ref, 2, target_ids.unsqueeze(-1)).squeeze(-1)

    # 2. Compute initial log-probs under "grpo_policy"
    model.set_adapter("grpo_policy")
    with torch.no_grad():
        out_pol_init = model(input_ids)
        logits_pol_init = out_pol_init.logits[:, :-1, :]
        log_probs_pol_init = torch.log_softmax(logits_pol_init, dim=-1)
        init_policy_lp = torch.gather(log_probs_pol_init, 2, target_ids.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(ref_lp, init_policy_lp, atol=1e-5), "Initial policy must match reference"

    # 3. Perturb / Train "grpo_policy" weights
    with torch.no_grad():
        model.adapter_layer.policy_lora_B.add_(torch.randn_like(model.adapter_layer.policy_lora_B) * 0.5)

    # Compute updated policy log-probs
    model.set_adapter("grpo_policy")
    with torch.no_grad():
        out_pol_up = model(input_ids)
        logits_pol_up = out_pol_up.logits[:, :-1, :]
        log_probs_pol_up = torch.log_softmax(logits_pol_up, dim=-1)
        updated_policy_lp = torch.gather(log_probs_pol_up, 2, target_ids.unsqueeze(-1)).squeeze(-1)

    assert not torch.allclose(ref_lp, updated_policy_lp, atol=1e-3), "Updated policy log-probs must diverge"

    # 4. Switch back to "reference" and verify zero gradient bleed / corruption
    model.set_adapter("reference")
    with torch.no_grad():
        out_ref_verify = model(input_ids)
        logits_ref_verify = out_ref_verify.logits[:, :-1, :]
        log_probs_ref_verify = torch.log_softmax(logits_ref_verify, dim=-1)
        ref_lp_verify = torch.gather(log_probs_ref_verify, 2, target_ids.unsqueeze(-1)).squeeze(-1)

    assert torch.allclose(ref_lp, ref_lp_verify, atol=1e-6), "Reference policy must be preserved to 6 decimal places"

    # 5. Verify Schulman k3 KL divergence: D_KL = exp(log_ratio) - log_ratio - 1 >= 0
    log_ratio = ref_lp - updated_policy_lp
    k3_kl = torch.exp(log_ratio) - log_ratio - 1.0

    assert (k3_kl >= -1e-6).all(), f"Schulman k3 KL penalty must be strictly non-negative: {k3_kl}"
    assert k3_kl.mean().item() > 0.0, "KL divergence must be positive for diverged policies"


def test_peft_dual_adapter_integration():
    """Verify integration when peft is installed."""
    peft = pytest.importorskip("peft")
    from peft import LoraConfig, get_peft_model

    class DummyBase(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 32)
        def forward(self, x):
            return self.linear(x)

    base = DummyBase()
    cfg_ref = LoraConfig(r=4, lora_alpha=8, target_modules=["linear"])
    model = get_peft_model(base, cfg_ref, adapter_name="reference")
    model.add_adapter("grpo_policy", cfg_ref)

    model.set_adapter("reference")
    assert model.active_adapter == "reference"
    model.set_adapter("grpo_policy")
    assert model.active_adapter == "grpo_policy"
