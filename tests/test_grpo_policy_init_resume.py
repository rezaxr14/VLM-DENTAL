"""GRPO policy starts at the SFT weights; step checkpoints resume. Runs against real peft on a tiny model."""

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import LoraConfig, PeftModel, get_peft_model  # noqa: E402

from dental_agent.model.checkpoints import save_checkpoint  # noqa: E402
from dental_agent.training.grpo_resume import (  # noqa: E402
    find_latest_step_checkpoint,
    init_policy_from_reference,
    load_policy_state,
    pool_fingerprint,
)


class TinyLM(nn.Module):
    def __init__(self, vocab=50, dim=16):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, vocab, bias=False)

    def forward(self, input_ids, **kw):
        return type("Out", (), {"logits": self.head(torch.tanh(self.proj(self.embed(input_ids))))})()


def _cfg(r=4, alpha=8):
    return LoraConfig(r=r, lora_alpha=alpha, target_modules=["proj"], lora_dropout=0.0, bias="none")


@pytest.fixture()
def sft_dir(tmp_path):
    """A saved 'SFT adapter' with non-zero lora_B (a zero B would make the test vacuous)."""
    torch.manual_seed(0)
    m = get_peft_model(TinyLM(), _cfg())
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn_like(p) * 0.3)
    m.save_pretrained(tmp_path / "sft")
    return tmp_path / "sft"


def _dual(sft_dir, base_seed=0):
    """Same construction as train_grpo: frozen 'reference' from disk + fresh 'grpo_policy'."""
    torch.manual_seed(base_seed)
    base = TinyLM()
    model = PeftModel.from_pretrained(base, str(sft_dir), adapter_name="reference", is_trainable=False)
    model.add_adapter("grpo_policy", _cfg())
    return model


def _logits(model, adapter, ids):
    model.set_adapter(adapter)
    with torch.no_grad():
        return model(ids).logits


IDS = torch.tensor([[1, 5, 9, 12, 20]])


def test_add_adapter_alone_starts_from_base_not_sft(sft_dir):
    """The bug: without a copy the policy equals the BASE model, while reference is the SFT model."""
    torch.manual_seed(0)
    base_out = TinyLM()(IDS).logits
    model = _dual(sft_dir)
    pol = _logits(model, "grpo_policy", IDS)
    ref = _logits(model, "reference", IDS)
    assert torch.allclose(pol, base_out, atol=1e-6)
    assert not torch.allclose(pol, ref, atol=1e-3)


def test_init_copies_sft_into_policy(sft_dir):
    model = _dual(sft_dir)
    n = init_policy_from_reference(model)
    assert n == 2  # lora_A + lora_B of the single target module
    ref = _logits(model, "reference", IDS)
    pol = _logits(model, "grpo_policy", IDS)
    assert torch.allclose(pol, ref, atol=1e-6)


def test_policy_trains_reference_stays_frozen(sft_dir):
    """compute_token_log_probs toggles adapters; only the policy may be trainable and the reference must not move."""
    grpo = pytest.importorskip("dental_agent.training.grpo")
    model = _dual(sft_dir)
    init_policy_from_reference(model)
    model.set_adapter("grpo_policy")
    enc = {"input_ids": IDS, "labels": IDS.clone()}

    ref_lp0, _ = grpo.compute_token_log_probs(model, enc, use_reference=True)
    pol_lp0, mask = grpo.compute_token_log_probs(model, enc, use_reference=False)
    assert torch.allclose(ref_lp0, pol_lp0, atol=1e-6)  # KL is exactly 0 at step 1

    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("grpo_policy" in n for n in trainable), trainable

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for _ in range(3):
        lp, mask = grpo.compute_token_log_probs(model, enc, use_reference=False)
        ref_lp, _ = grpo.compute_token_log_probs(model, enc, use_reference=True)  # toggles adapters mid-training
        loss = -(lp * mask).sum() + 0.1 * (lp - ref_lp).pow(2).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    ref_lp1, _ = grpo.compute_token_log_probs(model, enc, use_reference=True)
    pol_lp1, _ = grpo.compute_token_log_probs(model, enc, use_reference=False)
    assert torch.allclose(ref_lp0, ref_lp1, atol=1e-6), "reference drifted"
    assert not torch.allclose(pol_lp0, pol_lp1, atol=1e-4), "policy did not train"


def test_init_rejects_incompatible_adapters(sft_dir):
    torch.manual_seed(0)
    model = PeftModel.from_pretrained(TinyLM(), str(sft_dir), adapter_name="reference", is_trainable=False)
    model.add_adapter("grpo_policy", _cfg(r=8, alpha=16))
    with pytest.raises(ValueError, match="differs"):
        init_policy_from_reference(model)


def test_checkpoint_roundtrip_and_resume_state(sft_dir, tmp_path):
    out = tmp_path / "run"
    trained = _dual(sft_dir)
    init_policy_from_reference(trained)
    trained.set_adapter("grpo_policy")
    with torch.no_grad():
        for n, p in trained.named_parameters():
            if "grpo_policy" in n:
                p.add_(0.25)
    meta = {"step": 3, "baseline": 0.42, "group_size": 4, "pool_fingerprint": pool_fingerprint([1, 2, 3])}
    save_checkpoint(model=trained, processor=object(), tag="grpo-with_tools-step-3", checkpoint_dir=out, extra_metadata=meta)
    save_checkpoint(model=trained, processor=object(), tag="grpo-with_tools-step-10", checkpoint_dir=out, extra_metadata={**meta, "step": 10})
    save_checkpoint(model=trained, processor=object(), tag="grpo-no_tools-step-99", checkpoint_dir=out, extra_metadata={**meta, "step": 99})

    path, extra = find_latest_step_checkpoint(out, "with_tools")
    assert path.name == "grpo-with_tools-step-10"  # numeric, not lexicographic ("10" < "3" as text); other track ignored
    assert extra["baseline"] == 0.42 and extra["pool_fingerprint"] == meta["pool_fingerprint"]
    assert (path / "grpo_policy" / "adapter_config.json").is_file()
    assert find_latest_step_checkpoint(out, "no_tools")[0].name == "grpo-no_tools-step-99"
    assert find_latest_step_checkpoint(tmp_path / "missing", "with_tools") is None

    fresh = _dual(sft_dir)
    init_policy_from_reference(fresh)  # what train_grpo does first
    before = _logits(fresh, "grpo_policy", IDS)
    load_policy_state(fresh, path)
    after = _logits(fresh, "grpo_policy", IDS)
    assert not torch.allclose(before, after, atol=1e-4)
    assert torch.allclose(after, _logits(trained, "grpo_policy", IDS), atol=1e-6)
    # reference is never taken from the checkpoint: it still equals the SFT adapter
    assert torch.allclose(_logits(fresh, "reference", IDS), _logits(_dual(sft_dir), "reference", IDS), atol=1e-6)


def test_pool_fingerprint_is_order_sensitive():
    assert pool_fingerprint([1, 2, 3]) == pool_fingerprint([1.0, 2, 3])
    assert pool_fingerprint([1, 2, 3]) != pool_fingerprint([3, 2, 1])
    assert pool_fingerprint([1, 2, 3]) != pool_fingerprint([1, 2])


# ---------------------------------------------------------------------------
# train_grpo end to end (model, rollouts and rewards replaced by stand-ins): init from SFT, kill, resume.
# ---------------------------------------------------------------------------

def _run_train(monkeypatch, tmp_path, sft_dir, n_images=5, resume=False, kill_at=None, seen=None, start_probe=None,
               log_calls=None, **kw):
    import pandas as pd

    grpo = pytest.importorskip("dental_agent.training.grpo")
    seen = [] if seen is None else seen

    class Proc:
        def save_pretrained(self, *_a, **_k):
            pass

    def fake_load_model(cfg):
        torch.manual_seed(0)  # same base weights as _dual()
        return TinyLM(), Proc()

    monkeypatch.setattr(grpo, "load_model", fake_load_model)
    # wrap_distributed_model moves the model to CUDA whenever a GPU exists; keep the tiny model on CPU with IDS.
    monkeypatch.setattr(grpo, "wrap_distributed_model", lambda m, **kw: m)
    monkeypatch.setattr(grpo, "log_grpo_step", lambda *a, **k: log_calls.append(k) if log_calls is not None else None)

    def fake_step(model, image_ids_and_gts, running_ema_baseline, **_k):
        img_id = image_ids_and_gts[0][0]
        if start_probe is not None and not seen:
            start_probe["baseline"] = running_ema_baseline
            start_probe["logits"] = _logits(model, "grpo_policy", IDS)
        if kill_at is not None and img_id == kill_at:
            raise RuntimeError("session killed")
        seen.append(int(img_id))
        model.set_adapter("grpo_policy")
        with torch.no_grad():
            for n, p in model.named_parameters():
                if "grpo_policy" in n:
                    p.add_(0.01)
        return {"mean_reward": 0.5, "kl_divergence": 0.0}, running_ema_baseline + 1.0

    monkeypatch.setattr(grpo, "grpo_step", fake_step)
    images = pd.DataFrame({"id": list(range(1, n_images + 1)), "local_path": ["x.png"] * n_images})
    annots = pd.DataFrame({"image_id": [], "bbox": []})
    cats = pd.DataFrame({"id": [], "name": []})
    return grpo.train_grpo(
        images_df=images, annots_df=annots, categories_df=cats, sft_model_dir=str(sft_dir),
        checkpoint_dir=tmp_path / "ckpt", group_size=2, epochs_per_batch=1, kl_beta=0.04, clip_eps=0.2,
        learning_rate=1e-3, track="with_tools", push_every_steps=2, resume=resume, **kw,
    )


def test_train_grpo_starts_at_sft_then_resumes_after_kill(monkeypatch, tmp_path, sft_dir):
    # Fresh run: the first rollout must come from the SFT policy (== reference), not the bare base model.
    seen1, probe1 = [], {}
    with pytest.raises(RuntimeError, match="killed"):
        _run_train(monkeypatch, tmp_path, sft_dir, seen=seen1, kill_at=4, start_probe=probe1)
    assert seen1 == [1, 2, 3]
    assert torch.allclose(probe1["logits"], _logits(_dual(sft_dir), "reference", IDS), atol=1e-6)
    assert find_latest_step_checkpoint(tmp_path / "ckpt", "with_tools")[0].name == "grpo-with_tools-step-2"

    # Resume: skips images 1-2, restores EMA baseline (2.0 after two steps) and the step-2 policy weights.
    seen2, probe2 = [], {}
    final = _run_train(monkeypatch, tmp_path, sft_dir, resume=True, seen=seen2, start_probe=probe2)
    assert seen2 == [3, 4, 5]
    assert probe2["baseline"] == pytest.approx(2.0)
    expected = _dual(sft_dir)
    init_policy_from_reference(expected)
    with torch.no_grad():
        for n, p in expected.named_parameters():
            if "grpo_policy" in n:
                p.add_(0.02)  # two steps of +0.01
    assert torch.allclose(probe2["logits"], _logits(expected, "grpo_policy", IDS), atol=1e-5)
    assert final.endswith("grpo-with_tools-final")

    # A finished run is not trained again.
    seen3 = []
    _run_train(monkeypatch, tmp_path, sft_dir, resume=True, seen=seen3)
    assert seen3 == []


def test_resume_refuses_a_different_image_pool_or_missing_checkpoint(monkeypatch, tmp_path, sft_dir):
    with pytest.raises(FileNotFoundError):
        _run_train(monkeypatch, tmp_path, sft_dir, resume=True)
    with pytest.raises(RuntimeError, match="killed"):
        _run_train(monkeypatch, tmp_path, sft_dir, kill_at=4)
    with pytest.raises(ValueError, match="different image pool"):
        _run_train(monkeypatch, tmp_path, sft_dir, n_images=6, resume=True)
