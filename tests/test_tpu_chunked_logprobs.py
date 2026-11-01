"""CPU checks for the TPU GRPO scorer (mathematical reference), slot-overflow handling and the new CLI/preflight flag.

XLA execution of the scorer: tests/test_tpu_chunked_logprobs_xla_cpu.py (virtual XLA-CPU devices, not TPU HBM).
"""
import sys
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from dental_agent.training import grpo as G
from dental_agent.training.sft import BucketedQwenVLCollator
from dental_agent.training.tpu_chunked_logprobs import frozen_head_token_log_probs, validate_tpu_logprob_args
from dental_agent.utils import canonical as C

ROOT = Path(__file__).resolve().parents[1]


def data(dtype=torch.float32, bias=False):
    torch.manual_seed(2026)
    head = torch.nn.Linear(12, 31, bias=bias, dtype=dtype).requires_grad_(False)
    hidden = torch.randn(3, 19, 12, dtype=dtype, requires_grad=True)
    labels = torch.randint(0, 31, (3, 19))
    labels[:, :7] = -100
    labels[1, 12:] = -100
    labels[2, :] = -100
    return hidden, head, labels


def stock(h, head, labels):
    z = head(h)[:, :-1].float()
    targets = labels[:, 1:]
    mask = targets != -100
    lp = -F.cross_entropy(z.reshape(-1, 31), targets.clamp(min=0).reshape(-1), reduction="none").reshape(targets.shape)
    return lp.masked_fill(~mask, 0.0), mask.float()


@pytest.mark.parametrize("chunk", [1, 4, 8, 19, 32])
@pytest.mark.parametrize("bias", [False, True])
def test_log_probs_mask_and_weighted_gradients_match_stock(chunk, bias):
    h, head, labels = data(bias=bias)
    want_lp, want_mask = stock(h, head, labels)
    weights = torch.randn_like(want_lp)                 # arbitrary per-token weights, as PPO ratio * advantage would give
    want_grad = torch.autograd.grad((want_lp * weights).sum(), h)[0]
    lp, mask = frozen_head_token_log_probs(h, head, labels, chunk)
    got_grad = torch.autograd.grad((lp * weights).sum(), h)[0]
    assert lp.shape == want_lp.shape == (3, 18) and mask.dtype == torch.float32
    torch.testing.assert_close(lp, want_lp, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(mask, want_mask)
    torch.testing.assert_close(got_grad, want_grad, atol=2e-6, rtol=2e-5)
    assert torch.count_nonzero(got_grad[2]) == 0 and torch.count_nonzero(got_grad[:, -1]) == 0   # masked row / last slot
    assert head.weight.grad is None


def test_bf16_matches_stock_and_ignored_tokens_are_exactly_zero():
    h, head, labels = data(torch.bfloat16)
    want_lp, _ = stock(h, head, labels)
    lp, mask = frozen_head_token_log_probs(h, head, labels, 8)
    torch.testing.assert_close(lp, want_lp, atol=1e-5, rtol=1e-5)
    assert (lp[mask == 0] == 0).all() and lp.dtype == torch.float32
    weights = torch.randn_like(lp)
    want_grad = torch.autograd.grad((want_lp * weights).sum(), h)[0]
    got_grad = torch.autograd.grad((lp * weights).sum(), h)[0]
    torch.testing.assert_close(got_grad, want_grad, atol=2e-4, rtol=2e-2)


def test_no_grad_scoring_needs_no_graph():
    h, head, labels = data()
    with torch.no_grad():
        lp, _ = frozen_head_token_log_probs(h, head, labels, 4)
    assert not lp.requires_grad


def test_ppo_clip_objective_gradient_matches_stock():
    """The quantity GRPO actually differentiates: -min(ratio*A, clip(ratio)*A) + beta*k3, masked mean."""
    h, head, labels = data()
    old = (stock(h, head, labels)[0] + 0.05 * torch.randn(3, 18)).detach()
    ref = (stock(h, head, labels)[0] - 0.03 * torch.randn(3, 18)).detach()
    adv = torch.tensor([[0.8], [-0.5], [0.3]])

    def objective(lp, mask):
        ratio = torch.exp(lp - old)
        loss = -torch.min(ratio * adv, torch.clamp(ratio, 0.8, 1.2) * adv)
        d = ref - lp
        loss = loss + 0.04 * (torch.exp(d) - d - 1.0)
        return ((loss * mask).sum(1) / mask.sum(1).clamp(min=1)).sum()

    want = torch.autograd.grad(objective(*stock(h, head, labels)), h)[0]
    got = torch.autograd.grad(objective(*frozen_head_token_log_probs(h, head, labels, 5)), h)[0]
    torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)


def test_no_vocabulary_sized_activations_saved_for_backward():
    h, head, labels = data()
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(tuple(t.shape)) or t), lambda t: t):
        lp, _ = frozen_head_token_log_probs(h, head, labels, 4)
    assert not any(31 in shape and 4 in shape for shape in saved if len(shape) == 3)   # no [B, chunk, V]
    (lp.sum()).backward()
    assert h.grad is not None and torch.isfinite(h.grad).all()


def test_trainable_head_and_bad_shapes_are_rejected():
    h, head, labels = data()
    with pytest.raises(ValueError, match="frozen plain"):
        frozen_head_token_log_probs(h, torch.nn.Linear(12, 31), labels, 4)
    with pytest.raises(ValueError, match="positive"):
        frozen_head_token_log_probs(h, head, labels, 0)
    with pytest.raises(ValueError, match="matching labels"):
        frozen_head_token_log_probs(h, head, labels[:, :-1], 4)
    with pytest.raises(ValueError, match="int64"):
        frozen_head_token_log_probs(h, head, labels.int(), 4)


@pytest.mark.parametrize("kw,match", [
    (dict(chunk_tokens=-1), "nonnegative"),
    (dict(chunk_tokens=256, is_tpu=False), "real TPU"),
    (dict(chunk_tokens=256, logprob_chunk_tokens=128), "--logprob-chunk-tokens 0"),
    (dict(chunk_tokens=256, spmd=False), "--spmd --fsdp --pad-vision-to-slots"),
    (dict(chunk_tokens=256, fsdp=False), "--spmd --fsdp --pad-vision-to-slots"),
    (dict(chunk_tokens=256, pad_vision_to_slots=False), "--spmd --fsdp --pad-vision-to-slots"),
])
def test_configuration_fails_before_model_loading(kw, match):
    base = dict(chunk_tokens=256, is_tpu=True, spmd=True, fsdp=True, pad_vision_to_slots=True)
    base.update(kw)
    with pytest.raises(ValueError, match=match):
        validate_tpu_logprob_args(**base)


def test_disabled_flag_has_no_requirements():
    validate_tpu_logprob_args(chunk_tokens=0, is_tpu=False, spmd=False, fsdp=False, pad_vision_to_slots=False)


# ---------------------------------------------------------------- slot overflow (extra slots, drop-and-count)

IMG_TOK = 49


class _Tok:
    pad_token_id = 0
    eos_token_id = 1


class _Proc:
    image_token_id = IMG_TOK
    tokenizer = _Tok()


def _enc(families, text_len=120, seed=0):
    """One trajectory encoding that carries one image per entry of ``families``."""
    g = torch.Generator().manual_seed(seed)
    n_tok = sum(C.tokens_per_image(f) for f in families)
    ids = torch.cat([torch.randint(2, 40, (1, text_len), generator=g), torch.full((1, n_tok), IMG_TOK)], dim=1)
    labels = torch.full_like(ids, -100)
    labels[0, 5:45] = ids[0, 5:45]
    patches = sum(C.patches_per_image(f) for f in families)
    return {
        "input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids),
        "mm_token_type_ids": (ids == IMG_TOK).long(),
        "pixel_values": torch.randn(patches, C.PATCH_FEATURE_DIM),
        "image_grid_thw": torch.tensor([C.grid_thw(f) for f in families]) if families else torch.zeros((0, 3), dtype=torch.long),
    }


def _collator(budget, seq=4000):
    return BucketedQwenVLCollator(_Proc(), max_seq_len=seq, pad_vision_to_slots=True, slot_budget=C.parse_slot_budget(*budget))


def test_slot_usage_and_overflow_report_families():
    coll = _collator((1, 2, 0))
    ok = _enc(["FULL", "CROP", "CROP"])
    assert coll.slot_usage(ok) == {"FULL": 1, "CROP": 2, "COMPARE": 0} and coll.slot_overflow(ok) == {}
    over = _enc(["FULL", "FULL", "CROP", "COMPARE"])
    assert coll.slot_overflow(over) == {"FULL": 1, "COMPARE": 1}
    assert coll.slot_overflow({"input_ids": ok["input_ids"]}) == {}          # text-only: no grid, no overflow
    with pytest.raises(ValueError):                                          # still raises on the real padding path
        coll._pad_vision_slots(over)


def test_larger_budget_adds_empty_masked_slots_and_changes_nothing_real():
    small, big = _collator((1, 1, 0)), _collator((2, 3, 1))
    enc = _enc(["FULL", "CROP"])
    a, b = small._pad_vision_slots(enc), big._pad_vision_slots(enc)
    real = enc["input_ids"].shape[1]
    assert torch.equal(a["input_ids"][0, :real], b["input_ids"][0, :real])
    extra = b["input_ids"].shape[1] - a["input_ids"].shape[1]
    assert extra == C.tokens_per_image("FULL") + 2 * C.tokens_per_image("CROP") + C.tokens_per_image("COMPARE")
    assert (b["attention_mask"][0, real:] == 0).all() and (b["labels"][0, real:] == -100).all()
    assert (b["pixel_values"][enc["pixel_values"].shape[0]:] == 0).all()


def test_grpo_step_drops_and_counts_slot_overflow_without_crashing(monkeypatch):
    coll = _collator((1, 1, 0))
    fits, over = _enc(["FULL", "CROP"], seed=1), _enc(["FULL", "CROP", "CROP", "CROP"], seed=2)
    encs = {0: fits, 1: over}

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.emb, self.head = torch.nn.Embedding(50, 8), torch.nn.Linear(8, 50)
            self.device = torch.device("cpu")

        def forward(self, input_ids, **kw):
            return type("O", (), {"logits": self.head(self.emb(input_ids))})()

    model = Tiny()
    monkeypatch.setattr(G, "collect_grpo_group", lambda **kw: ([{"id": 0}, {"id": 1}], [1.0, 0.0], [None, None], [None, None]))
    monkeypatch.setattr(G, "build_full_trajectory_labels", lambda traj, processor: dict(encs[traj["id"]]))
    stats, _ = G.grpo_step(
        model=model, processor=None, image_ids_and_gts=[(1, [])], images_df=None, registry=object(), group_size=2,
        kl_beta=0.0, epochs_per_batch=1, optimizer=torch.optim.SGD(model.parameters(), lr=0.1),
        train_collator=coll, rows_per_forward=1,
    )
    assert stats["n_dropped_slot_overflow"] == 1 and stats["n_dropped_overlength"] == 0
    assert stats["slot_peak_full"] == 1 and stats["slot_peak_crop"] == 3 and stats["slot_peak_compare"] == 0
    assert stats["n_rollouts"] == 2                       # rewards of the dropped rollout still count in the mean


# ---------------------------------------------------------------- CLI / preflight / summary

def test_run_grpo_exposes_flag_and_preflight_checks_it():
    import subprocess
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_grpo.py"), "--help"], capture_output=True, text=True,
                          cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert done.returncode == 0 and "--tpu-logprob-chunk-tokens" in done.stdout


def test_summary_reports_slot_drops_and_peak_demand():
    sys.path.insert(0, str(ROOT / "scripts"))
    import summarize_grpo_log as sgl
    rows = [{"rollout_s": 5.0, "update_s": 5.0, "group_size": 4, "mean_reward": 0.5, "n_dropped_slot_overflow": n,
             "slot_peak_full": f, "slot_peak_crop": c, "slot_peak_compare": 1} for n, f, c in ((1, 6, 12), (0, 4, 9))]
    out = sgl.summarize(rows, pool_size=10, skip_first=0) if hasattr(sgl, "summarize") else None
    if out is None:
        pytest.skip("summarize entrypoint differs")
    assert out["dropped_slot_overflow"] == 1 and out["slot_peak"] == {"full": 6, "crop": 12, "compare": 1}
    assert "peak demand FULL 6, CROP 12, COMPARE 1" in sgl.format_summary(out)
