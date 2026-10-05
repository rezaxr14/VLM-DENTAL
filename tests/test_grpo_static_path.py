"""GRPO static-shape policy update: numerical equivalence with the per-trajectory path, row masking, SPMD flags."""

import sys

import pytest
import torch

from dental_agent.training import grpo as G
from dental_agent.training.sft import BucketedQwenVLCollator
from dental_agent.utils import canonical as C

VOCAB, IMG_TOK = 50, 49


class _Out:
    def __init__(self, logits):
        self.logits = logits


class TinyLM(torch.nn.Module):
    """Causal toy LM: logits depend only on the prefix, so right-padding cannot change real-token log-probs."""

    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(VOCAB, 16)
        self.rnn = torch.nn.GRU(16, 16, batch_first=True)
        self.head = torch.nn.Linear(16, VOCAB)

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, mm_token_type_ids=None, **kw):
        h, _ = self.rnn(self.emb(input_ids))
        return _Out(self.head(h))


class _Tok:
    pad_token_id = 0
    eos_token_id = 1


class _Proc:
    image_token_id = IMG_TOK

    def __init__(self):
        self.tokenizer = _Tok()


def _enc(text_len, n_label, seed):
    g = torch.Generator().manual_seed(seed)
    n_img = C.tokens_per_image("FULL")
    ids = torch.cat([torch.randint(2, 40, (1, text_len), generator=g), torch.full((1, n_img), IMG_TOK)], dim=1)
    labels = torch.full_like(ids, -100)
    labels[0, 5:5 + n_label] = ids[0, 5:5 + n_label]
    return {
        "input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids),
        "mm_token_type_ids": (ids == IMG_TOK).long(),
        "pixel_values": torch.randn(C.patches_per_image("FULL"), C.PATCH_FEATURE_DIM),
        "image_grid_thw": torch.tensor([C.grid_thw("FULL")]),
    }


BUDGET = C.parse_slot_budget(1, 1, 0)  # 1 FULL (4608 patches) + 1 CROP slot
SEQ = 1152 + 96 + 400


def _collator():
    return BucketedQwenVLCollator(_Proc(), max_seq_len=SEQ, pad_vision_to_slots=True, slot_budget=BUDGET)


def _legacy_loss_and_grads(model, encs, advs, old_lps, clip_eps, kl_beta, n_total):
    model.zero_grad()
    for enc, adv, old in zip(encs, advs, old_lps):
        new_lp, mask = G.compute_token_log_probs(_Wrap(model), enc, use_reference=False)
        ratio = torch.exp(new_lp - old)
        pt = -torch.min(ratio * adv, torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv)
        (((pt * mask).sum() / mask.sum().clamp(min=1)) / n_total).backward()
    return {n: p.grad.clone() for n, p in model.named_parameters()}


class _Wrap(torch.nn.Module):
    """compute_token_log_probs unwraps via .module and toggles adapters if present; the toy model has none."""

    def __init__(self, m):
        super().__init__()
        self.module = m

    def forward(self, **kw):
        return self.module(**kw)


def test_static_update_matches_per_trajectory_update():
    torch.manual_seed(1)
    model = TinyLM()
    encs = [_enc(300, 40, 1), _enc(250, 25, 2), _enc(380, 60, 3)]
    advs = torch.tensor([0.8, -0.5, 0.3])
    coll = _collator()

    # reference: per-trajectory, unpadded, as in the original loop
    with torch.no_grad():
        old_ref = [G.compute_token_log_probs(_Wrap(model), e)[0] for e in encs]
    ref_grads = _legacy_loss_and_grads(model, encs, advs, old_ref, 0.2, 0.0, len(encs))

    # static path: micro-batches of 2 rows with an inert filler row in the last batch
    rows = 2
    ident = lambda b: b
    old_static, _ = G.compute_static_old_log_probs(_Wrap(model), encs, coll, rows, ident)
    model.zero_grad()
    G.static_policy_epoch(_Wrap(model), coll, rows, [(encs, advs, old_static)], 0.2, 0.0, len(encs), ident)
    for n, p in model.named_parameters():
        assert torch.allclose(p.grad, ref_grads[n], atol=1e-5, rtol=1e-4), n


def test_filler_rows_have_zero_loss_and_gradient():
    model = TinyLM()
    coll = _collator()
    encs = [_enc(300, 40, 1)]
    batch = G.collate_static_rows(encs, coll, rows=4)
    assert batch["input_ids"].shape == (4, SEQ)
    assert (batch["labels"][1:] == -100).all() and (batch["labels"][0] != -100).any()
    assert batch["pixel_values"].shape[0] == 4 * C.slot_totals(BUDGET)["patches"]
    with pytest.raises(ValueError):
        G.collate_static_rows(encs * 5, coll, rows=4)


def test_every_trajectory_length_maps_to_one_shape():
    coll = _collator()
    shapes = {tuple(G.collate_static_rows([_enc(L, 10, L)], coll, rows=2)["input_ids"].shape) for L in (120, 300, 390)}
    assert shapes == {(2, SEQ)}


def test_padded_length_reports_text_plus_static_vision():
    coll = _collator()
    enc = _enc(300, 10, 5)
    assert coll.padded_length(enc) == 300 + C.slot_totals(BUDGET)["tokens"]


def test_grpo_cli_flags_explicit(monkeypatch):
    import importlib
    run_grpo = importlib.import_module("scripts.run_grpo")
    seen = {}

    class Stop(Exception):
        pass

    def fake_parse(self, *a, **k):
        raise Stop

    import argparse
    parser_cls = argparse.ArgumentParser
    monkeypatch.setattr(sys, "argv", ["run_grpo.py"])
    orig = parser_cls.parse_args
    monkeypatch.setattr(parser_cls, "parse_args", lambda self, *a, **k: (seen.setdefault("args", orig(self, *a, **k)), (_ for _ in ()).throw(Stop()))[0])
    with pytest.raises(Stop):
        run_grpo.main()
    a = seen["args"]
    assert a.spmd is True and a.canonical_resize is False and a.pad_vision_to_slots is False
    assert a.vision_slots == [5, 10, 4] and a.triangular_shim is True and a.max_seq_len is None


def test_sweep_cli_accepts_every_notebook_flag(monkeypatch):
    """Regression: the notebook passed --model-id in sweep mode, which the sweep launcher rejected."""
    from scripts.run_grpo_sweep import parse_args
    argv = ["run_grpo_sweep.py", "--track", "with_tools", "--model-id", "Qwen/Qwen3.5-9B", "--sft-stage", "dentex_alone",
            "--k-values", "1", "2", "--epochs", "2", "--lr", "5e-06", "--hf-repo", "x/y", "--num-cores", "8", "--fsdp",
            "--spmd", "--canonical-resize", "--pad-vision-to-slots", "--max-seq-len", "16384",
            "--vision-slots", "5", "10", "4", "--triangular-shim"]
    monkeypatch.setattr(sys, "argv", argv)
    a = parse_args()
    assert a.model_id == "Qwen/Qwen3.5-9B" and a.spmd and a.canonical_resize and a.pad_vision_to_slots
    assert a.vision_slots == [5, 10, 4] and a.max_seq_len == 16384
