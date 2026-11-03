"""Native MTP speculation on the TPU generator (CPU eager). The head here is randomly initialised: output equality with the
non-speculative generator must hold for ANY head (verification decides the tokens), and the head's own cache / chain are checked against
an independent full-sequence reference, so wrong pairing, positions or rollback show up as a numerical mismatch, not as low acceptance."""
import json

import pytest
import torch
from safetensors.torch import save_file

import test_tpu_generation as T
from dental_agent.model import tpu_generation as tg
from dental_agent.model import tpu_mtp
from dental_agent.model.tpu_speculative import SpeculativeConfig, parse_condition

EXPECTED_KEYS = {
    "fc.weight", "norm.weight", "pre_fc_norm_hidden.weight", "pre_fc_norm_embedding.weight",
    "layers.0.input_layernorm.weight", "layers.0.post_attention_layernorm.weight",
    "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight", "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight", "layers.0.self_attn.q_norm.weight", "layers.0.self_attn.k_norm.weight",
    "layers.0.mlp.gate_proj.weight", "layers.0.mlp.up_proj.weight", "layers.0.mlp.down_proj.weight"}


def _model(seed=1, **cfg_kw):
    model = T.tiny()
    cfg = T.cfg(**{"cache_tiers": (512,), **cfg_kw})
    tg.install_tpu_generation(model, cfg)
    text = model.config.get_text_config()
    torch.manual_seed(seed)
    head = tpu_mtp.MTPHead(text).eval().requires_grad_(False)
    for p in head.parameters():                                  # random weights, but not tiny: drafts must be non-degenerate
        p.data.normal_(0, 0.1)
    cfg.mtp_head = head
    return model, cfg, text


def _generate(model, cfg, name, ids, new, **kw):
    cfg.speculative = parse_condition(name)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), **kw.pop("extra", {})}
    return tg.tpu_generate(model, inputs, max_new_tokens=new, temperature=0.0, pad_token_id=0, **kw)


def test_conditions_and_the_head_layout():
    spec = parse_condition("mtp_3")
    assert (spec.method, spec.k, spec.width) == ("mtp", 3, 4) and spec.record() == dict(method="mtp", num_speculative_tokens=3)
    assert parse_condition("ngram_3").record()["prompt_lookup_max"] == 3
    for bad in ("mtp", "mtp_x", "mtp_0", "mtp_9"):
        with pytest.raises(ValueError):
            parse_condition(bad)
    _, _, text = _model()
    assert set(tpu_mtp.MTPHead(text).state_dict()) == EXPECTED_KEYS


def test_loader_reads_mtp_tensors_strictly(tmp_path):
    _, _, text = _model()
    head = tpu_mtp.MTPHead(text)
    state = {"mtp." + k: v.clone() for k, v in head.state_dict().items()}
    save_file({**state, "model.language_model.embed_tokens.weight": torch.zeros(2, 2)}, str(tmp_path / "model-00001.safetensors"))
    loaded = tpu_mtp.build_mtp_head(tmp_path, text, device=torch.device("cpu"), dtype=torch.float32)
    for k, v in head.state_dict().items():
        assert torch.equal(loaded.state_dict()[k], v)
    assert not any(p.requires_grad for p in loaded.parameters())
    bad = tmp_path / "bad"
    bad.mkdir()
    broken = dict(state)
    broken["mtp.fc.weight"] = torch.zeros(3, 3)
    del broken["mtp.norm.weight"]
    broken["mtp.layers.0.mlp.experts.gate_up_proj"] = torch.zeros(2)
    save_file(broken, str(bad / "m.safetensors"))
    with pytest.raises(ValueError) as err:
        tpu_mtp.build_mtp_head(bad, text, device=torch.device("cpu"), dtype=torch.float32)
    assert "norm.weight" in str(err.value) and "fc.weight" in str(err.value) and "experts.gate_up_proj" in str(err.value)
    none = tmp_path / "none"
    none.mkdir()
    save_file({"model.x": torch.zeros(1)}, str(none / "m.safetensors"))
    with pytest.raises(ValueError, match="no mtp"):
        tpu_mtp.build_mtp_head(none, text, device=torch.device("cpu"), dtype=torch.float32)


def test_mtp_needs_the_attached_head():
    model = T.tiny()
    cfg = T.cfg(cache_tiers=(512,))
    tg.install_tpu_generation(model, cfg)
    ids = T.text_prompt(40, 5).unsqueeze(0)
    with pytest.raises(ValueError, match="attach_mtp_head"):
        _generate(model, cfg, "mtp_2", ids, 8)


@pytest.mark.parametrize("name", ["mtp_1", "mtp_2", "mtp_3", "mtp_5"])
@pytest.mark.parametrize("seed,length", [(5, 100), (6, 37), (7, 128)])
def test_output_equals_the_non_speculative_generator_for_any_head(name, seed, length):
    model, cfg, _ = _model()
    ids = T.text_prompt(length, seed).unsqueeze(0)
    base = _generate(model, cfg, "baseline", ids, 60)
    out = _generate(model, cfg, name, ids, 60)
    assert torch.equal(base, out), (base, out)
    last = cfg.stats["last_call"]
    assert last["generated_tokens"] == 60 and last["speculative"] == dict(method="mtp", num_speculative_tokens=int(name[-1]))
    assert last["drafted"] == int(name[-1]) * last["rounds"] and last["plain_steps"] == 0


def test_eos_boundaries_penalty_and_images_stay_exact():
    model, cfg, _ = _model()
    ids = T.text_prompt(100, 5).unsqueeze(0)
    base = _generate(model, cfg, "baseline", ids, 60)
    for position in (2, 20, 41):
        eos = int(base[0, position])
        first = int((base[0] == eos).nonzero()[0])
        got = _generate(model, cfg, "mtp_3", ids, 60, eos_token_id=eos)
        assert torch.equal(_generate(model, cfg, "baseline", ids, 60, eos_token_id=eos), got) and got.shape[1] == first + 1
    for new in (1, 2, 3, 5):
        assert torch.equal(_generate(model, cfg, "baseline", ids, new), _generate(model, cfg, "mtp_3", ids, new))
    for penalty in (1.3, 2.0):
        assert torch.equal(_generate(model, cfg, "baseline", ids, 50, repetition_penalty=penalty),
                           _generate(model, cfg, "mtp_3", ids, 50, repetition_penalty=penalty))
    iids, mm = T.image_prompt(40, 20, 41)
    extra = dict(pixel_values=T.pixels(42), image_grid_thw=torch.tensor([T.grid_thw("CROP")]), mm_token_type_ids=mm[None])
    assert torch.equal(_generate(model, cfg, "baseline", iids[None], 40, extra=dict(extra)),
                       _generate(model, cfg, "mtp_2", iids[None], 40, extra=dict(extra)))


def test_the_cache_tier_must_hold_the_draft_overrun():
    model, cfg, _ = _model(cache_tiers=(256,))
    ids = T.text_prompt(150, 5).unsqueeze(0)
    with pytest.raises(tg.GenerationTooLong):
        _generate(model, cfg, "mtp_5", ids, 64)


class Oracle:
    """Replace the head's argmax by the true continuation, corrupted at chosen generated positions; records every head output."""

    def __init__(self, truth, corrupt):
        self.truth, self.corrupt, self.log = truth, corrupt, []

    def install(self, monkeypatch):
        oracle = self

        def propose(drafter, out, index):
            i = int(index)
            token = int(oracle.truth[min(i, len(oracle.truth) - 1)])
            if oracle.corrupt(i):
                token = (token + 7) % 128
            oracle.log.append((i, out.clone(), token))
            return torch.full((out.shape[0],), token, dtype=torch.long)
        monkeypatch.setattr(tpu_mtp.MTPDrafter, "propose", propose)


@pytest.mark.parametrize("k,corrupt", [(1, lambda i: False), (3, lambda i: False), (3, lambda i: i % 4 == 0), (5, lambda i: i % 3 == 1)])
def test_oracle_drafts_exercise_full_and_partial_acceptance(monkeypatch, k, corrupt):
    model, cfg, _ = _model()
    ids = T.text_prompt(100, 5).unsqueeze(0)
    base = _generate(model, cfg, "baseline", ids, 70)
    Oracle(base[0].tolist(), corrupt).install(monkeypatch)
    out = _generate(model, cfg, f"mtp_{k}", ids, 70)
    last = cfg.stats["last_call"]
    assert torch.equal(base, out)
    assert last["accepted"] > 0 and last["rounds"] < 70
    if corrupt(10) or k > 1 and any(corrupt(i) for i in range(70)):
        assert last["accepted"] < last["drafted"]
    else:
        assert last["accepted"] == last["drafted"] - 0 or last["accepted"] >= last["drafted"] - k   # perfect drafts: only the tail is cut


@pytest.mark.parametrize("k,corrupt,length", [(3, lambda i: i % 4 == 0, 128), (3, lambda i: i % 3 == 2, 64), (2, lambda i: False, 192)])
def test_the_heads_cache_and_chain_match_an_independent_full_sequence_reference(monkeypatch, k, corrupt, length):
    """Every head output of the decode (prefill fill, block refill, rewind after rejections, chain) equals running the head over the whole
    token sequence at once. Prompts are whole chunks so nothing is padded."""
    model, cfg, text = _model()
    ids = T.text_prompt(length, 5).unsqueeze(0)
    P = ids.shape[1]
    base = _generate(model, cfg, "baseline", ids, 48)
    oracle = Oracle(base[0].tolist(), corrupt)
    oracle.install(monkeypatch)
    _generate(model, cfg, f"mtp_{k}", ids, 48)
    lm, body, head = model.model.language_model, model.model.language_model, cfg.mtp_head
    gen = base[0].tolist()
    checked = 0
    pending = 0
    entries = list(oracle.log)
    while entries:
        round_entries, entries = entries[:k], entries[k:]
        idx1 = round_entries[0][0]
        pending = idx1 - 1                                         # generated index of the pending (not yet fed) token
        if pending >= len(gen):                                    # drafts after the last emitted token: nothing to compare with
            continue
        fed = torch.cat([ids[0], torch.tensor(gen[:pending], dtype=torch.long)])[None]
        with torch.no_grad():
            hidden = body(input_ids=fed).last_hidden_state[0]      # [L, H] after the final norm
            seq = torch.cat([fed[0], torch.tensor([gen[pending]])])
            pairs_tok = seq[1:]                                    # x_{i+1} for every fed position i
            H, E = hidden, body.embed_tokens(pairs_tok)
            ref_out = None
            for j, (i, out, token) in enumerate(round_entries):
                L = H.shape[0]
                pos = torch.arange(L).view(1, 1, L).expand(3, 1, L)
                mask = tg.attention_masks(torch.zeros(1, dtype=torch.long), torch.zeros((), dtype=torch.long), L, L, float_mask=False,
                                          dtype=torch.float32, linear=False)["full_attention"]
                pe = body.rotary_emb(H[None], pos)
                ref = head(H[None], E[None], pe, mask, None)[0, -1]
                assert torch.allclose(out[0], ref, atol=2e-4, rtol=2e-4), (i, j, float((out[0] - ref).abs().max()))
                checked += 1
                H = torch.cat([H, ref[None]])                      # next pair: (head output, the draft token that came out of it)
                E = torch.cat([E, body.embed_tokens(torch.tensor([token]))])
    assert checked >= 3 * k
