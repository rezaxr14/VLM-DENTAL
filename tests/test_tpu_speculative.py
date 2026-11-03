"""TPU speculative decoding (CPU eager): prompt-lookup drafts verified in one static block, hybrid recurrent state rolled back on rejection.

The tiny model is hybrid (three Gated-DeltaNet layers and one full-attention layer), so every partial acceptance exercises the
recurrent/conv rollback. Equality is against the non-speculative static generator, itself proven equal to stock ``generate``."""
import pytest
import torch

import test_tpu_generation as T
from dental_agent.model import tpu_generation as tg
from dental_agent.model.tpu_speculative import SpeculativeConfig, ngram_draft, parse_condition


def _generate(model, cfg, name, ids, new, **kw):
    cfg.speculative = parse_condition(name)
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), **kw.pop("extra", {})}
    return tg.tpu_generate(model, inputs, max_new_tokens=new, temperature=0.0, pad_token_id=0, **kw)


def _model(**cfg_kw):
    model = T.tiny()
    cfg = T.cfg(**{"cache_tiers": (512,), **cfg_kw})
    tg.install_tpu_generation(model, cfg)
    return model, cfg


def test_ngram_draft_picks_the_longest_most_recent_suffix_match():
    assert ngram_draft([1, 2, 3, 9, 1, 2, 3], k=3, min_n=1, max_n=3) == [9, 1, 2]
    assert ngram_draft([5, 6, 7, 5, 6, 8, 5, 6], k=2, min_n=1, max_n=2) == [8, 5]      # most recent earlier "5 6"
    assert ngram_draft([1, 2, 3, 4], k=3, min_n=1, max_n=3) == []                       # nothing repeats
    assert ngram_draft([4, 4], k=3, min_n=1, max_n=3) == [4]                            # an overlapping match is a match
    assert ngram_draft([7, 8, 7], k=5, min_n=1, max_n=5) == [8, 7]                      # truncated at the end of the context
    assert ngram_draft([3], k=2, min_n=1, max_n=2) == []


def test_conditions_match_the_vllm_benchmark_names():
    assert parse_condition("baseline") is None
    spec = parse_condition("ngram_5")
    assert (spec.k, spec.prompt_lookup_min, spec.prompt_lookup_max, spec.width) == (5, 1, 5, 6)
    for bad in ("mtp_1", "ngram", "ngram_x", "lookup_3"):
        with pytest.raises(ValueError):
            parse_condition(bad)
    with pytest.raises(ValueError):
        SpeculativeConfig(k=0)


@pytest.mark.parametrize("name", ["ngram_2", "ngram_3", "ngram_5"])
@pytest.mark.parametrize("seed,length", [(5, 100), (6, 37), (7, 150)])
def test_greedy_output_equals_the_non_speculative_generator(name, seed, length):
    model, cfg = _model()
    ids = T.text_prompt(length, seed).unsqueeze(0)
    base = _generate(model, cfg, "baseline", ids, 80)
    out = _generate(model, cfg, name, ids, 80)
    assert torch.equal(base, out), (base, out)
    last = cfg.stats["last_call"]
    assert last["generated_tokens"] == 80 and last["speculative"]["num_speculative_tokens"] == int(name[-1])


def test_rejections_happen_and_the_state_rollback_keeps_the_output_exact():
    model, cfg = _model()
    ids = T.text_prompt(100, 5).unsqueeze(0)
    base = _generate(model, cfg, "baseline", ids, 80)
    out = _generate(model, cfg, "ngram_3", ids, 80)
    last = cfg.stats["last_call"]
    assert torch.equal(base, out)
    assert last["accepted"] > 0 and last["accepted"] < last["drafted"]          # both full and partial acceptances occurred
    assert last["rounds"] + last["plain_steps"] < 80                             # fewer forwards than tokens
    assert cfg.stats["spec_rounds"] == last["rounds"]


def test_eos_inside_an_accepted_block_stops_exactly_where_baseline_stops():
    model, cfg = _model()
    ids = T.text_prompt(100, 5).unsqueeze(0)
    base = _generate(model, cfg, "baseline", ids, 80)
    for position in (3, 33, 55, 70):
        eos = int(base[0, position])
        first = int((base[0] == eos).nonzero()[0])
        want = _generate(model, cfg, "baseline", ids, 80, eos_token_id=eos)
        got = _generate(model, cfg, "ngram_3", ids, 80, eos_token_id=eos)
        assert torch.equal(want, got) and got.shape[1] == first + 1, (position, want, got)


@pytest.mark.parametrize("new", [1, 2, 3, 4, 7])
def test_max_new_tokens_boundaries(new):
    model, cfg = _model()
    ids = T.text_prompt(100, 5).unsqueeze(0)
    base = _generate(model, cfg, "baseline", ids, new)
    out = _generate(model, cfg, "ngram_5", ids, new)
    assert torch.equal(base, out) and out.shape[1] == new


def test_repetition_penalty_is_applied_position_by_position():
    model, cfg = _model()
    ids = T.text_prompt(100, 5).unsqueeze(0)
    for penalty in (1.3, 2.0):
        base = _generate(model, cfg, "baseline", ids, 60, repetition_penalty=penalty)
        out = _generate(model, cfg, "ngram_3", ids, 60, repetition_penalty=penalty)
        assert torch.equal(base, out), (penalty, base, out)


def test_image_prompt_with_an_empty_slot_speculates_exactly():
    model, cfg = _model()
    ids, mm = T.image_prompt(40, 20, 41)
    grid = torch.tensor([T.grid_thw("CROP")])
    extra = dict(pixel_values=T.pixels(42), image_grid_thw=grid, mm_token_type_ids=mm[None])
    base = _generate(model, cfg, "baseline", ids[None], 60, extra=dict(extra))
    out = _generate(model, cfg, "ngram_3", ids[None], 60, extra=dict(extra))
    assert torch.equal(base, out)


def test_the_cache_tier_must_hold_the_speculative_overrun():
    model, cfg = _model(cache_tiers=(256,))
    ids = T.text_prompt(150, 5).unsqueeze(0)
    cfg.speculative = None
    assert tg.tpu_generate(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, max_new_tokens=64, temperature=0.0,
                           pad_token_id=0).shape[1] > 0                               # 192 + 64 = 256 fits without speculation
    with pytest.raises(tg.GenerationTooLong):
        _generate(model, cfg, "ngram_5", ids, 64)                                     # + 5 draft positions does not
    assert cfg.stats["too_long"] == 1


def test_speculation_is_greedy_and_single_sequence_only():
    model, cfg = _model()
    ids = T.text_prompt(40, 5).unsqueeze(0)
    cfg.speculative = parse_condition("ngram_3")
    with pytest.raises(ValueError, match="greedy"):
        tg.tpu_generate(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, max_new_tokens=8, temperature=0.7, pad_token_id=0)
    two = torch.cat([ids, ids])
    with pytest.raises(ValueError, match="batch of 1"):
        tg.tpu_generate(model, {"input_ids": two, "attention_mask": torch.ones_like(two)}, max_new_tokens=8, temperature=0.0, pad_token_id=0)


def test_the_recorder_restores_the_model_module_afterwards():
    from transformers.models.qwen3_5 import modeling_qwen3_5 as m
    before = m.torch_chunk_gated_delta_rule
    model, cfg = _model()
    ids = T.text_prompt(60, 5).unsqueeze(0)
    _generate(model, cfg, "ngram_3", ids, 30)
    assert m.torch_chunk_gated_delta_rule is before
    base = _generate(model, cfg, "baseline", ids, 30)                                # and ordinary decoding is unaffected
    assert base.shape[1] == 30
