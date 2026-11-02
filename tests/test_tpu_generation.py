"""Static-shape TPU generation (CPU eager): token-for-token equal to stock ``generate`` on a tiny hybrid Qwen3.5.

Proves the numerics of chunked prefill, left padding, the pre-built mask dict, host-side M-RoPE positions, empty
masked vision slots and the device-counter decode loop. Compile behaviour is in ``test_tpu_generation_xla_cpu.py``.
"""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers import Qwen3_5ForConditionalGeneration  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import (  # noqa: E402
    Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig,
)

from dental_agent.model import tpu_generation as tg  # noqa: E402
from dental_agent.utils.canonical import PATCH_FEATURE_DIM, grid_thw, tokens_per_image, patches_per_image  # noqa: E402

IMG, VSTART, VEND, EOS = 100, 101, 102, 7
BUDGET = {"FULL": 0, "CROP": 2, "COMPARE": 0}


def tiny(seed=0, vocab=128):
    text = Qwen3_5TextConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=1024,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    vision = Qwen3_5VisionConfig(
        depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=64, patch_size=16,
        spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=2304,
    )
    torch.manual_seed(seed)
    cfg = Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=False,
                        image_token_id=IMG, vision_start_token_id=VSTART, vision_end_token_id=VEND)
    model = Qwen3_5ForConditionalGeneration(cfg).eval()
    model.generation_config.eos_token_id = EOS
    model.generation_config.pad_token_id = 0
    for k in ("top_k", "top_p", "temperature"):
        setattr(model.generation_config, k, None)
    return model


def cfg(**kw):
    base = dict(chunk_tokens=64, cache_tiers=(256, 512), slot_budget=BUDGET)
    base.update(kw)
    return tg.TPUGenerationConfig(**base)


def text_prompt(n, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(10, 90, (n,), generator=g)


def image_prompt(n_before, n_after, seed):
    n_vis = tokens_per_image("CROP")
    ids = torch.cat([text_prompt(n_before, seed), torch.tensor([VSTART]), torch.full((n_vis,), IMG),
                     torch.tensor([VEND]), text_prompt(n_after, seed + 1)])
    mm = torch.zeros_like(ids)
    mm[n_before + 1:n_before + 1 + n_vis] = 1
    return ids, mm


def pixels(seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((patches_per_image("CROP"), PATCH_FEATURE_DIM), generator=g)


def stock(model, ids, mask, new, **extra):
    with torch.no_grad():
        out = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=new, do_sample=False,
                             pad_token_id=0, use_cache=True, **extra)
    return out[:, ids.shape[1]:]


def mine(model, config, ids, mask, new, **extra):
    tg.install_tpu_generation(model, config)
    inputs = {"input_ids": ids, "attention_mask": mask, **extra}
    return tg.tpu_generate(model, inputs, max_new_tokens=new, temperature=0.0, pad_token_id=0)


def left_pad(rows, mm_rows=None):
    n = max(r.numel() for r in rows)
    ids = torch.zeros((len(rows), n), dtype=torch.long)
    mask = torch.zeros_like(ids)
    mm = torch.zeros_like(ids)
    for b, r in enumerate(rows):
        ids[b, n - r.numel():] = r
        mask[b, n - r.numel():] = 1
        if mm_rows is not None:
            mm[b, n - r.numel():] = mm_rows[b]
    return ids, mask, mm


def equal_until_eos(a, b):
    """Stock stops at the step the last row finishes and pads; compare the common length."""
    n = min(a.shape[1], b.shape[1])
    return torch.equal(a[:, :n], b[:, :n]) and a.shape[1] == b.shape[1]


@pytest.mark.parametrize("length", [5, 64, 70, 130])
def test_single_text_prompt_matches_stock(length):
    model = tiny()
    ids = text_prompt(length, 1).unsqueeze(0)
    mask = torch.ones_like(ids)
    ref = stock(model, ids, mask, 12)
    got = mine(model, cfg(), ids, mask, 12)
    assert equal_until_eos(ref, got), (ref, got)


def test_left_padded_batch_of_unequal_lengths_matches_stock():
    model = tiny()
    ids, mask, _ = left_pad([text_prompt(9, 2), text_prompt(75, 3), text_prompt(40, 4)])
    ref = stock(model, ids, mask, 10)
    got = mine(model, cfg(), ids, mask, 10)
    assert equal_until_eos(ref, got), (ref, got)


def test_prompt_is_chunked_when_longer_than_one_chunk():
    model = tiny()
    c = cfg()
    ids = text_prompt(150, 5).unsqueeze(0)
    got = mine(model, c, ids, torch.ones_like(ids), 4)
    assert c.stats["prefill_chunks"] == 3          # 150 -> padded to 192 = 3 chunks of 64
    assert equal_until_eos(stock(model, ids, torch.ones_like(ids), 4), got)


def test_image_prompt_with_empty_slot_matches_stock():
    model = tiny()
    ids, mm = image_prompt(11, 6, 6)
    grid = torch.tensor([grid_thw("CROP")])
    pix = pixels(7)
    ref = stock(model, ids.unsqueeze(0), torch.ones_like(ids).unsqueeze(0), 10, pixel_values=pix,
                image_grid_thw=grid, mm_token_type_ids=mm.unsqueeze(0))
    got = mine(model, cfg(), ids.unsqueeze(0), torch.ones_like(ids).unsqueeze(0), 10, pixel_values=pix,
               image_grid_thw=grid, mm_token_type_ids=mm.unsqueeze(0))
    assert equal_until_eos(ref, got), (ref, got)


def test_batch_with_image_and_text_only_rows_matches_stock():
    model = tiny()
    ids_a, mm_a = image_prompt(8, 5, 8)
    text = text_prompt(30, 9)
    ids, mask, mm = left_pad([ids_a, text], [mm_a, torch.zeros_like(text)])
    grid = torch.tensor([grid_thw("CROP")])
    pix = pixels(10)
    kw = dict(pixel_values=pix, image_grid_thw=grid, mm_token_type_ids=mm)
    ref = stock(model, ids, mask, 8, **kw)
    got = mine(model, cfg(), ids, mask, 8, **kw)
    assert equal_until_eos(ref, got), (ref, got)


def test_identical_rows_share_one_vision_pass_and_match_stock():
    model = tiny()
    ids1, mm1 = image_prompt(6, 4, 11)
    ids, mask, mm = left_pad([ids1] * 3, [mm1] * 3)
    grid = torch.tensor([grid_thw("CROP")] * 3)
    pix = torch.cat([pixels(12)] * 3, 0)
    kw = dict(pixel_values=pix, image_grid_thw=grid, mm_token_type_ids=mm)
    c = cfg(batch_multiple=4)
    calls = []
    orig = model.model.get_image_features

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)
    model.model.get_image_features = counting
    got = mine(model, c, ids, mask, 6, **kw)
    del model.model.get_image_features              # stock generate inspects the method; restore the original
    assert len(calls) == 1                           # one vision pass for 3 identical rows (+1 padding row is a copy)
    assert got.shape[0] == 3
    assert equal_until_eos(stock(model, ids, mask, 6, **kw), got)


def test_eos_stops_and_pads_like_stock():
    model = tiny()
    ids = text_prompt(20, 13).unsqueeze(0)
    mask = torch.ones_like(ids)
    free = stock(model, ids, mask, 6)
    # make the third generated token an EOS: rows then stop early and stay padded
    model.generation_config.eos_token_id = int(free[0, 2])
    ref = stock(model, ids, mask, 12)
    got = mine(model, cfg(), ids, mask, 12)
    assert got.shape[1] == 3 and equal_until_eos(ref, got), (ref, got)


def test_batch_padding_rows_never_change_the_result():
    model = tiny()
    ids, mask, _ = left_pad([text_prompt(17, 14), text_prompt(33, 15), text_prompt(5, 16)])
    a = mine(model, cfg(batch_multiple=1), ids, mask, 8)
    model2 = model
    model2.__dict__.pop(tg.ATTR, None)
    b = mine(model2, cfg(batch_multiple=4), ids, mask, 8)
    assert equal_until_eos(a, b)


def test_too_long_raises_and_is_counted():
    model = tiny()
    c = cfg(cache_tiers=(128,))
    tg.install_tpu_generation(model, c)
    ids = text_prompt(100, 17).unsqueeze(0)
    with pytest.raises(tg.GenerationTooLong):
        tg.tpu_generate(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, max_new_tokens=64)
    assert c.stats["too_long"] == 1


def test_slot_overflow_raises():
    model = tiny()
    ids1, mm1 = image_prompt(3, 3, 18)
    n = tokens_per_image("CROP")
    ids = torch.cat([ids1, ids1])[None]
    mm = torch.cat([mm1, mm1])[None]
    grid = torch.tensor([grid_thw("CROP")] * 3)     # 3 images > budget 2
    # build a prompt with three images
    ids3 = torch.cat([ids1, ids1, ids1])[None]
    mm3 = torch.cat([mm1, mm1, mm1])[None]
    pix = torch.cat([pixels(19)] * 3, 0)
    tg.install_tpu_generation(model, cfg(cache_tiers=(512, 1024)))
    with pytest.raises(tg.SlotOverflow):
        tg.tpu_generate(model, {"input_ids": ids3, "attention_mask": torch.ones_like(ids3), "pixel_values": pix,
                                "image_grid_thw": grid, "mm_token_type_ids": mm3}, max_new_tokens=4)


def test_non_canonical_grid_is_rejected():
    model = tiny()
    ids1, mm1 = image_prompt(3, 3, 20)
    tg.install_tpu_generation(model, cfg())
    bad = torch.tensor([[1, 16, 24]])           # right token count (96), not a canonical family
    with pytest.raises(ValueError, match="canonical"):
        tg.tpu_generate(model, {"input_ids": ids1[None], "attention_mask": torch.ones_like(ids1)[None],
                                "pixel_values": torch.zeros((384, PATCH_FEATURE_DIM)), "image_grid_thw": bad,
                                "mm_token_type_ids": mm1[None]}, max_new_tokens=2)


def test_sampling_top_k_one_is_greedy_and_top_p_keeps_only_the_nucleus():
    logits = torch.tensor([[5.0, 4.0, 0.0, -3.0]]).repeat(512, 1)
    torch.manual_seed(0)
    assert (tg._sample(logits, temperature=1.0, top_k=1, top_p=None) == 0).all()
    # probs ~ [0.73, 0.27, ...]; top_p 0.5 keeps only the largest token
    assert (tg._sample(logits, temperature=1.0, top_k=None, top_p=0.5) == 0).all()
    seen = set(tg._sample(logits, temperature=1.0, top_k=None, top_p=0.9).tolist())
    assert seen <= {0, 1} and seen == {0, 1}
    assert (tg._sample(logits, temperature=0.0, top_k=None, top_p=None) == 0).all()


def test_sampling_follows_the_distribution():
    torch.manual_seed(1)
    logits = torch.log(torch.tensor([[0.7, 0.2, 0.1]])).repeat(20000, 1)
    draws = tg._sample(logits, temperature=1.0, top_k=None, top_p=None)
    freq = torch.bincount(draws, minlength=3).float() / draws.numel()
    assert torch.allclose(freq, torch.tensor([0.7, 0.2, 0.1]), atol=0.02)


def test_config_validation():
    with pytest.raises(ValueError):
        tg.TPUGenerationConfig(chunk_tokens=100)
    with pytest.raises(ValueError):
        tg.TPUGenerationConfig(chunk_tokens=64, cache_tiers=(100,))
    with pytest.raises(ValueError):
        tg.install_tpu_generation(torch.nn.Linear(2, 2), tg.TPUGenerationConfig())


def test_generate_local_drop_in_shapes():
    model = tiny()
    tg.install_tpu_generation(model, cfg())
    ids = text_prompt(12, 21).unsqueeze(0)
    out = tg.tpu_generate_local(model, input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=5,
                                do_sample=False, pad_token_id=0, return_dict_in_generate=True)
    assert out.sequences.shape[0] == 1 and torch.equal(out.sequences[:, :12], ids) and out.past_key_values is None
    with pytest.raises(ValueError):
        tg.tpu_generate_local(model, input_ids=ids, past_key_values=object(), max_new_tokens=2)


def test_repetition_penalty_matches_stock_generate():
    model = tiny()
    ids = text_prompt(40, 61).unsqueeze(0)
    mask = torch.ones_like(ids)
    with torch.no_grad():
        ref = model.generate(input_ids=ids, attention_mask=mask, max_new_tokens=14, do_sample=False, pad_token_id=0,
                             repetition_penalty=1.3)[:, ids.shape[1]:]
    tg.install_tpu_generation(model, cfg())
    got = tg.tpu_generate(model, {"input_ids": ids, "attention_mask": mask}, max_new_tokens=14, temperature=0.0,
                          pad_token_id=0, repetition_penalty=1.3)
    assert equal_until_eos(ref, got), (ref, got)


def test_streamer_is_accepted_and_ignored():
    model = tiny()
    tg.install_tpu_generation(model, cfg(cache_tiers=(128, 256)))
    ids = text_prompt(10, 62).unsqueeze(0)
    out = tg.tpu_generate_local(model, input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=3, do_sample=False,
                                pad_token_id=0, streamer=object())
    assert torch.equal(out[:, :10], ids) and 10 < out.shape[1] <= 13
