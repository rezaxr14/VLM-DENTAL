"""The TPU evaluation bootstrap against the real XLA compiler on 8 virtual CPU devices: a merged (adapter-free) model on the host is
sharded by ``finalize_tpu_inference_model`` and generates exactly what the stock CPU ``generate`` does. NOT TPU HBM or throughput.
Run in its own process:

PJRT_DEVICE=CPU CPU_NUM_DEVICES=8 RUN_XLA_CPU_TESTS=1 PYTHONPATH=.:tests python -m pytest tests/test_tpu_evaluation_xla_cpu.py -q
"""
import copy
import importlib.util
import os
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_XLA_CPU_TESTS") != "1" or importlib.util.find_spec("torch_xla") is None,
                                reason="requires explicit XLA CPU test environment")


def test_finalize_shards_a_host_model_and_generates_like_the_cpu_reference_then_refuses_cleanly():
    import torch
    import torch_xla
    import torch_xla.runtime as xr
    assert os.environ.get("PJRT_DEVICE") == "CPU"
    from dental_agent.model.tpu_inference import enable_spmd, finalize_tpu_inference_model, generation_report, note_refusal
    enable_spmd()                                                       # before any tensor touches the device
    if not hasattr(torch, "xla"):
        torch._register_device_module("xla", torch_xla)
    assert xr.global_runtime_device_count() == 8
    import test_tpu_generation as T
    from dental_agent.model import tpu_generation as tg
    from dental_agent.training.xla_patches import install_xla_solve_triangular_shim
    install_xla_solve_triangular_shim()

    host = T.tiny()                                                     # merged model: no adapter, still on the host
    reference = copy.deepcopy(host).eval()
    options = SimpleNamespace(tpu_generation_chunk_tokens=64, tpu_generation_tiers=(256, 512), tpu_vision_slots=(1, 0, 0),
                              tpu_generation_warmup=False)
    model = finalize_tpu_inference_model(host, processor=None, options=options, max_new_tokens=16, repetition_penalty=None)
    assert all(not p.requires_grad for p in model.parameters())
    config = tg.find_attr(model, tg.ATTR)
    assert config is not None and config.batch_multiple == 8 and config.slot_budget["FULL"] == 1

    ids = T.text_prompt(100, 5).unsqueeze(0)
    got = tg.tpu_generate(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, max_new_tokens=12,
                         temperature=0.0, pad_token_id=0)
    want = T.stock(reference, ids, torch.ones_like(ids), 12)
    assert T.equal_until_eos(want, got), (want, got)

    # a prompt that cannot fit is refused (never truncated); the evaluator records and counts it
    too_long = T.text_prompt(600, 6).unsqueeze(0)
    with pytest.raises(tg.GenerationTooLong) as caught:
        tg.tpu_generate(model, {"input_ids": too_long, "attention_mask": torch.ones_like(too_long)}, max_new_tokens=12,
                        temperature=0.0, pad_token_id=0)
    note_refusal(model, 99, caught.value)
    report = generation_report(model)
    assert report["stats"]["rollouts_dropped"] == 1 and report["stats"]["refused_images"][0]["image_id"] == 99
    assert report["cache_tiers"] == [256, 512]


def test_finalize_rejects_a_budget_no_tier_can_hold_before_touching_the_model():
    from dental_agent.model.tpu_inference import finalize_tpu_inference_model
    options = SimpleNamespace(tpu_generation_chunk_tokens=64, tpu_generation_tiers=(256,), tpu_vision_slots=(1, 0, 0),
                              tpu_generation_warmup=False)
    with pytest.raises(ValueError, match="never lowered|does not fit"):
        finalize_tpu_inference_model(object(), None, options, max_new_tokens=16384)
