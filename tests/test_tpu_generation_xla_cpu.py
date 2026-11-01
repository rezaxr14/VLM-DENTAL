"""Static-shape TPU generation against the real XLA compiler on 8 virtual CPU devices (compile/sharding/recompile
behaviour, NOT TPU HBM, throughput or TPU-only lowering). Run in its own process (every XLA-CPU file calls xr.use_spmd()):

PJRT_DEVICE=CPU CPU_NUM_DEVICES=8 RUN_XLA_CPU_TESTS=1 PYTHONPATH=.:tests python -m pytest tests/test_tpu_generation_xla_cpu.py -q
"""
import copy
import importlib.util
import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_XLA_CPU_TESTS") != "1" or importlib.util.find_spec("torch_xla") is None,
                                reason="requires explicit XLA CPU test environment")


def _setup():
    import numpy as np
    import torch
    import torch_xla
    import torch_xla.runtime as xr
    assert os.environ.get("PJRT_DEVICE") == "CPU"
    xr.use_spmd()
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    assert xr.global_runtime_device_count() == 8 and xm.xla_device_hw(torch_xla.device()) == "CPU"
    mesh = xs.Mesh(np.arange(8), (8, 1), ("fsdp", "model"))
    xs.set_global_mesh(mesh)
    if not hasattr(torch, "xla"):
        torch._register_device_module("xla", torch_xla)
    return torch_xla, xm, xs, mesh


def _compiles():
    import torch_xla.debug.metrics as met
    metric = met.metric_data("CompileTime")
    return int(metric[0]) if metric else 0


def test_generation_matches_cpu_reference_through_peft_and_fsdp_and_stops_recompiling():
    import torch
    torch_xla, xm, xs, mesh = _setup()
    from peft import LoraConfig, get_peft_model
    import test_tpu_generation as T
    from dental_agent.model import tpu_generation as tg
    from dental_agent.training.sft import wrap_spmd_model
    from dental_agent.training.xla_patches import install_xla_solve_triangular_shim
    install_xla_solve_triangular_shim()

    cpu = get_peft_model(T.tiny(), LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=["q_proj", "v_proj"])).eval()
    model = copy.deepcopy(cpu)

    def reference(ids, new, **kw):
        mask = torch.ones_like(ids)
        return T.stock(cpu.base_model.model, ids, mask, new, **kw)

    cfg = T.cfg(batch_multiple=8, mesh=mesh, cache_tiers=(512,))
    tg.install_tpu_generation(model, cfg)
    model = model.to(torch_xla.device())
    wrapped = wrap_spmd_model(model, mesh=mesh)
    assert tg.find_attr(wrapped, tg.ATTR) is cfg                      # reachable through the FSDP + PEFT wrappers

    def run(ids, new, **kw):
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids), **kw}
        return tg.tpu_generate(wrapped, inputs, max_new_tokens=new, temperature=0.0, pad_token_id=0)

    # warm-up: XLA settles on its program set after the 3rd prefill chunk and the 3rd decode step (the first programs see
    # freshly created cache buffers), so warm with 4 chunks and 6 tokens. From there nothing depends on length.
    ids1 = T.text_prompt(200, 31).unsqueeze(0)
    got1 = run(ids1, 6)
    assert T.equal_until_eos(reference(ids1, 6), got1), (reference(ids1, 6), got1)
    xm.wait_device_ops()
    before = _compiles()
    # different prompt length (3 chunks) and different max_new_tokens, same tier -> no new graph
    ids2 = T.text_prompt(130, 32).unsqueeze(0)
    got2 = run(ids2, 9)
    xm.wait_device_ops()
    assert T.equal_until_eos(reference(ids2, 9), got2), (reference(ids2, 9), got2)
    new_text = _compiles() - before
    print("new compilations for a new length / new max_new (text):", new_text, flush=True)
    assert new_text == 0, "prompt length or max_new_tokens triggered a recompilation"

    # images: warm once, then different pixels and a different amount of text must reuse every graph
    ids_a, mm_a = T.image_prompt(150, 6, 41)
    grid = torch.tensor([T.grid_thw("CROP")])
    kw_a = dict(pixel_values=T.pixels(42), image_grid_thw=grid, mm_token_type_ids=mm_a[None])
    got_a = run(ids_a[None], 6, **kw_a)
    assert T.equal_until_eos(reference(ids_a[None], 6, **kw_a), got_a)
    xm.wait_device_ops()
    before = _compiles()
    ids_b, mm_b = T.image_prompt(30, 9, 43)
    kw_b = dict(pixel_values=T.pixels(44), image_grid_thw=grid, mm_token_type_ids=mm_b[None])
    got_b = run(ids_b[None], 8, **kw_b)
    xm.wait_device_ops()
    assert T.equal_until_eos(reference(ids_b[None], 8, **kw_b), got_b)
    new_img = _compiles() - before
    print("new compilations for new pixels / text length (image):", new_img, flush=True)
    assert new_img == 0, "image content or text length triggered a recompilation"

    # sampling (temperature / top-k / top-p) runs on the device and does not recompile for a new seed or prompt
    torch.manual_seed(0)
    s1 = tg.tpu_generate(wrapped, {"input_ids": ids1, "attention_mask": torch.ones_like(ids1)}, max_new_tokens=6,
                         temperature=0.8, top_k=20, top_p=0.9, pad_token_id=0)
    xm.wait_device_ops()
    before = _compiles()
    s2 = tg.tpu_generate(wrapped, {"input_ids": ids2, "attention_mask": torch.ones_like(ids2)}, max_new_tokens=7,
                         temperature=0.8, top_k=20, top_p=0.9, pad_token_id=0)
    xm.wait_device_ops()
    assert _compiles() - before == 0
    assert s1.shape[0] == 1 and s2.shape[0] == 1

    assert set(cfg.stats["tiers"]) == {512}


def test_k_identical_rows_with_batch_padding_on_the_mesh():
    import torch
    torch_xla, xm, xs, mesh = _setup()
    import test_tpu_generation as T
    from dental_agent.model import tpu_generation as tg
    from dental_agent.training.sft import wrap_spmd_model
    from dental_agent.training.xla_patches import install_xla_solve_triangular_shim
    install_xla_solve_triangular_shim()
    cpu = T.tiny()
    model = copy.deepcopy(cpu)
    tg.install_tpu_generation(model, T.cfg(batch_multiple=8, mesh=mesh))
    wrapped = wrap_spmd_model(model.to(torch_xla.device()), mesh=mesh)
    row = T.text_prompt(40, 51)
    ids = row.unsqueeze(0).repeat(5, 1)                                # K = 5 copies -> padded to 8 rows, 5 returned
    got = tg.tpu_generate(wrapped, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}, max_new_tokens=6,
                          temperature=0.0, pad_token_id=0)
    assert got.shape[0] == 5
    ref = T.stock(cpu, ids[:1], torch.ones_like(ids[:1]), 6)
    for r in range(5):
        assert T.equal_until_eos(ref, got[r:r + 1])
