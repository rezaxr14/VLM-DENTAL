"""Speculative decoding against the real XLA compiler on 8 virtual CPU devices: equal to non-speculative decoding through PEFT + FSDPv2
and no recompilation once XLA has settled, whatever the prompt length, draft length or number of accepted drafts. Not TPU performance.
Run in its own process:

PJRT_DEVICE=CPU CPU_NUM_DEVICES=8 RUN_XLA_CPU_TESTS=1 PYTHONPATH=.:tests python -m pytest tests/test_tpu_speculative_xla_cpu.py -q
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
    assert xr.global_runtime_device_count() == 8
    mesh = xs.Mesh(np.arange(8), (8, 1), ("fsdp", "model"))
    xs.set_global_mesh(mesh)
    if not hasattr(torch, "xla"):
        torch._register_device_module("xla", torch_xla)
    return torch_xla, xm, xs, mesh


def _compiles():
    """Distinct XLA programs compiled so far (``UncachedCompile``); ``CompileTime`` also counts per-execution handle creation."""
    import torch_xla.debug.metrics as met
    return int(met.counter_value("UncachedCompile") or 0)


def test_speculation_matches_baseline_through_fsdp_and_stops_recompiling():
    import torch
    torch_xla, xm, xs, mesh = _setup()
    import test_tpu_generation as T
    from dental_agent.model import tpu_generation as tg
    from dental_agent.model.tpu_speculative import parse_condition
    from dental_agent.training.sft import wrap_spmd_model
    from dental_agent.training.xla_patches import install_xla_solve_triangular_shim
    install_xla_solve_triangular_shim()

    cpu = T.tiny().eval()
    model = copy.deepcopy(cpu)
    cfg = T.cfg(batch_multiple=8, mesh=mesh, cache_tiers=(512,))
    tg.install_tpu_generation(model, cfg)
    model = model.to(torch_xla.device())
    wrapped = wrap_spmd_model(model, mesh=mesh)

    def run(name, ids, new, **kw):
        cfg.speculative = parse_condition(name)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        out = tg.tpu_generate(wrapped, inputs, max_new_tokens=new, temperature=0.0, pad_token_id=0, **kw)
        xm.wait_device_ops()
        return out

    ids = T.text_prompt(100, 5).unsqueeze(0)
    want = T.stock(cpu, ids, torch.ones_like(ids), 80)
    base = run("baseline", ids, 80)
    assert T.equal_until_eos(want, base)
    spec = run("ngram_3", ids, 80)
    assert torch.equal(base, spec), (base, spec)
    assert cfg.stats["last_call"]["accepted"] > 0

    # XLA settles on its program set after the first few rounds; from then on speculation compiles nothing new, whatever the prompt
    # length, draft length or number of accepted drafts. (A baseline run on a prompt with a new chunk count can still add a program
    # of the ordinary generator -- that is 0062 behaviour -- so baselines are run first, outside the measurement.)
    run("ngram_3", T.text_prompt(70, 8).unsqueeze(0), 60)
    cases = ((9, 130, 70), (10, 40, 50), (5, 100, 80))
    bases = {c: run("baseline", T.text_prompt(c[1], c[0]).unsqueeze(0), c[2]) for c in cases}
    before = _compiles()
    for c in cases:
        s = run("ngram_3", T.text_prompt(c[1], c[0]).unsqueeze(0), c[2])
        assert torch.equal(bases[c], s), (c, bases[c], s)
    extra = _compiles() - before
    print("new speculative compilations across prompts, draft lengths and acceptance patterns:", extra, flush=True)
    assert extra == 0, "speculation triggered a recompilation"
    assert cfg.stats["spec_rounds"] > 0 and cfg.stats["spec_plain_steps"] > 0
