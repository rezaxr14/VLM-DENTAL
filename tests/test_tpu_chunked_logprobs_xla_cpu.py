"""TPU GRPO scorer against the real XLA compiler on 8 virtual CPU devices (compile/sharding/recompile behaviour,
NOT TPU HBM or TPU-only lowering). Run this file in its own process (every XLA-CPU file calls xr.use_spmd()):

PJRT_DEVICE=CPU CPU_NUM_DEVICES=8 RUN_XLA_CPU_TESTS=1 PYTHONPATH=.:tests python -m pytest tests/test_tpu_chunked_logprobs_xla_cpu.py -q
"""
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


@pytest.mark.parametrize("dtype_name,bias", [("float32", False), ("bfloat16", True)])
def test_sharded_log_probs_gradients_hlo_and_cache(dtype_name, bias):
    import torch
    import torch.nn.functional as F
    torch_xla, xm, xs, mesh = _setup()
    from dental_agent.training.tpu_chunked_logprobs import frozen_head_token_log_probs
    torch.manual_seed(2026)
    dtype = getattr(torch, dtype_name)
    V, T, D = 32, 19, 8
    head_cpu = torch.nn.Linear(D, V, bias=bias, dtype=dtype).requires_grad_(False)
    head = torch.nn.Linear(D, V, bias=bias, dtype=dtype).requires_grad_(False)
    head.load_state_dict(head_cpu.state_dict())
    head = head.to(torch_xla.device())
    xs.mark_sharding(head.weight, mesh, ("fsdp", None))
    h_cpu = torch.randn(8, T, D, dtype=dtype)
    h = h_cpu.to(torch_xla.device()).requires_grad_(True)
    xs.mark_sharding(h, mesh, ("fsdp", None, None))
    torch_xla.sync()
    xm.wait_device_ops()
    new_compiles = []
    for step in range(3):
        labels = torch.randint(0, V, (8, T))
        for row in range(8):
            labels[row, : row + step + 1] = -100
        labels[-1, :] = -100
        weights = torch.randn(8, T - 1)
        hc = h_cpu.clone().requires_grad_(True)
        z = head_cpu(hc)[:, :-1].float()
        want = -F.cross_entropy(z.reshape(-1, V), labels[:, 1:].clamp(min=0).reshape(-1), reduction="none").reshape(8, T - 1)
        want = want.masked_fill(labels[:, 1:] == -100, 0.0)
        (want * weights).sum().backward()
        target = labels.to(torch_xla.device())
        xs.mark_sharding(target, mesh, ("fsdp", None))
        w = weights.to(torch_xla.device())
        h.grad = None
        before = _compiles()
        lp, mask = frozen_head_token_log_probs(h, head, target, 4, mesh=mesh)
        (lp * w).sum().backward()
        if step == 0:
            hlo = torch_xla._XLAC._get_xla_tensors_hlo([lp, h.grad])
            assert "while" in hlo and f"[8,{T + 1},{V}]" not in hlo and f"[8,{T},{V}]" not in hlo   # no full-sequence logits
        torch_xla.sync()
        xm.wait_device_ops()
        new_compiles.append(_compiles() - before)
        torch.testing.assert_close(lp.cpu().detach(), want.detach(), atol=0.03 if dtype == torch.bfloat16 else 1e-5, rtol=0.01)
        grad, ref = h.grad.cpu().detach(), hc.grad
        assert (grad.float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-12) < 0.05
        assert head.weight.grad is None
    print(dtype_name, "new compilations per step:", new_compiles, flush=True)
    assert new_compiles[-1] == 0, "fixed-shape scorer recompiled when only label values changed"


def test_installed_scorer_through_peft_and_fsdp_wrapper_matches_stock_log_probs_and_gradients():
    import copy
    import torch
    torch_xla, xm, xs, mesh = _setup()
    from peft import LoraConfig, get_peft_model
    from test_chunked_loss import _batch, _tiny
    from dental_agent.training.chunked_log_probs import token_log_probs
    from dental_agent.training.sft import apply_spmd_input_sharding, wrap_spmd_model
    from dental_agent.training.tpu_chunked_logprobs import install_tpu_chunked_logprobs
    from dental_agent.training.xla_patches import install_xla_solve_triangular_shim
    install_xla_solve_triangular_shim()
    cpu = get_peft_model(_tiny(), LoraConfig(r=2, lora_alpha=4, lora_dropout=0, target_modules=["q_proj", "v_proj"]))
    cpu.train()
    model = copy.deepcopy(cpu)
    batch = _batch(b=8, t=16, masked_prefix=6)
    weights = torch.randn(8, 15)
    want_lp, want_mask = token_log_probs(cpu, batch, 0)
    (want_lp * weights).sum().backward()
    expected = {n: p.grad.detach().clone() for n, p in cpu.named_parameters() if p.grad is not None}

    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True, "preserve_rng_state": False})
    install_tpu_chunked_logprobs(model, 4, mesh=mesh)
    wrapped = wrap_spmd_model(model, mesh=mesh)
    assert getattr(wrapped, "_dental_tpu_logprob_chunk_tokens", 0) == 4       # reachable through FSDP + PEFT wrappers
    dev = {k: v.to(torch_xla.device()) for k, v in batch.items()}
    apply_spmd_input_sharding(dev, mesh, num_cores=8)
    lp, mask = token_log_probs(wrapped, dev, 0)
    (lp * weights.to(torch_xla.device())).sum().backward()
    torch_xla.sync()
    xm.wait_device_ops()
    torch.testing.assert_close(lp.cpu().detach(), want_lp.detach(), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(mask.cpu(), want_mask)
    actual = {n.replace("_orig_module.", ""): p.grad.cpu().detach() for n, p in wrapped.named_parameters() if p.grad is not None}
    assert actual.keys() == expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=1e-4, rtol=1e-3, msg=name)
    # An ordinary (generation-style) forward keeps returning logits.
    plain = wrapped(input_ids=dev["input_ids"], attention_mask=dev["attention_mask"], use_cache=False)
    assert plain.logits is not None and plain.logits.shape[-1] == 300


def test_second_install_and_sft_loss_install_are_refused():
    import torch
    torch_xla, xm, xs, mesh = _setup()
    from test_chunked_loss import _tiny
    from dental_agent.training.tpu_chunked_logprobs import install_tpu_chunked_logprobs
    from dental_agent.training.tpu_chunked_loss import install_tpu_chunked_loss
    model = _tiny().requires_grad_(False)
    install_tpu_chunked_logprobs(model, 4, mesh=mesh)
    with pytest.raises(ValueError, match="already installed"):
        install_tpu_chunked_logprobs(model, 4, mesh=mesh)
    other = _tiny().requires_grad_(False)
    install_tpu_chunked_loss(other, 4, mesh=mesh)
    with pytest.raises(ValueError, match="fresh model"):
        install_tpu_chunked_logprobs(other, 4, mesh=mesh)
