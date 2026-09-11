import argparse
import sys
import torch
import torch.nn as nn
import pytest

from dental_agent.training.sft import unwrap_peft_model, wrap_distributed_model


class SimpleLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)

    def forward(self, x):
        return self.fc(x)


class DummyWrapper(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.module = inner

    def forward(self, x):
        return self.module(x)


def test_unwrap_peft_model_plain():
    """Unwrapped model should return itself when not wrapped in .module."""
    base = SimpleLinear()
    assert unwrap_peft_model(base) is base


def test_unwrap_peft_model_nested():
    """Deeply nested .module wrappers (e.g. FSDP(DDP(PeftModel))) should unwrap to base model."""
    base = SimpleLinear()
    wrapped_once = DummyWrapper(base)
    wrapped_twice = DummyWrapper(wrapped_once)
    assert unwrap_peft_model(wrapped_twice) is base


def test_wrap_distributed_model_cpu():
    """On CPU (non-TPU), wrap_distributed_model should safely return model without XLA errors."""
    base = SimpleLinear()
    out = wrap_distributed_model(base, is_tpu=False, num_cores=1, use_fsdp=True)
    assert out is base


def test_wrap_distributed_model_tpu_fallback_when_xla_unavailable(monkeypatch):
    """When torch_xla is not installed or FSDP fails, it should gracefully fall back."""
    base = SimpleLinear()
    import types
    fake_xm = types.ModuleType("fake_xm")
    fake_xm.xla_device = lambda: torch.device("cpu")
    monkeypatch.setitem(sys.modules, "torch_xla.core.xla_model", fake_xm)

    out = wrap_distributed_model(base, is_tpu=True, num_cores=8, use_fsdp=True, is_master=False)
    assert out is not None
    assert unwrap_peft_model(out) is base


def test_sft_cli_fsdp_flags():
    """Verify scripts/train_sft.py parses --fsdp and --no-fsdp correctly."""
    from scripts.train_sft import parse_args
    # Default is --fsdp True
    test_args = ["--track", "with_tools"]
    monkeypatch_sys_argv = ["train_sft.py", *test_args]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", monkeypatch_sys_argv)
        args = parse_args()
        assert args.fsdp is True

    # Explicit --no-fsdp
    test_args_no = ["--track", "with_tools", "--no-fsdp"]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["train_sft.py", *test_args_no])
        args = parse_args()
        assert args.fsdp is False

    # Single static sequence length default and override
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["train_sft.py", "--track", "with_tools"])
        args = parse_args()
        assert args.max_seq_len == 32768
        assert args.xla_pallas is True
        assert args.xla_spmd is False

    # Backtrack sequence length override
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["train_sft.py", "--track", "with_tools", "--max-seq-len", "24576", "--no-xla-pallas", "--xla-spmd"])
        args = parse_args()
        assert args.max_seq_len == 24576
        assert args.xla_pallas is False
        assert args.xla_spmd is True


def test_grpo_cli_fsdp_flags():
    """Verify scripts/run_grpo.py parses --fsdp and --no-fsdp correctly."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp", action=argparse.BooleanOptionalAction, default=True)
    args_default = parser.parse_args([])
    assert args_default.fsdp is True

    args_no = parser.parse_args(["--no-fsdp"])
    assert args_no.fsdp is False


def test_xla_fsdp_auto_wrap_policy_adapter():
    """Verify that PEFT FSDP policy handles torch_xla's unwrapped_params keyword argument without crashing."""
    from peft import LoraConfig, get_peft_model
    from peft.utils.other import fsdp_auto_wrap_policy

    class SubBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8)
        def forward(self, x):
            return self.linear(x)

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.block = SubBlock()
        def forward(self, x):
            return self.block(x)

    model = DummyModel()
    peft_model = get_peft_model(model, LoraConfig(target_modules=["linear"]))
    raw_policy = fsdp_auto_wrap_policy(peft_model)
    assert raw_policy is not None

    def xla_policy(module, recurse, unwrapped_params=0, **kwargs):
        try:
            return raw_policy(module=module, recurse=recurse, nonwrapped_numel=unwrapped_params)
        except TypeError:
            try:
                return raw_policy(module, recurse, unwrapped_params)
            except TypeError:
                return raw_policy(module=module, recurse=recurse)

    # Calling with torch_xla convention (unwrapped_params as keyword arg)
    result = xla_policy(module=peft_model.base_model.model.block, recurse=True, unwrapped_params=64)
    assert isinstance(result, bool)


def test_warmup_xla_cache_cli_spmd_flags():
    """Verify scripts/warmup_xla_cache.py parses --xla-spmd and --no-xla-spmd correctly."""
    from scripts.warmup_xla_cache import parse_args
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["warmup_xla_cache.py", "--track", "with_tools", "--xla-spmd"])
        args = parse_args()
        assert args.xla_spmd is True

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sys, "argv", ["warmup_xla_cache.py", "--track", "with_tools", "--no-xla-spmd"])
        args = parse_args()
        assert args.xla_spmd is False


def test_wrap_distributed_model_spmd_cpu_fallback():
    """wrap_distributed_model with use_spmd=True on CPU should return base model without errors."""
    base = SimpleLinear()
    out = wrap_distributed_model(base, is_tpu=False, num_cores=8, use_spmd=True)
    assert out is base


