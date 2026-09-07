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


def test_grpo_cli_fsdp_flags():
    """Verify scripts/run_grpo.py parses --fsdp and --no-fsdp correctly."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--fsdp", action=argparse.BooleanOptionalAction, default=True)
    args_default = parser.parse_args([])
    assert args_default.fsdp is True

    args_no = parser.parse_args(["--no-fsdp"])
    assert args_no.fsdp is False
