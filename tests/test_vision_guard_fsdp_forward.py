"""The vision guard must leave every vision block with a real bound ``forward``.

torch_xla's FSDPv2 refuses to wrap a module whose ``forward`` has no ``__func__`` ("The module wrapped by FSDP must define a
`forward` method"), and PEFT's auto-wrap policy wraps every Qwen3_5VisionBlock. A bare closure assigned to ``blk.forward``
therefore crashed TPU SFT with --lora-target-vision projector right after the model loaded.
"""
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from dental_agent.training.sft import freeze_and_guard_vision_tower


def _fsdp_v2_accepts(module) -> bool:
    """The check in torch_xla.experimental.spmd_fully_sharded_data_parallel.SpmdFullyShardedDataParallel.__init__ (torch_xla 2.8)."""
    return (hasattr(module, "forward") and hasattr(module.forward, "__func__")
            and module.forward.__func__ != nn.Module.forward)


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 4)

    def forward(self, x):
        return self.lin(x)


class _Visual(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList([_Block(), _Block()])
        self.merger = nn.Linear(4, 4)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.merger(x)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = _Visual()

    def forward(self, x):
        return self.visual(x)


def test_guarded_vision_blocks_stay_fsdp_wrappable_and_run_without_grad():
    model = _Model()
    freeze_and_guard_vision_tower(model, train_merger=True, is_master=False)
    for blk in model.visual.blocks:
        assert _fsdp_v2_accepts(blk)
    out = model.visual.blocks[0](torch.randn(3, 4, requires_grad=True))
    assert not out.requires_grad                      # the block still runs under torch.no_grad()
    assert all(not p.requires_grad for p in model.visual.blocks.parameters())
    assert model.visual.merger.weight.requires_grad   # the projector is left to PEFT / the caller
