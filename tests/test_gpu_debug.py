import pytest
import torch
from dental_agent.training.gpu_debug import check_adapter_gradients, check_adapter_update

class Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_B = torch.nn.Parameter(torch.ones(2, 3))


def test_debug_checks_real_optimizer_update_and_rejects_nonfinite_gradients():
    model = Adapter()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.lora_B.square().sum().backward()
    probe = check_adapter_gradients(model)
    optimizer.step()
    check_adapter_update(probe)
    model.lora_B.grad[0, 0] = float('nan')
    with pytest.raises(RuntimeError, match='nonfinite gradient'):
        check_adapter_gradients(model)


def test_debug_rejects_missing_gradients_and_missing_update():
    model = Adapter()
    with pytest.raises(RuntimeError, match='no nonzero LoRA gradient'):
        check_adapter_gradients(model)
    model.lora_B.sum().backward()
    probe = check_adapter_gradients(model)
    with pytest.raises(RuntimeError, match='did not update'):
        check_adapter_update(probe)
    check_adapter_update(probe, allow_unchanged=True)
