"""CPU contracts and numerical checks; these do not establish CUDA compatibility."""
import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from dental_agent.training.gpu_runtime import (
    model_loading_context, resolve_compute_dtype, sdpa_gqa_policy, _staged_materialize,
)


def test_explicit_compute_dtypes_do_not_force_fp16_on_modern_qlora():
    assert resolve_compute_dtype('qlora', 'bf16') == torch.bfloat16
    assert resolve_compute_dtype('qlora', 'fp16') == torch.float16
    assert resolve_compute_dtype('bf16') == torch.bfloat16
    assert resolve_compute_dtype('fp16') == torch.float16
    with pytest.raises(ValueError):
        resolve_compute_dtype('qlora', 'fp8')


def test_staged_loader_reads_real_safetensors_and_restores_hook_threads_on_error(tmp_path):
    from safetensors.torch import save_file
    from safetensors import safe_open
    import transformers.core_model_loading as cml
    path = tmp_path / 'weights.safetensors'
    expected = torch.arange(1024).reshape(32, 32).to(torch.bfloat16)
    save_file({'w': expected}, path)
    original, threads = cml._materialize_copy, torch.get_num_threads()
    with pytest.raises(RuntimeError, match='intentional'):
        with model_loading_context('staged', cpu_threads=1):
            assert torch.get_num_threads() == 1
            assert cml._materialize_copy is _staged_materialize
            with safe_open(path, framework='pt', device='cpu') as f:
                actual = cml._materialize_copy(f.get_slice('w'), device='cpu', dtype=torch.float16)
            torch.testing.assert_close(actual, expected.to(torch.float16), rtol=0, atol=0)
            raise RuntimeError('intentional')
    assert cml._materialize_copy is original
    assert torch.get_num_threads() == threads


def test_gqa_repetition_matches_native_gqa_output_and_gradients_and_restores_policy():
    import transformers.integrations.sdpa_attention as sdpa
    from types import SimpleNamespace
    torch.manual_seed(4)
    module = SimpleNamespace(num_key_value_groups=4, is_causal=True)
    tensors = [torch.randn(1, h, 17, 32, requires_grad=True) for h in (16, 4, 4)]
    original = sdpa.use_gqa_in_sdpa
    def run(force):
        for t in tensors:
            t.grad = None
        with sdpa_kernel(SDPBackend.MATH), sdpa_gqa_policy(force):
            out, _ = sdpa.sdpa_attention_forward(module, *tensors, None)
            out.square().sum().backward()
        return out.detach(), [t.grad.clone() for t in tensors]
    expected, eg = run('off')
    actual, ag = run('on')
    assert sdpa.use_gqa_in_sdpa is original
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    for a, e in zip(ag, eg):
        torch.testing.assert_close(a, e, rtol=1e-5, atol=2e-6)
    with pytest.raises(RuntimeError):
        with sdpa_gqa_policy('on'):
            raise RuntimeError('intentional')
    assert sdpa.use_gqa_in_sdpa is original
