"""Actual XLA CPU compilation/sharding checks, explicitly not TPU HBM validation.

Use a matching torch/torch_xla installation in an isolated environment:
PJRT_DEVICE=CPU CPU_NUM_DEVICES=8 RUN_XLA_CPU_TESTS=1 python -m pytest tests/test_tpu_chunked_loss_xla_cpu.py -q -s
"""
import importlib.util
import os
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('RUN_XLA_CPU_TESTS')!='1' or importlib.util.find_spec('torch_xla') is None,
                            reason='requires explicit XLA CPU test environment')


@pytest.mark.parametrize('dtype_name,bias',[('float32',False),('bfloat16',True)])
def test_real_xla_compilation_sharded_gradients_and_cache(dtype_name,bias):
    import torch
    import torch.nn.functional as F
    import torch_xla
    import torch_xla.runtime as xr
    assert os.environ.get('PJRT_DEVICE')=='CPU'
    xr.use_spmd()
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.debug.metrics as met
    import numpy as np
    from dental_agent.training.tpu_chunked_loss import frozen_head_causal_loss
    device=torch_xla.device()
    assert xm.xla_device_hw(device)=='CPU'
    assert xr.global_runtime_device_count()==8
    mesh=xs.Mesh(np.arange(8),(8,1),('fsdp','model'));xs.set_global_mesh(mesh)
    torch.manual_seed(2026)
    dtype=getattr(torch,dtype_name)
    head_cpu=torch.nn.Linear(8,32,bias=bias,dtype=dtype).requires_grad_(False)
    head=torch.nn.Linear(8,32,bias=bias,dtype=dtype).requires_grad_(False)
    head.load_state_dict(head_cpu.state_dict());head=head.to(device)
    xs.mark_sharding(head.weight,mesh,('fsdp',None))
    h_cpu=torch.randn(8,19,8,dtype=dtype)
    h=h_cpu.to(device).requires_grad_(True);xs.mark_sharding(h,mesh,('fsdp',None,None))
    torch_xla.sync();xm.wait_device_ops();met.clear_all()
    counts=[]
    for step in range(3):
        labels=torch.randint(0,32,(8,19))
        for row in range(8):labels[row,:row+step+1]=-100
        labels[-1,:]=-100
        hc=h_cpu.clone().requires_grad_(True)
        want=F.cross_entropy(head_cpu(hc)[:,:-1].float().reshape(-1,32),labels[:,1:].reshape(-1))
        want.backward()
        t=labels.to(device);xs.mark_sharding(t,mesh,('fsdp',None))
        h.grad=None
        metric=met.metric_data('CompileTime')
        before=int(metric[0]) if metric else 0
        got=frozen_head_causal_loss(h,head,t,4,mesh=mesh)
        got.backward()
        if step==0:
            hlo=torch_xla._XLAC._get_xla_tensors_hlo([got,h.grad])
            assert 'while' in hlo and '[8,20,32]' not in hlo
        torch_xla.sync();xm.wait_device_ops()
        # Whole-gradient host gathers may compile separate diagnostic graphs.
        # Count the actual loss/backward execution before doing those readbacks.
        after=met.metric_data('CompileTime')[0]
        counts.append(after-before)
        actual,grad=got.cpu().detach(),h.grad.cpu().detach()
        torch.testing.assert_close(actual,want.detach(),atol=.02,rtol=.01)
        torch.testing.assert_close(grad,hc.grad,atol=3e-4,rtol=.05)
        assert (grad.float()-hc.grad.float()).norm()/hc.grad.float().norm().clamp_min(1e-12)<.05
        assert head.weight.grad is None
    print(dtype_name,'new loss/backward compilations per step:',counts,flush=True)
    assert counts[-1]==0, 'Fixed-shape, changed-label loss/backward iteration recompiled'


def test_real_xla_cpu_qwen_peft_fsdp_wrapped_forward_and_backward():
    import copy
    import numpy as np
    import torch
    import torch_xla
    import torch_xla.runtime as xr
    assert os.environ.get('PJRT_DEVICE')=='CPU'
    xr.use_spmd()
    import torch_xla.distributed.spmd as xs
    import torch_xla.core.xla_model as xm
    assert xm.xla_device_hw(torch_xla.device())=='CPU'
    if not hasattr(torch,'xla'):torch._register_device_module('xla',torch_xla)
    from test_chunked_loss import _tiny,_batch
    from peft import LoraConfig,get_peft_model
    from dental_agent.training.sft import wrap_spmd_model,apply_spmd_input_sharding
    from dental_agent.training.tpu_chunked_loss import install_tpu_chunked_loss
    from dental_agent.training.xla_patches import install_xla_solve_triangular_shim
    install_xla_solve_triangular_shim()
    mesh=xs.Mesh(np.arange(8),(8,1),('fsdp','model'));xs.set_global_mesh(mesh)
    cpu=get_peft_model(_tiny(),LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules=['q_proj','v_proj']))
    cpu.train();model=copy.deepcopy(cpu)
    batch=_batch(b=8,t=16,masked_prefix=6)
    want=cpu(**batch,use_cache=False).loss;want.backward()
    expected={name:p.grad.detach().clone() for name,p in cpu.named_parameters() if p.grad is not None}
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True,'preserve_rng_state':False})
    install_tpu_chunked_loss(model,4,mesh=mesh)
    wrapped=wrap_spmd_model(model,mesh=mesh)
    assert wrapped is not model
    batch={k:v.to(torch_xla.device()) for k,v in batch.items()};apply_spmd_input_sharding(batch,mesh,num_cores=8)
    got=wrapped(**batch,use_cache=False)
    assert got.logits is None and got.loss.ndim==0
    got.loss.backward();torch_xla.sync();xm.wait_device_ops()
    torch.testing.assert_close(got.loss.cpu().detach(),want.detach(),atol=1e-4,rtol=1e-4)
    actual={name.replace('_orig_module.',''):p.grad.cpu().detach()
            for name,p in wrapped.named_parameters() if p.grad is not None}
    assert actual.keys()==expected.keys()
    for name in expected:
        torch.testing.assert_close(actual[name],expected[name],atol=1e-4,rtol=1e-3,msg=name)
