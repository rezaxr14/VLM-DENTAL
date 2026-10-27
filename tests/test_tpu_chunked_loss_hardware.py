"""Opt-in real TPU tests. Never substitute CPU/XLA mocks for this hardware gate.

PJRT_DEVICE=TPU RUN_REAL_TPU_TESTS=1 python -m pytest tests/test_tpu_chunked_loss_hardware.py -q -s
"""
import os
import importlib.util
import pytest

pytestmark=pytest.mark.skipif(os.environ.get('RUN_REAL_TPU_TESTS')!='1' or importlib.util.find_spec('torch_xla') is None,
                            reason='requires explicit real TPU run and matching torch_xla')


def test_real_tpu_output_head_probe(tmp_path,monkeypatch):
    from scripts import probe_tpu_chunked_loss
    monkeypatch.setattr('sys.argv',['probe_tpu_chunked_loss.py','--output-dir',str(tmp_path)])
    probe_tpu_chunked_loss.main()


def test_real_tpu_qwen_peft_fsdp_loss_and_update():
    import copy
    import numpy as np
    import torch
    import torch_xla
    import torch_xla.runtime as xr
    xr.use_spmd()
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    device=torch_xla.device()
    assert xm.xla_device_hw(device)=='TPU', 'No CPU fallback is permitted'
    if not hasattr(torch,'xla'):torch._register_device_module('xla',torch_xla)
    n=xr.global_runtime_device_count()
    assert n==8,'This integration recipe targets the user\'s 8-core TPU'
    mesh=xs.Mesh(np.arange(n),(n,1),('fsdp','model'));xs.set_global_mesh(mesh)
    from dental_agent.training.xla_patches import install_xla_solve_triangular_shim
    install_xla_solve_triangular_shim()
    from peft import LoraConfig,get_peft_model
    from test_chunked_loss import _tiny,_batch
    from dental_agent.training.sft import wrap_spmd_model,apply_spmd_input_sharding
    from dental_agent.training.tpu_chunked_loss import install_tpu_chunked_loss
    plain=_tiny();plain.config.image_token_id=299;plain.config.video_token_id=298;plain.config.vision_start_token_id=297
    cpu=get_peft_model(plain,LoraConfig(r=2,lora_alpha=4,lora_dropout=0,target_modules=['q_proj','v_proj','linear_fc1','linear_fc2']))
    cpu=cpu.to(torch.bfloat16).train()
    batch=_batch(b=n,t=40)
    batch['input_ids'].remainder_(290);batch['input_ids'][:,4]=297;batch['input_ids'][:,5:9]=299
    batch['labels']=batch['input_ids'].clone();batch['labels'][:,:25]=-100
    for row in range(n):batch['labels'][row,25:25+row]=-100
    batch['mm_token_type_ids']=torch.zeros_like(batch['input_ids']);batch['mm_token_type_ids'][:,5:9]=1
    batch['image_grid_thw']=torch.tensor([[1,4,4]]*n);batch['pixel_values']=torch.randn(n*16,3*2*16*16)
    model=copy.deepcopy(cpu)
    want=cpu(**batch,use_cache=False).loss
    want.backward()
    expected={name:p.grad.detach().float() for name,p in cpu.named_parameters() if p.grad is not None}
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':True,'preserve_rng_state':False})
    # Install while CPU-resident; the real FSDPv2 wrapper transfers/shards parameters.
    install_tpu_chunked_loss(model,7,mesh=mesh)
    wrapped=wrap_spmd_model(model,mesh=mesh)
    assert wrapped is not model
    batch={k:v.to(device) for k,v in batch.items()};apply_spmd_input_sharding(batch,mesh,num_cores=n)
    got=wrapped(**batch,use_cache=False)
    assert got.logits is None and got.loss.ndim==0
    got.loss.backward();xm.mark_step();xm.wait_device_ops()
    torch.testing.assert_close(got.loss.cpu().detach(),want.detach(),atol=.05,rtol=.02)
    actual={name.replace('_orig_module.',''):p.grad.cpu().detach().float()
            for name,p in wrapped.named_parameters() if p.grad is not None}
    assert actual.keys()==expected.keys() and any('merger' in name for name in actual)
    for name in expected:
        error=(actual[name]-expected[name]).norm()/expected[name].norm().clamp_min(1e-8)
        assert error<.15,(name,float(error))
    params=[p for p in wrapped.parameters() if p.requires_grad]
    probe=next(p for p in params if p.grad is not None and torch.count_nonzero(p.grad.cpu().detach()))
    before=probe.cpu().detach().clone()
    optimizer=torch.optim.AdamW(params,lr=1e-3)
    optimizer.step();xm.mark_step();xm.wait_device_ops()
    after=probe.cpu().detach()
    assert torch.isfinite(after).all() and not torch.equal(before,after)
