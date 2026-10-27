#!/usr/bin/env python3
"""Real TPU-only bounded-head loss/gradient, HLO and fixed-shape cache probe.

No pretrained weights/data downloads; no CPU fallback; no CUDA context.
CPU reference computations do not count as a TPU success. This is a small
output-head probe, not evidence that a complete 9B training graph fits HBM.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--chunk-tokens',type=int,default=256)
    p.add_argument('--sequence-length',type=int,default=512)
    p.add_argument('--vocab-size',type=int,default=4096)
    p.add_argument('--hidden-size',type=int,default=128)
    p.add_argument('--num-cores',type=int,default=8)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--output-dir',default='data/tpu_chunk_probe')
    a=p.parse_args()
    if min(a.chunk_tokens,a.sequence_length,a.vocab_size,a.hidden_size,a.num_cores)<=0 or a.steps<3:
        p.error('Positive dimensions and at least 3 steps are required')
    if a.sequence_length<2 or a.vocab_size<2:p.error('Sequence/vocabulary must contain at least two entries')
    os.environ.setdefault('PJRT_DEVICE','TPU')
    import torch
    import torch.nn.functional as F
    import torch_xla
    import torch_xla.runtime as xr
    xr.use_spmd()  # Must precede device initialization.
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.spmd as xs
    import torch_xla.debug.metrics as met
    device=torch_xla.device()
    if xm.xla_device_hw(device)!='TPU':raise RuntimeError('This probe requires real TPU hardware, with no CPU fallback')
    devices=xr.global_runtime_device_count()
    if devices!=a.num_cores:raise RuntimeError(f'Requested {a.num_cores} cores, runtime has {devices}')
    import numpy as np
    mesh=xs.Mesh(np.arange(devices),(devices,1),('fsdp','model'))
    xs.set_global_mesh(mesh)
    from dental_agent.training.tpu_chunked_loss import frozen_head_causal_loss,require_xla_loop
    require_xla_loop()
    out=Path(a.output_dir).expanduser().resolve();out.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(2026)
    # One global row per core. Small enough to compare against a full CPU oracle.
    head_cpu=torch.nn.Linear(a.hidden_size,a.vocab_size,bias=False,dtype=torch.bfloat16).requires_grad_(False)
    head=torch.nn.Linear(a.hidden_size,a.vocab_size,bias=False,dtype=torch.bfloat16)
    head.load_state_dict(head_cpu.state_dict());head.requires_grad_(False);head=head.to(device)
    # Reproduce a vocabulary-sharded FSDPv2 parameter. The logits are batch-sharded.
    xs.mark_sharding(head.weight,mesh,('fsdp',None))
    h_cpu=torch.randn(devices,a.sequence_length,a.hidden_size,dtype=torch.bfloat16)
    h=h_cpu.to(device).requires_grad_(True)
    xs.mark_sharding(h,mesh,('fsdp',None,None))
    xm.mark_step();xm.wait_device_ops();met.clear_all()
    records=[]
    def compile_count():
        metric=met.metric_data('CompileTime')
        return int(metric[0]) if metric else 0
    for step in range(a.steps):
        # Same shapes, different label values and supervised counts on each core.
        labels=torch.randint(0,a.vocab_size,(devices,a.sequence_length))
        for row in range(devices):labels[row,:min(a.sequence_length-1,1+step+row)]=-100
        if devices>1:labels[-1,:]=-100
        hc=h_cpu.clone().requires_grad_(True)
        expected=F.cross_entropy(head_cpu(hc)[:,:-1].float().reshape(-1,a.vocab_size),labels[:,1:].reshape(-1))
        expected.backward()
        target=labels.to(device);xs.mark_sharding(target,mesh,('fsdp',None))
        h.grad=None
        count_before=compile_count();start=time.perf_counter()
        loss=frozen_head_causal_loss(h,head,target,a.chunk_tokens,mesh=mesh)
        loss.backward()
        traced=time.perf_counter()-start
        if step==0:
            hlo=torch_xla._XLAC._get_xla_tensors_hlo([loss,h.grad])
            (out/'loss_and_backward.hlo.txt').write_text(hlo)
            # HLO shape check is before execution/fusion; the full [B,T,V] must
            # never appear when there are multiple chunks.
            padded=((a.sequence_length+a.chunk_tokens-1)//a.chunk_tokens)*a.chunk_tokens
            full=f'[{devices},{padded},{a.vocab_size}]'
            if padded>a.chunk_tokens and full in hlo:
                raise RuntimeError(f'Unexpected full-sequence vocabulary buffer {full}; inspect HLO')
            if not re.search(r'\bwhile\b',hlo):
                raise RuntimeError('Expected a compact XLA while-loop in the lowered graph')
        execution=time.perf_counter()
        xm.mark_step();xm.wait_device_ops()
        # Separate loop compilation from whole-gradient diagnostic host gathers.
        count_after_execute=compile_count()
        actual_loss=loss.cpu().detach();actual_grad=h.grad.cpu().detach()
        torch.testing.assert_close(actual_loss,expected.detach(),atol=.03,rtol=.01)
        torch.testing.assert_close(actual_grad,hc.grad,atol=3e-4,rtol=.05)
        relative_gradient_error=(actual_grad.float()-hc.grad.float()).norm()/hc.grad.float().norm().clamp_min(1e-12)
        if relative_gradient_error>.05:
            raise RuntimeError(f'Hidden-state relative gradient error {float(relative_gradient_error):.6f} exceeds 5%')
        if head.weight.grad is not None:raise RuntimeError('Frozen head acquired gradients')
        try:
            memory=xm.get_memory_info(device)
        except Exception as exc:
            memory={"unavailable":str(exc)}
        record={'step':step+1,'trace_s':traced,'execute_and_compare_s':time.perf_counter()-execution,
                'loss':float(actual_loss),'relative_gradient_error':float(relative_gradient_error),'compile_count_before_readback':count_after_execute,
                'new_compilations':count_after_execute-count_before,'memory':memory}
        records.append(record);print(json.dumps(record),flush=True)
    (out/'metrics.txt').write_text(met.metrics_report())
    report={'arguments':vars(a),'torch':torch.__version__,'torch_xla':torch_xla.__version__,
            'devices':devices,'results':records,'scope':'head-only real TPU probe; full 9B training fit unverified'}
    (out/'report.json').write_text(json.dumps(report,indent=2))
    if records[-1]['new_compilations']:
        raise RuntimeError('Fixed-shape final iteration compiled again; inspect metrics/HLO before loading 9B weights')
    print(f'[TPU LOSS PROBE PASSED] Real TPU loss/gradients and cache reuse; report: {out}',flush=True)


if __name__=='__main__':main()
