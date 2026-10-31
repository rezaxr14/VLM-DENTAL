"""Run-report helpers against the real compiler on XLA CPU devices (not TPU HBM).

PJRT_DEVICE=CPU CPU_NUM_DEVICES=8 RUN_XLA_CPU_TESTS=1 PYTHONPATH=.:tests python -m pytest tests/test_tpu_diagnostics_xla_cpu.py -q

Run each XLA-CPU test file in its own process: they all call xr.use_spmd() and two in one process segfault.
"""
import importlib.util
import json
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("RUN_XLA_CPU_TESTS") != "1" or importlib.util.find_spec("torch_xla") is None,
                                reason="requires explicit XLA CPU test environment")


def _report(tmp_path, every=1):
    from dental_agent.training.tpu_diagnostics import TPUReport
    return TPUReport(tmp_path / "rep", role="sft", heartbeat_seconds=0, stack_seconds=0, tpu_info_seconds=0,
                     step_snapshot_every=every)


def _run(report, shape_for_step, steps):
    import torch
    import torch_xla
    import torch_xla.core.xla_model as xm
    device = torch_xla.device()
    weight = torch.ones(16, 16, device=device)
    for step in range(1, steps + 1):
        x = torch.ones(shape_for_step(step), 16, device=device)
        (x @ weight).sum().item()                      # forces compile + execute
        xm.mark_step()
        report.step(step, loss=float(step))


def test_fixed_shapes_compile_once_and_raise_no_alert(tmp_path):
    from dental_agent.training import tpu_diagnostics as d
    report = _report(tmp_path)
    report.begin("fixed_loop")
    _run(report, lambda s: 8, 6)
    report.end("fixed_loop")
    snaps = [json.loads(l) for l in (tmp_path / "rep" / "snapshots.jsonl").read_text().splitlines()]
    counts = [s["compile"]["count"] for s in snaps]
    assert counts == sorted(counts) and counts[-1] == counts[2], counts          # no growth after step 3
    assert not (tmp_path / "rep" / "alerts.txt").exists()
    end = [json.loads(l) for l in (tmp_path / "rep" / "events.jsonl").read_text().splitlines()][-1]
    assert end["event"] == "end" and end["compiles_in_stage"] >= 1 and "device_memory" in end
    assert isinstance(d.fallback_counters(), dict)


def test_changing_shapes_after_warmup_raise_the_alert(tmp_path):
    report = _report(tmp_path)
    _run(report, lambda s: 8 if s <= 3 else 8 + s, 7)                              # a new shape every step from step 4
    alerts = (tmp_path / "rep" / "alerts.txt").read_text()
    assert "after step 3" in alerts
    snaps = [json.loads(l) for l in (tmp_path / "rep" / "snapshots.jsonl").read_text().splitlines()]
    assert any(s.get("recompiled_after_warmup") for s in snaps)


def test_final_report_contains_xla_metrics(tmp_path):
    report = _report(tmp_path)
    _run(report, lambda s: 8, 2)
    report.finish()
    assert "CompileTime" in (tmp_path / "rep" / "xla_metrics_final.txt").read_text()
    final = json.loads((tmp_path / "rep" / "final.json").read_text())
    assert final["status"] == "ok" and final["compile"]["count"] >= 1
