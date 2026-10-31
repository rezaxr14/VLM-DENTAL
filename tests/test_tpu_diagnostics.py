"""CPU tests for the opt-in TPU run report. No torch_xla needed (XLA behaviour: test_tpu_diagnostics_xla_cpu.py)."""
import argparse
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _args(tmp_path, **over):
    parser = argparse.ArgumentParser()
    from dental_agent.training.tpu_diagnostics import add_report_arguments
    add_report_arguments(parser)
    argv = ["--tpu-report-dir", str(tmp_path / "rep")]
    for key, value in over.items():
        argv += [f"--{key.replace('_', '-')}", str(value)]
    return parser.parse_args(argv)


def _lines(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def test_off_by_default_creates_nothing_and_imports_no_xla(tmp_path):
    from dental_agent.training.tpu_diagnostics import TPUReport, add_report_arguments
    parser = argparse.ArgumentParser()
    add_report_arguments(parser)
    args = parser.parse_args([])
    assert args.tpu_report_dir is None and TPUReport.from_args(args, "sft") is None
    code = (f"import sys; sys.path.insert(0, {str(ROOT)!r}); import dental_agent.training.tpu_diagnostics as d; "
            "assert 'torch_xla' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=tmp_path)


def test_events_environment_and_stage_lifecycle(tmp_path):
    from dental_agent.training.tpu_diagnostics import TPUReport
    report = TPUReport.from_args(_args(tmp_path, tpu_report_heartbeat_seconds=0, tpu_report_stack_seconds=0,
                                       tpu_report_tpu_info_seconds=0), "sft")
    report.environment()
    report.begin("load_model", note="x")
    report.mark("weights_read")
    report.end("load_model")
    report.step(1, loss=0.5)
    report.finish()
    rep = tmp_path / "rep"
    env = json.loads((rep / "environment.json").read_text())
    assert env["role"] == "sft" and "torch" in env["versions"] and "host_memory_gib" in env
    events = _lines(rep / "events.jsonl")
    assert [(e["event"], e["name"]) for e in events] == [("begin", "load_model"), ("mark", "weights_read"), ("end", "load_model")]
    assert events[2]["seconds"] >= 0 and "host" in events[2]
    final = json.loads((rep / "final.json").read_text())
    assert final["status"] == "ok" and final["step"] == 1 and final["fields"] == {"loss": 0.5}


def test_invalid_intervals_are_rejected(tmp_path):
    from dental_agent.training.tpu_diagnostics import TPUReport
    with pytest.raises(ValueError):
        TPUReport(tmp_path / "a", role="sft", heartbeat_seconds=-1)
    with pytest.raises(ValueError):
        TPUReport(tmp_path / "b", role="sft", step_snapshot_every=0)


def test_heartbeat_thread_writes_and_stops(tmp_path):
    from dental_agent.training.tpu_diagnostics import TPUReport
    report = TPUReport(tmp_path / "r", role="grpo", heartbeat_seconds=0.05, stack_seconds=0, tpu_info_seconds=0)
    report.start()
    report.begin("rollout")
    time.sleep(0.4)
    report.finish()
    beats = _lines(tmp_path / "r" / "heartbeat.jsonl")
    assert len(beats) >= 3 and beats[-1]["stage"] == "rollout"
    count = len(beats)
    time.sleep(0.2)
    assert len(_lines(tmp_path / "r" / "heartbeat.jsonl")) == count  # thread stopped


def test_upload_is_best_effort_and_whitelisted(tmp_path, monkeypatch):
    from dental_agent.training import tpu_diagnostics as d
    calls = []
    monkeypatch.setattr(d, "_upload_folder", lambda *a: calls.append(a))
    report = d.TPUReport(tmp_path / "r", role="sft", heartbeat_seconds=0, stack_seconds=0, tpu_info_seconds=0,
                         hub_repo="owner/repo", hub_subfolder="tpu_reports/sft/run1", upload_seconds=0)
    assert report.upload("m") is True
    assert calls[0][1:4] == ("owner/repo", "model", "tpu_reports/sft/run1")
    assert "*.safetensors" not in d.UPLOAD_PATTERNS and "*.pt" not in d.UPLOAD_PATTERNS

    def boom(*a):
        raise OSError("network down")
    monkeypatch.setattr(d, "_upload_folder", boom)
    assert report.upload("m") is False                      # never raises into training
    assert "network down" in (tmp_path / "r" / "upload_errors.txt").read_text()
    assert d.TPUReport(tmp_path / "q", role="sft", heartbeat_seconds=0, stack_seconds=0, tpu_info_seconds=0).upload("m") is False


def _child(tmp_path, body):
    script = tmp_path / "child.py"
    script.write_text(textwrap.dedent(f"""
        import sys, time
        sys.path.insert(0, {str(ROOT)!r})
        from dental_agent.training import tpu_diagnostics as d
        r = d.TPUReport({str(tmp_path / 'rep')!r}, role="sft", heartbeat_seconds=0.05, stack_seconds=0, tpu_info_seconds=0).start()
    """) + textwrap.dedent(body))
    return script


def test_uncaught_exception_is_recorded_with_stage_and_traceback(tmp_path):
    script = _child(tmp_path, """
        d.begin("aot_bucket_32768")
        raise RuntimeError("RESOURCE_EXHAUSTED: boom")
    """)
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)
    assert done.returncode != 0 and "RESOURCE_EXHAUSTED" in done.stderr            # original behaviour preserved
    final = json.loads((tmp_path / "rep" / "final.json").read_text())
    assert final["status"] == "crashed" and final["stage"] == "aot_bucket_32768"
    assert "RESOURCE_EXHAUSTED: boom" in final["error"] and "Traceback" in final["traceback"]


@pytest.mark.skipif(not hasattr(signal, "SIGTERM") or sys.platform == "win32", reason="POSIX signals")
def test_sigterm_while_blocked_dumps_stacks(tmp_path):
    script = _child(tmp_path, """
        d.begin("aot_bucket_16384")
        print("ready", flush=True)
        def long_native_wait():
            time.sleep(60)
        long_native_wait()
    """)
    child = subprocess.Popen([sys.executable, str(script)], stdout=subprocess.PIPE, text=True)
    assert child.stdout.readline().strip() == "ready"
    time.sleep(0.3)
    child.send_signal(signal.SIGTERM)
    child.wait(timeout=20)
    assert child.returncode == -signal.SIGTERM                                    # still terminated by the signal
    stacks = (tmp_path / "rep" / "stacks.txt").read_text()
    assert "long_native_wait" in stacks
    beats = _lines(tmp_path / "rep" / "heartbeat.jsonl")
    assert beats and beats[-1]["stage"] == "aot_bucket_16384"                     # where it was when it died


def test_scripts_expose_the_report_flags():
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    for script in ("train_sft.py", "warmup_xla_cache.py", "run_grpo.py"):
        done = subprocess.run([sys.executable, str(ROOT / "scripts" / script), "--help"], capture_output=True, text=True,
                              cwd=ROOT, env=env)
        assert done.returncode == 0, done.stderr[-500:]
        assert "--tpu-report-dir" in done.stdout and "--tpu-report-hub-repo" in done.stdout, script
