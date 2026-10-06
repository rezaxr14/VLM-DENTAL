"""Regression: the real train_sft.py runs end to end on a non-TPU machine with DEFAULT flags (--spmd defaults on).

Uses scripts/smoke_test_sft.py (tiny random Qwen3.5, synthetic traces, noise radiographs). Catches the crash where the
default --spmd reached wrap_spmd_model off-TPU, and flags that were silently ignored on GPU/CPU.
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _trace(i):
    calls = [
        {"tool": "window_level", "args": {"preset": "bone"}},
        {"tool": "zoom_crop", "args": {"bbox": [300.0 + 40 * i, 200.0, 100.0, 150.0], "padding_frac": 0.2}},
        {"tool": "zoom_crop", "args": {"bbox": [900.0, 500.0 + 30 * i, 120.0, 140.0], "padding_frac": 0.2}},
    ]
    return {
        "image_id": 100 + i, "dataset": "dentex",
        "messages": [
            {"role": "system", "content": "You are an expert dental radiologist. Use tools. " * 20},
            {"role": "user", "content": [{"type": "image", "image": "<Image>"}, {"type": "text", "text": "Analyze this panoramic X-ray."}]},
            {"role": "assistant", "content": json.dumps({"thought": "inspect teeth", "tool_calls": calls})},
            {"role": "user", "content": [
                {"type": "image", "image": "<Image>"}, {"type": "text", "text": "Result of window_level:"},
                {"type": "image", "image": "<Image>"}, {"type": "text", "text": "Result of zoom_crop:"},
                {"type": "image", "image": "<Image>"}, {"type": "text", "text": "Result of zoom_crop:"}]},
            {"role": "assistant", "content": json.dumps({"thought": "done", "final_answer": [
                {"tooth": 36, "diagnosis": "Caries", "confidence": 0.8}]})},
        ],
        "turns": [], "final_answer": [],
    }


def test_train_sft_default_flags_run_end_to_end_off_tpu(tmp_path):
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(4)) + "\n")
    work = tmp_path / "work"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test_sft.py"), "--traces", str(traces), "--num-traces", "4",
         "--work-dir", str(work), "--", "--canonical-resize", "--max-seq-len", "8192"],
        capture_output=True, text=True, timeout=600,
    )
    tail = (proc.stdout + proc.stderr)[-2500:]
    assert proc.returncode == 0, tail
    assert "[SMOKE] PASSED" in proc.stdout, tail
    assert (work / "out" / "adapter_config.json").exists()
