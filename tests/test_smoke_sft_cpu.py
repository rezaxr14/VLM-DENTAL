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


def test_pad_vision_to_slots_is_rejected_off_tpu(tmp_path):
    """Static vision slots exist only to avoid XLA recompilation; on GPU/CPU the flag is an error, not a silent no-op."""
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(2)) + "\n")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test_sft.py"), "--traces", str(traces), "--num-traces", "2",
         "--work-dir", str(tmp_path / "work"), "--", "--canonical-resize", "--pad-vision-to-slots"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode != 0
    assert "TPU-only" in proc.stdout + proc.stderr


def test_run_grpo_with_tools_runs_end_to_end_on_cpu(tmp_path):
    """Regression: launcher crash on cfg.data.data_dir, greedy (identical) rollouts, and unparseable replies being dropped
    from the transcript (all spans lost -> 'Zero assistant completion tokens')."""
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(3)) + "\n")
    work = tmp_path / "grpo"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test_grpo.py"), "--traces", str(traces), "--track", "with_tools",
         "--work-dir", str(work), "--", "--canonical-resize", "--logprob-chunk-tokens", "16"],
        capture_output=True, text=True, timeout=900,
    )
    tail = (proc.stdout + proc.stderr)[-2500:]
    assert proc.returncode == 0, tail
    assert "[SMOKE-GRPO] PASSED" in proc.stdout, tail


def test_train_sft_without_letterbox_runs_end_to_end_and_records_it(tmp_path):
    """GPU recipe: --canonical-resize --no-canonical-letterbox trains on unpadded views (variable token counts, dynamic padding)."""
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(4)) + "\n")
    work = tmp_path / "work"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test_sft.py"), "--traces", str(traces), "--num-traces", "4",
         "--work-dir", str(work), "--", "--canonical-resize", "--no-canonical-letterbox", "--max-seq-len", "8192"],
        capture_output=True, text=True, timeout=600,
    )
    tail = (proc.stdout + proc.stderr)[-2500:]
    assert proc.returncode == 0, tail
    assert "[SMOKE] PASSED" in proc.stdout, tail
    meta = json.loads((work / "out" / "sft_run_meta.json").read_text())
    assert meta["canonical_resize"] is True and meta["canonical_letterbox"] is False
    assert "canonical_letterbox=False" in proc.stdout + proc.stderr


def test_train_sft_with_chunked_loss_runs_end_to_end(tmp_path):
    """GPU recipe: --loss-chunk-tokens computes the loss on assistant tokens only (bounded logits memory) and still trains and evaluates."""
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(4)) + "\n")
    work = tmp_path / "work"
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test_sft.py"), "--traces", str(traces), "--num-traces", "4",
         "--work-dir", str(work), "--", "--canonical-resize", "--no-canonical-letterbox", "--max-seq-len", "8192",
         "--loss-chunk-tokens", "16"],
        capture_output=True, text=True, timeout=600,
    )
    tail = (proc.stdout + proc.stderr)[-2500:]
    assert proc.returncode == 0, tail
    assert "[SMOKE] PASSED" in proc.stdout, tail
    assert (work / "out" / "adapter_config.json").exists()
    meta = json.loads((work / "out" / "sft_run_meta.json").read_text())
    assert meta["precision"] == "bf16" and meta["qlora_skip_modules"] is None
    assert meta["compute_dtype"] == "torch.bfloat16" and meta["loss_chunk_tokens"] == 16
    assert meta["attention_implementation"] == "sdpa"
    assert "versions" in meta["runtime"]
    debug_out = work / "debug"
    debug = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_sft.py"),
         "--model-id", str(work / "model"), "--track", "with_tools",
         "--dataset-path", str(work / "data" / "smoke_traces.jsonl"),
         "--data-dir", str(work / "data"), "--output-dir", str(debug_out),
         "--canonical-resize", "--no-canonical-letterbox", "--max-seq-len", "8192",
         "--lora-r", "4", "--lora-alpha", "8", "--precision", "bf16",
         "--loss-chunk-tokens", "16", "--debug-steps", "2", "--gradient-accumulation-steps", "1",
         "--loader-mode", "staged", "--load-cpu-threads", "1", "--sdpa-repeat-kv", "on"],
        capture_output=True, text=True, timeout=600,
    )
    assert debug.returncode == 0, (debug.stdout + debug.stderr)[-4000:]
    for marker in ("[DEBUG COMPLETE]", "[HEAD]", "[GRAD]", "[UPDATE]"):
        assert marker in debug.stdout, debug.stdout[-4000:]
    assert not (debug_out / "adapter_config.json").exists()


def test_negative_loss_chunk_tokens_is_rejected(tmp_path):
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(2)) + "\n")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test_sft.py"), "--traces", str(traces), "--num-traces", "2",
         "--work-dir", str(tmp_path / "work"), "--", "--canonical-resize", "--loss-chunk-tokens", "-1"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode != 0
    assert "--loss-chunk-tokens" in proc.stdout + proc.stderr


def test_pad_vision_to_slots_without_letterbox_is_rejected(tmp_path):
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(2)) + "\n")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test_sft.py"), "--traces", str(traces), "--num-traces", "2",
         "--work-dir", str(tmp_path / "work"), "--", "--canonical-resize", "--no-canonical-letterbox", "--pad-vision-to-slots"],
        capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode != 0
    assert "TPU-only" in proc.stdout + proc.stderr or "canonical-letterbox" in proc.stdout + proc.stderr


def test_run_grpo_without_letterbox_runs_end_to_end_on_cpu(tmp_path):
    traces = tmp_path / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(3)) + "\n")
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "smoke_test_grpo.py"), "--traces", str(traces), "--track", "with_tools",
         "--work-dir", str(tmp_path / "grpo"), "--", "--canonical-resize", "--no-canonical-letterbox"],
        capture_output=True, text=True, timeout=900,
    )
    tail = (proc.stdout + proc.stderr)[-2500:]
    assert proc.returncode == 0, tail
    assert "[SMOKE-GRPO] PASSED" in proc.stdout, tail
    assert "canonical_letterbox=False" in proc.stdout + proc.stderr
