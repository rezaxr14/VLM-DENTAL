"""Speculative decoding on the TPU profile (patch 0064): coordinator gating / identity / command, the benchmark's parser and its
per-request recorder. Offline; the XLA behaviour is in test_tpu_speculative_xla_cpu.py."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import test_harness_grpo_mtp as G
from dental_agent.harness import notebook_eval as ne

ROOT = Path(__file__).resolve().parents[1]
TPU_DEVICE = [dict(name="TPU v5e-8", gib=0.0, capability=(0, 0), tpu=True)]
T4 = [dict(name="T4", gib=15.0, capability=(7, 5)), dict(name="T4", gib=15.0, capability=(7, 5))]


def test_tpu_runs_speculative_jobs_and_cuda_gating_is_unchanged():
    assert ne.hardware_reason("tpu", "speculative", TPU_DEVICE) is None
    assert "TPU" in ne.hardware_reason("tpu", "speculative", T4)
    assert "TPU" in ne.hardware_reason("tpu", "speculative", [])
    assert ne.hardware_reason("dual_t4_fp16", "speculative", T4) is None
    old_gpu = [dict(name="Volta", gib=32.0, capability=(7, 0))]
    assert "7.5" in ne.hardware_reason("dual_t4_fp16", "speculative", old_gpu)


def _session(tmp_path):
    stem = G.grpo_name("with_tools", 4, "dentex_alone", "bf16", "tpu_bf16", harness_mode="off")
    root = f"grpo/{stem}/grpo-with_tools-final"
    prefix = root + "/grpo_policy"
    hub = G.Hub(tmp_path)
    hub.files = {prefix + "/adapter_config.json": b"{}", prefix + "/adapter_model.safetensors": b"weights",
                 root + "/checkpoint_meta.json": json.dumps({"extra": {"harness": G.harness_identity("off")}}).encode()}
    session = ne.NotebookEvaluations(api=hub, download=hub.download, kinds=["speculative"], target_profiles="tpu", track="with_tools",
                                     speculative_target_kind="grpo", speculative_harness_mode="off", model_id="/base")
    session._saved_identity = lambda ident: None
    session._input = lambda split: dict(ids=[1, 2], fingerprint="cohort")
    session.inputs = {"dev": {"data_dir": "/dev-data"}}
    session.dev_split = tmp_path / "dev.json"
    session.dev_split.write_text(json.dumps({"image_ids": [1, 2]}))
    session._adapter = lambda job: "/exact/adapter"
    return session


def test_tpu_speculative_job_needs_no_vllm_and_no_merged_target(tmp_path):
    session = _session(tmp_path)
    jobs = [j for j in session.jobs if j["kind"] == "speculative"]
    assert len(jobs) == 1 and jobs[0]["profile"] == "tpu"
    identity = session._identity({**jobs[0], "size": "canonical"})
    assert identity["conditions"] == ["baseline", "mtp_1", "mtp_2", "mtp_3", "ngram_2", "ngram_3", "ngram_5"]
    assert identity["backend"] == "tpu_static_generation" and identity["merged_files"] is None and "serving" not in identity
    assert identity["tpu_generation"]["cache_tiers"] == [32768, 49152]
    command, result, adapter = session._command({**jobs[0], "size": "canonical"}, identity, tmp_path)
    assert command[1:3] == ["-u", "scripts/bench/spec_decode_bench_tpu.py"] and adapter == "/exact/adapter"
    assert "--vllm-executable" not in command and "--checkpoint" not in command and "--tpu-generation-chunk-tokens" in command
    assert command[command.index("--conditions") + 1:command.index("--conditions") + 8] == ["baseline", "mtp_1", "mtp_2", "mtp_3", "ngram_2", "ngram_3", "ngram_5"]
    assert result == tmp_path / "results.json"


def test_the_benchmark_parser_accepts_exactly_what_the_coordinator_sends(tmp_path):
    session = _session(tmp_path)
    job = {**[j for j in session.jobs if j["kind"] == "speculative"][0], "size": "canonical"}
    command, _, _ = session._command(job, session._identity(job), tmp_path)
    sys.path.insert(0, str(ROOT))
    from scripts.bench import spec_decode_bench_tpu as bench
    args = bench.parse_args(command[3:])
    assert args.conditions == ["baseline", "mtp_1", "mtp_2", "mtp_3", "ngram_2", "ngram_3", "ngram_5"] and args.tpu_generation_chunk_tokens == 512
    assert args.canonical_letterbox is True and args.max_new_tokens == 24576 and args.checkpoint_kind == "grpo"


def test_the_benchmark_refuses_unknown_conditions_and_a_missing_baseline():
    sys.path.insert(0, str(ROOT))
    from scripts.bench import spec_decode_bench_tpu as bench
    with pytest.raises(SystemExit):
        bench.parse_args(["--model-id", "m", "--adapter-path", "a", "--conditions", "baseline", "mtp_4"])
    with pytest.raises(ValueError, match="start with baseline"):
        bench.main(["--model-id", "m", "--adapter-path", "a", "--conditions", "ngram_3", "--tpu-generation-chunk-tokens", "512"])


def test_the_reply_recorder_reports_one_measurement_per_request(monkeypatch):
    sys.path.insert(0, str(ROOT))
    from scripts.bench import spec_decode_bench_tpu as bench
    from dental_agent.harness.specdec import episode_measurements
    config = SimpleNamespace(stats={})

    def fake_reply(model, processor, messages, **kwargs):
        config.stats["last_call"] = dict(prompt_tokens=900, generated_tokens=40, prefill_seconds=2.0, decode_seconds=3.0,
                                         speculative=dict(method="ngram", num_speculative_tokens=3), rounds=12, plain_steps=4, drafted=30, accepted=18)
        return "text", 900, [1, 2], None, None
    monkeypatch.setattr("dental_agent.model.inference.generate_agent_reply", fake_reply)
    reply = bench.TPUReply(config)
    assert reply(None, None, [{"role": "user", "content": "hi"}], max_new_tokens=8)[0] == "text"
    config.stats["last_call"] = None
    config.stats.pop("last_call")
    request = reply.requests[0]
    assert (request["drafted_tokens"], request["accepted_tokens"], request["acceptance_rate"]) == (30, 18, 0.6)
    assert request["generated_tokens"] == 40 and request["prefill_seconds"] == 2.0 and request["decode_seconds"] == 3.0
    summary = episode_measurements(reply.requests, 6.0, 0.5)
    assert summary["generated_tokens"] == 40 and summary["acceptance_rate"] == 0.6 and summary["tokens_per_second"] == pytest.approx(40 / 6.0)
    # a baseline request has no drafts: those fields are None, not zero, so pooled acceptance stays undefined
    config.stats["last_call"] = dict(prompt_tokens=10, generated_tokens=5, prefill_seconds=1.0, decode_seconds=1.0, speculative=None,
                                     rounds=0, plain_steps=4, drafted=0, accepted=0)
    bench.TPUReply(config)(None, None, [{"role": "user", "content": "hi"}], max_new_tokens=8)
