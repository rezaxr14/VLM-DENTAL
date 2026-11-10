"""scripts/run_external_benchmark.py end to end with the model, tools and judge stubbed: prepare -> policy -> answer (plain and with
findings) -> judge -> score -> compare, plus the refusals that protect a run (settings change, existing results, letterbox)."""
import importlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.test_external_benchmarks import FakeAPI, synthetic_tsvs  # noqa: E402


@pytest.fixture()
def script(monkeypatch):
    try:
        import torch  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    calls = SimpleNamespace(loads=[], prompts=[], trajectories=[])

    ev = types.ModuleType("scripts.evaluate_models")
    ev.load_env = lambda: None
    ev.setup_eval_hardware = lambda precision, qdtype="auto", **kw: ("cpu", "bf16", False)

    def load_eval_model(model_id, adapter, device, dtype, condition, quantize_4bit=False, runtime_options=None, **kw):
        calls.loads.append(dict(model_id=model_id, adapter=adapter, condition=condition))
        return "MODEL", "PROCESSOR"

    ev.load_eval_model = load_eval_model
    ev.eval_settings = lambda ns, cond, adapter: dict(condition=cond, harness=ns.harness_mode, model_id=ns.model_id, adapter_path=adapter or "",
                                                      max_new_tokens=int(ns.max_new_tokens), canonical_letterbox=bool(ns.canonical_letterbox))

    def run_condition_trajectory(model, processor, images_df, image_id, registry, ns):
        calls.trajectories.append((image_id, ns.harness_mode, ns.dataset, ns.canonical_resize, ns.max_tool_calls))
        answer = [dict(quadrant=3, tooth_position=6, diagnosis="Periapical Lesion", confidence=0.9)] if image_id == 0 else None
        return SimpleNamespace(final_answer=answer, format_ok=answer is not None, turns=[{"raw_output": "x"}], tool_calls=2)

    ev.run_condition_trajectory = run_condition_trajectory
    ev.findings_from_trajectory = lambda traj: list(traj.final_answer or [])
    ev.transcript_for_record = lambda traj: {"turns": len(traj.turns)}
    monkeypatch.setitem(sys.modules, "scripts.evaluate_models", ev)
    import scripts
    monkeypatch.setattr(scripts, "evaluate_models", ev, raising=False)

    lr = types.ModuleType("dental_agent.model.local_runtime")
    lr.add_runtime_args = lambda parser, **kw: parser.add_argument("--qlora-compute-dtype", default="auto")
    lr.runtime_from_args = lambda args: {}
    monkeypatch.setitem(sys.modules, "dental_agent.model.local_runtime", lr)

    inf = types.ModuleType("dental_agent.model.inference")

    def generate_agent_reply(model, processor, messages, max_new_tokens, temperature, return_ids):
        text = messages[0]["content"][1]["text"]
        calls.prompts.append(text)
        assert messages[0]["content"][0]["type"] == "image" and temperature == 0.0 and return_ids is True
        return ("<think>hm</think>(B)" if "Options:" in text else "30 teeth are visible"), 10, [1, 2, 3]

    inf.generate_agent_reply = generate_agent_reply
    monkeypatch.setitem(sys.modules, "dental_agent.model.inference", inf)

    reg = types.ModuleType("dental_agent.tools.registry")
    reg.ToolRegistry = SimpleNamespace(create_default=lambda: "REGISTRY")
    monkeypatch.setitem(sys.modules, "dental_agent.tools.registry", reg)
    abl = types.ModuleType("dental_agent.evaluation.tool_ablation")
    abl.AblationSpec = lambda: "SPEC"
    abl.ToolProbe = lambda registry, spec, image_id: SimpleNamespace(registry=registry)
    monkeypatch.setitem(sys.modules, "dental_agent.evaluation.tool_ablation", abl)

    mod = importlib.import_module("scripts.run_external_benchmark")
    mod.calls = calls
    return mod


def common(tmp_path):
    return ["--bench-dir", str(tmp_path / "bench"), "--results-dir", str(tmp_path / "res")]


def test_full_pipeline(script, tmp_path, monkeypatch):
    raw = tmp_path / "bench" / "raw"
    raw.mkdir(parents=True)
    synthetic_tsvs(raw)
    script.main(["prepare", "--bench-dir", str(tmp_path / "bench"), "--allow-md5-mismatch"])

    adapter = tmp_path / "grpo_policy"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    pol = ["policy", *common(tmp_path), "--policy-name", "grpo_k4_harness", "--adapter-path", str(adapter), "--harness-mode", "rules_skills_index",
           "--max-new-tokens", "16384"]
    with pytest.raises(SystemExit, match="canonical view explicitly"):
        script.main(pol)
    script.main([*pol, "--no-canonical-letterbox"])
    assert script.calls.loads[-1] == dict(model_id="Qwen/Qwen3.5-9B", adapter=str(adapter), condition="grpo_with_tools")
    assert {t[1:] for t in script.calls.trajectories} == {("rules_skills_index", "dentex", True, 50)}      # native prompt/taxonomy, canonical views
    findings = [json.loads(l) for l in (tmp_path / "res/policy/grpo_k4_harness/findings.jsonl").read_text().splitlines()]
    assert len(findings) == 2 and {f["answered"] for f in findings} == {True, False}
    assert "adapter_files_sha256" in findings[0]["settings"] and findings[0]["settings"]["canonical_letterbox"] is False
    with pytest.raises(SystemExit, match="already has"):
        script.main([*pol, "--no-canonical-letterbox"])                                                   # existing results: error by default
    script.main([*pol, "--no-canonical-letterbox", "--existing-results", "resume"])                      # resume: nothing left to run
    assert len(script.calls.trajectories) == 2

    ans = ["answer", *common(tmp_path), "--max-new-tokens", "16384", "--existing-results", "error"]
    script.main([*ans, "--system-name", "qwen3_5_9b_raw"])
    assert script.calls.loads[-1]["adapter"] is None and script.calls.loads[-1]["condition"] == "base_no_tools"
    assert all(p.startswith("Question:") for p in script.calls.prompts)                                  # raw baseline: official prompt only
    script.calls.prompts.clear()
    script.main([*ans, "--system-name", "optionA_grpo_k4_harness", "--findings-from", "grpo_k4_harness"])
    assert all(p.startswith("A dental radiograph analysis agent") for p in script.calls.prompts)
    assert any("#36: Periapical Lesion" in p for p in script.calls.prompts) and any("did not return a valid result" in p for p in script.calls.prompts)
    with pytest.raises(SystemExit, match="different settings"):                                          # a changed budget cannot be resumed
        script.main(["answer", *common(tmp_path), "--max-new-tokens", "8192", "--existing-results", "resume", "--system-name", "qwen3_5_9b_raw"])

    # judge with a fake endpoint: the raw baseline gets 0.2 on every open answer, Option A gets 0.6
    real_client = script.jg.JudgeClient
    for system, value in (("qwen3_5_9b_raw", "0.2"), ("optionA_grpo_k4_harness", "0.6")):
        monkeypatch.setattr(script.jg, "JudgeClient", lambda model, _v=value, **kw: real_client(model, client=FakeAPI([_v] * 50)))
        script.main(["judge", *common(tmp_path), "--system-name", system, "--judge-model", "judge-x", "--workers", "1"])
    assert (tmp_path / "res/judge/qwen3_5_9b_raw__judge-x.jsonl").is_file()

    script.main(["score", *common(tmp_path), "--systems", "qwen3_5_9b_raw", "optionA_grpo_k4_harness", "--judge-model", "judge-x", "--n-boot", "50"])
    md = (tmp_path / "res/tables/mmoral_opg.md").read_text()
    assert "qwen3_5_9b_raw [strict]" in md and "| optionA_grpo_k4_harness | judge-x |" in md
    scores = json.loads((tmp_path / "res/tables/mmoral_opg_scores.json").read_text())
    assert scores["official_format"]["qwen3_5_9b_raw"]["closed"]["n"] == 3
    script.main(["compare", *common(tmp_path), "--a", "optionA_grpo_k4_harness", "--b", "qwen3_5_9b_raw", "--judge-model", "judge-x", "--n-boot", "50"])
    cmp = json.loads((tmp_path / "res/tables/compare__optionA_grpo_k4_harness__minus__qwen3_5_9b_raw.json").read_text())
    assert cmp["open"]["Overall"]["mean"] == pytest.approx(40.0)                                          # 0.6 - 0.2, all questions, x100


def test_answer_needs_policy_results_for_findings(script, tmp_path):
    raw = tmp_path / "bench" / "raw"
    raw.mkdir(parents=True)
    synthetic_tsvs(raw)
    script.main(["prepare", "--bench-dir", str(tmp_path / "bench"), "--allow-md5-mismatch"])
    (tmp_path / "res/policy/ghost").mkdir(parents=True)
    (tmp_path / "res/policy/ghost/findings.jsonl").write_text("")
    with pytest.raises(SystemExit, match="no policy result"):
        script.main(["answer", *common(tmp_path), "--max-new-tokens", "16384", "--system-name", "x", "--findings-from", "ghost"])
    assert script.calls.loads == []                                  # refused before any model load


def test_policy_rejects_missing_adapter_before_loading(script, tmp_path):
    raw = tmp_path / "bench" / "raw"
    raw.mkdir(parents=True)
    synthetic_tsvs(raw)
    script.main(["prepare", "--bench-dir", str(tmp_path / "bench"), "--allow-md5-mismatch"])
    with pytest.raises(SystemExit, match="does not exist"):
        script.main(["policy", *common(tmp_path), "--policy-name", "p", "--adapter-path", str(tmp_path / "nope"), "--harness-mode", "off",
                     "--max-new-tokens", "16384", "--no-canonical-letterbox"])
    assert script.calls.loads == []


def test_leakage_uses_combined_train_and_stops_on_overlap(script, tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from PIL import Image
    raw = tmp_path / "bench" / "raw"
    raw.mkdir(parents=True)
    synthetic_tsvs(raw)
    script.main(["prepare", "--bench-dir", str(tmp_path / "bench"), "--allow-md5-mismatch"])
    images = pd.read_csv(tmp_path / "bench" / "prepared" / "images.csv")

    def save(name, seed):
        arr = np.random.default_rng(seed).integers(0, 255, size=(32, 64), dtype=np.uint8)
        Image.fromarray(arr).resize((256, 128)).convert("RGB").save(tmp_path / name)
        return str(tmp_path / name)

    seen = []
    other = save("d1.png", 123)

    def fake_loader(data_dir=None, split_name="train", **kw):
        seen.append((split_name, kw))
        return pd.DataFrame({"id": [1], "local_path": [other]}), None, None

    import dental_agent.data.dentex as dentex
    monkeypatch.setattr(dentex, "load_dentex_dataset", fake_loader)
    script.main(["leakage", "--bench-dir", str(tmp_path / "bench"), "--data-dir", str(tmp_path)])           # no copy: returns normally
    assert seen == [("train", {"combine_enumeration_splits": True}), ("validation", {})]
    report = json.loads((tmp_path / "bench" / "leakage_dentex.json").read_text())
    assert report["verdict"].startswith("no copy found") and report["n_reference"] == 2

    copy = images.local_path.iloc[0]                                                                         # a DENTEX 'image' that is a benchmark image
    monkeypatch.setattr(dentex, "load_dentex_dataset", lambda **kw: (pd.DataFrame({"id": [9], "local_path": [copy]}), None, None))
    with pytest.raises(SystemExit, match="overlap with DENTEX"):
        script.main(["leakage", "--bench-dir", str(tmp_path / "bench"), "--data-dir", str(tmp_path)])
