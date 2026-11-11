"""scripts/run_external_benchmark.py end to end with the model, tools, TPU generator and judge stubbed: prepare -> policy -> answer (plain and
with findings) -> judge -> score -> compare, the refusals that protect a run, and the TPU contract of docs/TPU_EVALUATION.md (static canonical
letterbox views required before any weight loads, SPMD before the device, max_new_tokens at load, refusals recorded and scored as wrong,
generation report written, external models refused)."""
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

TPU_FLAGS = ["--tpu-generation-chunk-tokens", "512", "--tpu-generation-tiers", "32768", "49152", "--tpu-vision-slots", "8", "16", "6"]


@pytest.fixture()
def script(monkeypatch):
    try:
        import torch  # noqa: F401
    except ImportError:
        torch_stub = types.ModuleType("torch")
        torch_stub.cuda = SimpleNamespace(is_available=lambda: False)
        monkeypatch.setitem(sys.modules, "torch", torch_stub)
    calls = SimpleNamespace(loads=[], prompts=[], trajectories=[], order=[], hw=[], images=[], refusals=[], reports=[], refuse_images=set(),
                            refuse_prompts_containing=None, is_tpu=False)

    class GenerationTooLong(ValueError):
        pass

    ti = types.ModuleType("dental_agent.model.tpu_inference")
    ti.GenerationTooLong = GenerationTooLong
    ti.GENERATION_REFUSED = (GenerationTooLong,)

    def require_static_views(*, canonical_resize, letterbox, what):
        calls.order.append(("require_static_views", what))
        if not (canonical_resize and letterbox):
            raise ValueError(f"{what}: TPU evaluation needs --canonical-resize --canonical-letterbox")

    def note_refusal(model, image_id, exc):
        calls.refusals.append(int(image_id))
        return f"{type(exc).__name__}: {exc}"

    def write_generation_report(model, path):
        calls.reports.append(str(path))
        return {}

    ti.require_static_views, ti.note_refusal, ti.write_generation_report = require_static_views, note_refusal, write_generation_report
    ti.refused_trajectory = lambda image_id, reason: SimpleNamespace(final_answer=None, format_ok=False, turns=[], tool_calls=0)
    monkeypatch.setitem(sys.modules, "dental_agent.model.tpu_inference", ti)

    ev = types.ModuleType("scripts.evaluate_models")
    ev.load_env = lambda: None

    def setup_eval_hardware(precision, qdtype="auto", spmd=False):
        calls.order.append(("setup_eval_hardware", spmd))
        calls.hw.append(dict(spmd=spmd))
        return "cpu", "bf16", calls.is_tpu

    def load_eval_model(model_id, adapter, device, dtype, condition, quantize_4bit=False, runtime_options=None, max_new_tokens=None, repetition_penalty=None):
        calls.order.append(("load_eval_model", max_new_tokens))
        calls.loads.append(dict(model_id=model_id, adapter=adapter, condition=condition, max_new_tokens=max_new_tokens))
        return "MODEL", "PROCESSOR"

    ev.setup_eval_hardware, ev.load_eval_model = setup_eval_hardware, load_eval_model
    ev.eval_settings = lambda ns, cond, adapter: dict(condition=cond, harness=ns.harness_mode, model_id=ns.model_id, adapter_path=adapter or "",
                                                      max_new_tokens=int(ns.max_new_tokens), canonical_letterbox=bool(ns.canonical_letterbox),
                                                      **ev.tpu_generation_settings(ns))
    ev.tpu_generation_settings = lambda a: ({"tpu_generation": {"chunk_tokens": int(a.tpu_generation_chunk_tokens), "cache_tiers": list(a.tpu_generation_tiers),
                                                                 "vision_slots": list(a.tpu_vision_slots)}} if int(a.tpu_generation_chunk_tokens) else {})

    def run_condition_trajectory(model, processor, images_df, image_id, registry, ns):
        calls.trajectories.append((image_id, ns.harness_mode, ns.dataset, ns.canonical_resize, ns.max_tool_calls))
        if image_id in calls.refuse_images:
            raise GenerationTooLong("prompt does not fit the largest tier")
        answer = [dict(quadrant=3, tooth_position=6, diagnosis="Periapical Lesion", confidence=0.9)] if image_id == 0 else None
        return SimpleNamespace(final_answer=answer, format_ok=answer is not None, turns=[{"raw_output": "x"}], tool_calls=2)

    ev.run_condition_trajectory = run_condition_trajectory
    ev.findings_from_trajectory = lambda traj: list(traj.final_answer or [])
    ev.transcript_for_record = lambda traj: {"turns": len(traj.turns)}
    monkeypatch.setitem(sys.modules, "scripts.evaluate_models", ev)
    import scripts
    monkeypatch.setattr(scripts, "evaluate_models", ev, raising=False)

    lr = types.ModuleType("dental_agent.model.local_runtime")

    def add_runtime_args(parser, **kw):
        parser.add_argument("--qlora-compute-dtype", default="auto")
        parser.add_argument("--tpu-generation-chunk-tokens", type=int, default=0)
        parser.add_argument("--tpu-generation-tiers", type=int, nargs="+", default=[32768, 49152])
        parser.add_argument("--tpu-vision-slots", type=int, nargs=3, default=[8, 16, 6])

    lr.add_runtime_args, lr.runtime_from_args = add_runtime_args, (lambda args: {})
    monkeypatch.setitem(sys.modules, "dental_agent.model.local_runtime", lr)

    inf = types.ModuleType("dental_agent.model.inference")

    def generate_agent_reply(model, processor, messages, max_new_tokens, temperature, return_ids):
        text = messages[0]["content"][1]["text"]
        image = messages[0]["content"][0]["image"]
        calls.prompts.append(text)
        calls.images.append(image.size)
        assert messages[0]["content"][0]["type"] == "image" and temperature == 0.0 and return_ids is True
        if calls.refuse_prompts_containing and calls.refuse_prompts_containing in text:
            raise GenerationTooLong("prompt does not fit")
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


def prepared(tmp_path, script):
    raw = tmp_path / "bench" / "raw"
    raw.mkdir(parents=True)
    synthetic_tsvs(raw)
    script.main(["prepare", "--bench-dir", str(tmp_path / "bench"), "--allow-md5-mismatch"])


def adapter_dir(tmp_path):
    adapter = tmp_path / "grpo_policy"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    (adapter / "adapter_model.safetensors").write_bytes(b"weights")
    return adapter


def policy_args(tmp_path, adapter, *extra):
    return ["policy", *common(tmp_path), "--policy-name", "grpo_k4_harness", "--adapter-path", str(adapter), "--harness-mode", "rules_skills_index",
            "--max-new-tokens", "16384", *extra]


# ----------------------------------------------------------------------------------------------------------------------- GPU path
def test_full_pipeline(script, tmp_path, monkeypatch):
    prepared(tmp_path, script)
    adapter = adapter_dir(tmp_path)
    pol = policy_args(tmp_path, adapter)
    with pytest.raises(SystemExit, match="canonical view explicitly"):
        script.main(pol)
    script.main([*pol, "--no-canonical-letterbox"])
    assert script.calls.loads[-1] == dict(model_id="Qwen/Qwen3.5-9B", adapter=str(adapter), condition="grpo_with_tools", max_new_tokens=None)
    assert script.calls.hw == [dict(spmd=False)] and not any(o[0] == "require_static_views" for o in script.calls.order)      # GPU: exactly the old calls
    assert {t[1:] for t in script.calls.trajectories} == {("rules_skills_index", "dentex", True, 50)}      # native prompt/taxonomy, canonical views
    findings = [json.loads(l) for l in (tmp_path / "res/policy/grpo_k4_harness/findings.jsonl").read_text().splitlines()]
    assert len(findings) == 2 and {f["answered"] for f in findings} == {True, False}
    assert "adapter_files_sha256" in findings[0]["settings"] and findings[0]["settings"]["canonical_letterbox"] is False
    assert "tpu_generation" not in findings[0]["settings"] and all(f["generation_refused"] is None for f in findings)
    with pytest.raises(SystemExit, match="already has"):
        script.main([*pol, "--no-canonical-letterbox"])                                                   # existing results: error by default
    script.main([*pol, "--no-canonical-letterbox", "--existing-results", "resume"])                      # resume: nothing left to run
    assert len(script.calls.trajectories) == 2 and script.calls.reports == []

    ans = ["answer", *common(tmp_path), "--max-new-tokens", "16384", "--existing-results", "error", "--image-view", "canonical", "--no-canonical-letterbox"]
    script.main([*ans, "--system-name", "qwen3_5_9b_raw"])
    assert script.calls.loads[-1]["adapter"] is None and script.calls.loads[-1]["condition"] == "base_no_tools"
    assert all(p.startswith("Question:") for p in script.calls.prompts)                                  # raw baseline: official prompt only
    assert all(w <= 1536 and h <= 768 for w, h in script.calls.images)                                    # canonical FULL view, no bars on a GPU
    script.calls.prompts.clear()
    script.main([*ans, "--system-name", "optionA_grpo_k4_harness", "--findings-from", "grpo_k4_harness"])
    assert all(p.startswith("A dental radiograph analysis agent") for p in script.calls.prompts)
    assert any("#36: Periapical Lesion" in p for p in script.calls.prompts) and any("did not return a valid result" in p for p in script.calls.prompts)
    with pytest.raises(SystemExit, match="different settings"):                                          # a changed budget cannot be resumed
        script.main(["answer", *common(tmp_path), "--max-new-tokens", "8192", "--existing-results", "resume", "--system-name", "qwen3_5_9b_raw",
                     "--image-view", "canonical", "--no-canonical-letterbox"])
    with pytest.raises(SystemExit, match="different settings"):                                          # nor a changed image view
        script.main(["answer", *common(tmp_path), "--max-new-tokens", "16384", "--existing-results", "resume", "--system-name", "qwen3_5_9b_raw",
                     "--image-view", "native"])

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
    prepared(tmp_path, script)
    (tmp_path / "res/policy/ghost").mkdir(parents=True)
    (tmp_path / "res/policy/ghost/findings.jsonl").write_text("")
    with pytest.raises(SystemExit, match="no policy result"):
        script.main(["answer", *common(tmp_path), "--max-new-tokens", "16384", "--system-name", "x", "--findings-from", "ghost",
                     "--image-view", "native"])
    assert script.calls.loads == []                                  # refused before any model load


def test_policy_rejects_missing_adapter_before_loading(script, tmp_path):
    prepared(tmp_path, script)
    with pytest.raises(SystemExit, match="does not exist"):
        script.main(["policy", *common(tmp_path), "--policy-name", "p", "--adapter-path", str(tmp_path / "nope"), "--harness-mode", "off",
                     "--max-new-tokens", "16384", "--no-canonical-letterbox"])
    assert script.calls.loads == []


def test_canonical_answer_requires_explicit_letterbox_and_native_does_not(script, tmp_path):
    prepared(tmp_path, script)
    base = ["answer", *common(tmp_path), "--max-new-tokens", "16384", "--system-name", "s"]
    with pytest.raises(SystemExit, match="canonical view explicitly"):
        script.main([*base, "--image-view", "canonical"])
    assert script.calls.loads == []
    script.main([*base, "--image-view", "native", "--limit", "1"])
    assert script.calls.images[0] == (64, 32)                          # the stored image, untouched


# ----------------------------------------------------------------------------------------------------------------------- TPU path
def test_tpu_policy_follows_the_evaluator_contract(script, tmp_path):
    prepared(tmp_path, script)
    adapter = adapter_dir(tmp_path)
    script.calls.refuse_images = {1}                                  # the generator refuses the second image
    script.main(policy_args(tmp_path, adapter, "--canonical-letterbox", *TPU_FLAGS))
    assert script.calls.order == [("require_static_views", "external benchmark (policy)"), ("setup_eval_hardware", True), ("load_eval_model", 16384)]
    recs = [json.loads(l) for l in (tmp_path / "res/policy/grpo_k4_harness/findings.jsonl").read_text().splitlines()]
    by_id = {r["image_id"]: r for r in recs}
    assert by_id[0]["answered"] is True and by_id[0]["generation_refused"] is None
    assert by_id[1]["answered"] is False and by_id[1]["findings"] == [] and by_id[1]["generation_refused"].startswith("GenerationTooLong")
    assert script.calls.refusals == [1]
    assert script.calls.reports == [str(tmp_path / "res/policy/grpo_k4_harness/tpu_generation_report.json")]
    assert by_id[0]["settings"]["tpu_generation"] == {"chunk_tokens": 512, "cache_tiers": [32768, 49152], "vision_slots": [8, 16, 6]}
    # a GPU result file is never resumed on a TPU
    with pytest.raises(SystemExit, match="different settings"):
        script.main([*policy_args(tmp_path, adapter, "--canonical-letterbox"), "--existing-results", "resume"])


def test_tpu_requires_letterbox_before_any_weight_loads(script, tmp_path):
    prepared(tmp_path, script)
    adapter = adapter_dir(tmp_path)
    with pytest.raises(ValueError, match="canonical-letterbox"):
        script.main(policy_args(tmp_path, adapter, "--no-canonical-letterbox", *TPU_FLAGS))
    assert script.calls.loads == [] and script.calls.hw == []        # stopped before hardware setup and before the weights


def test_tpu_answers_use_the_static_full_canvas_and_refusals_score_wrong(script, tmp_path, monkeypatch):
    prepared(tmp_path, script)
    script.calls.refuse_prompts_containing = "Describe teeth"         # one open question is refused
    ans = ["answer", *common(tmp_path), "--max-new-tokens", "16384", "--image-view", "canonical", "--canonical-letterbox", *TPU_FLAGS]
    script.main([*ans, "--system-name", "qwen3_5_9b_raw"])
    assert script.calls.order == [("require_static_views", "external benchmark (answer)"), ("setup_eval_hardware", True), ("load_eval_model", 16384)]
    assert set(script.calls.images) == {(1536, 768)}                  # letterboxed FULL canvas: the only shape the static vision slots accept
    open_rows = [json.loads(l) for l in (tmp_path / "res/answers/qwen3_5_9b_raw/open.jsonl").read_text().splitlines()]
    refused = [r for r in open_rows if r["generation_refused"]]
    assert len(refused) == 1 and refused[0]["prediction"] == "" and refused[0]["settings"]["tpu_generation"]["chunk_tokens"] == 512
    assert refused[0]["settings"]["image_view"] == "canonical" and refused[0]["settings"]["canonical_letterbox"] is True
    assert script.calls.reports == [str(tmp_path / "res/answers/qwen3_5_9b_raw/tpu_generation_report.json")]
    real_client = script.jg.JudgeClient
    fake = FakeAPI(["0.5"] * 50)
    monkeypatch.setattr(script.jg, "JudgeClient", lambda model, **kw: real_client(model, client=fake))
    script.main(["judge", *common(tmp_path), "--system-name", "qwen3_5_9b_raw", "--judge-model", "judge-x", "--workers", "1"])
    judged = {r["index"]: r for r in (json.loads(l) for l in (tmp_path / "res/judge/qwen3_5_9b_raw__judge-x.jsonl").read_text().splitlines())}
    assert judged[refused[0]["index"]]["score"] == 0.0 and judged[refused[0]["index"]]["tries"] == 0 and len(fake.kwargs) == len(open_rows) - 1   # no API call


def test_tpu_native_view_and_external_models_are_refused(script, tmp_path):
    prepared(tmp_path, script)
    base = ["answer", *common(tmp_path), "--max-new-tokens", "16384", "--canonical-letterbox", *TPU_FLAGS]
    with pytest.raises(ValueError, match="canonical-letterbox"):
        script.main([*base, "--system-name", "a", "--image-view", "native"])                 # native size has no static slot
    with pytest.raises(SystemExit, match="cannot run on the TPU"):
        script.main([*base, "--system-name", "b", "--image-view", "native", "--loader", "external", "--model-id", "OralGPT/OralGPT-Omni-7B-Instruct"])
    script.calls.is_tpu = True                                       # a TPU is detected although no TPU flag was given
    with pytest.raises(SystemExit, match="TPU was detected"):
        script.main(["answer", *common(tmp_path), "--max-new-tokens", "16384", "--system-name", "c", "--image-view", "native", "--loader", "external",
                     "--model-id", "OralGPT/OralGPT-Omni-7B-Instruct"])
    assert script.calls.loads == []


def test_refused_closed_answers_are_wrong_not_random():
    from dental_agent.benchmarks import mmoral_opg as mm, tables as tb
    rows = [dict(index=i, image_id=0, category="Teeth", answer="A", prediction="", generation_refused="GenerationTooLong: x",
                 option1="a", option2="b", option3="c", option4="d") for i in range(40)]
    res = mm.score_closed(rows)
    assert res["refused"] == 40 and res["unreadable"] == 0
    assert {r["category"]: r["acc"] for r in res["official"]["main"]}["Overall"] == 0.0           # 40 random picks would score about 25
    assert all(v[1] == 0.0 for v in tb.closed_vectors(rows, 0, "official").values())


def test_leakage_uses_combined_train_and_stops_on_overlap(script, tmp_path, monkeypatch):
    import numpy as np
    import pandas as pd
    from PIL import Image
    prepared(tmp_path, script)
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


# ------------------------------------------------------------------------------------------- the stubs match the REAL signatures
def _functions(path):
    import ast
    tree = ast.parse((REPO / path).read_text(encoding="utf-8"))
    out = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            a = node.args
            out[node.name] = dict(positional=[x.arg for x in a.posonlyargs + a.args], kwonly=[x.arg for x in a.kwonlyargs], has_varkw=a.kwarg is not None)
    return out


def test_script_calls_match_the_real_tpu_and_evaluator_signatures():
    """The stubs above are only as good as the signatures they imitate. This parses the real modules (no torch needed), so a change in the
    evaluator's TPU functions fails here instead of on the server."""
    ev = _functions("scripts/evaluate_models.py")
    assert ev["setup_eval_hardware"]["positional"][:3] == ["precision", "qlora_compute_dtype", "spmd"]
    load = ev["load_eval_model"]["positional"]
    assert load[:5] == ["model_id", "adapter_path", "device", "dtype", "condition"] and {"quantize_4bit", "runtime_options", "max_new_tokens"} <= set(load)
    assert ev["tpu_generation_settings"]["positional"] == ["args"] and ev["eval_settings"]["positional"] == ["args", "condition", "adapter_path"]
    assert ev["run_condition_trajectory"]["positional"] == ["model", "processor", "images_df", "image_id", "registry", "args"]
    assert {"findings_from_trajectory", "transcript_for_record"} <= set(ev)
    assert "load_env" in (REPO / "scripts/evaluate_models.py").read_text().split("def ")[0]            # imported into its namespace; run_eval_matrix calls ev.load_env()
    ti = _functions("dental_agent/model/tpu_inference.py")
    assert ti["require_static_views"]["kwonly"] == ["canonical_resize", "letterbox", "what"]
    assert ti["note_refusal"]["positional"] == ["model", "image_id", "exc"] and ti["refused_trajectory"]["positional"] == ["image_id", "reason"]
    assert ti["write_generation_report"]["positional"] == ["model", "path"]
    assert "GENERATION_REFUSED" in (REPO / "dental_agent/model/tpu_inference.py").read_text()
    cv = _functions("dental_agent/utils/canonical.py")
    assert cv["to_canonical"]["positional"][:2] == ["image", "family"] and {"cuda", "tpu", "what"} <= set(cv["check_gpu_image_policy"]["kwonly"])
    assert cv["check_gpu_image_policy"]["positional"] == ["canonical_resize", "letterbox"]
    inf = _functions("dental_agent/model/inference.py")
    assert {"max_new_tokens", "temperature", "return_ids"} <= set(inf["generate_agent_reply"]["positional"] + inf["generate_agent_reply"]["kwonly"])
