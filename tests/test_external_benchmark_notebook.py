"""The External Benchmarks notebook: valid JSON, compiling cells, the shared EVAL_PROFILES block, and every command its cells build is
accepted by the real argument parser of scripts/run_external_benchmark.py (the cells are executed with run_cmd replaced by a recorder)."""
import importlib
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tests.test_external_benchmarks import synthetic_tsvs  # noqa: E402

NOTEBOOK = REPO / "notebooks" / "VLM_Dental_Colab_External_Benchmarks.ipynb"


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def code_cells(nb):
    return [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


def strip_magics(src):
    return "\n".join((re.match(r"\s*", l).group() + "pass") if l.lstrip().startswith(("!", "%")) else l for l in src.split("\n"))


def profiles_of(nb):
    src = next(s for s in code_cells(nb) if "EVAL_PROFILES = {" in s)
    start = src.index("EVAL_PROFILES = {")
    end = src.index("\n}\n", start) + 2
    ns = {}
    exec(src[start:end], ns)
    return ns["EVAL_PROFILES"]


def test_notebook_is_valid_and_cells_compile():
    nb = load(NOTEBOOK)
    assert nb["nbformat"] == 4 and len(nb["cells"]) >= 40
    for i, src in enumerate(code_cells(nb), 1):
        compile(strip_magics(src), f"code cell {i}", "exec")
    assert all(c["outputs"] == [] and c["execution_count"] is None for c in nb["cells"] if c["cell_type"] == "code")      # no stored outputs


def test_eval_profiles_identical_to_final_evaluation():
    assert profiles_of(load(NOTEBOOK)) == profiles_of(load(REPO / "notebooks" / "VLM_Dental_Colab_Final_Evaluation.ipynb"))


def test_visible_settings_have_no_hidden_defaults():
    settings = next(s for s in code_cells(load(NOTEBOOK)) if "HARDWARE_PROFILE =" in s)
    assert "MAX_NEW_TOKENS = generation_budget(HARDWARE_PROFILE)" in settings and "JUDGE_MODEL = None" in settings and "MAX_TOOL_CALLS = 50" in settings
    assert "EXTERNAL_PRECISION = PROFILE[\"precision\"]" in settings
    assert 'EXISTING_RESULTS = "resume"' in settings and "POLICY_HARNESS_MODE = \"rules_skills_index\"" in settings


@pytest.fixture()
def script(monkeypatch):
    try:
        import torch  # noqa: F401
    except ImportError:
        monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    return importlib.import_module("scripts.run_external_benchmark")


@pytest.mark.parametrize("profile", ["quadro_rtx8000", "workstation48_lora", "tpu"])
def test_every_command_the_cells_build_is_accepted_by_the_script(script, tmp_path, monkeypatch, profile):
    pytest.importorskip("IPython")
    try:
        import dental_agent.training.experiment_profiles as ep
    except ImportError:                      # no torch here: load the pure-Python naming module by path
        import importlib.util
        spec = importlib.util.spec_from_file_location("dental_agent.training.experiment_profiles", REPO / "dental_agent/training/experiment_profiles.py")
        ep = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ep)
        monkeypatch.setitem(sys.modules, "dental_agent.training.experiment_profiles", ep)
    cfg = profiles_of(load(NOTEBOOK))[profile]
    tag, prec = cfg["tag"], cfg["precision"]
    data = tmp_path / "data"
    monkeypatch.setenv("DENTAL_AGENT_DATA_DIR", str(data))
    monkeypatch.setenv("HF_ARTIFACT_REPO", "repo/models")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.chdir(tmp_path)
    import dental_agent.utils.inputs as inputs
    monkeypatch.setattr(inputs, "resolve_qwen_model", lambda **kw: (str(tmp_path / "Qwen3.5-9B"), "env"))      # the real one may download 19 GB

    # a prepared benchmark, a fake policy checkpoint and policy findings, exactly where the cells look for them
    raw = data / "benchmarks" / "mmoral_opg" / "raw"
    raw.mkdir(parents=True)
    synthetic_tsvs(raw)
    from dental_agent.benchmarks import mmoral_opg as mm
    mm.prepare(raw, data / "benchmarks" / "mmoral_opg" / "prepared", allow_md5_mismatch=True)
    name = ep.grpo_name("with_tools", 4, "dentex_alone", prec, tag, "rules_skills_index")
    pol = data / "models" / name / "grpo-with_tools-final" / "grpo_policy"
    pol.mkdir(parents=True)
    (pol / "adapter_config.json").write_text("{}")
    (pol / "adapter_model.safetensors").write_bytes(b"w")
    results = data / "evaluations" / "external" / "mmoral_opg" / tag
    (results / "tables").mkdir(parents=True)
    (results / "tables" / "mmoral_opg.md").write_text("| table |")
    for system in ("qwen3_5_9b_raw", "optionA_grpo_k4_harness_rules_skills_index", "oralgpt_omni_7b", "oralgpt_opg_7b"):
        (results / "answers" / system).mkdir(parents=True)
        (results / "answers" / system / "open.jsonl").write_text("")
        (results / "answers" / system / "closed.jsonl").write_text("")

    captured = []
    parser = script.build_parser()

    def fake_run_cmd(cmd, check=True):
        cmd = [str(c) for c in cmd]
        assert cmd[1] == "scripts/run_external_benchmark.py"
        captured.append((cmd[2], parser.parse_args(cmd[2:])))          # a flag the script does not know raises SystemExit here
        return 0

    nb = load(NOTEBOOK)
    ns = {"__name__": "__notebook__"}
    for src in code_cells(nb):
        if "REPO_URL" in src or "AUDIT = [" in src or "TOKEN_SPECS" in src:
            continue                                                   # clone, dependency audit, secrets: environment, not commands
        assert "pip install" not in src                                # the notebook installs nothing (requirements.txt pins everything)
        if "tpu_device_entry" in src:                                 # the hardware cell: what the machine is is a property of the test
            ns["IS_TPU"] = profile == "tpu"
            ns["DEVICES"] = [dict(name="TPU v5e-8", gib=0.0, capability=(0, 0), tpu=True)] if profile == "tpu" else []
            continue
        if "ACCESS_CHECKS" in src:
            ns["GATE_STATUS"] = {"OralGPT/OralGPT-Omni-7B-Instruct": True, "OralGPT/OralGPT-OPG-7B-Instruct": False}
            continue
        if "def get_secret_multisource" in src:
            continue
        if "HARDWARE_PROFILE =" in src:
            src, n = re.subn(r'^HARDWARE_PROFILE = .*$', f'HARDWARE_PROFILE = "{profile}"', src, count=1, flags=re.M)
            assert n == 1
        ns.setdefault("os", os)
        ns.setdefault("sys", sys)
        ns.setdefault("Path", Path)
        ns.setdefault("IS_KAGGLE", False)
        exec(compile(strip_magics(src), "notebook cell", "exec"), ns)
        if "def run_cmd" in src:
            ns["run_cmd"] = fake_run_cmd                              # after the settings cell defined the real one
            ns["JUDGE_MODEL"] = "judge-x"
            ns["tabulate"] = __import__("tabulate").tabulate
            ns["LIMIT_FLAGS"] = []

    assert ns["TAG"] == tag and ns["RESULTS_DIR"].endswith(f"/external/mmoral_opg/{tag}")
    tpu = profile == "tpu"
    by_sub = {}
    for sub, args in captured:
        by_sub.setdefault(sub, []).append(args)
    assert [s for s, _ in captured].count("prepare") == 1 and by_sub["prepare"][0].download is True
    pol_args = by_sub["policy"][0]
    budget = 24576 if profile == "tpu" else 16384              # the repository's own generation_budget(profile)
    assert pol_args.harness_mode == "rules_skills_index" and pol_args.max_new_tokens == budget and pol_args.max_tool_calls == 50
    assert pol_args.canonical_letterbox is cfg["letterbox"] and pol_args.adapter_path == str(pol)       # the exact checkpoint of THIS profile
    assert pol_args.precision == prec and pol_args.attn_implementation == cfg["attention"]
    assert pol_args.existing_results == "resume"
    # the TPU flags are passed exactly when the profile is the TPU one (they are part of every result's identity)
    assert (pol_args.tpu_generation_chunk_tokens, list(pol_args.tpu_generation_tiers), list(pol_args.tpu_vision_slots)) == \
        ((512, [32768, 49152], [8, 16, 6]) if tpu else (0, [32768, 49152], [8, 16, 6]))
    answers = {a.system_name: a for a in by_sub["answer"]}
    raw, opt_a = answers["qwen3_5_9b_raw"], answers["optionA_grpo_k4_harness_rules_skills_index"]
    assert raw.findings_from is None and raw.loader == "repo" and raw.precision == prec
    assert raw.image_view == "canonical" and raw.canonical_letterbox is cfg["letterbox"]                # canonical view; letterbox as the profile
    assert opt_a.findings_from == "grpo_k4_harness_rules_skills_index" and opt_a.image_view == "canonical"
    assert raw.tpu_generation_chunk_tokens == pol_args.tpu_generation_chunk_tokens
    if tpu:
        assert set(answers) == {"qwen3_5_9b_raw", "optionA_grpo_k4_harness_rules_skills_index"}         # public models are never run on a TPU
    else:
        omni = answers["oralgpt_omni_7b"]
        assert omni.loader == "external" and omni.model_id == "OralGPT/OralGPT-Omni-7B-Instruct" and omni.precision == prec
        assert omni.image_view == "native" and omni.tpu_generation_chunk_tokens == 0
        assert "oralgpt_opg_7b" not in answers                           # not accessible in the audit: skipped, never run
    assert all(a.max_new_tokens == budget for a in answers.values())
    judges = {a.system_name: a for a in by_sub["judge"]}
    assert set(judges) == {"qwen3_5_9b_raw", "optionA_grpo_k4_harness_rules_skills_index", "oralgpt_omni_7b", "oralgpt_opg_7b"}   # one per system with open answers
    assert pol_args.model_id == str(tmp_path / "Qwen3.5-9B") and os.environ["QWEN35_9B_LOCAL_DIR"] == pol_args.model_id
    assert all(j.judge_model == "judge-x" and j.judge_send_temperature is True for j in judges.values())
    assert by_sub["score"][0].published.endswith("published_numbers.json") and by_sub["score"][0].judge_model == "judge-x"
    assert {(c.a, c.b) for c in by_sub["compare"]} >= {("optionA_grpo_k4_harness_rules_skills_index", "qwen3_5_9b_raw")}


def test_notebook_passes_nbformat_validation():
    nbformat = pytest.importorskip("nbformat")
    nbformat.validate(nbformat.read(NOTEBOOK, as_version=4))
