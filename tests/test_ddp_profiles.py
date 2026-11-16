"""Patch 0069: the older-notebook default (9B when nothing is set), the dual T4 profiles and the notebooks that carry them."""
import glob
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from dental_agent.harness import notebook_eval as ne
from dental_agent.training import ddp
from dental_agent.training.experiment_profiles import PROFILE_TAGS
from dental_agent.utils import model_size as ms

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted(glob.glob(str(ROOT / "notebooks" / "**" / "*.ipynb"), recursive=True))


def _code(path):
    nb = json.load(open(path, encoding="utf-8"))
    return [("".join(c["source"])) for c in nb["cells"] if c["cell_type"] == "code"]


def _strip_magics(src):
    """IPython `!cmd` / `%magic` lines (with backslash continuations) become `pass` at the same indentation."""
    out, skip = [], False
    for l in src.split("\n"):
        if skip or l.lstrip().startswith(("!", "%")):
            if not skip:
                out.append(l[: len(l) - len(l.lstrip())] + "pass")
            skip = l.rstrip().endswith("\\")
        else:
            out.append(l)
    return "\n".join(out)


# ---- an old notebook never sets DENTAL_MODEL_SIZE: everything must stay on 9B ----
def test_unset_env_is_9b_everywhere_in_a_fresh_process():
    env = {k: v for k, v in os.environ.items() if k != ms.SIZE_ENV}
    code = (
        "from dental_agent.utils import inputs, model_size as ms\n"
        "from dental_agent.config import ModelConfig\n"
        "from dental_agent.training.experiment_profiles import grpo_name\n"
        "assert ms.active_size()=='9b' and ms.default_model_id()=='Qwen/Qwen3.5-9B'\n"
        "assert inputs.BASE_MODEL_HUB_ID=='Qwen/Qwen3.5-9B' and inputs.QWEN_FOLDER_NAME=='Qwen3.5-9B' and inputs.LOCAL_BASE_MODEL_ENV=='QWEN35_9B_LOCAL_DIR'\n"
        "assert inputs.sft_adapter_name('with_tools','dentex_alone','fp16')=='qwen3_5_9b_sft_with_tools_dentex_alone_fp16'\n"
        "assert grpo_name('with_tools',4,'dentex_alone','fp16','quadro_rtx8000_fp16').startswith('qwen3_5_9b_grpo_')\n"
        "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True)
    assert out.returncode == 0 and "ok" in out.stdout, out.stderr[-2000:]


def test_empty_env_value_is_9b(monkeypatch):
    monkeypatch.setenv(ms.SIZE_ENV, "")
    assert ms.active_size() == "9b"


def test_invalid_size_is_an_error_not_a_fallback(monkeypatch):
    monkeypatch.setenv(ms.SIZE_ENV, "7b")
    with pytest.raises(ValueError):
        ms.active_size()


# ---- replica fit gate ----
def test_replica_fit_gate():
    ddp.check_replica_fits("fp16", 14.5, "4b")                 # 9.5 GiB + 1 fits a 15 GiB T4
    ddp.check_replica_fits("qlora", 14.5, "9b")                # QLoRA is not gated
    with pytest.raises(SystemExit):
        ddp.check_replica_fits("fp16", 14.5, "9b")             # 18 GiB of weights cannot sit on one T4


# ---- profiles ----
def test_profile_tags_are_separate_cohorts():
    assert PROFILE_TAGS["dual_t4"] == "dual_t4_fp16"
    assert PROFILE_TAGS["dual_t4_fp16"] == "dual_t4_unquantized_fp16"
    assert PROFILE_TAGS["dual_t4_ddp_fp16"] == "dual_t4_ddp_unquantized_fp16"
    assert len(set(PROFILE_TAGS.values())) >= len(set(PROFILE_TAGS.keys())) - 1   # only rtx4090 / rtx4090_qlora share a tag


def test_eval_profile_ddp_matches_the_single_process_runtime():
    a, b = ne.PROFILES["dual_t4_fp16"], ne.PROFILES["dual_t4_ddp_fp16"]
    assert b["tag"] != a["tag"]
    assert {k: v for k, v in a.items() if k != "tag"} == {k: v for k, v in b.items() if k != "tag"}


def test_ddp_hardware_gate_is_per_gpu(monkeypatch):
    t4 = dict(name="T4", capability=(7, 5), gib=15.0)
    monkeypatch.setenv(ms.SIZE_ENV, "4b")
    assert ne.hardware_reason("dual_t4_ddp_fp16", "sft", [t4, t4]) is None
    monkeypatch.setenv(ms.SIZE_ENV, "9b")
    assert ne.hardware_reason("dual_t4_ddp_fp16", "sft", [t4, t4]) is not None     # 30 GiB pooled is not what a replica needs
    assert ne.hardware_reason("dual_t4_fp16", "sft", [t4, t4]) is None            # the layer-split profile pools both GPUs (unchanged)


# ---- notebooks ----
@pytest.mark.parametrize("path", NOTEBOOKS, ids=[Path(p).name for p in NOTEBOOKS])
def test_notebook_code_cells_compile(path):
    for i, src in enumerate(_code(path)):
        try:
            compile(_strip_magics(src), f"{Path(path).name}[{i}]", "exec")
        except SyntaxError as e:
            pytest.fail(f"{Path(path).name} code cell {i}: {e}")


@pytest.mark.parametrize("path", [p for p in NOTEBOOKS if "TraceGen" not in p and "deprecated" not in p], ids=lambda p: Path(p).name)
def test_no_hardcoded_9b_name_in_notebook_code(path):
    for i, src in enumerate(_code(path)):
        for line in src.split("\n"):
            if line.lstrip().startswith("#"):
                continue
            code_part = line.split("  #")[0]
            assert "qwen3_5_9b" not in code_part and "Qwen/Qwen3.5-9B" not in code_part, f"{Path(path).name} cell {i}: {line.strip()[:120]}"


def test_every_eval_table_knows_the_ddp_profile():
    for path in NOTEBOOKS:
        for src in _code(path):
            if '"dual_t4_fp16": {' in src:
                assert '"dual_t4_ddp_fp16": {' in src and '"dual_t4_ddp_unquantized_fp16"' in src, Path(path).name


@pytest.mark.parametrize("name", ["VLM_Dental_Colab_SFT.ipynb", "VLM_Dental_Colab_GRPO.ipynb"])
def test_gpu_training_notebooks_have_both_dual_t4_profiles(name):
    text = "\n".join(_code(ROOT / "notebooks" / name))
    assert '"dual_t4_fp16"' in text and '"dual_t4_ddp_fp16"' in text
    assert "torch.distributed.run" in text and '"--data-parallel"' in text
    assert 'DATA_PARALLEL = WEIGHTS == "replicated" and not IS_TPU' in text      # the profile decides; no separate toggle
    assert "ddp=True" not in text and 'MODEL_SIZE != "4b"' not in text


def test_sft_notebook_halves_accumulation_for_the_ddp_profile():
    text = "\n".join(_code(ROOT / "notebooks" / "VLM_Dental_Colab_SFT.ipynb"))
    assert "GRAD_ACCUM_STEPS //= DDP_WORLD" in text
    assert "LAUNCH_CMD" in text and re.search(r"Popen\(\s*LAUNCH_CMD", text)
    assert re.search(r"preflight_cmd = cmd \+ \[\"--preflight\"\]", text)    # the read-only doctor stays plain python


def test_tpu_notebooks_never_launch_through_torchrun():
    for name in ("TPU_SFT.ipynb", "TPU_GRPO.ipynb"):
        text = "\n".join(_code(ROOT / "notebooks" / "TPU" / name))
        assert 'DATA_PARALLEL = WEIGHTS == "replicated" and not IS_TPU' in text and "torch.distributed.run" not in text


# ---- TPU data-parallel layout (--replicate-weights) ----
def test_replica_gate_for_a_tpu_core():
    ddp.check_replica_fits("bf16", 15.0, "4b")          # about 9.5 GiB + 1 fits a ~15 GiB v5e core
    with pytest.raises(SystemExit):
        ddp.check_replica_fits("bf16", 15.0, "9b")      # 18 GiB of weights cannot be replicated on a core


def _fake_torch_xla(monkeypatch, calls):
    import types
    import torch
    mods = {}
    for name in ("torch_xla", "torch_xla.core", "torch_xla.core.xla_model", "torch_xla.distributed", "torch_xla.distributed.spmd"):
        mods[name] = types.ModuleType(name)
    mods["torch_xla.core.xla_model"].xla_device = lambda: torch.device("cpu")
    mods["torch_xla.distributed.spmd"].mark_sharding = lambda t, mesh, spec: calls.append(spec)
    mods["torch_xla"].core, mods["torch_xla"].distributed = mods["torch_xla.core"], mods["torch_xla.distributed"]
    mods["torch_xla.core"].xla_model, mods["torch_xla.distributed"].spmd = mods["torch_xla.core.xla_model"], mods["torch_xla.distributed.spmd"]
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)


def test_replicated_layout_skips_fsdpv2_and_shards_only_the_batch(monkeypatch):
    import torch
    from dental_agent.training.sft import wrap_spmd_model
    calls = []
    _fake_torch_xla(monkeypatch, calls)               # no FSDPv2 module exists in the fake: touching it would raise
    model = torch.nn.Linear(4, 4)
    out = wrap_spmd_model(model, mesh=object(), is_master=False, replicate_weights=True)
    assert out is model                                # no wrapper, weights stay whole on the device
    out(torch.zeros(2, 3, 4))
    assert calls == [("fsdp", None, None)]             # only the activations carry a sharding: the batch axis


def test_replicate_flag_is_off_by_default_in_every_launcher():
    for script in ("train_sft.py", "run_grpo.py", "run_grpo_sweep.py", "warmup_xla_cache.py"):
        text = (ROOT / "scripts" / script).read_text(encoding="utf-8")
        assert '"--replicate-weights"' in text and "default=False" in text.split('"--replicate-weights"')[1][:120], script


def test_tpu_notebooks_take_the_weight_layout_from_the_profile():
    for name in ("TPU_SFT.ipynb", "TPU_GRPO.ipynb"):
        text = "\n".join(_code(ROOT / "notebooks" / "TPU" / name))
        assert 'REPLICATE_FLAG = ["--replicate-weights"] if (IS_TPU and WEIGHTS == "replicated") else []' in text, name
        assert "TPU_WEIGHT_LAYOUT" not in text and "*REPLICATE_FLAG" in text, name
    assert "*REPLICATE_FLAG" in "\n".join(_code(ROOT / "notebooks" / "TPU" / "TPU_SFT.ipynb")).split("warmup_cmd = [")[1]   # the AOT warmup compiles the same layout


# ---- the profile name is the only selector ----
def test_profile_parts():
    assert ms.profile_parts("tpu") == ("tpu", "9b", "sharded")
    assert ms.profile_parts("dual_t4_fp16_4b") == ("dual_t4_fp16", "4b", "sharded")
    assert ms.profile_parts("tpu_4b") == ("tpu", "4b", "sharded")
    assert ms.profile_parts("tpu_4b_dp") == ("tpu", "4b", "replicated")
    assert ms.profile_parts("dual_t4_ddp_fp16") == ("dual_t4_ddp_fp16", "4b", "replicated")
    assert ms.profile_parts("rtx4090") == ("rtx4090", "9b", "sharded")       # existing names stay 9B


def test_select_profile_exports_the_size_and_a_plain_name_is_always_9b():
    env = {}
    assert ms.select_profile("tpu_4b", env) == ("tpu", "4b", "sharded") and env[ms.SIZE_ENV] == "4b"
    assert ms.select_profile("tpu", env)[1] == "9b" and env[ms.SIZE_ENV] == "9b"     # a stale 4b in the environment cannot leak into a 9B profile


def test_eval_selection_normalises_names_and_refuses_mixed_sizes(monkeypatch):
    monkeypatch.delenv(ms.SIZE_ENV, raising=False)
    assert ne.select_profiles("tpu_4b_dp") == ["tpu"]
    assert ne.select_profiles(["dual_t4_4b", "dual_t4"]) == ["dual_t4"]
    assert ne.select_model_size("dual_t4_fp16_4b") == "4b" and os.environ[ms.SIZE_ENV] == "4b"
    assert ne.select_model_size("quadro_rtx8000") == "9b"
    assert ne.select_model_size("all") is None
    with pytest.raises(ValueError):
        ne.select_model_size(["tpu", "tpu_4b"])
    with pytest.raises(ValueError):
        ne.select_profiles("tpu_7b")
    monkeypatch.delenv(ms.SIZE_ENV, raising=False)


@pytest.mark.parametrize("path", [p for p in NOTEBOOKS if "TraceGen" not in p and "deprecated" not in p and "YOLO" not in p], ids=lambda p: Path(p).name)
def test_no_notebook_has_a_separate_model_size_switch(path):
    text = "\n".join(_code(path))
    assert 'MODEL_SIZE = "9b"' not in text and 'MODEL_SIZE = "4b"' not in text and "set_active_size(MODEL_SIZE)" not in text


@pytest.mark.parametrize("name", ["VLM_Dental_Colab_SFT.ipynb", "TPU/TPU_SFT.ipynb", "VLM_Dental_Colab_GRPO.ipynb", "TPU/TPU_GRPO.ipynb"])
def test_training_notebooks_select_the_profile_before_the_model_is_resolved(name):
    for src in _code(ROOT / "notebooks" / name):
        if "resolve_qwen_model(" in src and "FORCE_HF_MODEL_DOWNLOAD = False" in src:
            assert src.index("select_profile(HARDWARE_PROFILE)") < src.index("= resolve_qwen_model(")
            return
    pytest.fail("no model-resolution cell")
