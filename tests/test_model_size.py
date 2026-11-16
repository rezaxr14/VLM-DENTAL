"""DENTAL_MODEL_SIZE: 9b stays the default; 4b changes every name that identifies the model, and nothing crosses sizes."""
import json
import pytest

from dental_agent.utils import inputs, model_size as ms
from dental_agent.training.experiment_profiles import grpo_name


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv(ms.SIZE_ENV, raising=False)


def _size(monkeypatch, value):
    monkeypatch.setenv(ms.SIZE_ENV, value)


def test_default_is_9b_and_unchanged():
    assert ms.active_size() == "9b"
    assert inputs.BASE_MODEL_HUB_ID == "Qwen/Qwen3.5-9B"
    assert inputs.QWEN_FOLDER_NAME == "Qwen3.5-9B"
    assert inputs.LOCAL_BASE_MODEL_ENV == "QWEN35_9B_LOCAL_DIR"
    assert inputs.sft_adapter_name("with_tools", "dentex_alone", "qlora") == "qwen3_5_9b_sft_with_tools_dentex_alone_qlora"
    assert grpo_name("with_tools", 4, "dentex_alone", "qlora", "dual_t4_fp16") == "qwen3_5_9b_grpo_with_tools_k4_dentex_alone_qlora__dual_t4_fp16"


def test_4b_names(monkeypatch):
    _size(monkeypatch, "4b")
    assert inputs.BASE_MODEL_HUB_ID == "Qwen/Qwen3.5-4B"
    assert inputs.QWEN_FOLDER_NAME == "Qwen3.5-4B"
    assert inputs.LOCAL_BASE_MODEL_ENV == "QWEN35_4B_LOCAL_DIR"
    assert inputs.sft_adapter_name("with_tools", "x", "fp16") == "qwen3_5_4b_sft_with_tools_x_fp16"
    assert grpo_name("with_tools", 8, "x", "fp16", "dual_t4_unquantized_fp16").startswith("qwen3_5_4b_grpo_with_tools_k8_x_fp16__")


def test_unknown_size_raises(monkeypatch):
    _size(monkeypatch, "7b")
    with pytest.raises(ValueError):
        ms.active_size()


def test_4b_never_picks_up_a_9b_adapter_folder(monkeypatch):
    _size(monkeypatch, "4b")
    cands = [str(c) for c in inputs.sft_adapter_candidates("with_tools", "x", "fp16")]
    assert cands and all("9b" not in c for c in cands)
    assert not any(c.startswith("checkpoints/") for c in cands)          # the size-less legacy folder is 9B only
    monkeypatch.delenv(ms.SIZE_ENV)
    assert any(str(c).startswith("checkpoints/") for c in inputs.sft_adapter_candidates("with_tools", "x", "qlora"))


def _cfg(tmp_path, hidden):
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"hidden_size": hidden}}))
    (tmp_path / "tokenizer.json").write_text("{}")
    (tmp_path / "preprocessor_config.json").write_text("{}")
    (tmp_path / "model.safetensors").write_bytes(b"x")
    return tmp_path


def test_folder_of_the_other_size_is_refused(monkeypatch, tmp_path):
    folder = _cfg(tmp_path, 4096)                      # a 9B folder
    assert inputs.qwen_folder_problems(folder) == []
    _size(monkeypatch, "4b")
    problems = inputs.qwen_folder_problems(folder)
    assert problems and "9b" in problems[0] and "4b" in problems[0]
    monkeypatch.setenv("QWEN35_4B_LOCAL_DIR", str(folder))
    with pytest.raises(inputs.InputNotFound):
        inputs.resolve_qwen_model(download=False, say=lambda *_: None)


def test_4b_local_dir_env_is_its_own(monkeypatch, tmp_path):
    folder = _cfg(tmp_path, 2560)
    monkeypatch.setenv("QWEN35_9B_LOCAL_DIR", str(tmp_path / "does-not-matter"))   # a 9B setting must not leak into a 4B run
    _size(monkeypatch, "4b")
    monkeypatch.setenv("QWEN35_4B_LOCAL_DIR", str(folder))
    assert inputs.resolve_qwen_model(download=False, say=lambda *_: None) == (str(folder), "env")
    assert inputs.local_model_alias("Qwen/Qwen3.5-4B") == str(folder)
    assert inputs.local_model_alias("Qwen/Qwen3.5-9B") == "Qwen/Qwen3.5-9B"


def test_zero_shot_file_tag_follows_size(monkeypatch, tmp_path):
    folder = _cfg(tmp_path, 2560)
    _size(monkeypatch, "4b")
    monkeypatch.setenv("QWEN35_4B_LOCAL_DIR", str(folder))
    assert inputs.model_file_tag(str(folder)) == "Qwen--Qwen3.5-4B"


def test_eval_discovery_only_sees_the_active_size(monkeypatch):
    from dental_agent.harness import notebook_eval as ne
    files = {}
    for slug in ("qwen3_5_9b", "qwen3_5_4b"):
        base = f"sft/{slug}_sft_with_tools_all_fp16__dual_t4_unquantized_fp16"
        files[base + "/adapter_config.json"] = 1
        files[base + "/adapter_model.safetensors"] = 1
    profiles = ne.PROFILES
    nine = ne.discover(files, ["dual_t4_fp16"], profile_configs=profiles)
    assert [j["adapter"] for j in nine] == ["sft/qwen3_5_9b_sft_with_tools_all_fp16__dual_t4_unquantized_fp16"]
    _size(monkeypatch, "4b")
    four = ne.discover(files, ["dual_t4_fp16"], profile_configs=profiles)
    assert [j["adapter"] for j in four] == ["sft/qwen3_5_4b_sft_with_tools_all_fp16__dual_t4_unquantized_fp16"]


def test_memory_guard_uses_the_size_footprint(monkeypatch):
    from dental_agent.harness import notebook_eval as ne
    one_t4 = [dict(name="T4", gib=15.0, capability=(7, 5))]
    assert "pending" in ne.hardware_reason("dual_t4_fp16", "sft", one_t4)          # 9B fp16 needs about 18 GiB
    _size(monkeypatch, "4b")
    assert ne.hardware_reason("dual_t4_fp16", "sft", one_t4) is None               # 4B fp16 (about 9.5 GiB estimate) passes the guard


def test_grpo_activation_estimate_scales_with_hidden_size(monkeypatch):
    from dental_agent.model.backbone import estimate_grpo_memory
    a = estimate_grpo_memory(group_size=8, seq_len=16384)["activations_gb"] if "activations_gb" in estimate_grpo_memory() else None
    _size(monkeypatch, "4b")
    b = estimate_grpo_memory(group_size=8, seq_len=16384)["activations_gb"] if "activations_gb" in estimate_grpo_memory() else None
    if a is not None:
        assert b == pytest.approx(a * 2560 / 4096, rel=0.02)


def test_config_shapes_in_specs_match_published_configs():
    assert (ms.SPECS["9b"]["text_hidden"], ms.SPECS["9b"]["text_intermediate"], ms.SPECS["9b"]["vision_depth"]) == (4096, 12288, 27)
    assert (ms.SPECS["4b"]["text_hidden"], ms.SPECS["4b"]["text_intermediate"], ms.SPECS["4b"]["vision_depth"]) == (2560, 9216, 24)
