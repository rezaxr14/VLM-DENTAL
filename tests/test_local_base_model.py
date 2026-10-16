"""A local copy of Qwen3.5-9B (QWEN35_9B_LOCAL_DIR) is found, validated, wired through the notebooks, and works with every network
call refused: the processor/tokenizer load, SFT, GRPO (dual adapters), the token-length manifest and both preflights.

The offline tests run the real scripts in subprocesses with a sitecustomize that makes every non-loopback connection fail the way a
disconnected machine does and writes each attempt to a log; the assertion is that nothing but ultralytics' own import-time
connectivity probe (dns.google / one.one.one.one, not a download) tried to reach the network."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dental_agent.utils import inputs as inp  # noqa: E402

ENV = inp.LOCAL_BASE_MODEL_ENV


def model_folder(root: Path, *, config=True, shards=2, tokenizer=True, processor=True, index=True, missing_shard=False, empty_shard=False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    if config:
        (root / "config.json").write_text("{}")
    names = [f"model-{i + 1:05d}-of-{shards:05d}.safetensors" for i in range(shards)] if shards > 1 else (["model.safetensors"] if shards else [])
    for k, n in enumerate(names):
        if missing_shard and k == len(names) - 1:
            continue
        (root / n).write_bytes(b"" if (empty_shard and k == len(names) - 1) else b"x")
    if index and shards > 1:
        (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {f"w{i}": n for i, n in enumerate(names)}}))
    if tokenizer:
        (root / "tokenizer.json").write_text("{}")
    if processor:
        (root / "preprocessor_config.json").write_text("{}")
    return root


# ---------------------------------------------------------------------------------------------------------------------
# Validity of a folder
# ---------------------------------------------------------------------------------------------------------------------
def test_complete_folders_have_no_problems_sharded_or_single_file(tmp_path):
    assert inp.qwen_folder_problems(model_folder(tmp_path / "sharded")) == []
    assert inp.qwen_folder_problems(model_folder(tmp_path / "single", shards=1)) == []


def test_each_defect_is_reported(tmp_path):
    assert "is not a folder" in inp.qwen_folder_problems(tmp_path / "nope")[0]
    assert inp.qwen_folder_problems(model_folder(tmp_path / "a", config=False)) == ["no config.json"]
    assert inp.qwen_folder_problems(model_folder(tmp_path / "b", tokenizer=False)) == ["no tokenizer.json / tokenizer_config.json"]
    assert inp.qwen_folder_problems(model_folder(tmp_path / "c", processor=False)) == ["no preprocessor_config.json / processor_config.json"]
    assert inp.qwen_folder_problems(model_folder(tmp_path / "d", shards=0, index=False)) == ["no model.safetensors (and no model.safetensors.index.json)"]
    (problem,) = inp.qwen_folder_problems(model_folder(tmp_path / "e", missing_shard=True))               # an interrupted copy
    assert "1 of 2 weight shards are missing or empty" in problem and "model-00002-of-00002.safetensors" in problem
    (problem,) = inp.qwen_folder_problems(model_folder(tmp_path / "f", empty_shard=True))                 # a truncated 0-byte shard
    assert "1 of 2 weight shards are missing or empty" in problem


def test_bin_weights_do_not_count(tmp_path):
    d = model_folder(tmp_path / "m", shards=0, index=False)
    (d / "pytorch_model.bin").write_bytes(b"x")
    assert inp.qwen_folder_problems(d) == ["no model.safetensors (and no model.safetensors.index.json)"]


# ---------------------------------------------------------------------------------------------------------------------
# resolve_qwen_model(): candidates, priority, download
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture
def world(tmp_path, monkeypatch):
    """No Kaggle, no Drive, no HF cache; a tmp repo root; the working directory is somewhere else. Downloads are mocked."""
    repo, elsewhere = tmp_path / "repo", tmp_path / "elsewhere"
    repo.mkdir()
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    kaggle = tmp_path / "kaggle" / "input"
    kaggle.mkdir(parents=True)
    monkeypatch.setattr(inp, "REPO_ROOT", repo)
    monkeypatch.setattr(inp, "KAGGLE_INPUT", kaggle)
    monkeypatch.setattr(inp, "KAGGLE_WORKING", tmp_path / "kaggle" / "working")
    monkeypatch.setattr(inp, "KAGGLE_BASE_MODEL_DIRS", [str(kaggle / "qwen3-5-9b")])
    monkeypatch.setattr(inp, "_colab_models_dir", lambda: None)
    monkeypatch.setattr(inp.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    for var in (ENV, "HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HUB_OFFLINE"):
        monkeypatch.delenv(var, raising=False)
    calls = []

    def fake_snapshot(repo_id, local_dir):
        calls.append((repo_id, Path(local_dir)))
        model_folder(Path(local_dir))                                                          # what a finished download leaves
        return str(local_dir)

    monkeypatch.setattr(inp, "hub_snapshot_model", fake_snapshot)
    return types.SimpleNamespace(repo=repo, kaggle=kaggle, tmp=tmp_path, downloads=calls, canonical=repo / "data" / "models" / "Qwen3.5-9B",
                                 hf_snapshot=tmp_path / "home" / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots" / "abc123")


def resolve(**kw):
    lines = []
    out = inp.resolve_qwen_model(say=lines.append, **kw)
    return out, lines


def test_canonical_folder_comes_from_the_repo_not_from_the_working_directory(world):
    assert inp.qwen_canonical_dir() == world.canonical and os.getcwd() != str(world.repo)
    model_folder(world.canonical)
    (path, source), lines = resolve()
    assert (path, source) == (str(world.canonical), "repo-data")
    assert lines == [f"Qwen3.5-9B source: repo-data -> {world.canonical}"]


def test_priority_env_then_kaggle_then_colab_then_repo_then_hf_cache(world, monkeypatch):
    env_dir = model_folder(world.tmp / "my_copy")
    colab = model_folder(world.tmp / "drive" / "Qwen3.5-9B")
    monkeypatch.setattr(inp, "_colab_models_dir", lambda: colab)
    monkeypatch.setenv(ENV, str(env_dir))
    for d in (world.kaggle / "qwen3-5-9b", world.canonical, world.hf_snapshot):
        model_folder(d)
    order = [(str(env_dir), "env"), (str(world.kaggle / "qwen3-5-9b"), "kaggle-input"), (str(colab), "colab-drive"),
             (str(world.canonical), "repo-data"), (str(world.hf_snapshot), "hf-cache")]
    for expected in order:
        assert resolve()[0] == expected
        shutil.rmtree(expected[0])                                                              # take the winner away: the next one wins
        monkeypatch.delenv(ENV, raising=False)
    assert world.downloads == []


def test_env_folder_that_is_unusable_is_an_error_even_when_other_copies_exist(world, monkeypatch):
    model_folder(world.canonical)
    monkeypatch.setenv(ENV, str(world.tmp / "typo"))
    with pytest.raises(inp.InputNotFound, match=rf"(?s){ENV}=.*is not a folder"):
        resolve()
    monkeypatch.setenv(ENV, str(model_folder(world.tmp / "half", missing_shard=True)))
    with pytest.raises(inp.InputNotFound, match="weight shards are missing or empty"):
        resolve()
    assert world.downloads == []                                                                # a typo never becomes a download


def test_kaggle_input_is_used_in_place_read_only_and_never_copied(world):
    d = model_folder(world.kaggle / "qwen3-5-9b")

    def snapshot(root):
        return sorted((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in root.rglob("*"))

    before, tmp_before = snapshot(d), snapshot(world.tmp)
    for p in [d, *d.iterdir()]:
        p.chmod(0o555 if p.is_dir() else 0o444)                                                 # a Kaggle input is read-only
    try:
        assert resolve()[0] == (str(d), "kaggle-input")
    finally:
        for p in [d, *d.iterdir()]:
            p.chmod(0o755 if p.is_dir() else 0o644)
    assert snapshot(d) == before and snapshot(world.tmp) == tmp_before                         # nothing written, nothing copied
    assert not world.canonical.exists() and world.downloads == []


def test_partial_kaggle_input_is_skipped_with_a_reason_and_the_next_candidate_wins(world):
    model_folder(world.kaggle / "qwen3-5-9b", missing_shard=True)
    model_folder(world.canonical)
    assert resolve()[0] == (str(world.canonical), "repo-data")


def test_kaggle_models_layout_is_found_by_its_name(world):
    d = model_folder(world.kaggle / "models" / "rezanadimikj" / "Qwen3.5-9B" / "transformers" / "default" / "1")
    model_folder(world.kaggle / "some-adapter-dataset", shards=1)                               # unrelated folder: never picked
    assert resolve()[0] == (str(d), "kaggle-input")


def test_hf_cache_snapshot_is_used_in_place_incl_the_vllm_cache_of_the_repo(world):
    model_folder(world.repo / "data" / "models" / "vllm_cache" / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots" / "v1")
    (path, source), _ = resolve()
    assert source == "hf-cache" and path.endswith("vllm_cache/hub/models--Qwen--Qwen3.5-9B/snapshots/v1")
    assert not world.canonical.exists() and world.downloads == []


def test_hf_home_and_hf_hub_cache_variables_are_searched(world, monkeypatch):
    monkeypatch.setenv("HF_HOME", str(world.tmp / "hfhome"))
    snap = model_folder(world.tmp / "hfhome" / "hub" / "models--Qwen--Qwen3.5-9B" / "snapshots" / "x")
    assert resolve()[0] == (str(snap), "hf-cache")


def test_nothing_found_downloads_into_the_canonical_folder_once(world):
    (path, source), lines = resolve()
    assert (path, source) == (str(world.canonical), "downloaded")
    assert world.downloads == [("Qwen/Qwen3.5-9B", world.canonical)]
    assert any("downloading to" in line for line in lines)
    assert resolve()[0] == (str(world.canonical), "repo-data")                                  # the next run finds it, no second download
    assert len(world.downloads) == 1


def test_partial_canonical_folder_resumes_the_download_in_place(world):
    model_folder(world.canonical, missing_shard=True)
    (path, source), lines = resolve()
    assert source == "downloaded" and world.downloads == [("Qwen/Qwen3.5-9B", world.canonical)]
    assert any("resuming the partial download" in line for line in lines)


def test_force_download_skips_every_existing_copy(world):
    model_folder(world.canonical)
    model_folder(world.kaggle / "qwen3-5-9b")
    assert resolve(force_download=True)[0][1] == "downloaded" and len(world.downloads) == 1


def test_offline_or_no_download_flag_raises_a_clear_error_naming_where_to_put_the_model(world, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    with pytest.raises(inp.InputNotFound) as e:
        resolve()
    msg = str(e.value)
    assert str(world.canonical) in msg and "HF_HUB_OFFLINE=1" in msg and "QWEN35_9B_LOCAL_DIR" in msg
    monkeypatch.delenv("HF_HUB_OFFLINE")
    with pytest.raises(inp.InputNotFound, match="place the model in"):
        resolve(download=False)
    assert world.downloads == []


def test_download_failure_and_incomplete_download_are_reported_not_hidden(world, monkeypatch):
    def no_network(repo_id, local_dir):
        raise ConnectionError("Network is unreachable")

    monkeypatch.setattr(inp, "hub_snapshot_model", no_network)
    with pytest.raises(inp.InputNotFound, match=r"(?s)download of Qwen/Qwen3.5-9B failed: ConnectionError.*it resumes"):
        resolve()

    def truncated(repo_id, local_dir):
        model_folder(Path(local_dir), missing_shard=True)
        return str(local_dir)

    monkeypatch.setattr(inp, "hub_snapshot_model", truncated)
    with pytest.raises(inp.InputNotFound, match="weight shards are missing or empty"):
        resolve()


def test_unwritable_repo_on_kaggle_downloads_to_working_and_warns(world):
    (world.repo / "data").mkdir()
    (world.repo / "data" / "models").write_text("a file where the folder should be")            # cannot create data/models/Qwen3.5-9B
    world.tmp.joinpath("kaggle", "working").mkdir()
    (path, source), lines = resolve()
    target = world.tmp / "kaggle" / "working" / "models" / "Qwen3.5-9B"
    assert (path, source) == (str(target), "downloaded") and world.downloads == [("Qwen/Qwen3.5-9B", target)]
    assert any("19 GB" in line and "working-disk limit" in line for line in lines)


# ---------------------------------------------------------------------------------------------------------------------
# The environment variable as a path-only alias (zero-shot provider, cached zero-shot file name)
# ---------------------------------------------------------------------------------------------------------------------
def test_env_unset_or_blank_means_not_requested():
    assert inp.local_base_model_dir({}) is None
    assert inp.local_base_model_dir({ENV: "   "}) is None


def test_env_valid_returns_the_folder(tmp_path):
    d = model_folder(tmp_path / "Qwen3.5-9B")
    assert inp.local_base_model_dir({ENV: str(d)}) == d


def test_env_set_but_invalid_raises_never_none(tmp_path):
    for bad in (tmp_path / "typo", model_folder(tmp_path / "half", missing_shard=True)):
        with pytest.raises(inp.InputNotFound) as e:
            inp.local_base_model_dir({ENV: str(bad)})
        assert ENV in str(e.value) and "unset it" in str(e.value)


def test_alias_changes_only_the_base_model_id_and_keeps_the_hub_id_for_everything_else(tmp_path):
    d = model_folder(tmp_path / "m")
    env = {ENV: str(d)}
    assert inp.local_model_alias("Qwen/Qwen3.5-9B", env) == str(d)
    assert inp.local_model_alias("Qwen/Qwen2.5-VL-7B-Instruct", env) == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert inp.local_model_alias("Qwen/Qwen3.5-9B", {}) == "Qwen/Qwen3.5-9B"
    with pytest.raises(inp.InputNotFound):
        inp.local_model_alias("Qwen/Qwen3.5-9B", {ENV: str(tmp_path / "typo")})                 # no silent fall back to a download
    assert inp.local_model_alias("some/other-model", {ENV: str(tmp_path / "typo")}) == "some/other-model"   # unrelated ids never fail on it


def test_cached_zero_shot_file_name_is_the_same_for_the_hub_id_and_the_local_folder(tmp_path):
    d = model_folder(tmp_path / "Qwen3.5-9B")
    env = {ENV: str(d)}
    assert inp.model_file_tag("Qwen/Qwen3.5-9B", env) == "Qwen--Qwen3.5-9B"
    assert inp.model_file_tag(str(d), env) == "Qwen--Qwen3.5-9B"                                # --reuse-cached finds the zero-shot file
    other = model_folder(tmp_path / "other")
    assert inp.model_file_tag(str(other), env) == str(other).replace("/", "--").replace(":", "-")
    assert inp.model_file_tag(str(d), {ENV: str(tmp_path / "typo")}) == str(d).replace("/", "--").replace(":", "-")   # never raises here


def test_zero_shot_provider_loads_the_local_folder_while_the_model_name_stays_the_hub_id(tmp_path, monkeypatch):
    from dental_agent.training import api_pool

    d = model_folder(tmp_path / "m")
    seen = {}

    def fake_load(cfg, device_map="auto"):
        seen["name"] = cfg.name
        return types.SimpleNamespace(eval=lambda: None), object()

    monkeypatch.setattr("dental_agent.model.backbone.load_model", fake_load)
    monkeypatch.setenv(ENV, str(d))
    pool = api_pool._POOL
    pool._transformers_pipelines.pop("Qwen/Qwen3.5-9B", None)
    try:
        pool.get_transformers_pipeline("Qwen/Qwen3.5-9B")
        assert seen["name"] == str(d)
        assert "Qwen/Qwen3.5-9B" in pool._transformers_pipelines                                # memoised under the model NAME
    finally:
        pool._transformers_pipelines.pop("Qwen/Qwen3.5-9B", None)


# ---------------------------------------------------------------------------------------------------------------------
# The notebooks, executed from their real cells
# ---------------------------------------------------------------------------------------------------------------------
def code_of(nb: str, needle: str) -> str:
    cells = json.loads((ROOT / "notebooks" / nb).read_text(encoding="utf-8"))["cells"]
    return next("".join(c["source"]) for c in cells if c["cell_type"] == "code" and needle in "".join(c["source"]))


def code_cells(nb: str) -> list[str]:
    return ["".join(c["source"]) for c in json.loads((ROOT / "notebooks" / nb).read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def model_block(nb: str) -> str:
    src = code_of(nb, "resolve_qwen_model")
    start = re.search(r"^FORCE_HF_MODEL_DOWNLOAD = ", src, flags=re.M).start()
    return src[start: src.index("os.environ[LOCAL_BASE_MODEL_ENV] = MODEL_ID")] + "os.environ[LOCAL_BASE_MODEL_ENV] = MODEL_ID\n"


@pytest.mark.parametrize("nb", ["VLM_Dental_Colab_SFT.ipynb", "VLM_Dental_Colab_GRPO.ipynb"])
def test_sft_and_grpo_config_cells_resolve_one_folder_print_its_source_and_export_it(nb, world, monkeypatch, capsys):
    folder = model_folder(world.tmp / "D_models" / "Qwen3.5-9B")
    monkeypatch.setenv(ENV, str(folder))
    ns = {"os": os}
    exec(model_block(nb), ns)
    assert ns["MODEL_ID"] == str(folder) and ns["MODEL_SOURCE"] == "env" and os.environ[ENV] == str(folder)
    assert f"Qwen3.5-9B source: env -> {folder}" in capsys.readouterr().out


@pytest.mark.parametrize("nb", ["VLM_Dental_Colab_SFT.ipynb", "VLM_Dental_Colab_GRPO.ipynb"])
def test_config_cell_with_a_kaggle_input_still_uses_it_and_with_nothing_downloads_to_the_canonical_folder(nb, world, monkeypatch):
    kaggle = model_folder(world.kaggle / "qwen3-5-9b")
    ns = {"os": os}
    exec(model_block(nb), ns)
    assert (ns["MODEL_ID"], ns["MODEL_SOURCE"]) == (str(kaggle), "kaggle-input")                # the saved Kaggle dataset keeps working
    shutil.rmtree(kaggle)
    monkeypatch.delenv(ENV)
    ns = {"os": os}
    exec(model_block(nb), ns)
    assert (ns["MODEL_ID"], ns["MODEL_SOURCE"]) == (str(world.canonical), "downloaded") and len(world.downloads) == 1


@pytest.mark.parametrize("nb", ["VLM_Dental_Colab_SFT.ipynb", "VLM_Dental_Colab_GRPO.ipynb"])
def test_config_cell_force_download_toggle_goes_straight_to_the_canonical_folder(nb, world):
    model_folder(world.kaggle / "qwen3-5-9b")
    src = model_block(nb).replace("FORCE_HF_MODEL_DOWNLOAD = False", "FORCE_HF_MODEL_DOWNLOAD = True")
    ns = {"os": os}
    exec(src, ns)
    assert ns["MODEL_SOURCE"] == "downloaded"


@pytest.mark.parametrize("nb", ["VLM_Dental_Colab_SFT.ipynb", "VLM_Dental_Colab_GRPO.ipynb"])
def test_every_model_id_the_notebook_passes_is_the_resolved_one(nb):
    """One MODEL_ID, assigned once from resolve_qwen_model(); every --model-id argument is that variable, never a Hub id."""
    cells = code_cells(nb)
    assert sum(len(re.findall(r"^MODEL_ID, MODEL_SOURCE = resolve_qwen_model\(", s, flags=re.M)) for s in cells) == 1
    assert sum(len(re.findall(r"^MODEL_ID\s*=", s, flags=re.M)) for s in cells) == 0
    flags = [m for s in cells for m in re.findall(r'"--model-id",\s*([^,\n]+)', s)]
    assert flags and all(f.strip() == "MODEL_ID" for f in flags), flags
    assert not any(re.search(r'"--model-id",\s*["\']Qwen', s) for s in cells)


def test_grpo_launch_eval_matrix_and_ksweep_cells_all_receive_the_resolved_model_id():
    nb = "VLM_Dental_Colab_GRPO.ipynb"
    launch, matrix, ksweep = code_of(nb, "run_grpo.py"), code_of(nb, "run_eval_matrix"), code_of(nb, "KSWEEP_SIZES")
    assert launch.count('"--model-id", MODEL_ID') == 2                                           # single and sweep launch
    assert '"--model-id", MODEL_ID' in matrix and '"--model-id", MODEL_ID' in ksweep


def test_sft_manifest_warmup_launch_and_eval_cells_receive_the_resolved_model_id():
    nb = "VLM_Dental_Colab_SFT.ipynb"
    for script in ("scripts/compute_exact_trace_lengths.py", "scripts/warmup_xla_cache.py", "scripts/train_sft.py", "scripts/evaluate_models.py"):
        launching = [c for c in code_cells(nb) if script in c and '"--model-id"' in c]                  # the cells that build the command
        assert launching and all('"--model-id", MODEL_ID' in c for c in launching), script


def test_zero_shot_cell_keeps_the_name_and_hands_the_resolved_folder_to_the_provider(world, monkeypatch, capsys):
    folder = model_folder(world.tmp / "Qwen3.5-9B")
    monkeypatch.setenv(ENV, str(folder))
    monkeypatch.delenv("TRANSFORMERS_LOAD_IN_4BIT", raising=False)
    ns = {"os": os}
    exec(code_of("VLM_Dental_Colab_ZeroShot.ipynb", 'MODEL = "Qwen/Qwen3.5-9B"'), ns)
    assert ns["MODEL"] == "Qwen/Qwen3.5-9B" and ns["QWEN_PATH"] == str(folder) and ns["PROVIDER"] == "transformers"
    assert os.environ[ENV] == str(folder) and f"Qwen3.5-9B source: env -> {folder}" in capsys.readouterr().out
    assert inp.local_model_alias(ns["MODEL"]) == str(folder)                                      # what run_zero_shot.py's provider loads


def test_tracegen_vllm_start_serves_the_local_folder_under_the_logical_name(world, monkeypatch, capsys):
    folder = model_folder(world.tmp / "Qwen3.5-9B")
    monkeypatch.setenv(ENV, str(folder))
    monkeypatch.setenv("GENERATOR_MODEL", "Qwen/Qwen3.5-9B")
    src = code_of("VLM_Dental_Colab_TraceGen.ipynb", "vllm.entrypoints.openai.api_server',")
    seg = src[src.index("model_name = os.environ.get('GENERATOR_MODEL'"): src.index("vllm_process = subprocess.Popen(")]
    ns = {"os": os, "IS_COLAB": False, "vllm_cache_dir": str(world.tmp / "vllm_cache")}
    exec(seg, ns)
    cmd = ns["vllm_cmd"]
    assert cmd[cmd.index("--model") + 1] == str(folder)                                           # vLLM loads the folder ...
    assert cmd[cmd.index("--served-model-name") + 1] == "Qwen/Qwen3.5-9B"                         # ... and answers to the name the client sends
    monkeypatch.setenv("GENERATOR_MODEL", "some-org/other-model")                                 # other models: unchanged
    ns = {"os": os, "IS_COLAB": False, "vllm_cache_dir": str(world.tmp / "vllm_cache")}
    exec(seg, ns)
    assert ns["vllm_cmd"][ns["vllm_cmd"].index("--model") + 1] == "some-org/other-model" and "--served-model-name" not in ns["vllm_cmd"]


def test_no_notebook_sets_hf_hub_offline_globally():
    for nb in ("SFT", "GRPO", "ZeroShot", "TraceGen"):
        assert not any("HF_HUB_OFFLINE" in s for s in code_cells(f"VLM_Dental_Colab_{nb}.ipynb")), nb      # datasets still download from the Hub


# ---------------------------------------------------------------------------------------------------------------------
# Real scripts, local folder, every network call refused
# ---------------------------------------------------------------------------------------------------------------------
BLOCKER = '''import os, socket
_ok = {"127.0.0.1", "::1", "localhost"}
_log = os.environ.get("NET_ATTEMPT_LOG")
def _note(what):
    if _log:
        with open(_log, "a") as f:
            f.write(what + "\\n")
def _guard(real):
    def f(self, address, *a, **k):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, str) and (host in _ok or host.startswith("/") or host.startswith("\\0")):
            return real(self, address, *a, **k)
        _note("connect " + repr(address))
        raise OSError(101, "Network is unreachable (blocked by test)")
    return f
socket.socket.connect = _guard(socket.socket.connect)
socket.socket.connect_ex = _guard(socket.socket.connect_ex)
_gai = socket.getaddrinfo
def gai(host, *a, **k):
    if host is not None and host not in _ok:
        _note("dns " + str(host))
        raise socket.gaierror(-2, "Name or service not known (blocked by test)")
    return _gai(host, *a, **k)
socket.getaddrinfo = gai
'''
ULTRALYTICS_PROBES = {"dns.google", "one.one.one.one"}      # ultralytics' import-time "am I online" check; it downloads nothing


@pytest.fixture(scope="module")
def offline(tmp_path_factory):
    root = tmp_path_factory.mktemp("offline")
    (root / "blk").mkdir()
    (root / "blk" / "sitecustomize.py").write_text(BLOCKER)
    log = root / "net.log"
    env = dict(os.environ, PYTHONPATH=str(root / "blk"), HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=str(root / "hf_empty"),
               NET_ATTEMPT_LOG=str(log))
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", ENV):
        env.pop(var, None)
    from test_smoke_sft_cpu import _trace

    traces = root / "traces.jsonl"
    traces.write_text("\n".join(json.dumps(_trace(i)) for i in range(4)) + "\n")

    def run(cmd, timeout=900, cwd=None):
        return subprocess.run([sys.executable, *map(str, cmd)], capture_output=True, text=True, timeout=timeout, env=env, cwd=cwd)

    def attempts():
        return sorted({line.split(" ", 1)[1].strip("()'\", ").split("'")[0] for line in log.read_text().splitlines()}) if log.exists() else []

    return type("Offline", (), {"root": root, "traces": traces, "run": staticmethod(run), "attempts": staticmethod(attempts), "log": log})


def assert_no_download_attempt(offline):
    hosts = offline.attempts()
    assert set(hosts) <= ULTRALYTICS_PROBES, f"something tried to reach the network: {hosts}"


@pytest.fixture(scope="module")
def sft_run(offline):
    work = offline.root / "sft"
    proc = offline.run([ROOT / "scripts" / "smoke_test_sft.py", "--traces", offline.traces, "--num-traces", "4", "--work-dir", work,
                        "--", "--canonical-resize", "--max-seq-len", "8192"])
    return proc, work


def test_sft_trains_from_a_local_folder_with_the_network_refused(offline, sft_run):
    proc, work = sft_run
    tail = (proc.stdout + proc.stderr)[-2000:]
    assert proc.returncode == 0 and "[SMOKE] PASSED" in proc.stdout, tail
    assert f"Loading {work / 'model'}" in proc.stdout                                           # the processor + model came from the folder
    assert (work / "out" / "adapter_config.json").exists()
    assert_no_download_attempt(offline)


def test_sft_preflight_and_token_length_manifest_use_the_local_folder_offline(offline, sft_run):
    _, work = sft_run
    flags = ["--model-id", work / "model", "--data-dir", work / "data", "--canonical-resize"]
    pf = offline.run([ROOT / "scripts" / "train_sft.py", *flags, "--track", "with_tools", "--stage", "dentex_alone", "--dataset-path",
                      work / "data" / "smoke_traces.jsonl", "--output-dir", work / "out_pf", "--precision", "bf16", "--max-seq-len", "8192",
                      "--hf-repo", "", "--preflight"])
    assert pf.returncode == 0, pf.stdout[-2000:] + pf.stderr[-1000:]
    assert re.search(r"\[PASS\] base model: .*model \(attached/local folder\)", pf.stdout) and "[FAIL]" not in pf.stdout
    lengths = offline.run([ROOT / "scripts" / "compute_exact_trace_lengths.py", *flags, "--trace-files", work / "data" / "smoke_traces.jsonl",
                           "--output-file", work / "lengths.json", "--strict"])
    assert lengths.returncode == 0, lengths.stdout[-1500:] + lengths.stderr[-1500:]
    meta = json.loads((work / "lengths.json").read_text())["_meta"]
    assert meta["complete"] is True                                                             # tokenizer + image processor ran on the folder
    assert_no_download_attempt(offline)


def test_grpo_dual_adapters_run_from_a_local_folder_and_its_preflight_accepts_it_offline(offline):
    work = offline.root / "grpo"
    proc = offline.run([ROOT / "scripts" / "smoke_test_grpo.py", "--traces", offline.traces, "--track", "with_tools", "--work-dir", work,
                        "--", "--canonical-resize"])
    tail = (proc.stdout + proc.stderr)[-2000:]
    assert proc.returncode == 0 and "[SMOKE-GRPO] PASSED" in proc.stdout, tail
    pf = offline.run([ROOT / "scripts" / "run_grpo.py", "--track", "with_tools", "--sft-stage", "dentex_alone", "--sft-model-dir", work / "adapter",
                      "--model-id", work / "model", "--dataset", "dentex", "--group-size", "2", "--output-dir", work / "out_pf", "--hf-repo", "",
                      "--canonical-resize", "--preflight"], cwd=offline.root)
    # The only FAIL allowed is the missing real DENTEX train data of this sandbox; the local base model must PASS.
    assert re.search(r"\[PASS\] base model: .*model \(attached/local folder\)", pf.stdout), pf.stdout[-2000:]
    fails = [line for line in pf.stdout.splitlines() if line.startswith("[FAIL]")]
    assert all("dataset" in line for line in fails), fails
    assert_no_download_attempt(offline)
