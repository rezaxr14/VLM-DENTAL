"""Input resolution (dental_agent/utils/inputs.py): every input is found by itself, in a fixed order, from any working
directory, and a missing one fails loudly BEFORE a model loads. No network: the three Hub shims are replaced."""

from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from dental_agent.utils import inputs as inp  # noqa: E402

DEV_KEY = "dentex::7"


# ---------------------------------------------------------------------------------------------------------------------
# A hermetic world: tmp repo root, tmp cwd, no Kaggle, empty HF cache, Hub unreachable unless a test installs a fake.
# ---------------------------------------------------------------------------------------------------------------------
class Hub:
    """In-memory Hub: ``files[(repo, type)] = {path: bytes}``. Counts what was downloaded."""

    def __init__(self):
        self.files: dict[tuple[str, str], dict[str, bytes]] = {}
        self.downloads: list[str] = []
        self.reachable = True

    def list(self, repo, repo_type="model"):
        if not self.reachable:
            raise ConnectionError("hub unreachable")
        if (repo, repo_type) not in self.files:
            raise FileNotFoundError(f"{repo} not found")
        return list(self.files[(repo, repo_type)])

    def file(self, repo, repo_type, filename, local_dir):
        self.downloads.append(f"{repo}:{filename}")
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.files[(repo, repo_type)][filename])
        return str(dest)

    def snapshot(self, repo, repo_type, allow_patterns, local_dir):
        import fnmatch

        for path, blob in self.files[(repo, repo_type)].items():
            if any(fnmatch.fnmatch(path, pat) for pat in allow_patterns):
                self.downloads.append(f"{repo}:{path}")
                dest = Path(local_dir) / path
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(blob)
        return str(local_dir)


@pytest.fixture
def world(tmp_path, monkeypatch):
    repo_root, cwd = tmp_path / "repo", tmp_path / "elsewhere"
    repo_root.mkdir()
    cwd.mkdir()
    for rel in ("dental_agent/training/sft.py", "dental_agent/utils/canonical.py", "dental_agent/agent/prompts.py"):
        (repo_root / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo_root / rel).write_text(f"# {rel} v1\n")
    monkeypatch.chdir(cwd)
    monkeypatch.setattr(inp, "REPO_ROOT", repo_root)
    monkeypatch.setattr(inp, "KAGGLE_INPUT", tmp_path / "no_kaggle")
    monkeypatch.setattr(inp.Path, "home", classmethod(lambda cls: tmp_path / "home"))
    for var in ("HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_TOKEN", "GROUNDING_MODEL_PATH", "HF_ARTIFACT_REPO",
                "HF_TRACES_REPO", "DENTEX_IMAGES_REPO", "TUFTS_IMAGES_REPO"):
        monkeypatch.delenv(var, raising=False)
    inp._IMG_CACHE.clear()
    hub = Hub()
    monkeypatch.setattr(inp, "hub_list_files", hub.list)
    monkeypatch.setattr(inp, "hub_file", hub.file)
    monkeypatch.setattr(inp, "hub_snapshot", hub.snapshot)
    # one-image Hub download used by resolve_image(): serve from the fake Hub into data_dir
    def fake_slice(ds, ids, data_dir, repo):
        out = {}
        rid = inp.images_repo(ds, repo)
        for i in ids:
            name = inp.hub_image_filename(ds, i)
            if (rid, "dataset") in hub.files and name in hub.files[(rid, "dataset")]:
                dest = Path(data_dir or "data") / ds / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(hub.files[(rid, "dataset")][name])
                hub.downloads.append(f"{rid}:{name}")
                out[i] = dest
            else:
                out[i] = None
        return out

    monkeypatch.setattr(inp, "_download_slice", fake_slice)
    return Namespace(root=repo_root, cwd=cwd, tmp=tmp_path, hub=hub, kaggle=tmp_path / "no_kaggle")


def touch(p: Path, data: bytes = b"x") -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return p


def write_traces(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------------------------------------------------
def test_image_found_from_any_working_directory(world):
    img = touch(world.root / "data" / "dentex" / "images" / "12.png")
    # recorded on another machine, and the notebook is NOT in the repo folder
    got = inp.resolve_image("dentex", 12, "/content/VLM-DENTAL/data/dentex/images/12.png")
    assert got == img


def test_image_windows_path_is_rebased(world):
    img = touch(world.root / "data" / "dentex" / "images" / "13.png")
    got = inp.resolve_image("dentex", 13, "C:\\Users\\reza\\VLM-DENTAL\\data\\dentex\\images\\13.png")
    assert got == img


def test_image_by_id_when_recorded_path_is_useless(world):
    img = touch(world.root / "data" / "images" / "14.png")
    assert inp.resolve_image("dentex", "14", "nonsense/14.png") == img


def test_image_in_kaggle_input(world, monkeypatch):
    img = touch(world.kaggle / "dentex-images" / "images" / "15.png")
    got = inp.resolve_image("dentex", 15, None)
    assert got == img


def test_image_in_hf_cache(world, monkeypatch):
    cache = world.tmp / "hfcache"
    img = touch(cache / "datasets--Reza-Nadimi--dentex-train-images" / "snapshots" / "abc" / "images" / "16.png")
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    assert inp.resolve_image("dentex", 16, None) == img


def test_image_hub_download_of_a_single_file(world):
    world.hub.files[("Reza-Nadimi/dentex-train-images", "dataset")] = {"images/17.png": b"png", "images/18.png": b"png"}
    got = inp.resolve_image("dentex", 17, None, data_dir=world.cwd / "data")
    assert got is not None and got.is_file()
    assert world.hub.downloads == ["Reza-Nadimi/dentex-train-images:images/17.png"]  # only the one image


def test_train_image_never_resolves_to_a_validation_file(world):
    # ids collide: validation image 21 exists, TRAIN image 21 does not -> must NOT be returned
    touch(world.root / "data" / "dentex" / "validation_images" / "21.png")
    touch(world.root / "data" / "dentex" / "validation_data" / "xrays" / "21.png")
    assert inp.resolve_image("dentex", 21, "data/dentex/validation_images/21.png", allow_hub=False) is None
    # ...but an ordinary train file with the same id is found
    train = touch(world.root / "data" / "dentex" / "images" / "21.png")
    assert inp.resolve_image("dentex", 21, None, allow_hub=False) == train


def test_tufts_extensions(world):
    img = touch(world.root / "data" / "tufts" / "Radiographs" / "5.JPG")
    assert inp.resolve_image("tufts", 5, None, allow_hub=False) == img


def test_check_trace_images_counts_missing(world):
    touch(world.root / "data" / "dentex" / "images" / "1.png")
    f = write_traces(world.root / "data" / "traces" / "a.jsonl", [
        {"dataset": "dentex", "image_id": 1, "image_path": "x/1.png"},
        {"dataset": "dentex", "image_id": 1, "image_path": "x/1.png"},   # same image twice
        {"dataset": "dentex", "image_id": 2, "image_path": "x/2.png"},
        {"dataset": "tufts", "image_id": 3, "image_path": "x/3.JPG"},
    ])
    chk = inp.check_trace_images([f], "data", hub="off")
    assert (chk.n_records, chk.n_unique, chk.n_missing) == (4, 3, 2)
    assert chk.by_dataset() == {"dentex": 1, "tufts": 1}
    assert "2 of 3 images missing" in chk.describe()


def test_check_trace_images_excludes_dev_keys(world):
    f = write_traces(world.root / "data" / "traces" / "a.jsonl", [{"dataset": "dentex", "image_id": 7, "image_path": "x"}])
    assert inp.check_trace_images([f], "data", exclude_keys={DEV_KEY}, hub="off").n_unique == 0


def test_ensure_trace_images_fetches_only_the_missing_delta(world):
    touch(world.root / "data" / "dentex" / "images" / "1.png")
    world.hub.files[("Reza-Nadimi/dentex-train-images", "dataset")] = {"images/2.png": b"p"}
    f = write_traces(world.root / "data" / "traces" / "a.jsonl", [
        {"dataset": "dentex", "image_id": 1, "image_path": "p"}, {"dataset": "dentex", "image_id": 2, "image_path": "p"}])
    chk = inp.ensure_trace_images([f], world.root / "data")
    assert chk.fetched == 1 and chk.n_missing == 0
    assert world.hub.downloads == ["Reza-Nadimi/dentex-train-images:images/2.png"]


def test_ensure_trace_images_raises_with_search_list_and_fix(world):
    f = write_traces(world.root / "data" / "traces" / "a.jsonl", [{"dataset": "dentex", "image_id": 99, "image_path": "p"}])
    with pytest.raises(inp.InputNotFound) as ei:
        inp.ensure_trace_images([f], world.root / "data")
    msg = str(ei.value)
    assert "1 of 1 images missing" in msg and "dentex::99" in msg
    assert "Reza-Nadimi/dentex-train-images" in msg and "fetch_hf_assets.py --images" in msg


def test_ensure_dataset_images_repairs_local_path(world):
    import pandas as pd

    img = touch(world.root / "data" / "dentex" / "images" / "3.png")
    df = pd.DataFrame({"id": [3, 4], "file_name": ["3.png", "4.png"], "local_path": [None, None]})
    out, chk = inp.ensure_dataset_images(df, "dentex", world.root / "data", hub="off")
    assert out.loc[0, "local_path"] == str(img) and out.loc[1, "local_path"] is None
    assert chk.n_missing == 1


# ---------------------------------------------------------------------------------------------------------------------
# SFT adapter
# ---------------------------------------------------------------------------------------------------------------------
def make_adapter(d: Path, **meta) -> Path:
    touch(d / "adapter_config.json", b"{}")
    touch(d / "adapter_model.safetensors")
    if meta:
        (d / "sft_run_meta.json").write_text(json.dumps(meta))
    return d


def test_adapter_explicit_wrong_fails_and_never_falls_through(world):
    make_adapter(world.root / "data" / "models" / "qwen3_5_9b_sft_no_tools_dentex_alone_bf16")   # a good one exists...
    with pytest.raises(inp.InputNotFound) as ei:
        inp.resolve_sft_adapter("data/models/typo", "no_tools", "dentex_alone", "bf16")           # ...but the flag is wrong
    assert "--sft-model-dir data/models/typo: no such folder" in str(ei.value)


def test_adapter_explicit_ok(world):
    d = make_adapter(world.tmp / "mine")
    r = inp.resolve_sft_adapter(str(d), "no_tools", "dentex_alone", "bf16")
    assert r.path == d and r.source == "flag"


def test_adapter_suffixed_folder_from_any_cwd(world):
    d = make_adapter(world.root / "data" / "models" / "qwen3_5_9b_sft_with_tools_dentex_alone_qlora")
    r = inp.resolve_sft_adapter(None, "with_tools", "dentex_alone", "qlora")
    assert r.path == d and r.source == "local"


def test_adapter_other_precision_is_never_picked(world):
    make_adapter(world.root / "data" / "models" / "qwen3_5_9b_sft_with_tools_dentex_alone_qlora")
    with pytest.raises(inp.InputNotFound):
        inp.resolve_sft_adapter(None, "with_tools", "dentex_alone", "bf16", hub="off")


def test_adapter_plain_name_still_found(world):
    d = make_adapter(world.root / "data" / "models" / "qwen3_5_9b_sft_with_tools_dentex_alone")
    assert inp.resolve_sft_adapter(None, "with_tools", "dentex_alone", "bf16").path == d


def test_adapter_in_kaggle_input_by_name_and_by_meta(world):
    k = world.kaggle
    d = make_adapter(k / "datasets" / "reza" / "my-sft" / "qwen3_5_9b_sft_no_tools_dentex_alone_bf16")
    assert inp.resolve_sft_adapter(None, "no_tools", "dentex_alone", "bf16").path == d
    import shutil
    shutil.rmtree(k)
    root = make_adapter(k / "some-dataset", track="no_tools", stage="dentex_alone", precision="bf16")
    r = inp.resolve_sft_adapter(None, "no_tools", "dentex_alone", "bf16")
    assert r.path == root and r.source == "kaggle"


def test_adapter_from_hub_when_absent_locally(world):
    pfx = "sft/qwen3_5_9b_sft_no_tools_dentex_alone_bf16/"
    world.hub.files[(inp.DEFAULT_MODELS_REPO, "model")] = {
        pfx + "adapter_config.json": b"{}", pfx + "adapter_model.safetensors": b"w", pfx + "sft_run_meta.json": b"{}",
        pfx + "optimizer.pt": b"huge"}
    r = inp.resolve_sft_adapter(None, "no_tools", "dentex_alone", "bf16")
    assert r.source == "hub" and (r.path / "adapter_config.json").is_file()
    assert not any("optimizer.pt" in d for d in world.hub.downloads)  # adapter files only


def test_adapter_hub_probe_does_not_download(world):
    pfx = "sft/qwen3_5_9b_sft_no_tools_dentex_alone_bf16/"
    world.hub.files[(inp.DEFAULT_MODELS_REPO, "model")] = {pfx + "adapter_config.json": b"{}"}
    r = inp.resolve_sft_adapter(None, "no_tools", "dentex_alone", "bf16", hub="probe")
    assert r.source == "hub-available" and world.hub.downloads == []


def test_adapter_not_found_names_everything_searched_and_the_fix(world):
    world.hub.reachable = False
    with pytest.raises(inp.InputNotFound) as ei:
        inp.resolve_sft_adapter(None, "with_tools", "dentex_alone", "qlora")
    msg = str(ei.value)
    assert "qwen3_5_9b_sft_with_tools_dentex_alone_qlora" in msg
    assert "unreachable" in msg and "--sft-model-dir" in msg
    assert "train_sft.py --track with_tools --stage dentex_alone --precision qlora" in msg


def test_assert_adapter_loadable_reports_missing_weights_and_best_adapter_hint(world):
    d = touch(world.root / "a" / "adapter_config.json", b"{}").parent
    assert "no adapter_model" in inp.adapter_problem(d)
    (world.root / "b" / "best_adapter").mkdir(parents=True)
    assert "best_adapter" in inp.adapter_problem(world.root / "b")
    with pytest.raises(inp.InputNotFound):
        inp.assert_adapter_loadable(d)
    make_adapter(d)
    inp.assert_adapter_loadable(d)


# ---------------------------------------------------------------------------------------------------------------------
# Traces
# ---------------------------------------------------------------------------------------------------------------------
def test_stage_trace_names_table():
    assert inp.stage_trace_names("dentex_alone", "with_tools") == ["train_cot_traces_dentex.jsonl", "train_cot_traces_healthy_dentex.jsonl"]
    assert inp.stage_trace_names("dentex_alone", "no_tools")[0] == "train_cot_traces_dentex_no_tools.jsonl"
    assert len(inp.stage_trace_names("dentex_tufts_overlap", "with_tools")) == 4
    assert "train_cot_traces_tufts_all.jsonl" in inp.stage_trace_names("full_curriculum", "with_tools")


def test_traces_all_local(world):
    for n in inp.stage_trace_names("dentex_alone", "no_tools"):
        touch(world.root / "data" / "traces" / n)
    res = inp.resolve_stage_traces("dentex_alone", "no_tools", "data")
    assert [p.name for p in res.files] == inp.stage_trace_names("dentex_alone", "no_tools") and not res.legacy_fallback


def test_traces_partial_local_fetches_only_the_missing_file(world):
    names = inp.stage_trace_names("dentex_alone", "with_tools")
    touch(world.root / "data" / "traces" / names[0])
    world.hub.files[(inp.DEFAULT_TRACES_REPO, "dataset")] = {n: b"{}\n" for n in names + ["train_cot_traces_tufts.jsonl"]}
    res = inp.resolve_stage_traces("dentex_alone", "with_tools", world.root / "data")
    assert len(res.files) == 2 and all(f.is_file() for f in res.files)
    assert world.hub.downloads == [f"{inp.DEFAULT_TRACES_REPO}:{names[1]}"]


def test_traces_probe_lists_pending_without_downloading(world):
    names = inp.stage_trace_names("dentex_alone", "with_tools")
    world.hub.files[(inp.DEFAULT_TRACES_REPO, "dataset")] = {n: b"{}\n" for n in names}
    res = inp.resolve_stage_traces("dentex_alone", "with_tools", world.root / "data", hub="probe")
    assert sorted(res.hub_pending) == sorted(names) and world.hub.downloads == []


def test_traces_missing_everywhere_raises_with_the_fix(world):
    world.hub.files[(inp.DEFAULT_TRACES_REPO, "dataset")] = {}
    with pytest.raises(inp.InputNotFound) as ei:
        inp.resolve_stage_traces("dentex_alone", "with_tools", world.root / "data")
    assert "train_cot_traces_healthy_dentex.jsonl" in str(ei.value) and "sync_traces_hf.py --download" in str(ei.value)


def test_traces_one_of_two_missing_is_an_error_not_a_silent_half_dataset(world):
    names = inp.stage_trace_names("dentex_alone", "with_tools")
    touch(world.root / "data" / "traces" / names[0])         # disease file only; the healthy one is nowhere
    world.hub.reachable = False
    with pytest.raises(inp.InputNotFound) as ei:
        inp.resolve_stage_traces("dentex_alone", "with_tools", world.root / "data")
    assert names[1] in str(ei.value)


def test_traces_legacy_single_file_only_when_no_stage_file_exists(world):
    touch(world.root / "data" / "traces" / "train_cot_traces_no_tools.jsonl")
    world.hub.reachable = False
    res = inp.resolve_stage_traces("dentex_alone", "no_tools", world.root / "data")
    assert res.legacy_fallback and res.files[0].name == "train_cot_traces_no_tools.jsonl"


def test_trace_parity_lists_images_missing_in_a_track(world):
    d = world.root / "data" / "traces"
    rows = lambda ids: [{"dataset": "tufts", "image_id": i, "image_path": "p"} for i in ids]  # noqa: E731
    write_traces(d / "train_cot_traces_tufts.jsonl", rows(range(1, 6)))
    write_traces(d / "train_cot_traces_tufts_no_tools.jsonl", rows(range(1, 4)))
    (p,) = inp.trace_parity(d)
    assert p["with_tools"] == 5 and p["no_tools"] == 3 and p["only_with_tools"] == ["tufts::4", "tufts::5"]


# ---------------------------------------------------------------------------------------------------------------------
# Token-length manifest
# ---------------------------------------------------------------------------------------------------------------------
def manifest_for(files, canonical=True, complete=True, estimated=None):
    return {"canonical_resize": canonical, "lengths_by_file_and_id": {},
            "_meta": {"fingerprint": inp.manifest_fingerprint(files, canonical), "complete": complete, "estimated_traces": estimated or []}}


def test_manifest_fresh_then_each_staleness_reason(world):
    f = write_traces(world.root / "data" / "traces" / "t.jsonl", [{"dataset": "dentex", "image_id": 1}])
    assert inp.manifest_staleness(manifest_for([f]), [f], True) == []

    assert "no fingerprint" in inp.manifest_staleness({"canonical_resize": True}, [f], True)[0]
    assert any("did not finish" in r for r in inp.manifest_staleness(manifest_for([f], complete=False), [f], True))
    # resuming a partial measurement is allowed
    assert inp.manifest_staleness(manifest_for([f], complete=False), [f], True, require_complete=False) == []
    assert any("estimates" in r for r in inp.manifest_staleness(manifest_for([f], estimated=["dentex::9"]), [f], True))
    assert any("canonical_resize" in r for r in inp.manifest_staleness(manifest_for([f], canonical=False), [f], True))

    m = manifest_for([f])
    write_traces(f, [{"dataset": "dentex", "image_id": 1}, {"dataset": "dentex", "image_id": 2}])      # traces edited (patch 0028 style)
    assert any("t.jsonl changed (1 -> 2 traces)" in r for r in inp.manifest_staleness(m, [f], True))

    m = manifest_for([f])
    (world.root / "dental_agent/training/sft.py").write_text("# sft.py v2\n")                          # dataset code edited
    assert any("code changed" in r for r in inp.manifest_staleness(m, [f], True))


def test_manifest_crlf_checkout_does_not_change_the_fingerprint(world):
    a = inp.code_fingerprint()
    p = world.root / "dental_agent/training/sft.py"
    p.write_bytes(p.read_bytes().replace(b"\n", b"\r\n"))
    assert inp.code_fingerprint() == a


def test_check_token_manifest_states(world):
    f = write_traces(world.root / "data" / "traces" / "t.jsonl", [{"dataset": "dentex", "image_id": 1}])
    st = inp.check_token_manifest(world.root / "data", [f], True)
    assert st.state == "missing"
    mp = world.root / "data" / "traces" / inp.MANIFEST_NAME
    mp.write_text(json.dumps(manifest_for([f])))
    assert inp.check_token_manifest(world.root / "data", [f], True).state == "fresh"
    mp.write_text(json.dumps({"canonical_resize": True}))
    assert inp.check_token_manifest(world.root / "data", [f], True).state == "stale"
    mp.write_text("{not json")
    assert "not readable" in inp.check_token_manifest(world.root / "data", [f], True).reasons[0]


def test_manifest_command_carries_every_explicit_setting():
    cmd = inp.manifest_command("data", True, "Qwen/Qwen3.5-9B", [5, 10, 4])
    assert "--recompute" in cmd and "--canonical-resize" in cmd and "--vision-slots 5 10 4" in cmd
    assert "--canonical-resize" not in inp.manifest_command("data", False, "m", [1, 0, 0])


# ---------------------------------------------------------------------------------------------------------------------
# Dev split, YOLO, base model
# ---------------------------------------------------------------------------------------------------------------------
def test_dev_split_found_from_other_cwd_and_none_means_no_holdout(world):
    p = touch(world.root / "data" / "splits" / "dentex_dev.json", b'{"keys": ["dentex::7"]}')
    assert inp.resolve_dev_split("data/splits/dentex_dev.json", "data") == p
    assert inp.resolve_dev_split(None) is None


def test_dev_split_missing_names_the_command(world):
    with pytest.raises(inp.InputNotFound) as ei:
        inp.resolve_dev_split("data/splits/dentex_dev.json", "data")
    assert "make_dev_split.py" in str(ei.value) and "--seed 42" in str(ei.value)


def test_yolo_env_then_local_then_hub(world, monkeypatch):
    # nothing anywhere, Hub unreachable -> loud
    world.hub.reachable = False
    with pytest.raises(inp.InputNotFound) as ei:
        inp.resolve_yolo_checkpoint()
    assert "fetch_hf_assets.py --models" in str(ei.value) and "GROUNDING_MODEL_PATH" in str(ei.value)
    world.hub.reachable = True
    # Hub has it -> copied to the default place
    world.hub.files[(inp.DEFAULT_MODELS_REPO, "model")] = {inp.YOLO_HUB_FILE: b"pt"}
    probe = inp.resolve_yolo_checkpoint(hub="probe")
    assert probe.source == "hub-available" and world.hub.downloads == []
    got = inp.resolve_yolo_checkpoint()
    assert got.source == "hub" and got.path.is_file()
    # now local wins and nothing more is downloaded
    n = len(world.hub.downloads)
    assert inp.resolve_yolo_checkpoint().source == "local" and len(world.hub.downloads) == n
    # env var wins over the default folder
    mine = touch(world.tmp / "mine" / "best.pt")
    monkeypatch.setenv("GROUNDING_MODEL_PATH", str(mine))
    assert inp.resolve_yolo_checkpoint().path == mine


def test_yolo_in_kaggle_dataset(world):
    k = touch(world.kaggle / "tooth-yolo" / "yolo_cv_best" / "weights" / "best.pt")
    r = inp.resolve_yolo_checkpoint(hub="off")
    assert r.path == k and r.source == "kaggle"


def test_base_model_cases(world):
    d = world.tmp / "qwen"
    touch(d / "config.json")
    with pytest.raises(inp.InputNotFound, match="no weight files"):
        inp.resolve_base_model(str(d))
    touch(d / "model.safetensors")
    assert inp.resolve_base_model(str(d)).source == "local"
    # an explicit Hub id is accepted and never re-routed to a Kaggle copy
    assert inp.resolve_base_model("Qwen/Qwen3.5-9B").source == "hub-available"
    # a path that does not exist fails before any load
    with pytest.raises(inp.InputNotFound) as ei:
        inp.resolve_base_model("/kaggle/input/qwen3-5-9b-typo")
    assert "no such folder" in str(ei.value)
    # probe mode: unreachable / config missing
    world.hub.reachable = False
    with pytest.raises(inp.InputNotFound, match="unreachable"):
        inp.resolve_base_model("Qwen/Qwen3.5-9B", hub="probe")
    world.hub.reachable = True
    world.hub.files[("Qwen/Qwen3.5-9B", "model")] = {"config.json": b"{}"}
    assert inp.resolve_base_model("Qwen/Qwen3.5-9B", hub="probe").source == "hub-available"


def test_base_model_hf_cache_snapshot(world, monkeypatch):
    cache = world.tmp / "hfc"
    snap = touch(cache / "models--Qwen--Qwen3.5-9B" / "snapshots" / "h1" / "config.json").parent
    monkeypatch.setenv("HF_HUB_CACHE", str(cache))
    r = inp.resolve_base_model("Qwen/Qwen3.5-9B")
    assert r.path == snap and r.source == "hub-cache"


# ---------------------------------------------------------------------------------------------------------------------
# Dataset: a missing image is an error, never a gray square
# ---------------------------------------------------------------------------------------------------------------------
def test_sft_dataset_raises_on_missing_image_and_rejects_bad_mode(world):
    from dental_agent.training.sft import DentalSFTDataset

    f = write_traces(world.root / "data" / "traces" / "t.jsonl", [{
        "dataset": "dentex", "image_id": 404, "image_path": "data/dentex/images/404.png",
        "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}])
    with pytest.raises(ValueError, match="missing_image"):
        DentalSFTDataset(f, processor=object(), data_dir=world.root / "data", missing_image="ignore")
    ds = DentalSFTDataset(f, processor=object(), data_dir=world.root / "data")
    with pytest.raises(FileNotFoundError, match=r"\[MISSING IMAGE\].*dentex::404"):
        ds[0]


def test_legacy_resolve_image_path_never_returns_a_validation_file(world):
    from dental_agent.training.sft import resolve_image_path

    touch(world.root / "data" / "dentex" / "validation_images" / "30.png")
    rec = {"dataset": "dentex", "image_id": 30, "image_path": "data/dentex/validation_images/30.png"}
    assert not resolve_image_path(rec, world.root / "data")


# ---------------------------------------------------------------------------------------------------------------------
# train_sft.py: resolve_sft_inputs + the SFT doctor (--preflight)
# ---------------------------------------------------------------------------------------------------------------------
def sft_args(world, **over):
    import train_sft

    argv = ["x", "--track", "no_tools", "--stage", "dentex_alone", "--precision", "bf16", "--data-dir", str(world.root / "data"),
            "--model-id", "Qwen/Qwen3.5-9B", "--dev-split", str(world.root / "data" / "splits" / "dentex_dev.json")]
    old = sys.argv
    sys.argv = argv
    try:
        args = train_sft.parse_args()
    finally:
        sys.argv = old
    for k, v in over.items():
        setattr(args, k, v)
    return args


def sft_world(world, with_manifest=True, missing_image=False):
    d = world.root / "data"
    world.hub.files[("Qwen/Qwen3.5-9B", "model")] = {"config.json": b"{}"}      # the base model is on the (fake) Hub
    touch(d / "splits" / "dentex_dev.json", json.dumps({"keys": [DEV_KEY]}).encode())
    rows = [{"dataset": "dentex", "image_id": i, "image_path": f"data/dentex/images/{i}.png"} for i in (1, 2, 7)]
    files = []
    for n in inp.stage_trace_names("dentex_alone", "no_tools"):
        files.append(write_traces(d / "traces" / n, rows))
    for i in (1, 2, 7):
        if not (missing_image and i == 2):
            touch(d / "dentex" / "images" / f"{i}.png")
    if with_manifest:
        (d / "traces" / inp.MANIFEST_NAME).write_text(json.dumps(manifest_for(files, canonical=True)))
    return files


def test_resolve_sft_inputs_success_writes_resolved_paths_and_skips_dev_images(world, capsys):
    import train_sft

    files = sft_world(world)
    (world.root / "data" / "dentex" / "images" / "7.png").unlink()          # dev image: need not exist
    args = sft_args(world, canonical_resize=True)
    train_sft.resolve_sft_inputs(args, is_tpu=False)
    assert args.dataset_path.split(",") == [str(f) for f in files]
    assert "all 2 resolve" in capsys.readouterr().out


def test_resolve_sft_inputs_missing_image_stops_before_model(world):
    import train_sft

    sft_world(world, missing_image=True)
    with pytest.raises(SystemExit) as ei:
        train_sft.resolve_sft_inputs(sft_args(world, canonical_resize=True), is_tpu=False)
    assert "[INPUT NOT FOUND]" in str(ei.value) and "dentex::2" in str(ei.value)


def test_resolve_sft_inputs_stale_manifest_is_fatal_only_with_slot_padding(world, capsys):
    import train_sft

    files = sft_world(world)
    write_traces(files[0], [{"dataset": "dentex", "image_id": 1, "image_path": "data/dentex/images/1.png"}])   # traces changed
    train_sft.resolve_sft_inputs(sft_args(world, canonical_resize=True), is_tpu=False)
    assert "[MANIFEST WARNING]" in capsys.readouterr().out                                                    # GPU path: warn
    with pytest.raises(SystemExit) as ei:
        train_sft.resolve_sft_inputs(sft_args(world, canonical_resize=True, pad_vision_to_slots=True), is_tpu=True)
    assert "compute_exact_trace_lengths.py" in str(ei.value) and "--vision-slots 5 10 4" in str(ei.value)


def test_resolve_sft_inputs_explicit_dataset_path_must_exist(world):
    import train_sft

    sft_world(world)
    with pytest.raises(SystemExit, match="--dataset-path"):
        train_sft.resolve_sft_inputs(sft_args(world, dataset_path="nope.jsonl"), is_tpu=False)


def test_sft_stage_trace_shim_in_train_sft_uses_the_shared_table(world):
    import train_sft

    sft_world(world)
    got = train_sft.resolve_stage_traces("dentex_alone", "no_tools", world.root / "data")
    assert [Path(p).name for p in got] == inp.stage_trace_names("dentex_alone", "no_tools")


def run_doctor(args):
    from dental_agent.training.preflight import run_sft_preflight

    lines: list[str] = []
    n_fail = run_sft_preflight(args, lines.append)
    return n_fail, "\n".join(lines)


def test_sft_doctor_passes_on_a_complete_world_and_is_read_only(world):
    sft_world(world)
    before = sorted(p for p in world.tmp.rglob("*"))
    n, out = run_doctor(sft_args(world, canonical_resize=True))
    assert n == 0, out
    assert "ready" in out
    assert sorted(p for p in world.tmp.rglob("*")) == before       # read-only: nothing created


def test_sft_doctor_fails_on_missing_images_missing_traces_and_slot_manifest(world):
    sft_world(world, with_manifest=False, missing_image=True)
    world.hub.files[("Reza-Nadimi/dentex-train-images", "dataset")] = {"images/1.png": b"p"}    # Hub does not hold image 2 either
    n, out = run_doctor(sft_args(world, canonical_resize=True, pad_vision_to_slots=True))
    assert n >= 2, out
    assert "[FAIL] trace images" in out and "dentex::2" in out and "not on the Hub either" in out
    assert "[FAIL] token-length manifest" in out and "compute_exact_trace_lengths.py" in out    # the manifest FAIL names the one command

    import shutil

    shutil.rmtree(world.root / "data" / "traces")                                               # traces nowhere, not on the Hub
    world.hub.files[(inp.DEFAULT_TRACES_REPO, "dataset")] = {}
    n2, out2 = run_doctor(sft_args(world))
    assert n2 >= 1 and "[FAIL] traces" in out2 and "sync_traces_hf.py --download" in out2


def test_sft_doctor_image_on_the_hub_is_a_warning_not_a_failure(world):
    sft_world(world, missing_image=True)
    world.hub.files[("Reza-Nadimi/dentex-train-images", "dataset")] = {"images/2.png": b"p"}
    n, out = run_doctor(sft_args(world, canonical_resize=True))
    assert n == 0, out
    assert "[WARN] trace images" in out and "fetched from the Hub at start-up" in out


def test_sft_doctor_flag_conflicts(world):
    sft_world(world)
    n, out = run_doctor(sft_args(world, pad_vision_to_slots=True, canonical_resize=False))
    assert n >= 1 and "requires --canonical-resize" in out
    n, out = run_doctor(sft_args(world, pad_vision_to_slots=True, canonical_resize=True, max_seq_len=7000))
    assert n >= 1 and "cannot hold the 7488" in out


def test_sft_doctor_warns_about_slot_waste_for_no_tools(world):
    sft_world(world)
    n, out = run_doctor(sft_args(world, canonical_resize=True, pad_vision_to_slots=True))
    assert "6336" in out and "1152" in out                         # 7488 padded vs 1152 used


def test_sft_doctor_reports_track_parity_gap_for_the_stage_that_uses_tufts(world):
    d = world.root / "data" / "traces"
    sft_world(world)
    rows = lambda ids: [{"dataset": "tufts", "image_id": i, "image_path": "p"} for i in ids]  # noqa: E731
    for n in inp.stage_trace_names("dentex_tufts_overlap", "no_tools"):
        if "tufts" in n and "healthy" not in n:
            write_traces(d / n, rows([1]))                       # no-tools track: 1 Tufts image
        elif not (d / n).is_file():
            write_traces(d / n, [{"dataset": n.split("_")[-3] if False else "dentex", "image_id": 1, "image_path": "p"}])
    write_traces(d / "train_cot_traces_tufts.jsonl", rows([1, 2, 3]))                    # with-tools track: 3
    touch(world.root / "data" / "tufts" / "Radiographs" / "1.JPG")
    n, out = run_doctor(sft_args(world, stage="dentex_tufts_overlap", canonical_resize=True))
    assert "[WARN] track parity" in out and "train_cot_traces_tufts.jsonl" in out and "tufts::2" in out
    assert "run_trace_gen.py --mode generate --no-tools" in out
    # a stage that does not train on Tufts does not nag about it
    _, out2 = run_doctor(sft_args(world, stage="dentex_alone", canonical_resize=True))
    assert "track parity" not in out2


# ---------------------------------------------------------------------------------------------------------------------
# run_grpo.py: resolve_grpo_inputs / ensure_pool_images
# ---------------------------------------------------------------------------------------------------------------------
def grpo_args(world, **over):
    base = dict(sft_model_dir=None, track="no_tools", sft_stage="dentex_alone", hf_repo=None, precision="bf16",
                model_id="Qwen/Qwen3.5-9B", dev_split=str(world.root / "data" / "splits" / "dentex_dev.json"))
    base.update(over)
    return Namespace(**base)


def test_resolve_grpo_inputs_finds_suffixed_adapter_and_dev_split(world):
    import run_grpo

    d = make_adapter(world.root / "data" / "models" / "qwen3_5_9b_sft_no_tools_dentex_alone_bf16")
    touch(world.root / "data" / "splits" / "dentex_dev.json", b'{"keys": []}')
    args = grpo_args(world)
    got = run_grpo.resolve_sft_reference.__module__ and run_grpo.resolve_grpo_inputs(args, str(world.root / "data"))
    assert Path(got) == d


def test_resolve_grpo_inputs_missing_adapter_is_one_clear_exit(world):
    import run_grpo

    world.hub.reachable = False
    with pytest.raises(SystemExit) as ei:
        run_grpo.resolve_grpo_inputs(grpo_args(world), str(world.root / "data"))
    msg = str(ei.value)
    assert msg.startswith("[INPUT NOT FOUND]") and "SFT adapter" in msg and "train_sft.py" in msg


def test_resolve_grpo_inputs_with_tools_sets_absolute_yolo_path(world, monkeypatch):
    import run_grpo

    make_adapter(world.root / "data" / "models" / "qwen3_5_9b_sft_with_tools_dentex_alone_bf16")
    touch(world.root / "data" / "splits" / "dentex_dev.json", b'{"keys": []}')
    yolo = touch(world.root / "data" / "models" / "yolo_cv_best" / "weights" / "best.pt")
    # run_grpo writes GROUNDING_MODEL_PATH into os.environ on purpose. setenv-then-delenv makes monkeypatch restore "absent"
    # at teardown, so the value cannot leak into later tests (the ToothGrounder singleton would pick it up).
    monkeypatch.setenv("GROUNDING_MODEL_PATH", "placeholder")
    monkeypatch.delenv("GROUNDING_MODEL_PATH")
    run_grpo.resolve_grpo_inputs(grpo_args(world, track="with_tools"), str(world.root / "data"))
    import os
    assert Path(os.environ["GROUNDING_MODEL_PATH"]) == yolo.resolve()


def test_ensure_pool_images_raises_when_a_pool_image_stays_missing(world):
    import pandas as pd
    import run_grpo

    touch(world.root / "data" / "dentex" / "images" / "1.png")
    df = pd.DataFrame({"id": [1, 2], "file_name": ["1.png", "2.png"], "local_path": [None, None]})
    with pytest.raises(SystemExit) as ei:
        run_grpo.ensure_pool_images(df, "dentex", str(world.root / "data"))
    assert "1 of 2 images missing" in str(ei.value)
    touch(world.root / "data" / "dentex" / "images" / "2.png")
    out = run_grpo.ensure_pool_images(df, "dentex", str(world.root / "data"))
    assert out["local_path"].notna().all()


# ---------------------------------------------------------------------------------------------------------------------
# scripts/dataset_statistics.py
# ---------------------------------------------------------------------------------------------------------------------
def test_dataset_statistics_counts_from_files_not_from_old_docs(tmp_path):
    import dataset_statistics as ds

    d = tmp_path / "traces"
    gt = lambda *dx: [{"tooth": 11, "diagnosis": x} for x in dx]  # noqa: E731
    write_traces(d / "train_cot_traces_dentex.jsonl", [
        {"dataset": "dentex", "image_id": 1, "ground_truth": gt("caries", "caries")},
        {"dataset": "dentex", "image_id": 2, "ground_truth": gt("deep caries")},
        {"dataset": "dentex", "image_id": 2, "ground_truth": gt("deep caries")},          # duplicate key
        {"dataset": "dentex", "image_id": 7, "ground_truth": gt("caries")},               # dev image
    ])
    write_traces(d / "train_cot_traces_dentex_no_tools.jsonl", [
        {"dataset": "dentex", "image_id": 1, "ground_truth": gt("caries", "caries")},
        {"dataset": "dentex", "image_id": 2, "ground_truth": gt("deep caries")},
    ])
    write_traces(d / "train_cot_traces_healthy_dentex.jsonl", [{"dataset": "dentex", "image_id": 9, "ground_truth": []}])
    write_traces(d / "train_cot_traces_unverified.jsonl", [{"dataset": "dentex", "image_id": 99}])   # skipped
    stats = {s["file"]: s for s in ds.collect(d, {DEV_KEY})}
    assert "train_cot_traces_unverified.jsonl" not in stats
    s = stats["train_cot_traces_dentex.jsonl"]
    assert (s["traces"], s["images"], s["findings"], s["in_dev_split"]) == (4, 3, 4, 1)   # 1 + 1 + 1 + 2 = 4 findings over 3 images
    assert s["duplicate_keys"] == ["dentex::2"] and s["by_diagnosis"] == {"caries": 3, "deep caries": 1}
    h = stats["train_cot_traces_healthy_dentex.jsonl"]
    assert h["healthy"] == 1 and h["findings"] == 0
    (p,) = ds.parity(list(stats.values()))
    assert p["only_with_tools"] == ["dentex::7"]
    text = ds.render(list(stats.values()), [p], "data/splits/dentex_dev.json")
    assert "only with tools: dentex::7" in text and "Do not edit by hand" in text
