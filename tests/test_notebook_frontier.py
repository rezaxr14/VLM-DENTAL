"""Executes the code cells of notebooks/VLM_Dental_Kaggle_Frontier_Benchmarks.ipynb end to end against fakes:
a fake ``kaggle`` executable, a fake hosted runner (the REAL rendered task source runs through ``run_shard``), a fake Hub repo and a
synthetic cohort. Nothing touches the network, Kaggle or quota. It proves the notebook's wiring, not Kaggle's behaviour."""
import json
import os
import re
import stat
import sys
import types
from pathlib import Path

import pytest

from dental_agent import config
from dental_agent.external import hosted_driver
from dental_agent.external.persist import ExternalHub
from test_external_extensions import FakeApi, hub  # noqa: F401
from test_external_hosted import FakeCLI, IDS, MODELS, world  # noqa: F401

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "VLM_Dental_Kaggle_Frontier_Benchmarks.ipynb"


def cells():
    nb = json.loads(NOTEBOOK.read_text())
    return [("".join(c["source"]), c["cell_type"]) for c in nb["cells"]]


def override(src, **values):
    for name, value in values.items():
        src, n = re.subn(rf"^{name} = .*?(?=\s{{2,}}#|$)", f"{name} = {value!r}", src, count=1, flags=re.M)
        assert n == 1, f"{name} is not a settings line"
    return src


@pytest.fixture
def runner(world, monkeypatch, hub, tmp_path):
    ext, api = hub
    # a fake `kaggle` executable (version / benchmarks help / quota)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "kaggle"
    exe.write_text("#!/bin/sh\ncase \"$*\" in\n *--version*) echo 'Kaggle API 2.2.5';;\n *'benchmarks --help'*) echo usage;;\n"
                   " *quota*) echo '[{\"period\":\"Daily\",\"remaining\":\"$9.5\",\"used\":\"$0.5\",\"total\":\"$10\",\"refillAt\":\"2026-11-16T00:00:00Z\"}]';;\nesac\n")
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    # the attached datasets, in the interactive layout
    mount = tmp_path / "in" / "datasets" / "rezanadimikj" / "dentex-panoramic"
    mount.parent.mkdir(parents=True)
    import shutil
    shutil.copytree(world["mount"], mount)
    world["input"] = tmp_path / "in" / "datasets" / "rezanadimikj"          # where the fake workers look for attached datasets
    envdir = tmp_path / "in" / "datasets" / "rezanadimikj" / "vlm-dental-env"
    envdir.mkdir(parents=True)
    (envdir / "env").write_text("HF_TOKEN=hf_fake\nHF_EVAL_REPO=owner/existing-results\n")
    images = world["images"].assign(local_path=[str(mount / "images" / f"{i}.png") for i in IDS])
    monkeypatch.setattr(config, "discover_env_datasets", lambda roots=(): [("rezanadimikj/vlm-dental-env", envdir / "env")])
    for key in ("HF_TOKEN", "HF_EVAL_REPO", "HF_TRACES_REPO"):
        monkeypatch.setenv(key, "x")
        monkeypatch.delenv(key)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "login", lambda token=None, **k: None, raising=False)
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda *a, **k: types.SimpleNamespace(whoami=lambda: {"name": "tester"}), raising=False)
    from dental_agent.utils import inputs
    monkeypatch.setattr(inputs, "KAGGLE_INPUT", tmp_path / "in")
    cli = FakeCLI(world, monkeypatch)
    monkeypatch.setattr(hosted_driver, "KaggleCLI", lambda: cli)
    real_build = hosted_driver.build_campaign                       # pre-staged checkout = pre-staged environment: no git clone, no pip
    monkeypatch.setattr(hosted_driver, "build_campaign", lambda *a, **k: real_build(*a, repo_path=str(Path(__file__).resolve().parents[1]), **k))
    fake_ev = types.SimpleNamespace(load_eval_split=lambda a: (images, world["annots"], world["cats"]),
                                    eligible_images=lambda df, limit=None: df if not limit else df.iloc[:limit])
    import scripts
    monkeypatch.setitem(sys.modules, "scripts.evaluate_models", fake_ev)
    monkeypatch.setattr(scripts, "evaluate_models", fake_ev, raising=False)
    monkeypatch.setattr(ExternalHub, "from_env", classmethod(lambda cls, repo=None, env=os.environ, repo_type=None: ext if repo else None))
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    return types.SimpleNamespace(cli=cli, api=api, tmp=tmp_path)


def run_notebook(r, monkeypatch, tmp_path, **settings):
    settings.setdefault("SPLIT", "test")                          # the shipped default is the 250-image cohort; its own test is below
    ns = {"__name__": "__notebook__"}
    for i, (src, kind) in enumerate(cells()):
        if kind != "code":
            continue
        src = src.replace("INSTALL_MISSING = True", "INSTALL_MISSING = False")
        src = src.replace('os.path.exists("/kaggle")', "False")                     # behave as a local run whatever this machine has
        if "# ----------------------------------------------------------------------------- WHAT TO RUN" in src:
            src = override(src, **settings)
            src = src.replace('ROOT = Path("/kaggle/working/kaggle_hosted") if IS_KAGGLE else Path("results/kaggle/hosted")', f"ROOT = Path({str(tmp_path / 'root')!r})")
        if i == 0:
            pass
        exec(compile(src, f"notebook-cell-{i}", "exec"), ns)
        if "HF_OK = False" in src:                                   # the fake login cannot succeed; the Hub is the fake repo
            pass
    return ns


def test_notebook_runs_end_to_end_and_writes_its_report(runner, world, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    monkeypatch.setenv("HF_EVAL_REPO", "owner/existing-results")
    ns = run_notebook(runner, monkeypatch, tmp_path, CONDITIONS=["kaggle_zero_shot"], MODELS=MODELS, SMOKE_MODEL="claude-haiku-5-5",
                      SHARD_SIZE=2, WORKERS=2, RUN_CAMPAIGN=True, WAIT_MINUTES=1, POLL_SECONDS=0, N_BOOT=50)
    assert ns["IMAGES_ON_KAGGLE"] and ns["IMAGE_DATASETS"] == ["rezanadimikj/dentex-panoramic"]
    assert ns["SMOKE_OK"] is True and ns["CAMPAIGN_STATE"] == "done"
    assert all(v == "4/4" for v in ns["REPORT"]["cells"].values()) and len(ns["REPORT"]["cells"]) == 2
    assert set(ns["EVALUATED"]) == {f"{m}/kaggle_zero_shot" for m in MODELS} and len(ns["CELLS"]) == 2
    assert ns["STATUS"]["scored_cells"] == 2 and ns["STATUS"]["smoke_ok"] is True
    report = (tmp_path / "root" / "report.md").read_text()
    assert "smoke test: ok" in report and "| Cell |" in report and (tmp_path / "root" / "comparison.md").is_file()
    assert any(n.startswith("external/dentex/test/") for n in runner.api.files)          # collected results reached the (fake) Hub
    assert any(n.startswith("external/campaign/") for n in runner.api.files)             # and so did the campaign state
    assert all(s.env_dataset == "rezanadimikj/vlm-dental-env" for s in ns["CAMPAIGN"].specs.values())


def test_campaign_is_blocked_until_the_smoke_test_has_passed(runner, monkeypatch, tmp_path):
    ns = run_notebook(runner, monkeypatch, tmp_path, CONDITIONS=["kaggle_zero_shot"], MODELS=MODELS, RUN_SMOKE_TEST=False, RUN_CAMPAIGN=True,
                      SHARD_SIZE=2, POLL_SECONDS=0, N_BOOT=50)
    assert ns["CAMPAIGN_STATE"] == "blocked by smoke test" and runner.cli.pushes == []        # nothing pushed, nothing scheduled
    assert "NOT OK" in (tmp_path / "root" / "report.md").read_text()                          # and the report still exists


def test_a_second_session_resumes_from_the_hub_without_spending_again(runner, monkeypatch, tmp_path):
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    monkeypatch.setenv("HF_EVAL_REPO", "owner/existing-results")
    kw = dict(CONDITIONS=["kaggle_zero_shot"], MODELS=MODELS, SMOKE_MODEL="claude-haiku-5-5", SHARD_SIZE=2, WORKERS=2, RUN_CAMPAIGN=True,
              WAIT_MINUTES=1, POLL_SECONDS=0, N_BOOT=50)
    run_notebook(runner, monkeypatch, tmp_path / "s1", **kw)
    pushes, runs = len(runner.cli.pushes), sum(len(v) for v in runner.cli.runs.values())
    ns = run_notebook(runner, monkeypatch, tmp_path / "s2", **kw)                           # a fresh container: empty working directory
    assert len(runner.cli.pushes) == pushes and sum(len(v) for v in runner.cli.runs.values()) == runs
    assert len(ns["CELLS"]) == 2 and all(len(r) == 4 for r in ns["CELLS"].values())


def test_run_all_degrades_to_a_report_when_the_cohort_cannot_load(runner, monkeypatch, tmp_path):
    def boom(args):
        raise FileNotFoundError("DENTEX data not found")
    monkeypatch.setattr(sys.modules["scripts.evaluate_models"], "load_eval_split", boom)
    ns = run_notebook(runner, monkeypatch, tmp_path, CONDITIONS=["kaggle_zero_shot"], MODELS=MODELS, RUN_CAMPAIGN=True, N_BOOT=50)
    assert ns["CAMPAIGN"] is None and "cohort_error" in ns["STATUS"] and runner.cli.pushes == []
    report = (tmp_path / "root" / "report.md").read_text()
    assert "preflight: NOT OK" in report and "(no campaign was built)" in report


def test_run_all_with_images_not_on_kaggle_stops_before_spending_and_says_so(runner, world, monkeypatch, tmp_path):
    from dental_agent.utils import inputs
    monkeypatch.setattr(inputs, "KAGGLE_INPUT", tmp_path / "nowhere")                     # no attached image dataset
    ns = run_notebook(runner, monkeypatch, tmp_path, CONDITIONS=["kaggle_zero_shot"], MODELS=MODELS, RUN_CAMPAIGN=True, N_BOOT=50)
    assert ns["IMAGES_ON_KAGGLE"] is False and ns["CAMPAIGN"] is None and ns["SMOKE_OK"] is False and runner.cli.pushes == []
    assert "images on Kaggle: NOT OK" in (tmp_path / "root" / "report.md").read_text()


def test_notebook_on_the_250_image_cohort_carries_its_settings_prompts_and_scoring(runner, world, monkeypatch, tmp_path):
    from dental_agent.data import official_test250 as o
    monkeypatch.setenv("HF_TOKEN", "hf_fake")
    monkeypatch.setenv("HF_EVAL_REPO", "owner/existing-results")
    settings = dict(coco_version="v1", mapping_version="v1", mapping_sha256="m" * 64, source_zip_sha256=None, scoring="test250_scoring_v2",
                    prompt_classes=list(o.PROMPT_CLASSES))
    import dental_agent.harness.notebook_eval as ne
    resolved = []
    monkeypatch.setattr(ne, "prepare_inputs", lambda data, split, dev, api, download, token, files=None: resolved.append(split) or {"data_dir": str(data)})
    fake_ev = sys.modules["scripts.evaluate_models"]
    plain = fake_ev.load_eval_split

    def cohort_split(args):
        args.prompt_dataset, args.test250_settings = o.SPLIT, settings        # what official_test250.load_cohort_split puts on args
        return plain(args)
    monkeypatch.setattr(fake_ev, "load_eval_split", cohort_split)
    ns = run_notebook(runner, monkeypatch, tmp_path, SPLIT=o.SPLIT, CONDITIONS=["kaggle_zero_shot"], MODELS=MODELS, SMOKE_MODEL="claude-haiku-5-5",
                      SHARD_SIZE=2, WORKERS=2, RUN_CAMPAIGN=True, WAIT_MINUTES=1, POLL_SECONDS=0, N_BOOT=50)
    assert resolved == [o.SPLIT] and ns["COHORT"] == settings
    assert all(sp.split == o.SPLIT and sp.cohort_settings == settings for sp in ns["CAMPAIGN"].specs.values())
    recs = [r for cell in ns["CELLS"].values() for r in cell]
    assert recs and all(r["split"] == o.SPLIT and r["scoring_profile"] == "test250_scoring_v2" and r["eval_settings"]["official_test250"] == settings for r in recs)
    assert ns["CAMPAIGN_STATE"] == "done" and any(f"/{o.SPLIT}/" in n for n in runner.api.files)          # results are filed under the cohort's split
