"""Hosted (non-live) campaign, tested offline end to end: the REAL rendered task source runs against a fake ``kaggle_benchmarks`` and a
fake ``KaggleCLI`` whose "remote" is a folder; the model endpoint is scripted. No network, no Kaggle, no quota."""
import json
import shutil
import sys
import types
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from dental_agent.external import hosted_driver, hosted_runtime, importer, sharding
from dental_agent.external.hosted_driver import HostedCampaign
from dental_agent.external.kaggle_cli import CLIError, KaggleCLI, RunRow, TaskStatus, parse_status
from dental_agent.external.kaggle_provider import OpenAICompatBackend, ProviderError
from dental_agent.external.model_profiles import HOSTED_CATALOG, hosted_model
from dental_agent.external.sharding import ShardSpec, build_specs, plan_shards, relative_paths, render_task_source, task_name
from dental_agent.tools.synthetic import make_synthetic_dental_image
from test_external_kaggle import EMPTY, FINAL
from test_external_extensions import hub  # noqa: F401  (a fake of an existing Hub repo)
from test_external_resume import Scripted, ok

REPO = Path(__file__).resolve().parents[1]
IDS = [100, 101, 102, 103]
MODELS = ["claude-haiku-5-5", "gemini-3.8-flash"]


@pytest.fixture
def world(tmp_path):
    """A mounted dataset (``<input>/dentex-panoramic/images/<id>.png``), the local image table, annotations and categories."""
    mount = tmp_path / "input" / "dentex-panoramic"
    rows = []
    for i in IDS:
        p = mount / "images" / f"{i}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        make_synthetic_dental_image(width=512, height=256, seed=i).save(p)
        rows.append({"id": i, "local_path": str(p)})
    images = pd.DataFrame(rows)
    annots = pd.DataFrame([
        {"id": 1, "image_id": 100, "category_id_1": 1, "category_id_2": 6, "category_id_3": 1, "bbox": [200, 100, 40, 50]},
        {"id": 2, "image_id": 101, "category_id_1": 3, "category_id_2": 8, "category_id_3": 4, "bbox": [300, 150, 45, 55]}])
    cats = pd.DataFrame([{"id": 1, "name": "Caries"}, {"id": 2, "name": "Deep Caries"}, {"id": 3, "name": "Periapical Lesion"}, {"id": 4, "name": "Impacted Tooth"}])
    return {"input": tmp_path / "input", "mount": mount, "images": images, "annots": annots, "cats": cats, "tmp": tmp_path}


def make_specs(world, condition="kaggle_zero_shot", shard_size=2, **kw):
    rel = relative_paths(world["images"], IDS, world["mount"])
    return build_specs(condition=condition, image_ids=IDS, rel_paths=rel, image_datasets=["owner/dentex-panoramic"],
                       cohort_sha256="c" * 64, split="test", shard_size=shard_size, workers=2, repo_path=str(REPO),
                       **kw)


class FakeCLI(KaggleCLI):
    """Executes pushed task sources through the real ``run_shard`` and keeps outputs in a 'remote' folder."""

    def __init__(self, world, monkeypatch, scripts=None, default_model="gemini-3.8-flash"):
        self.world, self.scripts, self.default_model = world, scripts or {}, default_model
        self.remote = world["tmp"] / "remote"
        self.sources, self.versions, self.runs, self.attached, self.pushes, self.counter = {}, {}, {}, {}, [], 0
        kb = types.ModuleType("kaggle_benchmarks")
        kb.task = lambda name=None, description=None: (lambda fn: types.SimpleNamespace(run=lambda llm: fn(llm)))
        kb.llm = None
        self.kb = kb
        monkeypatch.setitem(sys.modules, "kaggle_benchmarks", kb)
        real = hosted_runtime.run_shard
        self._real = real

        def patched(spec, model, **kw):
            cli_slug = next(m.cli_slug for m in HOSTED_CATALOG if m.raw_slug == model)
            backend = OpenAICompatBackend(model, base_url="http://x", api_key="k", post=self.scripts.get(cli_slug) or Scripted([ok(FINAL)]), adaptive=True)
            return real(spec, model, working=self._working, roots=(str(world["input"]),), backend=backend, hub=None, **kw)
        monkeypatch.setattr(hosted_runtime, "run_shard", patched)

    def push(self, task, source, datasets=(), wait_s=60):
        self.versions[task] = self.versions.get(task, 0) + 1
        self.sources[task] = Path(source).read_text()
        self.attached[task] = list(datasets)
        self.pushes.append(task)
        self._exec(task, self.default_model)                   # a real push also runs the default model once
        return self.versions[task]

    def _exec(self, task, model):
        self.counter += 1
        self._working = self.remote / task / str(self.versions[task]) / model / str(1000 + self.counter)
        self._working.mkdir(parents=True)
        self.kb.llm = types.SimpleNamespace(model=hosted_model(model).raw_slug)
        exec(compile(self.sources[task], task, "exec"), {"__name__": "hosted_task"})
        self.runs.setdefault(task, {})[model] = "completed"

    def run_text(self, task, model):
        self._exec(task, model)
        return True, "Submitted run(s)\n  %s: Scheduled" % model

    def status(self, task):
        return TaskStatus(task, self.versions[task], {m: RunRow(m, s) for m, s in self.runs.get(task, {}).items()})

    def download(self, task, out_dir, model=None):
        (Path(out_dir) / task).mkdir(parents=True, exist_ok=True)
        if (self.remote / task).is_dir():
            shutil.copytree(self.remote / task, Path(out_dir) / task, dirs_exist_ok=True)
        return Path(out_dir) / task


def campaign(world, monkeypatch, cli=None, **kw):
    cli = cli or FakeCLI(world, monkeypatch)
    return HostedCampaign(world["tmp"] / "campaign", cli, make_specs(world, **kw), MODELS, log=lambda s: None), cli


# ----------------------------------------------------------------------------- planning and task source

def test_shards_are_stable_contiguous_slices():
    assert plan_shards([3, 1, 2, 5, 4], 2) == [[1, 2], [3, 4], [5]]
    assert task_name("abcdef123", "kaggle_tools_rules_skills", 3) == "vlmdental-abcdef-tools-rules-skills-s03"
    assert len(task_name("abcdef123", "kaggle_tools_rules_skills", 3)) <= 40


def test_spec_round_trip_and_attachments(world):
    specs = make_specs(world, condition="kaggle_tools", env_dataset="owner/vlm-dental-env", hf_results_repo="Reza-Nadimi/results")
    s = specs[0]
    assert ShardSpec.from_json(s.to_json()) == s and s.sha() == ShardSpec.from_json(s.to_json()).sha()
    assert s.attach == ["owner/dentex-panoramic", "owner/vlm-dental-env"] and s.needs_grounder
    assert [step["args"][0] for step in s.install_steps] == ["torch", "torch", "ultralytics"] and "huggingface_hub" in s.requirements
    assert [st["args"][0] for st in make_specs(world)[0].install_steps] == ["torch", "torch"] and not make_specs(world)[0].needs_grounder


def test_rendered_task_source_compiles_and_carries_no_secret(world):
    src = render_task_source(make_specs(world, env_dataset="owner/vlm-dental-env")[0])
    compile(src, "t.py", "exec")
    assert "HF_TOKEN" not in src and "MODEL_PROXY_API_KEY" not in src and "owner/vlm-dental-env" in src


def test_relative_paths_rejects_images_outside_the_dataset(world, tmp_path):
    with pytest.raises(ValueError):
        relative_paths(world["images"], IDS, tmp_path / "elsewhere")


# ----------------------------------------------------------------------------- the whole campaign

def test_campaign_runs_every_model_and_scores_locally(world, monkeypatch):
    camp, cli = campaign(world, monkeypatch)
    pushed = camp.push_missing()
    assert set(pushed.values()) == {"pushed v1"} and cli.attached[next(iter(cli.attached))] == ["owner/dentex-panoramic"]
    # the default-model run created by the push is adopted, not scheduled a second time
    n = camp.schedule()
    assert n == {"scheduled": 2, "rejected": 0, "skipped_cap": 0}          # 2 shards x (2 models - the adopted one)
    assert not any(v for v in camp.refresh().values() if v is None)
    out = world["tmp"] / "out"
    assert camp.collect(out)["collected"] == 4
    report = camp.report(out)
    assert report["cells"] == {f"{m}/kaggle_zero_shot": "4/4" for m in MODELS} and not report["paused"]
    results = camp.evaluate(out, world["annots"], world["cats"])
    assert set(results) == {f"{m}/kaggle_zero_shot" for m in MODELS}
    for res in results.values():
        assert len(res["records"]) == 4 and not res["errors"] and not res["pending"]
    # identity built from the hosted run equals the one rebuilt locally
    runs = [r for r in camp.state["runs"].values() if r.get("collected")]
    assert {r["manifest"]["settings_fingerprint"] for r in runs if r["model_raw_slug"] == "anthropic/claude-haiku-5-5"} == \
        {importer.fingerprint(camp.expected_settings("claude-haiku-5-5", "kaggle_zero_shot"))}


def test_campaign_is_idempotent_across_sessions(world, monkeypatch):
    camp, cli = campaign(world, monkeypatch)
    camp.push_missing(); camp.schedule(); camp.refresh(); camp.collect(world["tmp"] / "out")
    n_pushes = len(cli.pushes)
    again, _ = campaign(world, monkeypatch, cli=cli)                         # a new session reads the state file
    assert set(again.push_missing().values()) == {"up to date"} and len(cli.pushes) == n_pushes
    assert again.schedule() == {"scheduled": 0, "rejected": 0, "skipped_cap": 0}
    assert again.collect(world["tmp"] / "out") == {"collected": 0, "failed": 0, "no_output": 0}


def test_inflight_cap_and_rejected_runs(world, monkeypatch):
    camp, cli = campaign(world, monkeypatch)
    camp.push_missing()
    cli.runs = {}                                                            # nothing has finished yet
    for task in list(camp.state["runs"]):
        camp.state["runs"][task]["status"] = "running"
    assert camp.schedule(max_inflight=1)["skipped_cap"] >= 1
    camp2, cli2 = campaign(world, monkeypatch)
    camp2.push_missing()
    cli2.run_text = lambda task, model: (False, "400 Client Error: Bad Request")
    assert camp2.schedule()["rejected"] == 2 and all(r["status"] == "rejected" for r in camp2.state["runs"].values() if r["status"] == "rejected")


# ----------------------------------------------------------------------------- pause, remainders

def test_a_capped_shard_keeps_its_finished_images_and_a_remainder_finishes_the_model(world, monkeypatch):
    capped = Scripted([ok(FINAL), ProviderError("HTTP 429: daily quota exceeded", kind="rate_limit", status=429)])
    cli = FakeCLI(world, monkeypatch, scripts={"claude-haiku-5-5": capped})
    camp, _ = campaign(world, monkeypatch, cli=cli, shard_size=4)
    camp.push_missing(); camp.schedule(); camp.refresh()
    out = world["tmp"] / "out"
    camp.collect(out)
    assert camp.report(out)["paused"] and camp.paused_until()[0]["kind"] == "quota"
    assert camp.missing(out, "claude-haiku-5-5", "kaggle_zero_shot") and not camp.missing(out, "gemini-3.8-flash", "kaggle_zero_shot")
    new = camp.plan_remainders(out)
    assert len(new) == 1 and camp.task_models[new[0]] == ["claude-haiku-5-5"]
    cli.scripts["claude-haiku-5-5"] = Scripted([ok(FINAL)])                  # the allowance refilled
    camp.push_missing(); camp.schedule(); camp.refresh(); camp.collect(out)
    assert not camp.missing(out, "claude-haiku-5-5", "kaggle_zero_shot")
    assert camp.plan_remainders(out) == []
    assert len(camp.evaluate(out, world["annots"], world["cats"])["claude-haiku-5-5/kaggle_zero_shot"]["records"]) == 4


def test_remainder_plan_survives_a_new_session(world, monkeypatch):
    capped = Scripted([ok(FINAL), ProviderError("HTTP 429: daily quota exceeded", kind="rate_limit", status=429)])
    cli = FakeCLI(world, monkeypatch, scripts={"claude-haiku-5-5": capped})
    camp, _ = campaign(world, monkeypatch, cli=cli, shard_size=4)
    camp.push_missing(); camp.schedule(); camp.refresh(); camp.collect(world["tmp"] / "out")
    name = camp.plan_remainders(world["tmp"] / "out")[0]
    again, _ = campaign(world, monkeypatch, cli=cli, shard_size=4)
    assert name in again.specs and again.task_models[name] == ["claude-haiku-5-5"]


# ----------------------------------------------------------------------------- failure is explicit, never silent

def test_a_missing_dataset_fails_before_any_request(world, tmp_path):
    spec = make_specs(world)[0]
    post = Scripted([ok(FINAL)])
    backend = OpenAICompatBackend("m", base_url="http://x", api_key="k", post=post)
    work = tmp_path / "w"
    manifest = hosted_runtime.run_shard(spec, "m", working=work, roots=(str(tmp_path / "nothing"),), backend=backend, hub=None, log=lambda s: None)
    assert manifest["status"] == "failed" and manifest["stage"] == "images" and not post.payloads
    assert json.loads((work / "shard_manifest.json").read_text())["status"] == "failed"


def test_failed_run_without_output_is_reported_not_merged(world, monkeypatch):
    camp, cli = campaign(world, monkeypatch)
    camp.push_missing(); camp.schedule()
    shutil.rmtree(cli.remote)                                                # Kaggle returned nothing for these runs
    cli.remote.mkdir()
    for run in camp.state["runs"].values():
        run["status"] = "running"
    cli.runs = {t: {m: "failed" for m in MODELS} for t in camp.specs}
    camp.refresh()
    assert camp.collect(world["tmp"] / "out")["no_output"] == 4
    assert camp.report(world["tmp"] / "out")["runs"] == {"no_output": 4}


def test_hosted_clone_that_differs_from_this_checkout_is_rejected_on_import(world, monkeypatch):
    camp, cli = campaign(world, monkeypatch)
    camp.push_missing(); camp.schedule(); camp.refresh()
    out = world["tmp"] / "out"
    camp.collect(out)
    from dental_agent.external import identity
    monkeypatch.setattr(identity, "prompt_hashes", lambda *a, **k: {"system_prompt": "x" * 64, "first_user_prompt": "y" * 64})
    with pytest.raises(importer.IdentityMismatch):
        camp.evaluate(out, world["annots"], world["cats"], models=["claude-haiku-5-5"])


# ----------------------------------------------------------------------------- the Hub and the image datasets

def test_collected_results_round_trip_through_the_hub_to_a_fresh_machine(world, monkeypatch, hub):
    ext, api = hub
    camp, cli = campaign(world, monkeypatch)
    camp.push_missing(); camp.schedule(); camp.refresh()
    out = world["tmp"] / "out"
    camp.collect(out)
    assert camp.push_to_hub(ext, out, world["tmp"] / "work") == {f"{m}/kaggle_zero_shot": 4 for m in MODELS}
    assert camp.push_to_hub(ext, out, world["tmp"] / "work") == {f"{m}/kaggle_zero_shot": 4 for m in MODELS}     # idempotent
    fresh_out = world["tmp"] / "fresh"                                           # a new session on another machine: no local results
    camp2 = HostedCampaign(world["tmp"] / "campaign2", cli, make_specs(world), MODELS, log=lambda s: None)
    camp2.state["runs"] = {k: {**v, "model_raw_slug": v.get("model_raw_slug")} for k, v in camp.state["runs"].items()}
    assert camp2.restore_from_hub(ext, fresh_out, world["tmp"] / "work2") == {f"{m}/kaggle_zero_shot": 4 for m in MODELS}
    assert camp2.restore_from_hub(ext, fresh_out, world["tmp"] / "work3") == {}                                   # nothing new the second time
    results = camp2.evaluate(fresh_out, world["annots"], world["cats"])
    assert all(len(r["records"]) == 4 for r in results.values())


def test_image_datasets_are_read_back_from_where_they_are_mounted(world, tmp_path):
    mount = tmp_path / "input2" / "datasets" / "rezanadimikj" / "dentex-panoramic"
    mount.parent.mkdir(parents=True)
    shutil.copytree(world["mount"], mount)
    images = world["images"].assign(local_path=[str(mount / "images" / f"{i}.png") for i in IDS])
    ids, rel = sharding.locate_image_datasets(images, IDS, [mount])
    assert ids == ["rezanadimikj/dentex-panoramic"] and rel["100"] == "images/100.png"
    with pytest.raises(ValueError, match="not inside an attached"):
        sharding.locate_image_datasets(world["images"], IDS, [mount])
    flat = tmp_path / "input3" / "dentex-panoramic"
    shutil.copytree(world["mount"], flat)
    with pytest.raises(ValueError, match="cannot tell the owner"):
        sharding.locate_image_datasets(world["images"].assign(local_path=[str(flat / "images" / f"{i}.png") for i in IDS]), IDS, [flat])


# ----------------------------------------------------------------------------- the .env dataset

def test_env_dataset_loads_names_only_and_never_overrides_the_runner(world, tmp_path):
    folder = tmp_path / "input" / "vlm-dental-env"
    folder.mkdir(parents=True)
    (folder / ".env").write_text("# comment\nHF_TOKEN=hf_secretvalue\nHF_EVAL_REPO='Reza-Nadimi/vlm-dental-evaluations'\nMODEL_PROXY_API_KEY=evil\nEMPTY=\n")
    env = {"MODEL_PROXY_API_KEY": "real"}
    loaded = hosted_runtime.load_env_dataset("owner/vlm-dental-env", (str(tmp_path / "input"),), env)
    assert loaded == ["HF_EVAL_REPO", "HF_TOKEN"] and env["MODEL_PROXY_API_KEY"] == "real" and env["HF_TOKEN"] == "hf_secretvalue"
    with pytest.raises(hosted_runtime.ShardFailed):
        hosted_runtime.load_env_dataset("owner/not-attached", (str(tmp_path / "input"),), {})
    assert hosted_runtime.load_env_dataset(None, (), {}) == []


# ----------------------------------------------------------------------------- CLI parsing (real output samples)

STATUS = """Task:     vlm-dental-probe
Version:  3
Status:   Completed
Created:  2026-11-14 20:53:28

Datasets: rezanadimikj/dentex-panoramic
Model                         Status          Started             Ended
───────────────────────────── ─────────────── ─────────────────── ───────────────────
grok-4.7                      Completed       2026-11-14 20:58:41 2026-11-14 20:58:52
gemini-3.1-flash-lite-preview Running         2026-11-14 20:57:46
claude-sonnet-5-5-default     Failed          2026-11-14 20:57:27 2026-11-14 20:57:37

View logs: kaggle b t log vlm-dental-probe [-m <model>]"""


def test_status_parsing_matches_real_output():
    s = parse_status("vlm-dental-probe", STATUS)
    assert s.version == 3 and set(s.rows) == {"grok-4.7", "gemini-3.1-flash-lite-preview", "claude-sonnet-5-5-default"}
    assert s.rows["grok-4.7"].terminal and s.rows["grok-4.7"].ended == "2026-11-14 20:58:52"
    assert not s.rows["gemini-3.1-flash-lite-preview"].terminal and s.rows["claude-sonnet-5-5-default"].status == "failed"


def test_cli_push_run_download_and_error_handling():
    calls = []

    def runner(args, timeout):
        calls.append(args)
        verb = args[3]
        if verb == "push":
            return 0, "Pushed new version of t1\n   Task Details:  https://www.kaggle.com/benchmarks/tasks/rezanadimikj/t1/7\nAttached Kaggle dataset(s): a/b"
        if verb == "run":
            return 0, "Submitted run(s) for task 't1'.\n   m1: Scheduled" if args[-1] == "m1" else "400 Client Error: Bad Request"
        return 0, ""
    cli = KaggleCLI(runner=runner)
    assert cli.push("t1", "f.py", datasets=["a/b", "c/d"]) == 7
    assert calls[0].count("-d") == 2 and calls[0][:5] == ["kaggle", "benchmarks", "tasks", "push", "t1"]
    assert cli.run("t1", "m1") is True and cli.run("t1", "m2") is False
    with pytest.raises(CLIError):
        KaggleCLI(runner=lambda a, t: (1, "Unauthorized")).push("t1", "f.py")
    assert KaggleCLI(runner=lambda a, t: (0, "Slug  Display\n-----\nm1  M One\nm2  M Two\nm1  dup")).models() == ["m1", "m2"]
    assert cli.download("t1", "/tmp/o", model="m1") == Path("/tmp/o/t1") and calls[-1][-2:] == ["-s", "-f"] and "-m" in calls[-1]


def test_build_campaign_reads_datasets_back_and_fails_early_without_grounder_weights(world, tmp_path, monkeypatch):
    mount = tmp_path / "in" / "datasets" / "rezanadimikj" / "dentex-panoramic"
    mount.parent.mkdir(parents=True)
    shutil.copytree(world["mount"], mount)
    images = world["images"].assign(local_path=[str(mount / "images" / f"{i}.png") for i in IDS])
    envs = tmp_path / "in" / "datasets" / "rezanadimikj" / "vlm-dental-env"
    envs.mkdir(parents=True)
    (envs / "env").write_text("HF_TOKEN=x\n")
    from dental_agent import config
    monkeypatch.setattr(config, "KAGGLE_INPUT", str(tmp_path / "in"))
    monkeypatch.setattr(config, "discover_env_datasets", lambda roots=(str(tmp_path / "in"),): [("rezanadimikj/vlm-dental-env", envs / "env")])
    camp = hosted_driver.build_campaign(tmp_path / "c", images, conditions=["kaggle_zero_shot", "kaggle_tools"], models=MODELS, shard_size=2,
                                        dataset_dirs=[mount], cli=FakeCLI(world, monkeypatch), log=lambda s: None)
    assert len(camp.specs) == 4 and {s.env_dataset for s in camp.specs.values()} == {"rezanadimikj/vlm-dental-env"}
    assert {tuple(s.attach) for s in camp.specs.values()} == {("rezanadimikj/dentex-panoramic", "rezanadimikj/vlm-dental-env")}
    monkeypatch.setattr(config, "discover_env_datasets", lambda roots=(): [])
    with pytest.raises(ValueError, match="YOLO weights"):
        hosted_driver.build_campaign(tmp_path / "c2", images, conditions=["kaggle_tools"], models=MODELS, dataset_dirs=[mount], log=lambda s: None)
    zero = hosted_driver.build_campaign(tmp_path / "c3", images, conditions=["kaggle_zero_shot"], models=MODELS, dataset_dirs=[mount],
                                        cli=FakeCLI(world, monkeypatch), log=lambda s: None)
    assert all(s.env_dataset is None and not s.needs_grounder for s in zero.specs.values())     # zero-shot needs no weights, no tokens


# ----------------------------------------------------------------------------- the worker can import the project

def _third_party_imports(entry="dental_agent.external.hosted_runtime"):
    """Module-level third-party imports reachable from ``entry`` (what a bare worker must have installed). Walks the real repository."""
    import ast
    stdlib = set(sys.stdlib_module_names)
    seen, third = set(), set()

    def modfile(name):
        p = REPO / name.replace(".", "/")
        return p / "__init__.py" if (p / "__init__.py").is_file() else (p.with_suffix(".py") if p.with_suffix(".py").is_file() else None)

    def visit(name):
        f = modfile(name)
        if name in seen or f is None:
            return
        seen.add(name)
        parts = name.split(".")
        for i in range(1, len(parts)):
            visit(".".join(parts[:i]))                       # importing a submodule runs every parent package __init__

        def walk(nodes):
            for n in nodes:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if isinstance(n, ast.Import):
                    found = [(a.name, []) for a in n.names]
                elif isinstance(n, ast.ImportFrom):
                    if n.level:
                        base = parts if f.name == "__init__.py" else parts[:-1]
                        mod = ".".join(base[:len(base) - (n.level - 1)] + ([n.module] if n.module else []))
                    else:
                        mod = n.module
                    found = [(mod, [a.name for a in n.names])]
                else:
                    for field in ("body", "orelse", "finalbody", "handlers"):
                        walk(getattr(n, field, []) or [])
                    continue
                for mod, names in found:
                    top = (mod or "").split(".")[0]
                    if top == "dental_agent":
                        visit(mod)
                        for nm in names:
                            visit(mod + "." + nm)
                    elif top and top not in stdlib and top != "__future__":
                        third.add(top)
        walk(ast.parse(f.read_text()).body)

    visit(entry)
    return third


def test_worker_requirements_cover_everything_the_import_chain_needs():
    pip_names = {"PIL": "pillow", "yaml": "pyyaml", "sklearn": "scikit-learn", "dotenv": "python-dotenv"}
    installed = {r.lower().replace("_", "-") for r in sharding.BASE_REQUIREMENTS}
    installed |= {a.lower() for st in sharding.TORCH_INSTALL_STEPS for a in st["args"]}
    optional_at_import = {"ultralytics"}                      # imported inside try/except; only tool conditions need it
    needed = {pip_names.get(m, m).lower().replace("_", "-") for m in _third_party_imports()} - optional_at_import
    assert needed <= installed, f"a bare worker cannot import: {sorted(needed - installed)}"


def test_every_worker_installs_torch_and_a_tool_worker_also_installs_the_grounder_package(world):
    zero = make_specs(world)[0]
    tools = make_specs(world, condition="kaggle_tools", env_dataset="owner/vlm-dental-env")[0]
    assert [st["args"][0] for st in zero.install_steps] == ["torch", "torch"] and "langgraph" in zero.requirements
    assert [st["args"][0] for st in tools.install_steps] == ["torch", "torch", "ultralytics"]


def test_a_worker_that_cannot_import_the_project_leaves_a_manifest_that_says_why(world, tmp_path, monkeypatch):
    src = render_task_source(make_specs(world)[0])
    kb = types.ModuleType("kaggle_benchmarks")
    kb.task = lambda name=None, description=None: (lambda fn: types.SimpleNamespace(run=lambda llm: fn(llm)))
    kb.llm = types.SimpleNamespace(model="x/y")
    monkeypatch.setitem(sys.modules, "kaggle_benchmarks", kb)
    monkeypatch.setitem(sys.modules, "dental_agent.external.hosted_runtime", None)       # makes the import fail
    monkeypatch.setenv("VLM_WORKING_DIR", str(tmp_path / "w"))
    with pytest.raises(ImportError):
        exec(compile(src, "task", "exec"), {"__name__": "t"})
    m = json.loads((tmp_path / "w" / "shard_manifest.json").read_text())
    assert m["status"] == "failed" and m["stage"] == "import" and "bootstrap" in m


# ----------------------------------------------------------------------------- staging the image dataset

def test_image_dataset_is_staged_with_images_only_and_published_privately(world, tmp_path):
    from dental_agent.external import image_dataset
    staged = image_dataset.stage_image_dataset(world["images"], IDS[:2], tmp_path / "stage", dataset_id="rezanadimikj/dentex-test-250")
    assert staged["images"] == 2 and sorted(p.name for p in (tmp_path / "stage" / "images").iterdir()) == ["100.png", "101.png"]
    meta = json.loads((tmp_path / "stage" / "dataset-metadata.json").read_text())
    assert meta["id"] == "rezanadimikj/dentex-test-250" and "isPrivate" not in meta and not list((tmp_path / "stage").glob("*.json.*"))
    assert not any(p.suffix in (".json",) and p.name != "dataset-metadata.json" for p in (tmp_path / "stage").rglob("*"))   # no annotations
    with pytest.raises(ValueError, match="owner/slug"):
        image_dataset.stage_image_dataset(world["images"], IDS, tmp_path / "x", dataset_id="noslash")
    calls = []

    def runner(args, timeout):
        calls.append(args)
        return (1, "404 not found") if args[2] == "status" and len(calls) == 1 else (0, "Starting upload\nYour private Dataset is being created")

    cli = KaggleCLI(runner)
    assert cli.dataset_publish(tmp_path / "stage", "rezanadimikj/dentex-test-250") == "created"
    assert calls[1][:3] == ["kaggle", "datasets", "create"] and "--public" not in calls[1]
    calls.clear()
    assert KaggleCLI(lambda a, t: (calls.append(a) or (0, "ref: x\nstatus: ready"))).dataset_publish(tmp_path / "stage", "o/s", "more") == "versioned"
    assert calls[1][:3] == ["kaggle", "datasets", "version"]
    with pytest.raises(CLIError):
        KaggleCLI(lambda a, t: (0, "ok") if a[2] == "status" else (1, "Invalid dataset slug")).dataset_publish(tmp_path / "stage", "o/s")


def test_a_new_session_adopts_the_hub_state_and_does_not_push_or_schedule_again(world, monkeypatch, hub):
    ext, _ = hub
    camp, cli = campaign(world, monkeypatch)
    camp.push_missing(); camp.schedule(); camp.refresh()
    camp.sync_state_to_hub(ext, world["tmp"] / "s1", label="rezanadimikj")
    camp.sync_state_to_hub(ext, world["tmp"] / "s2", label="rezanadimikj")                  # idempotent
    pushes_before, runs_before = len(cli.pushes), sum(len(v) for v in cli.runs.values())
    fresh = HostedCampaign(world["tmp"] / "fresh-campaign", cli, make_specs(world), MODELS, log=lambda s: None)
    assert fresh.restore_state_from_hub(ext, world["tmp"] / "s3", label="someone_else") is False        # another account's state is not adopted
    assert fresh.restore_state_from_hub(ext, world["tmp"] / "s4", label="rezanadimikj") is True
    assert set(fresh.push_missing().values()) == {"up to date"}
    assert fresh.schedule() == {"scheduled": 0, "rejected": 0, "skipped_cap": 0}
    assert len(cli.pushes) == pushes_before and sum(len(v) for v in cli.runs.values()) == runs_before    # nothing was spent twice
    assert fresh.restore_state_from_hub(ext, world["tmp"] / "s5", label="rezanadimikj") is False        # local state is never replaced
