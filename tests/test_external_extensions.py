"""The extension seams (conditions, tool packs, evaluators, plugins) and the Hugging Face persistence, tested offline.

The Hub is a fake of an EXISTING repo: it has no ``create_repo``, so any attempt to create one fails the test."""
import json
import sys
import textwrap
from types import SimpleNamespace

import pytest

from dental_agent.external import conditions, coordinator, evaluators, identity, importer, persist, plugins, tool_packs
from dental_agent.external.conditions import Condition, register_condition
from dental_agent.external.persist import ExternalHub
from test_external_kaggle import cohort, settings_for  # noqa: F401
from test_external_resume import backend_for, ok


# ----------------------------------------------------------------------------- a fake of an existing private Hub repo

class FakeApi:
    def __init__(self, tmp):
        self.tmp, self.files, self.n, self.commits = tmp, {}, 0, 0

    def repo_info(self, repo, repo_type=None):
        return SimpleNamespace(sha=str(self.n))

    def list_repo_files(self, repo, repo_type=None, revision=None):
        return list(self.files)

    def create_commit(self, repo, repo_type=None, operations=None, commit_message=None, parent_commit=None):
        assert parent_commit == str(self.n), "stale parent commit"
        for op in operations:
            self.files[op.path_in_repo] = op.path_or_fileobj
        self.n += 1
        self.commits += 1

    def download(self, repo, name, repo_type=None, revision=None, token=None):
        path = self.tmp / "cache" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.files[name])
        return str(path)


@pytest.fixture
def hub(tmp_path, monkeypatch):
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "CommitOperationAdd",
                        lambda path_in_repo, path_or_fileobj: SimpleNamespace(path_in_repo=path_in_repo, path_or_fileobj=path_or_fileobj), raising=False)
    from dental_agent.harness.notebook_eval import HubResults
    api = FakeApi(tmp_path)
    return ExternalHub(HubResults("owner/existing-results", api, api.download, repo_type="dataset")), api


def row(i, **kw):
    return {"kind": "transport", "image_id": i, "condition": "kaggle_zero_shot", "final_answer": f"answer {i}", **kw}


def test_round_trip_resume_and_shards_from_two_accounts(hub, tmp_path):
    ext, api = hub
    settings = settings_for("kaggle_zero_shot", [1, 2, 3, 4])
    assert ext.save(tmp_path / "a", settings, [], complete=False, shard="s00") is False            # nothing to push yet
    assert ext.save(tmp_path / "a", settings, [row(1), row(2)], complete=True, shard="s00")
    assert ext.save(tmp_path / "b", settings, [row(3)], complete=False, shard="s01")
    assert ext.save(tmp_path / "b", settings, [row(3), row(4)], complete=True, shard="s01")         # a later chunk extends the bundle
    assert [r["image_id"] for r in ext.restore(tmp_path / "r", settings, "s00")] == [1, 2]
    assert ext.restore(tmp_path / "x", settings, "s09") == []                                       # no bundle: empty, not an error
    assert [s.rsplit("/", 1)[1] for s in ext.shards(settings)] == ["s00", "s01"]
    merged = ext.restore_all_shards(tmp_path / "all", settings)
    assert sorted(merged) == [1, 2, 3, 4]
    assert all(name.startswith("external/dentex/test/fake_model-1/kaggle_zero_shot/run_") for name in api.files)


def test_the_hub_refuses_conflicts_and_never_loses_progress(hub, tmp_path):
    ext, _ = hub
    settings = settings_for("kaggle_zero_shot", [1, 2])
    ext.save(tmp_path / "a", settings, [row(1), row(2)], complete=True, shard="s00")
    with pytest.raises(ValueError):                                                                  # shorter copy
        ext.save(tmp_path / "b", settings, [row(1)], complete=False, shard="s00")
    with pytest.raises(ValueError):                                                                  # a different answer for a saved image
        ext.save(tmp_path / "c", settings, [row(1, final_answer="changed"), row(2)], complete=True, shard="s00")
    other = settings_for("kaggle_zero_shot", [1, 2], slug="other/model")
    assert ext.restore(tmp_path / "d", other, "s00") == []                                           # another model never sees this bundle


def test_two_different_records_for_one_image_are_never_merged(hub, tmp_path):
    ext, _ = hub
    settings = settings_for("kaggle_zero_shot", [1, 2])
    ext.save(tmp_path / "a", settings, [row(1)], complete=False, shard="s00")
    ext.save(tmp_path / "b", settings, [row(1, final_answer="other account")], complete=False, shard="s01")
    with pytest.raises(ValueError, match="two different records"):
        ext.restore_all_shards(tmp_path / "all", settings)


def test_persistence_is_off_without_a_token_or_a_repository():
    assert ExternalHub.from_env(None, {"HF_TOKEN": "hf_x"}) is None
    assert ExternalHub.from_env("owner/repo", {}) is None
    assert ExternalHub.from_env("owner/repo", {"HF_TOKEN": "your_token_here"}) is None


# ----------------------------------------------------------------------------- seams: conditions, packs, evaluators

@pytest.fixture
def registries(monkeypatch):
    monkeypatch.setattr(conditions, "CONDITIONS", dict(conditions.CONDITIONS))
    monkeypatch.setattr(tool_packs, "_PACKS", dict(tool_packs._PACKS))
    monkeypatch.setattr(evaluators, "_EVALUATORS", dict(evaluators._EVALUATORS))


def edit_pack():
    from dental_agent.tools.registry import ToolRegistry
    reg = ToolRegistry.create_default()
    reg.register("sharpen", lambda image, amount=1.0: image, "Unsharp mask the current view.", {"amount": 1.0})
    return reg


def test_existing_identities_do_not_change(cohort):
    s = settings_for("kaggle_tools", [100, 101, 102])
    assert "pack" not in s and "evaluator" not in s                    # saved checkpoints / transports keep their fingerprints


def test_registering_a_condition_protects_existing_names(registries):
    register_condition(Condition("kaggle_tools_edit_v1", True, "off", "unified", required=False, pack="edit_v1"))
    assert "kaggle_tools_edit_v1" in conditions.CONDITIONS and "kaggle_tools_edit_v1" not in conditions.required_conditions()
    with pytest.raises(ValueError, match="already registered"):
        register_condition(Condition("kaggle_tools", True, "off", "unified"))
    with pytest.raises(ValueError, match="already registered"):
        tool_packs.register_pack("default", edit_pack)


def test_a_tool_pack_changes_the_identity_and_reaches_the_loop(registries, cohort, tmp_path):
    images, _, _ = cohort
    tool_packs.register_pack("edit_v1", edit_pack)
    register_condition(Condition("kaggle_tools_edit_v1", True, "off", "unified", required=False, pack="edit_v1"))
    base, edit = settings_for("kaggle_tools", [100]), settings_for("kaggle_tools_edit_v1", [100])
    assert edit["pack"] == "edit_v1" and edit["prompt_sha256"] != base["prompt_sha256"]               # the tool descriptions are in the hash
    assert identity.fingerprint(edit) != identity.fingerprint(base)
    seen = []

    def run_image(cond, image_id, images_df, reply, cfg, registry):
        seen.append(sorted(registry.tools) if hasattr(registry, "tools") else registry.format_tool_descriptions())
        return SimpleNamespace(final_answer="x", format_ok=True, turns=[], tool_calls=0)

    backend, _ = backend_for([ok("x")])
    coordinator.run_cell(backend=backend, condition="kaggle_tools_edit_v1", images_df=images, image_ids=[100], out_dir=tmp_path,
                         settings=edit, log=lambda s: None, run_image=run_image)
    assert seen and "sharpen" in json.dumps(seen[0])
    with pytest.raises(ValueError, match="Unknown tool pack"):
        tool_packs.get_registry("missing")


def test_a_registered_evaluator_scores_its_own_condition(registries, cohort, tmp_path):
    images, annots, cats = cohort
    evaluators.register_evaluator("echo", lambda rec, a, c, **kw: {"image_id": rec["image_id"], "echo": rec["final_answer"]})
    register_condition(Condition("kaggle_zero_shot_echo", False, "off", "unified", required=False, evaluator="echo"))
    settings = settings_for("kaggle_zero_shot_echo", [100])
    assert settings["evaluator"] == "echo"
    backend, _ = backend_for([ok(EMPTY_ANSWER)])
    coordinator.run_cell(backend=backend, condition="kaggle_zero_shot_echo", images_df=images, image_ids=[100], out_dir=tmp_path,
                         settings=settings, log=lambda s: None)
    paths = importer.cell_paths(tmp_path, "fake/model-1", "kaggle_zero_shot_echo")
    out = importer.import_transport(paths["transport"], annots, cats, settings=settings, cohort_ids={100})
    assert [r["echo"] for r in out["records"]] == [importer.read_jsonl(paths["transport"])[0]["final_answer"]]
    with pytest.raises(ValueError, match="Unknown evaluator"):
        evaluators.get_evaluator("nope")


EMPTY_ANSWER = "no findings"


def test_plugins_are_plain_imports_that_register_things(registries, tmp_path, monkeypatch):
    (tmp_path / "my_seg_plugin.py").write_text(textwrap.dedent("""
        from dental_agent.external.conditions import Condition, register_condition
        register_condition(Condition("plugin_condition", False, "off", "unified", required=False))
    """))
    monkeypatch.syspath_prepend(str(tmp_path))
    assert plugins.load_plugins(["my_seg_plugin"]) == ["my_seg_plugin"] and "plugin_condition" in conditions.CONDITIONS
    sys.modules.pop("my_seg_plugin", None)
    with pytest.raises(ImportError):
        plugins.load_plugins(["not_a_plugin_module"])


# ----------------------------------------------------------------------------- hindsight export

def test_hindsight_rows_cover_successes_and_failures_and_export_to_the_traces_repo(hub, cohort, tmp_path):
    from dental_agent.external import hindsight
    images, annots, cats = cohort
    ext, api = hub
    settings = settings_for("kaggle_zero_shot", [100, 101])
    backend, _ = backend_for([ok("no findings")])
    coordinator.run_cell(backend=backend, condition="kaggle_zero_shot", images_df=images, image_ids=[100, 101], out_dir=tmp_path / "o",
                         settings=settings, log=lambda s: None)
    paths = importer.cell_paths(tmp_path / "o", "fake/model-1", "kaggle_zero_shot")
    evaluated = importer.import_transport(paths["transport"], annots, cats, settings=settings, cohort_ids={100, 101})["records"]
    summary = hindsight.export_hindsight(ext, tmp_path / "h", settings, evaluated)
    assert summary["pushed"] and summary["rows"] == 2 and sum(summary["outcomes"].values()) == 2
    assert all(n.startswith("hindsight/external/dentex/test/") for n in api.files)
    stored = [json.loads(l) for l in api.files[next(n for n in api.files if n.endswith("results.jsonl"))].decode().splitlines()]
    assert {r["image_id"] for r in stored} == {100, 101} and all(r["run_fingerprint"] == identity.fingerprint(settings) for r in stored)
    assert hindsight.export_hindsight(None, tmp_path / "h2", settings, evaluated)["pushed"] is False      # off without a hub


@pytest.mark.parametrize("rec,expected", [
    ({"external": {"error": {"x": 1}}}, "error"),
    ({"format_ok": False}, "format_failure"),
    ({"format_ok": True, "all_exact_match": True}, "success"),
    ({"format_ok": True, "fdi_correct": True}, "partial"),
    ({"format_ok": True}, "failure"),
])
def test_outcome_labels(rec, expected):
    from dental_agent.external import hindsight
    assert hindsight.outcome(rec) == expected
