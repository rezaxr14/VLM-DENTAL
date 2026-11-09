"""Flatten notebook results into separate area folders, aggregate them, and publish each area as separate Hub files."""

import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from dental_agent.evaluation import matrix  # noqa: E402

flatten = importlib.import_module("scripts.flatten_results")
publish = importlib.import_module("scripts.publish_aggregates")
TAG, STAGE, HARNESS = "workstation48_lora_bf16", "dentex_alone", "rules_skills_index"


def F(q, pos, diag):
    return {"quadrant": q, "tooth_position": pos, "diagnosis": diag, "confidence": 0.9}


def records(n, hits):
    return [{"image_id": i, "dataset": "dentex", "format_ok": True, "turns": 3, "tool_calls": 2, "ground_truth": [F(1, 6, "Caries")],
             "predictions": [F(1, 6, "Caries")] if i < hits else []} for i in range(n)]


def cohort(src, split, kind, condition, hits, size="canonical", mode="off", k=None, run="run_a", tag=TAG, stage=STAGE, n=20, ablation=""):
    """One run folder as NotebookEvaluations lays it out, holding the eval file the evaluator writes."""
    parts = [split, tag, stage, kind, condition] + ([f"harness_{mode}"] if mode != "off" else []) + ([f"k{k}"] if k is not None else []) + [size, run]
    folder = src.joinpath(*parts)
    folder.mkdir(parents=True, exist_ok=True)
    matrix.cell_path(folder, "dentex", split, condition, size, ablation).write_text("\n".join(json.dumps(r) for r in records(n, hits)) + "\n")
    return folder


def raw(src, split, kind, condition, mode="off", k=None, size="canonical", run="run_r", tag=TAG, stage=STAGE):
    parts = [split, tag, stage, kind, condition] + ([f"harness_{mode}"] if mode != "off" else []) + ([f"k{k}"] if k is not None else []) + [size, run]
    folder = src.joinpath(*parts)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "results.json").write_text(json.dumps({"kind": kind, "condition": condition, "mode": mode}))
    (folder / "summary.json").write_text("{}")
    (folder / "command.json").write_text("[]")  # not a result document: must not be copied
    return folder


@pytest.fixture
def tree(tmp_path):
    src = tmp_path / "evaluations" / "dentex"
    for split, n in (("test", 50), ("dev", 20)):
        cohort(src, split, "baseline", "base_with_tools", 4, n=n)
        cohort(src, split, "sft", "sft_with_tools", 10, n=n)
        cohort(src, split, "sft", "sft_with_tools", 9, size="original", n=n)
        cohort(src, split, "grpo", "grpo_with_tools", 14, k=4, n=n)
        cohort(src, split, "grpo", "grpo_with_tools", 16, k=4, mode=HARNESS, n=n)
        cohort(src, split, "grpo", "grpo_with_tools", 11, k=1, n=n)      # K other than the chosen one
    cohort(src, "dev", "grpo", "grpo_with_tools", 12, k=2)
    cohort(src, "dev", "grpo", "grpo_with_tools", 15, k=4, ablation="__no-nudge_crop", run="run_p")  # probe: not in the K sweep
    cohort(src, "test", "grpo", "grpo_no_tools", 13, k=4, n=50)
    cohort(src, "test", "sft", "sft_with_tools", 1, tag="dual_t4_fp16")                  # another profile: ignored
    cohort(src, "test", "sft", "sft_with_tools", 1, stage="other_stage")                 # another stage: ignored
    raw(src, "dev", "rules_skills", "rules_skills_sft_with_tools")
    raw(src, "dev", "rules_skills", "rules_skills_grpo_with_tools", k=4)
    raw(src, "dev", "rules_skills", "rules_skills_grpo_with_tools", k=4, mode=HARNESS)
    raw(src, "dev", "rules_skills", "rules_skills_grpo_with_tools", k=1)
    raw(src, "dev", "speculative", "speculative_grpo_with_tools", k=4, mode=HARNESS)
    return src


def names(folder):
    return sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file() and p.name != flatten.MANIFEST)


def run_flatten(tree, *extra):
    out = tree.parent / "flat"
    flatten.main(["--src", str(tree), "--out", str(out), *extra])
    return out


def test_harness_modes_get_separate_folders_and_controls(tree):
    out = run_flatten(tree, "--k", "4")
    assert sorted(p.name for p in out.iterdir()) == ["dev", "dev__harness_rules_skills_index", "k_sweep", "k_sweep__harness_rules_skills_index",
                                                       "rules_skills", "speculative", "test", "test__harness_rules_skills_index"]
    off = names(out / "test")
    assert off == ["eval_dentex_test_base_with_tools_canonical.jsonl", "eval_dentex_test_grpo_no_tools_canonical.jsonl",
                   "eval_dentex_test_grpo_with_tools_canonical.jsonl", "eval_dentex_test_sft_with_tools.jsonl",
                   "eval_dentex_test_sft_with_tools_canonical.jsonl"]
    hot = names(out / "test__harness_rules_skills_index")
    assert "eval_dentex_test_grpo_with_tools_canonical.jsonl" in hot and "eval_dentex_test_base_with_tools_canonical.jsonl" in hot
    assert not any("grpo_no_tools" in n for n in hot)                       # only what has a harness cohort plus the controls
    a = (out / "test" / "eval_dentex_test_grpo_with_tools_canonical.jsonl").read_text()
    b = (out / "test__harness_rules_skills_index" / "eval_dentex_test_grpo_with_tools_canonical.jsonl").read_text()
    assert a != b and len(a.splitlines()) == len(b.splitlines()) == 50     # the two policies never overwrite each other
    assert sorted(names(out / "k_sweep")) == ["k1/eval_dentex_dev_grpo_with_tools_canonical.jsonl", "k2/eval_dentex_dev_grpo_with_tools_canonical.jsonl",
                                              "k4/eval_dentex_dev_grpo_with_tools_canonical.jsonl"]
    assert names(out / "k_sweep__harness_rules_skills_index") == ["k4/eval_dentex_dev_grpo_with_tools_canonical.jsonl"]
    assert any(n.endswith("__no-nudge_crop.jsonl") for n in names(out / "dev"))        # probes ride along in dev, not in the sweep
    assert not any("no-nudge" in n for n in names(out / "k_sweep"))
    assert sorted(names(out / "rules_skills")) == [
        "rules_skills_grpo_with_tools__harness_rules_skills_index__k4__canonical/results.json",
        "rules_skills_grpo_with_tools__harness_rules_skills_index__k4__canonical/summary.json",
        "rules_skills_grpo_with_tools__k4__canonical/results.json", "rules_skills_grpo_with_tools__k4__canonical/summary.json",
        "rules_skills_sft_with_tools__canonical/results.json", "rules_skills_sft_with_tools__canonical/summary.json"]
    assert len(names(out / "speculative")) == 2
    manifest = json.loads((out / "test" / flatten.MANIFEST).read_text())
    assert manifest["tag"] == TAG and manifest["stage"] == STAGE and manifest["k"] == 4
    assert all(len(v["sha256"]) == 64 for v in manifest["files"].values())


def test_without_k_only_the_sweep_gets_grpo(tree):
    out = run_flatten(tree)
    assert not any("grpo" in n for n in names(out / "test")) and not any("grpo" in n for n in names(out / "dev"))
    assert len(names(out / "k_sweep")) == 3
    assert names(out / "rules_skills") == ["rules_skills_sft_with_tools__canonical/results.json", "rules_skills_sft_with_tools__canonical/summary.json"]
    assert not (out / "speculative").exists()                                         # its only cohort is a K=4 one


def test_changed_k_removes_stale_files_and_touches_nothing_else(tree):
    out = run_flatten(tree, "--k", "4")
    keep = out / "notes.txt"
    keep.write_text("mine")
    (out / "test" / "aggregator_output.csv").write_text("mine too")
    out = run_flatten(tree, "--k", "1")
    assert keep.read_text() == "mine" and (out / "test" / "aggregator_output.csv").is_file()
    assert json.loads((out / "test" / flatten.MANIFEST).read_text())["k"] == 1
    k1 = (out / "test" / "eval_dentex_test_grpo_with_tools_canonical.jsonl").read_text()
    assert sum('"predictions": [{' in line for line in k1.splitlines()) == 11            # K=1 file (11 hits), not the K=4 one (14)
    assert "grpo_no_tools" not in " ".join(names(out / "test"))                      # that K=4-only file is gone, not left behind
    assert not (out / "rules_skills" / "rules_skills_grpo_with_tools__k4__canonical").exists()


def test_conflicting_run_folders_stop_before_any_copy(tree):
    out = run_flatten(tree, "--k", "4")
    before = {p: p.read_bytes() for p in out.rglob("*") if p.is_file()}
    cohort(tree, "test", "sft", "sft_with_tools", 3, run="run_other")                  # same cohort again under another run id
    with pytest.raises(SystemExit, match="stopped before copying"):
        run_flatten(tree, "--k", "4")
    assert {p: p.read_bytes() for p in out.rglob("*") if p.is_file()} == before


def test_empty_selection_and_missing_source_fail_clearly(tree):
    with pytest.raises(SystemExit, match="no result files"):
        run_flatten(tree, "--tag", "nonexistent")
    with pytest.raises(SystemExit, match="not found"):
        flatten.main(["--src", str(tree / "missing"), "--out", str(tree.parent / "x")])


class FakeHub:
    """Minimal huggingface_hub double: optimistic parent commit, add-only operations."""
    def __init__(self):
        self.files, self.commits = {}, []

    def repo_info(self, repo, repo_type=None):
        return SimpleNamespace(sha=f"sha{len(self.commits)}")

    def list_repo_files(self, repo, repo_type=None, revision=None):
        return list(self.files)

    def create_commit(self, repo, repo_type=None, operations=None, commit_message=None, parent_commit=None):
        assert parent_commit == f"sha{len(self.commits)}"
        for op in operations:
            assert op.path_in_repo not in self.files, "an existing Hub file must never be overwritten"
            self.files[op.path_in_repo] = op.path_or_fileobj
        self.commits.append(commit_message)


def aggregate(out, monkeypatch, capsys):
    ae = importlib.import_module("scripts.aggregate_eval")
    ak = importlib.import_module("scripts.aggregate_k_sweep")
    for area, split in (("test", "test"), ("dev", "dev"), ("test__harness_rules_skills_index", "test"), ("dev__harness_rules_skills_index", "dev")):
        monkeypatch.setattr(sys, "argv", ["aggregate_eval", "--output-dir", str(out / area), "--dataset", "dentex", "--split", split, "--n-boot", "50"])
        ae.main()
    for area, ref in (("k_sweep", "dev"), ("k_sweep__harness_rules_skills_index", "dev__harness_rules_skills_index")):
        monkeypatch.setattr(sys, "argv", ["aggregate_k_sweep", "--sweep-dir", str(out / area), "--reference-dir", str(out / ref), "--n-boot", "50"])
        ak.main()
    capsys.readouterr()


def test_end_to_end_aggregate_then_publish_separate_files(tree, monkeypatch, capsys):
    out = run_flatten(tree, "--k", "4")
    aggregate(out, monkeypatch, capsys)
    hub = FakeHub()
    argv = ["--flat-dir", str(out), "--k", "4", "--tag", TAG, "--stage", STAGE, "--repo", "owner/results", "--repo-type", "model"]
    with pytest.raises(SystemExit, match="--k is required"):
        publish.main(["--flat-dir", str(out), "--tag", TAG], api=hub)
    results = publish.main(argv, api=hub)
    assert {r["area"] for r in results} == set(p.name for p in out.iterdir() if p.is_dir()) and {r["status"] for r in results} == {"uploaded"}
    paths = sorted(hub.files)
    assert all(p.startswith("aggregates/dentex/") and not p.startswith("evaluations/") for p in paths)   # per-cohort bundles are never touched
    by_area = {}
    for p in paths:
        by_area.setdefault(p.split("/")[2], []).append(p)
    assert len(by_area) == 8
    assert any(p.endswith("/matrix_dentex_test/matrix_summary.md") for p in by_area["test"])
    assert any(p.endswith("/matrix_dentex_test/matrix_summary.md") for p in by_area["test__harness_rules_skills_index"])
    assert any("/sweep/" in p and "/ksweep_dentex_dev_grpo_with_tools_canonical/k_sweep_summary.md" in p for p in by_area["k_sweep"])
    assert any(p.endswith("/results.json") for p in by_area["rules_skills"]) and any(p.endswith("/results.json") for p in by_area["speculative"])
    assert all(f"/{TAG}/{STAGE}/k4/run_" in p for p in by_area["test"] + by_area["dev"])
    assert not any(p.endswith("published.json") for p in paths)
    commits = len(hub.commits)
    again = publish.main(argv, api=hub)                                                 # identical inputs: nothing new
    assert {r["status"] for r in again} == {"unchanged"} and len(hub.commits) == commits
    (out / "test" / "matrix_dentex_test" / "matrix_summary.md").write_text("edited table")  # changed output: new folder, old one kept
    changed = publish.main(argv + ["--areas", "test"], api=hub)
    assert changed[0]["status"] == "uploaded" and changed[0]["hub_path"] != next(r["hub_path"] for r in results if r["area"] == "test")
    assert len(hub.commits) == commits + 1
    assert (out / "test" / "published.json").is_file()


def test_publish_refuses_a_k_that_differs_from_the_flattened_one_and_dry_run_needs_no_hub(tree, monkeypatch, capsys):
    out = run_flatten(tree, "--k", "4")
    aggregate(out, monkeypatch, capsys)
    with pytest.raises(SystemExit, match="flattened with --k 4"):
        publish.main(["--flat-dir", str(out), "--areas", "test", "--k", "2", "--tag", TAG], api=FakeHub())
    with pytest.raises(SystemExit, match="not other_tag"):
        publish.main(["--flat-dir", str(out), "--areas", "test", "--k", "4", "--tag", "other_tag"], api=FakeHub())
    dry = publish.main(["--flat-dir", str(out), "--areas", "test", "k_sweep", "--k", "4", "--tag", TAG, "--dry-run"])
    assert [r["status"] for r in dry] == ["dry_run", "dry_run"] and not (out / "test" / "published.json").exists()


def test_publish_needs_the_aggregation_first(tree):
    out = run_flatten(tree, "--k", "4")
    with pytest.raises(SystemExit, match="no aggregation output"):
        publish.main(["--flat-dir", str(out), "--areas", "test", "--k", "4", "--tag", TAG], api=FakeHub())
