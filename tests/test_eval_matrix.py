"""Pooled finding-level metrics, paired bootstrap, the matrix tables, resumable evaluation and the matrix runner."""

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dental_agent.evaluation import matrix


def F(q, pos, diag, conf=0.9):
    return {"quadrant": q, "tooth_position": pos, "diagnosis": diag, "confidence": conf}


def rec(image_id, gt, pred, format_ok=True, **kw):
    return {"image_id": image_id, "dataset": "dentex", "ground_truth": gt, "predictions": pred, "format_ok": format_ok,
            "turns": 3, "tool_calls": 2, **kw}


# The docstring's worked example: A (2 GT, 2 pred, 1 exact), B (1 GT, nothing predicted), C (healthy, 1 invented finding).
EXAMPLE = [
    rec(1, [F(1, 6, "Caries"), F(2, 5, "Periapical Lesion")], [F(1, 6, "Caries"), F(3, 1, "Impacted")]),
    rec(2, [F(4, 8, "Impacted")], []),
    rec(3, [], [F(1, 1, "Caries")]),
]


def test_pooled_counts_match_the_worked_example():
    counts = matrix.counts_for(EXAMPLE)
    m = matrix.pooled_prf(counts, "exact")
    assert (m["tp"], m["n_pred"], m["n_gt"]) == (1, 3, 3)
    assert m["precision"] == pytest.approx(1 / 3) and m["recall"] == pytest.approx(1 / 3) and m["f1"] == pytest.approx(2 / 6)
    # Pooling is not the mean of per-image F1: image A alone has F1 0.5, B 0.0, C 0.0 -> mean 1/6.
    assert matrix.pooled_prf(counts[:1], "exact")["f1"] == pytest.approx(0.5)


def test_every_finding_counts_not_the_first_row():
    # Two ground-truth findings, both predicted exactly: recall must be 1.0 (first-row matching would give 0.5).
    r = rec(1, [F(1, 6, "Caries"), F(2, 5, "Deep Caries")], [F(2, 5, "Deep Caries"), F(1, 6, "Caries")])
    m = matrix.pooled_prf(matrix.counts_for([r]), "exact")
    assert m["recall"] == 1.0 and m["precision"] == 1.0


def test_fdi_level_ignores_the_diagnosis():
    r = rec(1, [F(1, 6, "Caries")], [F(1, 6, "Periapical Lesion")])
    counts = matrix.counts_for([r])
    assert matrix.pooled_prf(counts, "fdi")["f1"] == 1.0 and matrix.pooled_prf(counts, "exact")["f1"] == 0.0


def test_per_class_and_healthy_reports():
    counts = matrix.counts_for(EXAMPLE)
    pc = matrix.per_class_prf(counts)
    assert pc["Caries"]["tp"] == 1 and pc["Caries"]["n_pred"] == 2 and pc["Caries"]["n_gt"] == 1
    assert pc["Caries"]["precision"] == 0.5 and pc["Caries"]["recall"] == 1.0
    assert pc["Periapical Lesion"]["n_gt"] == 1 and pc["Periapical Lesion"]["n_pred"] == 0
    assert pc["Impacted"]["n_gt"] == 1 and pc["Impacted"]["n_pred"] == 1 and pc["Impacted"]["tp"] == 0
    h = matrix.healthy_report(counts)
    assert h == {"n_healthy": 1, "false_positive_images": 1, "false_positive_image_rate": 1.0,
                 "false_positive_findings": 1, "correct_empty_answers": 0, "format_failures": 0}


def test_healthy_image_with_empty_answer_is_correct_unless_the_format_failed():
    ok = matrix.image_counts(rec(1, [], [], format_ok=True))
    bad = matrix.image_counts(rec(2, [], [], format_ok=False))
    assert ok["all_exact"] is True and bad["all_exact"] is False
    h = matrix.healthy_report([ok, bad])
    assert h["correct_empty_answers"] == 1 and h["format_failures"] == 1 and h["false_positive_images"] == 0


def test_summarize_cell_reports_pathological_only_and_ci():
    s = matrix.summarize_cell(EXAMPLE, n_boot=200, seed=1)
    assert s["n_images"] == 3 and s["n_healthy"] == 1 and s["n_pathological"] == 2
    assert s["exact"]["f1"] == pytest.approx(1 / 3)
    # without the healthy image's false positive: TP 1, predicted 2, GT 3 -> F1 = 2/5
    assert s["exact_pathological_only"]["f1"] == pytest.approx(2 / 5)
    assert s["exact"]["f1_ci_low"] <= s["exact"]["f1"] <= s["exact"]["f1_ci_high"]


def test_load_records_keeps_the_last_record_of_an_image(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [rec(1, [], [F(1, 1, "Caries")]), rec(1, [], []), rec(2, [], [])]) + "\n")
    recs = matrix.load_records(p)
    assert len(recs) == 2 and [r for r in recs if r["image_id"] == 1][0]["predictions"] == []


def _paired_pair(n=30):
    good = [rec(i, [F(1, 6, "Caries")], [F(1, 6, "Caries")]) for i in range(n)]
    half = [rec(i, [F(1, 6, "Caries")], [F(1, 6, "Caries")] if i % 2 else []) for i in range(n)]
    return good, half


def test_paired_bootstrap_detects_a_real_difference_and_is_deterministic():
    good, half = _paired_pair()
    r1 = matrix.paired_bootstrap_diff(good, half, "exact", "f1", n_boot=500, seed=3)
    r2 = matrix.paired_bootstrap_diff(good, half, "exact", "f1", n_boot=500, seed=3)
    assert r1 == r2
    assert r1["n_paired"] == 30 and r1["a_value"] == 1.0
    assert r1["diff"] > 0.2 and r1["ci_low"] > 0 and r1["p_value"] < 0.05
    same = matrix.paired_bootstrap_diff(good, good, "exact", "f1", n_boot=300, seed=3)
    assert same["diff"] == 0.0 and same["ci_low"] <= 0 <= same["ci_high"] and same["p_value"] == 1.0


def test_paired_bootstrap_uses_only_shared_images():
    good, half = _paired_pair(10)
    res = matrix.paired_bootstrap_diff(good, half[:6], n_boot=100)
    assert res["n_paired"] == 6 and res["n_only_a"] == 4 and res["n_only_b"] == 0


def _write_cell(tmp_path, cond, size, recs, split="test"):
    path = matrix.cell_path(tmp_path, "dentex", split, cond, size)
    path.write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def test_matrix_tables_end_to_end(tmp_path):
    good, half = _paired_pair(12)
    _write_cell(tmp_path, "base_no_tools", "canonical", half)
    _write_cell(tmp_path, "sft_no_tools", "canonical", good)
    _write_cell(tmp_path, "sft_with_tools", "canonical", good[:-1])           # one image short
    _write_cell(tmp_path, "sft_no_tools", "original", half)
    assert matrix.cell_path(tmp_path, "dentex", "test", "sft_no_tools", "canonical").name == "eval_dentex_test_sft_no_tools_canonical.jsonl"
    assert matrix.cell_path(tmp_path, "dentex", "test", "sft_no_tools", "original").name == "eval_dentex_test_sft_no_tools.jsonl"

    cells = matrix.load_matrix(tmp_path, "dentex", "test")
    assert set(cells) == {"base_no_tools@canonical", "sft_no_tools@canonical", "sft_with_tools@canonical", "sft_no_tools@original"}
    summary = matrix.build_summary(cells, n_boot=100, seed=0)
    labels = {c["label"] for c in summary["comparisons"]}
    assert "SFT vs base (no_tools, canonical)" in labels and "tools vs no tools (sft, canonical)" in labels
    assert "canonical vs original (sft_no_tools)" in labels
    assert summary["coverage"]["cells_missing_images"] == {"sft_with_tools@canonical": [matrix.record_key(good[-1])]}

    paths = matrix.write_matrix_outputs(summary, tmp_path / "out")
    assert all(p.is_file() for p in paths.values())
    md = paths["markdown"].read_text()
    assert "## Canonical size (primary)" in md and "Coverage warning" in md and "Paired comparisons" in md
    assert "sft_no_tools" in paths["pooled"].read_text()


# ---------------------------------------------------------------------------
# evaluate_condition: transcript, resume, settings guard
# ---------------------------------------------------------------------------

ev = pytest.importorskip("scripts.evaluate_models")


@pytest.fixture
def eval_world(tmp_path, monkeypatch):
    rows = []
    for i in (1, 2, 3):
        f = tmp_path / f"img{i}.png"
        Image.new("RGB", (8, 8)).save(f)
        rows.append({"id": i, "local_path": str(f)})
    imgs = pd.DataFrame(rows)
    gts = {1: [F(1, 6, "Caries")], 2: [], 3: [F(2, 5, "Deep Caries")]}
    monkeypatch.setattr(ev, "extract_ground_truth", lambda image_id, a, c: gts[image_id])
    calls = []

    def fake_traj(model, processor, images_df, image_id, registry, args):
        calls.append(image_id)
        if getattr(fake_traj, "die_at", None) == image_id:
            raise RuntimeError("session killed")
        answer = [] if image_id == 2 else gts[image_id]
        return SimpleNamespace(
            final_answer=answer, format_ok=True, tool_calls=1,
            turns=[{"turn": 0, "raw_output": "{}", "status": "final_answer", "tool_calls_this_turn": []}],
            messages=[{"role": "system", "content": "SYSTEM"},
                      {"role": "user", "content": [{"type": "image", "image": object()}, {"type": "text", "text": "look"}]},
                      {"role": "assistant", "content": "{}"}])

    monkeypatch.setattr(ev, "run_condition_trajectory", fake_traj)
    args = argparse.Namespace(
        dataset="dentex", split="test", output_dir=str(tmp_path / "out"), reuse_cached=False, limit=None,
        model_id="m", adapter_path=None, precision="bf16", canonical_resize=True, max_tool_calls=50, max_turns=None,
        max_new_tokens=2048, malformed_retries=0, existing_results="error", condition="sft_no_tools")
    return SimpleNamespace(args=args, imgs=imgs, calls=calls, die=fake_traj, tmp=tmp_path)


def _evaluate(w, cond="base_no_tools", **over):
    for k, v in over.items():
        setattr(w.args, k, v)
    return ev.evaluate_condition(cond, w.args, w.imgs, pd.DataFrame(), pd.DataFrame(), loaded=(object(), object()))


def test_record_holds_transcript_settings_and_timing(eval_world):
    recs = _evaluate(eval_world)
    r = recs[0]
    assert [m["role"] for m in r["trajectory"]["messages"]] == ["user", "assistant"]   # system prompt left out
    assert r["trajectory"]["messages"][0]["content"] == "<image>\nlook"                  # image replaced by a marker
    assert r["trajectory"]["turns"][0]["status"] == "final_answer"
    assert r["eval_settings"]["canonical_resize"] is True and r["eval_settings"]["max_new_tokens"] == 2048
    assert r["seconds"] >= 0 and r["prompt"] == "unified"


def test_existing_results_error_resume_overwrite(eval_world):
    w = eval_world
    w.die.die_at = 3
    with pytest.raises(RuntimeError, match="killed"):
        _evaluate(w)                                   # images 1 and 2 are on disk, 3 died
    out = matrix.cell_path(w.args.output_dir, "dentex", "test", "base_no_tools", "canonical")
    assert len(matrix.load_records(out)) == 2

    w.die.die_at = None
    with pytest.raises(SystemExit, match="already has results"):
        _evaluate(w)                                   # default: refuse instead of appending duplicates

    w.calls.clear()
    recs = _evaluate(w, existing_results="resume")
    assert w.calls == [3] and len(recs) == 3           # only the unfinished image ran
    assert len(matrix.load_records(out)) == 3

    with pytest.raises(SystemExit, match="different settings"):
        _evaluate(w, existing_results="resume", max_new_tokens=512)

    w.calls.clear()
    _evaluate(w, existing_results="overwrite")
    assert w.calls == [1, 2, 3] and len(matrix.load_records(out)) == 3


def test_dev_split_selects_only_dev_images_and_never_the_test_file(tmp_path, monkeypatch):
    train = pd.DataFrame({"id": [1, 2, 3, 4], "local_path": ["a"] * 4, "source_file": ["train.json"] * 4})
    annots = pd.DataFrame({"image_id": [1, 2, 3, 4], "x": [0] * 4})
    seen = {}

    def fake_loader(data_dir=None, split_name="validation", **kw):
        seen["split"] = split_name
        return train.copy(), annots, pd.DataFrame()

    monkeypatch.setattr(ev, "load_dentex_dataset", fake_loader)
    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"dataset": "dentex", "keys": ["dentex::2", "dentex::4"]}))
    args = argparse.Namespace(split="dev", dev_split=str(dev), data_dir="d", dataset="dentex")
    imgs, an, _ = ev.load_eval_split(args)
    assert seen["split"] == "train" and list(imgs["id"]) == [2, 4] and set(an["image_id"]) == {2, 4}

    with pytest.raises(SystemExit, match="needs --dev-split"):
        ev.load_eval_split(argparse.Namespace(split="dev", dev_split=None, data_dir="d", dataset="dentex"))
    args.split = "test"
    ev.load_eval_split(args)
    assert seen["split"] == "test"


# ---------------------------------------------------------------------------
# The matrix runner
# ---------------------------------------------------------------------------

rm = pytest.importorskip("scripts.run_eval_matrix")


@pytest.fixture
def runner(tmp_path, monkeypatch):
    imgs = pd.DataFrame({"id": [1, 2], "local_path": [str(tmp_path / f"{i}.png") for i in (1, 2)]})
    for p in imgs["local_path"]:
        Image.new("RGB", (4, 4)).save(p)
    log = SimpleNamespace(loads=[], cells=[])
    monkeypatch.setattr(ev, "setup_eval_hardware", lambda precision, qlora_compute_dtype="auto": ("cpu", None, False))

    def fake_load(model_id, adapter, device, dtype, condition, quantize_4bit=False, runtime_options=None):
        log.loads.append(adapter)
        return ("model", "proc")

    def fake_eval(cond, ns, i, a, c, loaded=None):
        assert loaded == ("model", "proc")
        log.cells.append((cond, ns.canonical_resize, ns.adapter_path))
        return []

    monkeypatch.setattr(ev, "load_eval_model", fake_load)
    monkeypatch.setattr(ev, "evaluate_condition", fake_eval)
    return SimpleNamespace(imgs=imgs, log=log, tmp=tmp_path)


def _margs(tmp, *extra):
    return rm.parse_args(["--output-dir", str(tmp / "out"), "--sft-adapter", "SFT", "--grpo-adapter", "GRPO", *extra])


def test_runner_loads_each_checkpoint_once_and_runs_every_cell(runner):
    rm.run_matrix(_margs(runner.tmp), runner.imgs, None, None)
    assert runner.log.loads == [None, "SFT", "GRPO"]                       # three loads for twelve cells
    assert len(runner.log.cells) == 12
    assert {(c, canon) for c, canon, _ in runner.log.cells} == {(c, z) for c in matrix.CONDITIONS for z in (True, False)}
    assert all(a == {"base": None, "sft": "SFT", "grpo": "GRPO"}[c.split("_")[0]] for c, _, a in runner.log.cells)


def test_runner_stops_before_loading_when_an_adapter_is_missing(runner):
    args = rm.parse_args(["--output-dir", str(runner.tmp / "out"), "--sft-adapter", "SFT"])
    with pytest.raises(SystemExit, match="--grpo-adapter was not given"):
        rm.run_matrix(args, runner.imgs, None, None)
    assert runner.log.loads == []
    rm.run_matrix(rm.parse_args(["--output-dir", str(runner.tmp / "out"), "--conditions", "sft_no_tools", "sft_with_tools",
                                 "--sizes", "canonical", "--sft-adapter", "SFT"]), runner.imgs, None, None)
    assert runner.log.loads == ["SFT"] and len(runner.log.cells) == 2


def test_runner_refuses_existing_results_before_any_model_load(runner):
    out = matrix.cell_path(runner.tmp / "out", "dentex", "test", "sft_no_tools", "canonical")
    out.parent.mkdir(parents=True)
    out.write_text(json.dumps(rec(1, [], [])) + "\n")
    with pytest.raises(SystemExit, match="already have records"):
        rm.run_matrix(_margs(runner.tmp), runner.imgs, None, None)
    assert runner.log.loads == []


def test_runner_skips_a_finished_cell_without_loading_the_model(runner):
    args = _margs(runner.tmp, "--conditions", "base_no_tools", "--sizes", "canonical", "--existing-results", "resume")
    ns = rm.cell_args(args, "base_no_tools", "canonical")
    settings = ev.eval_settings(ns, "base_no_tools", None)
    out = matrix.cell_path(args.output_dir, "dentex", "test", "base_no_tools", "canonical")
    out.parent.mkdir(parents=True)
    out.write_text("\n".join(json.dumps(rec(i, [], [], eval_settings=settings)) for i in (1, 2)) + "\n")
    res = rm.run_matrix(args, runner.imgs, None, None)
    assert runner.log.loads == [] and runner.log.cells == [] and len(res["base_no_tools@canonical"]) == 2
