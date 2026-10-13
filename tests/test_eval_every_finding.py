"""Every finding counts in the ablation, batch and sweep evaluators (no ``.iloc[0]`` first-row ground truth), healthy images
are evaluated with ground truth [], and calibration is per predicted finding."""

from pathlib import Path

import pandas as pd
import pytest

from dental_agent.evaluation import ablations, batch_runner, ground_truth, sweep
from dental_agent.evaluation.metrics import expected_calibration_error, finding_calibration_pairs


def F(q, pos, diag, conf=0.9):
    return {"quadrant": q, "tooth_position": pos, "diagnosis": diag, "confidence": conf}


# DENTEX image 1 has three annotation rows (raw labels are 0-indexed), image 2 is healthy (no row).
ANNOTS = pd.DataFrame([
    {"image_id": 1, "category_id_1": 0, "category_id_2": 5, "category_id_3": 1, "bbox": [10, 20, 30, 40]},
    {"image_id": 1, "category_id_1": 0, "category_id_2": 5, "category_id_3": 3, "bbox": [10, 20, 30, 40]},
    {"image_id": 1, "category_id_1": 3, "category_id_2": 7, "category_id_3": 0, "bbox": [50, 60, 70, 80]},
])
IMAGES = pd.DataFrame([{"id": 1, "local_path": "a.png"}, {"id": 2, "local_path": "b.png"}])
CATS = pd.DataFrame([{"id": 0, "name": "Impacted"}, {"id": 1, "name": "Caries"}, {"id": 2, "name": "Periapical Lesion"},
                     {"id": 3, "name": "Deep Caries"}])

ALL_THREE = [F(1, 6, "Caries"), F(1, 6, "Deep Caries"), F(4, 8, "Impacted")]
FIRST_ONLY = [F(1, 6, "Caries")]


def traj(final_answer):
    return {"final_answer": final_answer, "format_ok": True, "tool_calls": 0, "turns": []}


# ---------------------------------------------------------------------------
# the shared ground truth
# ---------------------------------------------------------------------------

def test_ground_truth_worked_example_every_row_is_a_finding():
    gt = ground_truth.extract_ground_truth(1, ANNOTS, None)
    assert [(g["quadrant"], g["tooth_position"], g["diagnosis"]) for g in gt] == [
        (1, 6, "Caries"), (1, 6, "Deep Caries"), (4, 8, "Impacted")]
    assert gt[0]["bbox"] == [10.0, 20.0, 30.0, 40.0]
    assert ground_truth.extract_ground_truth(2, ANNOTS, None) == []          # healthy image: ground truth []
    assert [g["diagnosis"] for g in ground_truth.extract_ground_truth(1, ANNOTS, CATS)] == ["Caries", "Deep Caries", "Impacted"]


def test_ground_truth_diag_col_is_a_parameter():
    annots = ANNOTS.rename(columns={"category_id_3": "dx"})
    gt = ground_truth.extract_ground_truth(1, annots, None, diag_col="dx")
    assert [g["diagnosis"] for g in gt] == ["Caries", "Deep Caries", "Impacted"]


def test_evaluate_models_exposes_the_same_function():
    ev = pytest.importorskip("scripts.evaluate_models")
    assert ev.extract_ground_truth is ground_truth.extract_ground_truth
    assert ev.DENTEX_DEFAULT_DIAGNOSES is ground_truth.DENTEX_DEFAULT_DIAGNOSES


# ---------------------------------------------------------------------------
# calibration per predicted finding
# ---------------------------------------------------------------------------

def test_finding_calibration_pairs_worked_example():
    gt = [F(1, 6, "Caries"), F(2, 6, "Impacted")]
    pred = [F(1, 6, "Caries", 0.9), F(2, 6, "Caries", 0.8)]          # second: right tooth, wrong diagnosis
    assert finding_calibration_pairs(gt, pred) == [(0.9, 1), (0.8, 0)]
    assert finding_calibration_pairs(gt, [{"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries", "confidence": None}]) == [(0.5, 1)]
    assert finding_calibration_pairs([], [F(1, 1, "Caries", 0.7)]) == [(0.7, 0)]      # a finding on a healthy image is wrong
    assert finding_calibration_pairs(gt, []) == []


def test_identical_duplicate_predictions_are_not_both_correct():
    gt = [F(1, 6, "Caries")]
    twin_a, twin_b = F(1, 6, "Caries", 0.9), F(1, 6, "Caries", 0.9)
    assert sorted(a for _, a in finding_calibration_pairs(gt, [twin_a, twin_b])) == [0, 1]


# ---------------------------------------------------------------------------
# run_agent_batch
# ---------------------------------------------------------------------------

def test_run_agent_batch_scores_all_findings_and_evaluates_the_healthy_image(tmp_path):
    answers = {1: ALL_THREE, 2: []}
    res = batch_runner.run_agent_batch([1, 2], IMAGES, ANNOTS, agent_fn=lambda i: traj(answers[i]), registry=object(),
                                       categories_df=CATS, cache_path=tmp_path / "c.json")
    assert [r["image_id"] for r in res] == [1, 2]                              # the healthy image is no longer skipped
    assert len(res[0]["ground_truth"]) == 3 and res[1]["ground_truth"] == []
    assert res[0]["reward_components"]["accuracy"] == 1.0 and res[1]["reward_components"]["accuracy"] == 1.0


def test_predicting_only_the_first_finding_no_longer_scores_perfect():
    res = batch_runner.run_agent_batch([1], IMAGES, ANNOTS, agent_fn=lambda i: traj(FIRST_ONLY), registry=object())
    # recall 1/3, precision 1 -> F1-style accuracy 0.5 (a first-row ground truth would have scored this 1.0)
    assert res[0]["reward_components"]["accuracy"] == pytest.approx(0.5)


def test_run_agent_batch_resumes_by_image(tmp_path):
    cache = tmp_path / "c.json"
    calls = []

    def agent(i):
        calls.append(i)
        return traj([] if i == 2 else ALL_THREE)

    batch_runner.run_agent_batch([1], IMAGES, ANNOTS, agent_fn=agent, registry=object(), cache_path=cache)
    batch_runner.run_agent_batch([1, 2], IMAGES, ANNOTS, agent_fn=agent, registry=object(), cache_path=cache)
    assert calls == [1, 2]


# ---------------------------------------------------------------------------
# ablations: H1, checkpoint comparison, legacy wrapper helper
# ---------------------------------------------------------------------------

@pytest.fixture
def stub_agents(monkeypatch):
    monkeypatch.setattr(ablations.ToolRegistry, "create_default", classmethod(lambda cls: object()))
    full = {1: ALL_THREE, 2: []}
    first = {1: FIRST_ONLY, 2: []}
    monkeypatch.setattr(ablations, "run_agent", lambda img_id, *a, **k: traj(full[img_id]))
    monkeypatch.setattr(ablations, "run_agent_no_tools", lambda img_id, *a, **k: traj(first[img_id]))


def test_h1_ablation_counts_every_finding_and_the_healthy_image(stub_agents):
    m_with, m_without = ablations.run_h1_ablation([1, 2], ANNOTS, model=object(), processor=object(), images_df=IMAGES,
                                                  categories_df=CATS)
    assert m_with["n_examples"] == m_without["n_examples"] == 2               # healthy image included
    assert m_with["closeness_score"] > m_without["closeness_score"]          # the two later findings are recovered, not ignored
    assert m_with["mean_reward"] > m_without["mean_reward"]


def test_compare_checkpoints_uses_all_findings(stub_agents):
    out = ablations.compare_checkpoints(["current (no reload)"], [1, 2], IMAGES, ANNOTS, categories_df=CATS,
                                        current_model=object(), current_processor=object())
    m = out["current (no reload)"]
    assert m["n_examples"] == 2 and m["fdi_accuracy"] == 0.5               # image 1 right; the healthy image has no finding to match


def test_prepare_eval_samples_gives_a_finding_list_per_image():
    images, gts = ablations._prepare_eval_samples(IMAGES, ANNOTS, categories_df=CATS)
    assert len(images) == 2 and [len(g) for g in gts] == [3, 0]
    assert gts[0][1]["diagnosis"] == "Deep Caries"


# ---------------------------------------------------------------------------
# evaluate_dataset: multi-finding answers and per-finding calibration
# ---------------------------------------------------------------------------

def test_evaluate_dataset_handles_list_answers_and_calibrates_per_finding(monkeypatch):
    answers = {
        1: [F(1, 6, "Caries", 0.9), F(1, 6, "Deep Caries", 0.8), F(4, 8, "Caries", 0.6)],     # third: right tooth, wrong diagnosis
        2: [],
    }
    monkeypatch.setattr(batch_runner, "run_agent", lambda img_id, *a, **k: traj(answers[img_id]))
    m = batch_runner.evaluate_dataset(IMAGES, ANNOTS, model=object(), processor=object(), registry=object(), categories_df=CATS)
    # the old single-answer code raised AttributeError on a list answer; now: three predicted findings, correct 1, 1, 0
    assert m["ece"] == pytest.approx(expected_calibration_error([0.9, 0.8, 0.6], [1, 1, 0]))
    assert m["ece"] == pytest.approx((0.1 + 0.2 + 0.6) / 3, abs=1e-5)
    assert m["exact_match_accuracy"] == 0.5 and m["fdi_localization_accuracy"] == 0.5
    assert m["total_samples"] == 2.0


# ---------------------------------------------------------------------------
# sweep
# ---------------------------------------------------------------------------

def test_reward_sweep_scores_all_findings_and_includes_healthy(monkeypatch):
    monkeypatch.setattr(sweep.ToolRegistry, "create_default", classmethod(lambda cls: object()))
    answers = {1: ALL_THREE, 2: []}
    df = sweep.sweep_reward_weights([1, 2], weight_grid=[{"acc": 1.0, "fmt": 0.0, "tool": 0.0, "eff": 0.0}],
                                    annots_df=ANNOTS, categories_df=CATS, agent_fn=lambda i: traj(answers[i]))
    assert df.loc[0, "n"] == 2 and df.loc[0, "mean_reward"] == pytest.approx(1.0)
    first_only = sweep.sweep_reward_weights([1], weight_grid=[{"acc": 1.0, "fmt": 0.0, "tool": 0.0, "eff": 0.0}],
                                            annots_df=ANNOTS, categories_df=CATS, agent_fn=lambda i: traj(FIRST_ONLY))
    assert first_only.loc[0, "mean_reward"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# guard: the first-row pattern stays out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("module", [ablations, batch_runner, sweep])
def test_no_first_row_ground_truth_in_the_evaluators(module):
    assert ".iloc[0]" not in Path(module.__file__).read_text(encoding="utf-8")
