"""Evaluation-time tool ablations: box geometry, the per-image probe, localization summary, tables and the wiring."""

import argparse
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from dental_agent.evaluation import matrix
from dental_agent.evaluation.tool_ablation import (
    AblationSpec, ToolProbe, box_iou, coverage, perturb_box, perturbation_angle, spec_from_args, zoom_rect,
)
from dental_agent.tools.registry import ToolRegistry
from dental_agent.tools.nudge import tool_nudge_crop
from dental_agent.tools.zoom_crop import tool_zoom_crop

GT_BOX = [100.0, 100.0, 100.0, 100.0]


def test_iou_worked_example_from_the_module_docstring():
    assert box_iou(GT_BOX, [150, 100, 100, 100]) == pytest.approx(1 / 3)
    assert box_iou(GT_BOX, GT_BOX) == 1.0
    assert box_iou(GT_BOX, [500, 500, 10, 10]) == 0.0


def test_coverage_and_zoom_rect_follow_the_tool_padding_rule():
    # padding is max(25% of the box, 50 px) on each side: 100x100 box -> 50 px on each side
    assert zoom_rect(GT_BOX, 0.25, 1000, 1000) == [50.0, 50.0, 200.0, 200.0]
    assert zoom_rect([0, 0, 100, 100], 0.25, 120, 1000) == [0.0, 0.0, 120.0, 150.0]          # clamped to the image: right edge 120, bottom 0+100+50
    assert coverage([50, 50, 200, 200], GT_BOX) == 1.0
    assert coverage([0, 0, 150, 200], GT_BOX) == pytest.approx(0.5)
    img = Image.new("RGB", (1000, 1000))
    view = tool_zoom_crop(img, GT_BOX, padding_frac=0.25)
    assert view.size == (200, 200)                                                             # the real tool agrees


def test_perturb_box_is_clamped_deterministic_and_scale_is_about_the_centre():
    a = perturbation_angle(0, 7, 38)
    assert a == perturbation_angle(0, 7, 38) and a != perturbation_angle(1, 7, 38)
    b = perturb_box(GT_BOX, 1000, 1000, 0.0, 2.0, a)                                           # scale only
    assert b == [50.0, 50.0, 200.0, 200.0]
    edge = perturb_box([0, 0, 100, 100], 1000, 1000, 0.5, 1.0, 3.14159)                        # pushed left off the image
    assert edge[0] == 0.0 and edge[2:] == [100.0, 100.0]
    shifted = perturb_box(GT_BOX, 1000, 1000, 0.5, 1.0, 0.0)                                   # angle 0 = right by half a width
    assert shifted == [150.0, 100.0, 100.0, 100.0]


def test_spec_tag_and_args():
    assert AblationSpec().tag() == "" and not AblationSpec().active
    assert AblationSpec(remove_tools=("nudge_crop",)).tag() == "__no-nudge_crop"
    t = AblationSpec(locate_shift_frac=0.3, seed=2).tag()
    assert t == "__loc-shift0.3-scale1-seed2"
    ns = argparse.Namespace(remove_tools=["nudge_crop", "zoom_crop"], locate_shift_frac=0.3, locate_scale=1.0, ablation_seed=0)
    spec = spec_from_args(ns)
    assert spec.remove_tools == ("nudge_crop", "zoom_crop") and spec.tag() == "__no-nudge_crop+no-zoom_crop+loc-shift0.3-scale1-seed0"
    assert not spec_from_args(argparse.Namespace()).active                                     # flags absent = no ablation


def _stub_registry(located):
    reg = ToolRegistry()
    reg.register("locate_tooth", lambda image, tooth=None: {"bbox": list(located[int(tooth)])} if int(tooth) in located
                 else {"error": "not found"}, "locate", {"tooth": 38})
    reg.register("nudge_crop", tool_nudge_crop, "nudge", {})
    reg.register("zoom_crop", tool_zoom_crop, "zoom", {})
    reg.register("denoise", lambda image, **kw: image, "denoise", {})
    return reg


GT = [{"quadrant": 3, "tooth_position": 8, "diagnosis": "Caries", "bbox": GT_BOX}]
IMG = Image.new("RGB", (1000, 1000))


def test_probe_removes_tools_and_leaves_the_base_registry_untouched():
    base = _stub_registry({38: GT_BOX})
    probe = ToolProbe(base, AblationSpec(remove_tools=("nudge_crop",)), 1)
    assert probe.registry.get("nudge_crop") is None and "nudge_crop" not in probe.registry.format_tool_descriptions()
    assert probe.registry.get("zoom_crop") and base.get("nudge_crop")


def test_unablated_probe_changes_nothing_and_logs_localization():
    probe = ToolProbe(_stub_registry({38: GT_BOX}), AblationSpec(), 1)
    assert probe.registry.execute("locate_tooth", image=IMG, tooth=38)["bbox"] == GT_BOX
    probe.registry.execute("zoom_crop", image=IMG, bbox=GT_BOX)
    (r,) = probe.box_vs_gt(GT)
    assert r == {"tooth": 38, "found": True, "iou_initial": 1.0, "iou_final": 1.0, "n_nudges": 0, "crop_coverage": 1.0}


def test_perturbed_locate_then_nudge_recovers_the_box():
    # the docstring example, built by hand: a located box shifted half a width, then a nudge back
    probe = ToolProbe(_stub_registry({38: [150, 100, 100, 100]}), AblationSpec(), 1)
    out = probe.registry.execute("locate_tooth", image=IMG, tooth=38)
    nud = probe.registry.execute("nudge_crop", image=IMG, bbox=out["bbox"], dx_frac=-0.5)
    assert nud["bbox"] == GT_BOX
    probe.registry.execute("zoom_crop", image=IMG, bbox=nud["bbox"])
    (r,) = probe.box_vs_gt(GT)
    assert r["iou_initial"] == pytest.approx(1 / 3) and r["iou_final"] == 1.0 and r["n_nudges"] == 1 and r["crop_coverage"] == 1.0


def test_locate_perturbation_is_applied_logged_and_paired_across_runs():
    spec = AblationSpec(locate_shift_frac=0.5, seed=3)
    boxes = []
    for _ in range(2):
        probe = ToolProbe(_stub_registry({38: GT_BOX}), spec, 9)
        boxes.append(probe.registry.execute("locate_tooth", image=IMG, tooth=38)["bbox"])
    assert boxes[0] == boxes[1] != GT_BOX                                                      # same wrong box on every run
    (r,) = probe.box_vs_gt(GT)
    assert r["iou_initial"] == pytest.approx(box_iou(boxes[0], GT_BOX)) and r["iou_initial"] < 0.7
    assert probe.calls[0]["true_box"] == GT_BOX                                                # the detector's own box is kept in the log


def test_unlocated_tooth_is_reported_not_found_and_failed_locate_is_not_a_crash():
    probe = ToolProbe(_stub_registry({}), AblationSpec(locate_shift_frac=0.5), 1)
    assert "error" in probe.registry.execute("locate_tooth", image=IMG, tooth=38)
    assert probe.box_vs_gt(GT) == [{"tooth": 38, "found": False}]
    assert probe.box_vs_gt([{"quadrant": 1, "tooth_position": 1, "diagnosis": "x"}]) == []      # no annotated box: not measurable


def test_nudge_of_an_unrelated_box_does_not_count_for_the_located_tooth():
    probe = ToolProbe(_stub_registry({38: GT_BOX}), AblationSpec(), 1)
    probe.registry.execute("locate_tooth", image=IMG, tooth=38)
    probe.registry.execute("nudge_crop", image=IMG, bbox=[700, 700, 100, 100], dx_frac=0.2)   # asserted box, not a located one
    (r,) = probe.box_vs_gt(GT)
    assert r["n_nudges"] == 0 and r["iou_final"] == 1.0


def test_summary_counts_attempts_at_a_removed_tool():
    probe = ToolProbe(_stub_registry({38: GT_BOX}), AblationSpec(remove_tools=("nudge_crop",)), 1)
    turns = [{"tool_calls_this_turn": [{"tool_name": "nudge_crop"}, {"tool_name": "zoom_crop"}]},
             {"tool_calls_this_turn": [{"tool_name": "nudge_crop"}]}, {"tool_calls_this_turn": None}]
    s = probe.summary(GT, turns)
    assert s["removed_tool_attempts"] == 2 and s["tag"] == "__no-nudge_crop" and s["spec"]["remove_tools"] == ["nudge_crop"]


# ---------------------------------------------------------------------------
# localization summary and tables
# ---------------------------------------------------------------------------

def _loc_rec(image_id, rows, attempts=0, **kw):
    return {"image_id": image_id, "dataset": "dentex", "ground_truth": [], "predictions": [], "format_ok": True, "turns": 3,
            "tool_calls": 2, "ablation": {"box_vs_gt": rows, "removed_tool_attempts": attempts}, **kw}


def _row(i0, i1, nudges=0, cov=None):
    return {"tooth": 38, "found": True, "iou_initial": i0, "iou_final": i1, "n_nudges": nudges, "crop_coverage": cov}


def test_localization_summary_worked_example():
    recs = [_loc_rec(1, [_row(0.33, 1.0, 1), _row(0.8, 0.8)]), _loc_rec(2, [_row(0.2, 0.3, 2), {"tooth": 11, "found": False}], attempts=2)]
    m = matrix.localization_summary(recs)
    assert m["n_findings_with_box"] == 4 and m["located_rate"] == pytest.approx(0.75)
    assert m["hit_initial"] == pytest.approx(1 / 3) and m["hit_final"] == pytest.approx(2 / 3)
    assert (m["n_recoverable"], m["n_recovered"], m["recovery_rate"]) == (2, 1, 0.5)               # two below 0.5 at first, one fixed
    assert m["removed_tool_attempts_per_image"] == 1.0
    assert matrix.localization_summary([rec_plain()]) is None                                       # no box data at all


def rec_plain():
    return {"image_id": 1, "dataset": "dentex", "ground_truth": [], "predictions": [], "format_ok": True}


def _f(q, pos, diag):
    return {"quadrant": q, "tooth_position": pos, "diagnosis": diag, "confidence": 0.9}


def _paired(n, hit):
    return [{"image_id": i, "dataset": "dentex", "format_ok": True, "turns": 3, "tool_calls": 2,
             "ground_truth": [_f(1, 6, "Caries")], "predictions": [_f(1, 6, "Caries")] if i < hit else []} for i in range(n)]


def test_ablation_file_gives_effects_localization_and_csvs(tmp_path):
    base = _paired(10, 10)
    abl = [{**r, **_loc_rec(r["image_id"], [_row(0.2, 0.7, 1)], attempts=1)} for r in _paired(10, 6)]
    matrix.cell_path(tmp_path, "dentex", "test", "sft_with_tools", "canonical").write_text("\n".join(json.dumps(r) for r in base) + "\n")
    tag = AblationSpec(remove_tools=("nudge_crop",)).tag()
    p = matrix.cell_path(tmp_path, "dentex", "test", "sft_with_tools", "canonical", tag)
    assert p.name == "eval_dentex_test_sft_with_tools_canonical__no-nudge_crop.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in abl) + "\n")

    assert set(matrix.load_matrix(tmp_path, "dentex", "test")) == {"sft_with_tools@canonical"}   # the baseline never absorbs the ablation
    assert matrix.discover_ablation_tags(tmp_path, "dentex", "test") == [tag]
    s = matrix.build_full_summary(tmp_path, "dentex", "test", n_boot=50)
    eff = [e for e in s["ablations"][tag]["effects"] if e["level"] == "exact"]
    assert len(eff) == 1 and eff[0]["diff"] < 0 and eff[0]["n_paired"] == 10                        # ablated minus baseline
    assert s["ablations"][tag]["cells"]["sft_with_tools@canonical"]["localization"]["hit_final"] == 1.0
    paths = matrix.write_matrix_outputs(s, tmp_path / "out")
    md = paths["markdown"].read_text()
    assert f"Ablation `{tag}`" in md and "Localization under this ablation" in md
    assert tag in paths["ablations"].read_text() and tag in paths["localization"].read_text()


# ---------------------------------------------------------------------------
# wiring: evaluate_condition and the matrix runner
# ---------------------------------------------------------------------------

ev = pytest.importorskip("scripts.evaluate_models")
rm = pytest.importorskip("scripts.run_eval_matrix")


@pytest.fixture
def world(tmp_path, monkeypatch):
    f = tmp_path / "img1.png"
    Image.new("RGB", (400, 400)).save(f)
    imgs = pd.DataFrame([{"id": 1, "local_path": str(f)}])
    gt = [{**_f(3, 8, "Caries"), "bbox": [100.0, 100.0, 100.0, 100.0]}]
    monkeypatch.setattr(ev, "extract_ground_truth", lambda image_id, a, c: gt)
    seen = {}

    class FakeRegistry:
        @staticmethod
        def create_default():
            return _stub_registry({38: [100.0, 100.0, 100.0, 100.0]})

    monkeypatch.setattr(ev, "ToolRegistry", FakeRegistry)

    def fake_traj(model, processor, images_df, image_id, registry, args):
        seen["registry"] = registry
        if registry is not None and registry.get("locate_tooth"):
            registry.execute("locate_tooth", image=IMG, tooth=38)
        return SimpleNamespace(final_answer=gt, format_ok=True, tool_calls=1,
                               turns=[{"turn": 0, "raw_output": "{}", "status": "final_answer",
                                       "tool_calls_this_turn": [{"tool_name": "nudge_crop"}]}],
                               messages=[{"role": "user", "content": "x"}])

    monkeypatch.setattr(ev, "run_condition_trajectory", fake_traj)
    args = argparse.Namespace(
        dataset="dentex", split="test", output_dir=str(tmp_path / "out"), reuse_cached=False, limit=None,
        model_id="m", adapter_path=None, precision="bf16", canonical_resize=True, max_tool_calls=50, max_turns=None,
        max_new_tokens=2048, malformed_retries=0, existing_results="error", condition="sft_with_tools",
        remove_tools=[], locate_shift_frac=0.0, locate_scale=1.0, ablation_seed=0)
    return SimpleNamespace(args=args, imgs=imgs, seen=seen, tmp=tmp_path)


def _run(w, cond="base_with_tools"):
    return ev.evaluate_condition(cond, w.args, w.imgs, pd.DataFrame(), pd.DataFrame(), loaded=(object(), object()))


def test_evaluate_condition_without_ablation_keeps_the_old_file_and_settings(world):
    (r,) = _run(world)
    assert (world.tmp / "out" / "eval_dentex_test_base_with_tools_canonical.jsonl").is_file()
    assert "ablation" not in r["eval_settings"]                                                     # 0030-era resume fingerprints stay valid
    assert r["ablation"]["tag"] == "" and r["ablation"]["box_vs_gt"][0]["iou_initial"] == 1.0


def test_evaluate_condition_with_ablation_writes_a_tagged_file_and_records_the_probe(world):
    world.args.remove_tools = ["nudge_crop"]
    (r,) = _run(world)
    assert (world.tmp / "out" / "eval_dentex_test_base_with_tools_canonical__no-nudge_crop.jsonl").is_file()
    assert not (world.tmp / "out" / "eval_dentex_test_base_with_tools_canonical.jsonl").exists()
    assert r["eval_settings"]["ablation"]["remove_tools"] == ["nudge_crop"]
    assert world.seen["registry"].get("nudge_crop") is None
    assert r["ablation"]["removed_tool_attempts"] == 1 and r["ablation"]["logged_calls"] == {"locate_tooth": 1}


def test_ablation_on_a_no_tools_condition_is_refused(world):
    world.args.locate_shift_frac = 0.3
    with pytest.raises(SystemExit, match="with_tools"):
        _run(world, "base_no_tools")


def test_runner_runs_only_with_tools_cells_for_an_ablation_and_uses_tagged_paths(tmp_path, monkeypatch):
    imgs = pd.DataFrame({"id": [1], "local_path": [str(tmp_path / "1.png")]})
    Image.new("RGB", (4, 4)).save(imgs["local_path"][0])
    ran = []
    monkeypatch.setattr(ev, "setup_eval_hardware", lambda precision: ("cpu", None, False))
    monkeypatch.setattr(ev, "load_eval_model", lambda *a, **k: ("model", "proc"))
    monkeypatch.setattr(ev, "evaluate_condition", lambda cond, ns, i, a, c, loaded=None: ran.append((cond, ns.remove_tools)) or [])
    args = rm.parse_args(["--output-dir", str(tmp_path / "out"), "--sft-adapter", "SFT", "--grpo-adapter", "GRPO",
                          "--remove-tools", "nudge_crop", "--sizes", "canonical"])
    rm.run_matrix(args, imgs, None, None)
    assert [c for c, _ in ran] == ["base_with_tools", "sft_with_tools", "grpo_with_tools"] and all(t == ["nudge_crop"] for _, t in ran)
    # an existing UNABLATED file does not block the ablation, a tagged one does
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    matrix.cell_path(out, "dentex", "test", "sft_with_tools", "canonical").write_text('{"image_id": 1}\n')
    rm.run_matrix(args, imgs, None, None)
    matrix.cell_path(out, "dentex", "test", "sft_with_tools", "canonical", "__no-nudge_crop").write_text('{"image_id": 1}\n')
    with pytest.raises(SystemExit, match="already have records"):
        rm.run_matrix(args, imgs, None, None)
