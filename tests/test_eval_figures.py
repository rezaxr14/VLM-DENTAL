"""Figures of the evaluation matrix: plot data read from the aggregate tables, tag names, footnotes, and that every figure renders."""

import importlib
import json
import sys

import pytest

from dental_agent.evaluation import figures, matrix
from dental_agent.evaluation.tool_ablation import AblationSpec


def F(q, pos, diag):
    return {"quadrant": q, "tooth_position": pos, "diagnosis": diag, "confidence": 0.9}


def _records(n, hit, probe_rows=None, cond_prompt=None):
    """n images with one Caries finding each; the first ``hit`` are predicted exactly. ``probe_rows`` adds the ablation probe."""
    out = []
    for i in range(n):
        r = {"image_id": i, "dataset": "dentex", "format_ok": True, "turns": 3, "tool_calls": 2,
             "ground_truth": [F(1, 6, "Caries")], "predictions": [F(1, 6, "Caries")] if i < hit else []}
        if probe_rows is not None:
            r["ablation"] = {"box_vs_gt": probe_rows(i), "removed_tool_attempts": 0}
        if cond_prompt:
            r["prompt"] = cond_prompt
        out.append(r)
    return out


def _row(i0, i1):
    return {"tooth": 16, "found": True, "iou_initial": i0, "iou_final": i1, "n_nudges": 1, "crop_coverage": None}


def _write(tmp_path, cond, size, recs, tag="", split="dev"):
    matrix.cell_path(tmp_path, "dentex", split, cond, size, tag).write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def _matrix_dir(tmp_path, ablation=True, split="dev"):
    """A small dev matrix on disk, aggregated the way ``aggregate_eval.py`` does it; returns the matrix_<dataset>_<split> folder."""
    _write(tmp_path, "base_no_tools", "canonical", _records(12, 3), split=split)
    _write(tmp_path, "sft_no_tools", "canonical", _records(12, 8), split=split)
    _write(tmp_path, "sft_with_tools", "canonical", _records(12, 11, probe_rows=lambda i: [_row(0.33, 1.0), _row(0.8, 0.8)]), split=split)
    _write(tmp_path, "sft_no_tools", "original", _records(12, 6), split=split)
    if ablation:
        tag = AblationSpec(remove_tools=("nudge_crop",)).tag()
        _write(tmp_path, "sft_with_tools", "canonical", _records(12, 7, probe_rows=lambda i: [_row(0.33, 0.33), _row(0.8, 0.8)]), tag, split=split)
    summary = matrix.build_full_summary(tmp_path, "dentex", split, n_boot=60, seed=0)
    out = tmp_path / f"matrix_dentex_{split}"
    matrix.write_matrix_outputs(summary, out)
    return out


# ---------------------------------------------------------------------------
# reading and naming
# ---------------------------------------------------------------------------

def test_read_table_types_and_empty_cells(tmp_path):
    p = tmp_path / "t.csv"
    p.write_text("ablation,cell,hit_final,recovery_rate\n,sft_with_tools@canonical,0.5,\n")
    (row,) = figures.read_table(p)
    assert row == {"ablation": "", "cell": "sft_with_tools@canonical", "hit_final": 0.5, "recovery_rate": None}


def test_split_cell():
    assert figures.split_cell("grpo_no_tools@original") == ("grpo", "no_tools", "original")


def test_describe_tag_matches_what_the_ablation_spec_writes():
    assert figures.describe_tag("") == "Unablated"
    assert figures.describe_tag(AblationSpec(remove_tools=("nudge_crop",)).tag()) == "without nudge_crop"
    assert figures.describe_tag(AblationSpec(locate_shift_frac=0.3, seed=2).tag()) == "locate_tooth box shifted by 30% of its size (seed 2)"
    assert figures.describe_tag(AblationSpec(locate_scale=1.5).tag()) == "locate_tooth box scaled x1.5 (seed 0)"
    both = AblationSpec(remove_tools=("zoom_crop",), locate_shift_frac=0.5, locate_scale=2.0).tag()
    assert figures.describe_tag(both) == "without zoom_crop, locate_tooth box shifted by 50% of its size and scaled x2 (seed 0)"


# ---------------------------------------------------------------------------
# plot data
# ---------------------------------------------------------------------------

def test_forest_cells_order_values_and_level(tmp_path):
    t = figures.load_tables(_matrix_dir(tmp_path, ablation=False))
    by_size = figures.forest_cells(t["pooled"], "exact")
    assert list(by_size) == ["canonical", "original"]
    assert [r["cell"] for r in by_size["canonical"]] == ["base_no_tools@canonical", "sft_no_tools@canonical", "sft_with_tools@canonical"]
    assert [r["cell"] for r in by_size["original"]] == ["sft_no_tools@original"]
    # the plotted number IS the table number: 8 of 12 exact, nothing extra -> F1 = 2*8 / (8 + 12)
    sft = next(r for r in by_size["canonical"] if r["cell"] == "sft_no_tools@canonical")
    assert sft["value"] == pytest.approx(16 / 20) and sft["images"] == 12
    assert sft["low"] <= sft["value"] <= sft["high"]
    assert figures.forest_cells(t["pooled"], "fdi")["canonical"][0]["model"] == "base"


def test_comparison_rows_mark_intervals_that_exclude_zero(tmp_path):
    t = figures.load_tables(_matrix_dir(tmp_path, ablation=False))
    rows = figures.comparison_rows(t["comparisons"], "exact")
    assert {r["block"] for r in rows} == {"canonical", "size"}
    sft_vs_base = next(r for r in rows if r["label"] == "SFT vs base (no_tools, canonical)")
    assert sft_vs_base["diff"] == pytest.approx(16 / 20 - 6 / 15)           # 3 of 12 hit -> F1 = 6 / (3 + 12)
    assert sft_vs_base["significant"] == (sft_vs_base["low"] > 0 or sft_vs_base["high"] < 0)
    assert all(r["significant"] == (r["low"] > 0 or r["high"] < 0) for r in rows)


def test_localization_groups_worked_example(tmp_path):
    # two located findings per image with (initial, final) IoU (0.33, 1.0) and (0.8, 0.8): hit 1/2 -> 2/2, one of the one misses recovered per image
    t = figures.load_tables(_matrix_dir(tmp_path, ablation=True))
    groups = dict(figures.localization_groups(t["localization"]))
    assert list(dict(figures.localization_groups(t["localization"]))) == ["", "__no-nudge_crop"]      # unablated first
    (plain,) = groups[""]
    assert plain["label"] == "SFT, canonical"
    assert plain["initial"] == pytest.approx(0.5) and plain["final"] == pytest.approx(1.0)
    assert (plain["recovered"], plain["recoverable"], plain["n_located"]) == (12, 12, 24)
    (nudge_off,) = groups["__no-nudge_crop"]
    assert nudge_off["initial"] == nudge_off["final"] == pytest.approx(0.5) and nudge_off["recovered"] == 0


def test_ablation_groups_effect_is_ablated_minus_unablated(tmp_path):
    t = figures.load_tables(_matrix_dir(tmp_path, ablation=True))
    ((tag, rows),) = figures.ablation_groups(t["ablations"], "exact")
    assert tag == "__no-nudge_crop" and rows[0]["label"] == "SFT, canonical"
    # 7 of 12 findings predicted (7 predictions): F1 = 2*7 / (7 + 12);  11 of 12: 2*11 / (11 + 12)
    assert rows[0]["ablated"] == pytest.approx(14 / 19) and rows[0]["unablated"] == pytest.approx(22 / 23)
    assert rows[0]["diff"] == pytest.approx(14 / 19 - 22 / 23) and rows[0]["n_paired"] == 12


def test_footnote_flags_cached_prompt_and_coverage_gaps(tmp_path):
    _write(tmp_path, "base_no_tools", "canonical", _records(10, 2, cond_prompt="zero_shot_cached"))
    _write(tmp_path, "sft_no_tools", "canonical", _records(9, 6))                       # one image short
    out = tmp_path / "m"
    matrix.write_matrix_outputs(matrix.build_full_summary(tmp_path, "dentex", "dev", n_boot=30), out)
    t = figures.load_tables(out)
    notes = " ".join(figures._image_note(t["summary"], [r for r in t["pooled"] if r["level"] == "exact"]))
    assert "(*) cached zero-shot run" in notes and "base_no_tools@canonical" in notes
    assert "Cells cover different image sets" in notes and "9-10 images per cell" in notes and "30 resamples" in notes


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------

def test_make_all_writes_every_figure_in_both_formats(tmp_path):
    out = _matrix_dir(tmp_path, ablation=True)
    written = figures.make_all(out, formats=("png", "pdf"), dpi=40)
    assert set(written) == set(figures.figure_names())
    for paths in written.values():
        assert [p.suffix for p in paths] == [".png", ".pdf"] and all(p.is_file() and p.stat().st_size > 1000 for p in paths)
    assert written["matrix_forest_exact"][0].parent == out / "figures"


def test_figures_without_data_are_skipped_not_empty(tmp_path):
    out = _matrix_dir(tmp_path, ablation=False)
    written = figures.make_all(out, formats=("png",), levels=("exact",), dpi=40)
    assert "ablation_effects_exact" not in written
    assert {"matrix_forest_exact", "paired_comparisons_exact", "localization_before_after"} <= set(written)   # unablated tool cell has the probe
    assert figures.plot_ablation_effects([], {}, "exact") is None and figures.plot_localization([], {}) is None
    assert figures.plot_matrix_forest([], {}, "exact") is None and figures.plot_paired_comparisons([], {}, "exact") is None


def test_make_all_honours_out_dir_and_levels(tmp_path):
    out = _matrix_dir(tmp_path, ablation=True)
    written = figures.make_all(out, tmp_path / "paper", formats=("png",), levels=("fdi",), dpi=40)
    assert all(n.endswith("_fdi") or n == "localization_before_after" for n in written)
    assert (tmp_path / "paper" / "matrix_forest_fdi.png").is_file() and not (out / "figures").exists()


# ---------------------------------------------------------------------------
# the script
# ---------------------------------------------------------------------------

def _script():
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
    return importlib.import_module("scripts.plot_eval_figures")


def test_script_reads_the_aggregate_folder_and_writes_figures(tmp_path, monkeypatch, capsys):
    _matrix_dir(tmp_path, ablation=False, split="dev")
    monkeypatch.setattr(sys, "argv", ["plot_eval_figures.py", "--output-dir", str(tmp_path), "--split", "dev", "--formats", "png",
                                      "--levels", "exact", "--dpi", "40"])
    _script().main()
    printed = capsys.readouterr().out
    assert "matrix_forest_exact" in printed and "skipped (no rows in the table): ablation_effects_exact" in printed
    assert (tmp_path / "matrix_dentex_dev" / "figures" / "matrix_forest_exact.png").is_file()


def test_script_tells_you_to_aggregate_first(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["plot_eval_figures.py", "--output-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="aggregate_eval.py"):
        _script().main()
