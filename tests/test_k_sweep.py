"""GRPO group-size (K) sweep: tie counts in the step log, per-K dev results, paired tables, figures, and the sweep launcher
(flag parity with run_grpo.py, SFT adapter search with the precision suffix, resume, preflight)."""

import importlib
import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from dental_agent.evaluation import figures, k_sweep, matrix

REPO = Path(__file__).resolve().parent.parent


def F(q, pos, diag):
    return {"quadrant": q, "tooth_position": pos, "diagnosis": diag, "confidence": 0.9}


def records(n, hits, extra_pred=0):
    """n dev images with one Caries finding each; the first ``hits`` are predicted exactly, the first ``extra_pred`` also get an
    invented second finding."""
    out = []
    for i in range(n):
        pred = [F(1, 6, "Caries")] if i < hits else []
        if i < extra_pred:
            pred.append(F(2, 3, "Impacted"))
        out.append({"image_id": i, "dataset": "dentex", "format_ok": True, "turns": 3, "tool_calls": 2,
                    "ground_truth": [F(1, 6, "Caries")], "predictions": pred})
    return out


def write_cell(folder, recs, condition="grpo_with_tools", size="canonical", split="dev"):
    folder.mkdir(parents=True, exist_ok=True)
    matrix.cell_path(folder, "dentex", split, condition, size).write_text("\n".join(json.dumps(r) for r in recs) + "\n")


def step_rows(k, n_steps, ties, hours_per_step=0.5):
    """n_steps log rows at group size k, the first ``ties`` of them tied; reward rises 0.2 -> 0.6 linearly."""
    rows = []
    for s in range(1, n_steps + 1):
        rows.append({"step": s, "group_size": k, "mean_reward": 0.2 + 0.4 * (s - 1) / (n_steps - 1), "kl_divergence": 0.001 * s,
                     "rollout_s": hours_per_step * 3600 * 0.75, "update_s": hours_per_step * 3600 * 0.25, "format_ok_rate": 1.0,
                     "mean_tool_calls": 4.0, "n_dropped_overlength": 0, "n_groups": 1, "n_tied_groups": 1 if s <= ties else 0})
    return rows


@pytest.fixture
def sweep_dir(tmp_path):
    """K = 1, 2, 4 on 20 dev images: 8, 14 and 15 exact hits; the SFT reference gets 10."""
    sweep = tmp_path / "k_sweep"
    for k, hits in ((1, 8), (2, 14), (4, 15)):
        write_cell(sweep / f"k{k}", records(20, hits))
    ref = tmp_path / "evals"
    write_cell(ref, records(20, 10), condition="sft_with_tools")
    return sweep, ref


# ---------------------------------------------------------------------------
# training log
# ---------------------------------------------------------------------------

def test_training_summary_worked_example_tie_rate():
    # 40 steps at K=2, 14 tied groups: tie rate 14/40 = 0.35; 40 steps x 0.5 h = 20 h; 80 rollouts
    t = k_sweep.summarize_training_log(step_rows(2, 40, ties=14), k=2)
    assert t["tie_rate"] == pytest.approx(0.35) and t["steps"] == 40
    assert t["hours"] == pytest.approx(20.0) and t["rollouts_total"] == 80 and t["rollouts_per_step"] == 2
    assert t["reward_first_quarter"] == pytest.approx(0.2 + 0.4 * (4.5 / 39)) and t["reward_last_quarter"] > t["reward_first_quarter"]


def test_tie_rate_is_undefined_for_k1_and_for_old_logs():
    assert k_sweep.summarize_training_log(step_rows(1, 8, ties=0), k=1)["tie_rate"] is None
    old = [{k: v for k, v in r.items() if k not in ("n_groups", "n_tied_groups")} for r in step_rows(4, 8, ties=3)]
    assert k_sweep.summarize_training_log(old, k=4)["tie_rate"] is None


def test_load_training_log_keeps_the_last_record_of_a_resumed_step(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [{"step": 1, "mean_reward": 0.1}, {"step": 2, "mean_reward": 0.2},
                                                   {"step": 2, "mean_reward": 0.9}]) + "\nnot json\n")
    rows = k_sweep.load_training_log(p)
    assert [(r["step"], r["mean_reward"]) for r in rows] == [(1, 0.1), (2, 0.9)]


def test_grpo_step_counts_tied_groups(monkeypatch):
    grpo = pytest.importorskip("dental_agent.training.grpo")
    import torch
    import torch.nn as nn

    class Scalar(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))
            self.device = torch.device("cpu")

    def run(rewards, k):
        trajs = [{"turns": [1], "tool_calls": 0, "format_ok": True} for _ in rewards]
        monkeypatch.setattr(grpo, "collect_grpo_group", lambda **kw: (trajs, rewards, [torch.zeros(1, 3)] * k, [torch.ones(1, 3)] * k))
        monkeypatch.setattr(grpo, "build_full_trajectory_labels", lambda traj, processor: {})
        monkeypatch.setattr(grpo, "compute_token_log_probs", lambda m, enc, use_reference=False: (m.w.expand(1, 3) * 1.0, torch.ones(1, 3)))
        model = Scalar()
        stats, _ = grpo.grpo_step(model=model, processor=None, image_ids_and_gts=[(1, [])], images_df=None, registry=object(),
                                  group_size=k, kl_beta=0.0, epochs_per_batch=1, optimizer=torch.optim.SGD(model.parameters(), lr=0.1))
        return stats

    assert (run([0.5, 0.5], 2)["n_groups"], run([0.5, 0.5], 2)["n_tied_groups"]) == (1, 1)     # all rewards equal: no signal
    assert run([1.0, 0.0], 2)["n_tied_groups"] == 0
    assert run([0.5], 1)["n_tied_groups"] == 0                                                   # K=1 uses the EMA baseline, no tie


# ---------------------------------------------------------------------------
# per-K tables
# ---------------------------------------------------------------------------

def test_discover_and_load_k_cells(sweep_dir, tmp_path):
    sweep, _ = sweep_dir
    (sweep / "notes").mkdir()
    (sweep / "k8").mkdir()                                                           # a K folder without results is left out
    assert list(k_sweep.discover_k_dirs(sweep)) == [1, 2, 4, 8]
    cells = k_sweep.load_k_cells(sweep, "dentex", "dev", "grpo_with_tools", "canonical")
    assert sorted(cells) == [1, 2, 4] and all(len(v) == 20 for v in cells.values())
    assert sorted(k_sweep.load_k_cells(sweep, "dentex", "dev", "grpo_with_tools", "canonical", ks=[2])) == [2]
    with pytest.raises(FileNotFoundError):
        k_sweep.load_reference(tmp_path / "nowhere", "dentex", "dev", "sft_with_tools", "canonical")


def test_build_k_sweep_worked_example(sweep_dir):
    sweep, ref_dir = sweep_dir
    cells = k_sweep.load_k_cells(sweep, "dentex", "dev", "grpo_with_tools", "canonical")
    ref = k_sweep.load_reference(ref_dir, "dentex", "dev", "sft_with_tools", "canonical")
    s = k_sweep.build_k_sweep(cells, ref, n_boot=200, seed=0)
    rows = {(r["k"], r["level"]): r for r in s["rows"]}
    # one finding per image, 20 images, nothing invented: F1 = 2*hits / (hits + 20) -> 16/28, 28/34, 30/35
    assert rows[(1, "exact")]["f1"] == pytest.approx(16 / 28)
    assert rows[(2, "exact")]["f1"] == pytest.approx(28 / 34) and rows[(4, "exact")]["f1"] == pytest.approx(30 / 35)
    assert s["reference"]["exact"]["f1"] == pytest.approx(20 / 30)
    # paired against SFT: K=4 minus SFT = 30/35 - 20/30, same 20 images
    vr = rows[(4, "exact")]["vs_reference"]
    assert vr["diff"] == pytest.approx(30 / 35 - 20 / 30) and vr["n_paired"] == 20 and vr["ci_low"] <= vr["diff"] <= vr["ci_high"]
    # best K is 4; K=2 vs best is the paired difference, K=4 has no 'vs best'
    assert s["candidates"]["exact"]["best_k"] == 4 and rows[(4, "exact")]["vs_best"] is None and rows[(4, "exact")]["is_best"]
    assert rows[(2, "exact")]["vs_best"]["diff"] == pytest.approx(28 / 34 - 30 / 35)
    assert rows[(1, "exact")]["within_noise_of_best"] is False                              # 8 vs 15 hits on the same 20 images
    assert 4 in s["candidates"]["exact"]["within_noise_of_best"]
    assert s["candidates"]["exact"]["cheapest_within_noise"] == min(s["candidates"]["exact"]["within_noise_of_best"])
    adj = [a for a in s["adjacent"] if a["level"] == "exact"]
    assert [(a["k"], a["k_smaller"]) for a in adj] == [(2, 1), (4, 2)]
    assert adj[0]["diff"] == pytest.approx(28 / 34 - 16 / 28)


def test_a_tie_for_best_goes_to_the_smaller_k(tmp_path):
    cells = {2: records(10, 6), 4: records(10, 6), 8: records(10, 3)}
    s = k_sweep.build_k_sweep(cells, None, n_boot=50)
    assert s["candidates"]["exact"]["best_k"] == 2
    assert all(r["vs_reference"] is None for r in s["rows"])                                 # no reference given


def test_precision_counts_invented_findings_per_k(tmp_path):
    cells = {1: records(10, 10), 2: records(10, 10, extra_pred=5)}                            # K=2 invents 5 extra findings
    s = k_sweep.build_k_sweep(cells, None, n_boot=50)
    rows = {r["k"]: r for r in s["rows"] if r["level"] == "exact"}
    assert rows[1]["f1"] == pytest.approx(1.0)
    assert rows[2]["precision"] == pytest.approx(10 / 15) and rows[2]["f1"] == pytest.approx(2 * 10 / (15 + 10))


def test_coverage_gap_is_reported():
    s = k_sweep.build_k_sweep({1: records(10, 5), 2: records(9, 5)}, None, n_boot=50)
    assert s["coverage"]["k_missing_images"] == {2: 1}
    assert "Coverage warning" in k_sweep.summary_markdown(s)


def test_outputs_markdown_and_csv(sweep_dir, tmp_path):
    sweep, ref_dir = sweep_dir
    cells = k_sweep.load_k_cells(sweep, "dentex", "dev", "grpo_with_tools", "canonical")
    logs = {k: step_rows(k, 8, ties=k) for k in cells}
    s = k_sweep.build_k_sweep(cells, k_sweep.load_reference(ref_dir, "dentex", "dev", "sft_with_tools", "canonical"), logs, n_boot=100)
    paths = k_sweep.write_k_sweep_outputs(s, tmp_path / "out", "grpo_with_tools")
    md = paths["markdown"].read_text()
    assert "Not distinguishable from the best K" in md and "Reported, not applied" in md and "Tie rate" in md
    assert "| 4 (best) |" in md and "n/a" in md                                              # K=1 tie rate is n/a
    table = figures.read_table(paths["table"])
    assert len(table) == 6 and {r["level"] for r in table} == {"exact", "fdi"}
    assert figures.read_table(paths["training"])[0]["k"] == 1.0
    assert json.loads(paths["json"].read_text())["ks"] == [1, 2, 4]


# ---------------------------------------------------------------------------
# figures and scripts
# ---------------------------------------------------------------------------

def aggregate(tmp_path, sweep, ref_dir, with_logs=True):
    agg = importlib.import_module("scripts.aggregate_k_sweep")
    log_dir = tmp_path / "models"
    for k in (1, 2, 4):
        d = log_dir / f"k{k}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "log.jsonl").write_text("\n".join(json.dumps(r) for r in step_rows(k, 8, ties=k)) + "\n")
    argv = ["--sweep-dir", str(sweep), "--n-boot", "100"]
    if ref_dir is not None:
        argv += ["--reference-dir", str(ref_dir)]
    if with_logs:
        argv += ["--log-template", str(log_dir / "k{k}" / "log.jsonl")]
    sys.path.insert(0, str(REPO))
    agg.main(argv)
    return sweep / k_sweep.out_dir_name("dentex", "dev", "grpo_with_tools", "canonical")


def test_k_sweep_rows_plot_the_table_numbers(sweep_dir, tmp_path, capsys):
    sweep, ref_dir = sweep_dir
    out = aggregate(tmp_path, sweep, ref_dir)
    t = figures.load_k_sweep_tables(out)
    rows = figures.k_sweep_rows(t["table"], "exact")
    assert [r["k"] for r in rows] == [1, 2, 4]
    assert rows[2]["value"] == pytest.approx(30 / 35) and rows[2]["reference"] == pytest.approx(20 / 30)
    assert rows[2]["diff"] == pytest.approx(30 / 35 - 20 / 30)
    assert all(r["significant"] == (r["diff_low"] > 0 or r["diff_high"] < 0) for r in rows)
    assert rows[2]["is_best"] and not rows[0]["is_best"]


def test_make_k_sweep_figures_renders_all(sweep_dir, tmp_path):
    sweep, ref_dir = sweep_dir
    out = aggregate(tmp_path, sweep, ref_dir)
    written = figures.make_k_sweep_figures(out, formats=("png", "pdf"), dpi=40)
    assert set(written) == set(figures.k_sweep_figure_names())
    assert all(p.is_file() and p.stat().st_size > 1000 for paths in written.values() for p in paths)
    assert written["k_sweep_exact"][0].parent == out / "figures"


def test_figures_degrade_without_reference_or_logs(sweep_dir, tmp_path):
    sweep, _ = sweep_dir
    out = aggregate(tmp_path, sweep, None, with_logs=False)
    written = figures.make_k_sweep_figures(out, formats=("png",), levels=("exact",), dpi=40)
    assert set(written) == {"k_sweep_exact"}                                                  # one panel, no training figure
    assert figures.plot_k_sweep([], {}) is None and figures.plot_k_sweep_training([], {}) is None


def test_aggregate_script_errors_and_plot_script(sweep_dir, tmp_path, capsys):
    agg = importlib.import_module("scripts.aggregate_k_sweep")
    with pytest.raises(SystemExit, match="No eval_dentex_dev_grpo_with_tools"):
        agg.main(["--sweep-dir", str(tmp_path)])
    sweep, ref_dir = sweep_dir
    aggregate(tmp_path, sweep, ref_dir)
    plot = importlib.import_module("scripts.plot_k_sweep")
    plot.main(["--sweep-dir", str(sweep), "--formats", "png", "--levels", "exact", "--dpi", "40"])
    printed = capsys.readouterr().out
    assert "k_sweep_exact" in printed and "k_sweep_training" in printed
    assert (sweep / k_sweep.out_dir_name("dentex", "dev", "grpo_with_tools", "canonical") / "figures" / "k_sweep_exact.png").is_file()
    with pytest.raises(SystemExit, match="aggregate_k_sweep.py"):
        plot.main(["--sweep-dir", str(tmp_path / "empty")])


# ---------------------------------------------------------------------------
# the sweep launcher
# ---------------------------------------------------------------------------

@pytest.fixture
def sweep_mod():
    sys.path.insert(0, str(REPO))
    return importlib.import_module("scripts.run_grpo_sweep")


def test_sweep_cli_has_the_run_grpo_flags(sweep_mod, monkeypatch):
    argv = ["run_grpo_sweep.py", "--k-values", "2", "4", "--sft-model-dir", "x/sft_qlora", "--kl-beta", "0.1", "--clip-eps", "0.3",
            "--push-every-steps", "10", "--checkpoint-dir", "ck", "--resume", "--resume-hf", "a/b", "--preflight"]
    monkeypatch.setattr(sys, "argv", argv)
    a = sweep_mod.parse_args()
    assert (a.sft_model_dir, a.kl_beta, a.clip_eps, a.push_every_steps, a.checkpoint_dir) == ("x/sft_qlora", 0.1, 0.3, 10, "ck")
    assert a.resume and a.resume_hf == "a/b" and a.preflight and a.k_values == [2, 4]
    monkeypatch.setattr(sys, "argv", ["run_grpo_sweep.py"])
    d = sweep_mod.parse_args()
    assert (d.kl_beta, d.clip_eps, d.push_every_steps, d.checkpoint_dir, d.resume, d.max_seq_len) == (0.04, 0.2, 25, "data/models", False, 16384)


def test_sft_search_uses_the_precision_suffix_of_the_run(tmp_path, monkeypatch):
    run_grpo = importlib.import_module("scripts.run_grpo")
    monkeypatch.chdir(tmp_path)
    assert [str(p) for p in run_grpo.sft_reference_candidates("with_tools", "dentex_alone")][0].endswith("sft_with_tools_dentex_alone")
    cands = [str(p) for p in run_grpo.sft_reference_candidates("with_tools", "dentex_alone", "qlora")]
    assert cands[0].endswith("sft_with_tools_dentex_alone_qlora") and not any(c.endswith("_bf16") for c in cands)
    for suffix in ("qlora", "bf16"):
        d = tmp_path / "data" / "models" / f"qwen3_5_9b_sft_with_tools_dentex_alone_{suffix}"
        d.mkdir(parents=True)
        (d / "adapter_config.json").write_text("{}")
    # each run finds the adapter of its own precision, nothing is switched for it
    assert run_grpo.resolve_sft_reference(None, "with_tools", "dentex_alone", None, "bf16").endswith("_bf16")
    assert run_grpo.resolve_sft_reference(None, "with_tools", "dentex_alone", None, "qlora").endswith("_qlora")


def test_sweep_preflight_checks_every_k_with_its_own_group_size(sweep_mod, monkeypatch, capsys):
    from dental_agent.training import preflight
    seen = []

    def fake(args, candidates, emit=print):
        seen.append((args.group_size, args.output_dir, args.dataset, str(candidates[0])))
        return 1 if args.group_size == 4 else 0

    monkeypatch.setattr(preflight, "run_preflight", fake)
    monkeypatch.setattr(sys, "argv", ["run_grpo_sweep.py", "--k-values", "2", "4", "--precision", "qlora", "--checkpoint-dir", "ck", "--preflight"])
    with pytest.raises(SystemExit) as e:
        sweep_mod.main()
    assert e.value.code == 1
    assert [(k, out, ds) for k, out, ds, _ in seen] == [(2, "ck", "dentex"), (4, "ck", "dentex")]
    assert seen[0][3].endswith("_dentex_alone_qlora") and "FAIL over K = [2, 4]" in capsys.readouterr().out


def test_sweep_resume_skips_a_finished_k_and_passes_every_setting(sweep_mod, monkeypatch, tmp_path):
    import pandas as pd
    calls = []
    imgs = pd.DataFrame([{"id": 1, "local_path": "a.png"}])
    monkeypatch.setattr(sweep_mod, "load_dentex_dataset", lambda *a, **k: (imgs, pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(sweep_mod, "select_training_pool", lambda df, *a, **k: df)
    monkeypatch.setattr(sweep_mod, "resolve_grpo_inputs", lambda *a: "sft_dir")
    monkeypatch.setattr(sweep_mod, "ensure_pool_images", lambda df, *a: df)
    monkeypatch.setattr(sweep_mod, "train_grpo", lambda **kw: (calls.append(kw), f"{kw['checkpoint_dir']}/final")[1])
    done = tmp_path / "ck" / "qwen3_5_9b_grpo_with_tools_k2_dentex_alone" / "grpo-with_tools-final"
    done.mkdir(parents=True)
    (done / "checkpoint_meta.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["run_grpo_sweep.py", "--k-values", "2", "4", "--checkpoint-dir", str(tmp_path / "ck"),
                                      "--output-dir", str(tmp_path / "out"), "--resume", "--kl-beta", "0.1", "--clip-eps", "0.3",
                                      "--push-every-steps", "7", "--epochs", "2", "--no-spmd", "--no-triangular-shim"])
    sweep_mod.main()
    assert [c["group_size"] for c in calls] == [4]                                           # K=2 was complete: no model load
    c = calls[0]
    assert (c["kl_beta"], c["clip_eps"], c["push_every_steps"], c["epochs_per_batch"], c["resume"]) == (0.1, 0.3, 7, 2, True)
    assert c["sft_model_dir"] == "sft_dir" and c["checkpoint_dir"].startswith(str(tmp_path / "ck"))
    log = [json.loads(line) for line in (tmp_path / "out" / "grpo_k_sweep_results.jsonl").read_text().splitlines()]
    assert [r["group_size"] for r in log] == [4] and log[0]["kl_beta"] == 0.1


def test_resume_hf_needs_resume(sweep_mod, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_grpo_sweep.py", "--resume-hf", "a/b"])
    with pytest.raises(SystemExit, match="--resume"):
        sweep_mod.main()
