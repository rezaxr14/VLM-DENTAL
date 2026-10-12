"""Per-step GRPO timing/diagnostics, the log summarizer's projection, and the --preflight checks."""

import json
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd
import pytest
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import summarize_grpo_log as sgl  # noqa: E402
from dental_agent.training import preflight as pf  # noqa: E402
from dental_agent.training.grpo_resume import pool_fingerprint  # noqa: E402
from dental_agent.training.run_meta import keys_fingerprint, write_sft_run_meta  # noqa: E402


# ---------------------------------------------------------------------------
# grpo_step stats
# ---------------------------------------------------------------------------

class _Scalar(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(1))
        self.device = torch.device("cpu")


def test_grpo_step_reports_timing_and_rollout_diagnostics(monkeypatch):
    grpo = pytest.importorskip("dental_agent.training.grpo")
    trajs = [
        {"turns": [1, 2, 3], "tool_calls": 4, "format_ok": True},
        {"turns": [1, 2], "tool_calls": 2, "format_ok": False},
    ]

    def fake_collect(**kw):
        masks = [torch.ones(1, 3), torch.ones(1, 3)]
        return trajs, [1.0, 0.0], [torch.zeros(1, 3), torch.zeros(1, 3)], masks

    model = _Scalar()
    monkeypatch.setattr(grpo, "collect_grpo_group", fake_collect)
    monkeypatch.setattr(grpo, "build_full_trajectory_labels", lambda traj, processor: {})
    monkeypatch.setattr(grpo, "compute_token_log_probs", lambda m, enc, use_reference=False: (m.w.expand(1, 3) * 1.0, torch.ones(1, 3)))
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    stats, _ = grpo.grpo_step(
        model=model, processor=None, image_ids_and_gts=[(1, [])], images_df=None, registry=object(),
        group_size=2, kl_beta=0.0, epochs_per_batch=1, optimizer=opt,
    )
    assert stats["mean_turns"] == pytest.approx(2.5)          # (3 + 2) / 2
    assert stats["mean_tool_calls"] == pytest.approx(3.0)     # (4 + 2) / 2
    assert stats["format_ok_rate"] == pytest.approx(0.5)
    assert stats["rollout_s"] >= 0 and stats["update_s"] >= 0
    assert set(stats) >= {"mean_reward", "kl_divergence", "n_dropped_overlength"}


def test_train_grpo_writes_every_step_to_the_given_log_path(monkeypatch, tmp_path):
    peft = pytest.importorskip("peft")  # noqa: F841
    import test_grpo_policy_init_resume as base

    sft = base.sft_dir.__wrapped__(tmp_path) if hasattr(base.sft_dir, "__wrapped__") else None
    if sft is None:
        pytest.skip("fixture not callable directly")
    calls = []
    base._run_train(monkeypatch, tmp_path, sft, n_images=3, log_calls=calls, log_path=str(tmp_path / "probe.jsonl"))
    assert [c["log_path"] for c in calls] == [str(tmp_path / "probe.jsonl")] * 3
    assert [c["extra"]["step"] for c in calls] == [1, 2, 3]


# ---------------------------------------------------------------------------
# summarize_grpo_log
# ---------------------------------------------------------------------------

def _rows(n, k=4, rollout=400.0, update=200.0):
    return [
        {"step": i, "group_size": k, "rollout_s": rollout, "update_s": update, "mean_reward": 0.5,
         "mean_turns": 11.0, "mean_tool_calls": 20.0, "format_ok_rate": 1.0, "n_dropped_overlength": 0}
        for i in range(1, n + 1)
    ]


def test_projection_scales_linearly_in_k():
    rows = _rows(10)
    rows[0]["rollout_s"] = 3000.0  # XLA compile in the first step must not enter the mean
    s = sgl.summarize(rows, pool_size=100, skip_first=2, project_k=[2, 8])
    assert s["mean_step_s"] == pytest.approx(600.0)
    assert s["projection_hours"][4] == pytest.approx(100 * 600 / 3600)       # 16.67 h
    assert s["projection_hours"][8] == pytest.approx(2 * 100 * 600 / 3600)   # 33.33 h
    assert s["projection_hours"][2] == pytest.approx(0.5 * 100 * 600 / 3600)
    assert "K=8" in sgl.format_summary(s)


def test_summarize_refuses_mixed_k_and_old_logs():
    with pytest.raises(SystemExit, match="mixes group sizes"):
        sgl.summarize(_rows(4, k=4) + _rows(4, k=8), pool_size=10)
    with pytest.raises(SystemExit, match="rollout_s"):
        sgl.summarize([{"step": 1, "mean_reward": 0.1, "group_size": 4}], pool_size=10)


def test_load_log_roundtrip(tmp_path):
    p = tmp_path / "log.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in _rows(3)) + "\n")
    assert len(sgl.load_log(p)) == 3


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------

@pytest.fixture
def world(tmp_path, monkeypatch):
    """A tiny DENTEX-like train set on disk, an SFT adapter folder with run metadata, and a dev split file."""
    imgs = []
    for i in range(1, 9):
        f = tmp_path / f"img{i}.png"
        f.write_bytes(b"x")
        imgs.append({"id": i, "local_path": str(f), "source_file": "training_data/train_triple.json"})
    images_df = pd.DataFrame(imgs)
    annots_df = pd.DataFrame({"image_id": [1, 2, 3, 4, 5, 6]})  # images 7 and 8 are healthy
    cats_df = pd.DataFrame({"id": [], "name": []})
    monkeypatch.setattr("dental_agent.data.dentex.load_dentex_dataset", lambda *a, **k: (images_df.copy(), annots_df, cats_df))
    monkeypatch.setattr("dental_agent.data.dentex.find_local_dentex_dir", lambda *a, **k: tmp_path)

    dev = tmp_path / "dev.json"
    dev.write_text(json.dumps({"dataset": "dentex", "image_ids": [2, 8], "keys": ["dentex::2", "dentex::8"]}))
    sft = tmp_path / "sft"
    sft.mkdir()
    (sft / "adapter_config.json").write_text("{}")
    (sft / "adapter_model.safetensors").write_bytes(b"x")
    write_sft_run_meta(sft, track="with_tools", canonical_resize=True, max_seq_len=16384,
                       dev_split_fingerprint=keys_fingerprint(["dentex::2", "dentex::8"]), vision_slots=[5, 10, 4])
    args = Namespace(
        track="with_tools", group_size=4, sft_stage="dentex_alone", dataset="dentex", config=None, sft_model_dir=str(sft),
        hf_repo=None, output_dir=str(tmp_path / "out"), dev_split=str(dev), max_images=None, seed=42, include_healthy=True,
        canonical_resize=True, pad_vision_to_slots=False, max_seq_len=16384, vision_slots=[5, 10, 4], spmd=True,
        num_cores=1, precision="bf16", resume=False, resume_hf=None, temperature=0.7, push_every_steps=25,
    )
    return Namespace(args=args, tmp=tmp_path, sft=sft, dev=dev)


def _run(world, **over):
    for k, v in over.items():
        setattr(world.args, k, v)
    lines: list[str] = []
    n_fail = pf.run_preflight(world.args, [], emit=lines.append)
    return n_fail, "\n".join(lines)


def test_preflight_clean_command_has_no_fail(world):
    n_fail, text = _run(world)
    assert n_fail == 0, text
    assert "6 images = 6 GRPO steps (1 healthy)" in text  # 8 train - 2 dev = 6; image 7 is healthy
    assert "dev split vs SFT: same held-out images" in text


def test_preflight_catches_canonical_and_dev_split_mismatch_with_sft(world):
    n_fail, text = _run(world, canonical_resize=False)
    assert n_fail == 1 and "[FAIL] canonical-resize vs SFT" in text
    other = world.tmp / "dev2.json"
    other.write_text(json.dumps({"dataset": "dentex", "keys": ["dentex::3"]}))
    n_fail, text = _run(world, canonical_resize=True, dev_split=str(other))
    assert n_fail == 1 and "[FAIL] dev split vs SFT" in text


def test_preflight_sft_adapter_missing_is_fail_without_hub_and_warn_with_it(world):
    n_fail, text = _run(world, sft_model_dir=str(world.tmp / "nowhere"))
    assert n_fail == 1 and "[FAIL] SFT adapter" in text
    n_fail, text = _run(world, hf_repo="user/repo")
    assert n_fail == 0 and "[WARN] SFT adapter" in text


def test_preflight_old_adapter_without_metadata_warns(world):
    (world.sft / "sft_run_meta.json").unlink()
    n_fail, text = _run(world)
    assert n_fail == 0 and "[WARN] SFT run metadata" in text


def test_preflight_flags_missing_images_missing_dev_file_and_test_split(world, monkeypatch):
    Path(world.args.output_dir).mkdir(exist_ok=True)
    (world.tmp / "img3.png").unlink()
    n_fail, text = _run(world)
    assert n_fail == 1 and "[FAIL] image files: 1 of 6" in text

    n_fail, text = _run(world, dev_split=str(world.tmp / "missing.json"))
    assert "[FAIL] dev split" in text and n_fail >= 1

    bad = pd.DataFrame({"id": [1], "local_path": [str(world.tmp / "img1.png")], "source_file": ["validation_triple.json"]})
    monkeypatch.setattr("dental_agent.data.dentex.load_dentex_dataset",
                        lambda *a, **k: (bad, pd.DataFrame({"image_id": [1]}), pd.DataFrame()))
    n_fail, text = _run(world, dev_split=str(world.dev))
    assert n_fail >= 1 and "[FAIL] train pool" in text and "held-out test split" in text


def test_preflight_without_dev_split_warns(world):
    n_fail, text = _run(world, dev_split=None)
    assert "[WARN] dev split: none given" in text
    assert "dev split vs SFT" in text and "[FAIL] dev split vs SFT" in text  # the SFT run held images out, GRPO would not


def _step_ckpt(root, step, fp, k):
    d = root / f"grpo-with_tools-step-{step}"
    (d / "grpo_policy").mkdir(parents=True)
    (d / "grpo_policy" / "adapter_config.json").write_text("{}")
    (d / "checkpoint_meta.json").write_text(json.dumps({"extra": {"step": step, "pool_fingerprint": fp, "group_size": k}}))


def test_preflight_resume_matches_or_refuses(world):
    out = Path(world.args.output_dir) / "qwen3_5_9b_grpo_with_tools_k4_dentex_alone"
    out.mkdir(parents=True)
    pool_ids = None

    # Learn the pool the preflight will build, then store a checkpoint for it.
    lines: list[str] = []
    world.args.resume = False
    pf.run_preflight(world.args, [], emit=lines.append)
    from dental_agent.data.dev_split import select_training_pool
    images = pd.DataFrame({"id": range(1, 9), "local_path": [str(world.tmp / f"img{i}.png") for i in range(1, 9)]})
    pool = select_training_pool(images, pd.DataFrame({"image_id": [1, 2, 3, 4, 5, 6]}), dev_keys={"dentex::2", "dentex::8"}, seed=42)
    pool_ids = pool["id"]

    _step_ckpt(out, 3, pool_fingerprint(pool_ids), 4)
    n_fail, text = _run(world, resume=True)
    assert n_fail == 0 and "continues at step 4" in text
    n_fail, text = _run(world, resume=False)
    assert "overwrites those checkpoints" in text

    for d in out.iterdir():
        for f in sorted(d.rglob("*"), reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        d.rmdir()
    _step_ckpt(out, 3, "deadbeef", 4)
    n_fail, text = _run(world, resume=True)
    assert n_fail == 1 and "[FAIL] resume" in text


def test_preflight_flag_checks(world):
    n_fail, text = _run(world, pad_vision_to_slots=True, canonical_resize=True, max_seq_len=1000)
    assert "[FAIL] max-seq-len" in text
    n_fail, text = _run(world, pad_vision_to_slots=False, max_seq_len=16384, resume_hf="u/r", resume=False)
    assert "[FAIL] resume-hf" in text
    n_fail, text = _run(world, resume_hf=None, temperature=0.0)
    assert "[FAIL] temperature" in text


def test_preflight_never_downloads_missing_dentex_data(world, monkeypatch):
    monkeypatch.setattr("dental_agent.data.dentex.find_local_dentex_dir", lambda *a, **k: None)

    def boom(*a, **k):
        raise AssertionError("preflight must not call the downloading loader")

    monkeypatch.setattr("dental_agent.data.dentex.load_dentex_dataset", boom)
    n_fail, text = _run(world)
    assert n_fail >= 1 and "[FAIL] dataset: DENTEX train data not found" in text
