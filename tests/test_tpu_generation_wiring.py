"""CPU checks for how TPU static generation is wired in: generate_local dispatch, drop-and-count of refused rollouts,
argument validation, CLI/preflight/summary. (Numerics: test_tpu_generation.py; compile behaviour: *_xla_cpu.py.)"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from dental_agent.model import tpu_generation as tg

ROOT = Path(__file__).resolve().parents[1]


class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(2, 2)
        self.device = torch.device("cpu")


def test_generate_local_uses_stock_generate_unless_installed(monkeypatch):
    from dental_agent.model import local_runtime as lr

    calls = []

    class Stock(Dummy):
        def generate(self, **kw):
            calls.append(("stock", sorted(kw)))
            return "stock-out"

    m = Stock()
    assert lr.generate_local(m, input_ids=torch.zeros((1, 2), dtype=torch.long)) == "stock-out"   # GPU/CPU path untouched
    object.__setattr__(m, "_dental_tpu_generation", tg.TPUGenerationConfig())
    monkeypatch.setattr(tg, "tpu_generate_local", lambda model, **kw: ("tpu-out", sorted(kw)))
    out = lr.generate_local(m, input_ids=torch.zeros((1, 2), dtype=torch.long), max_new_tokens=3)
    assert out[0] == "tpu-out" and "max_new_tokens" in out[1] and len(calls) == 1


def test_validate_args_matrix():
    ok = dict(chunk_tokens=512, tiers=(8192, 32768), is_tpu=True, spmd=True, fsdp=True, pad_vision_to_slots=True,
              canonical_resize=True)
    tg.validate_tpu_generation_args(**ok)
    tg.validate_tpu_generation_args(**{**ok, "chunk_tokens": 0, "is_tpu": False})      # off: anything goes
    for bad in ({"is_tpu": False}, {"spmd": False}, {"fsdp": False}, {"pad_vision_to_slots": False},
                {"canonical_resize": False}, {"chunk_tokens": -1}, {"chunk_tokens": 100}, {"tiers": (8000,)},
                {"tiers": ()}):
        with pytest.raises(ValueError):
            tg.validate_tpu_generation_args(**{**ok, **bad})


def test_collector_drops_and_counts_a_refused_rollout(monkeypatch):
    from dental_agent.training import grpo as G

    m = Dummy()
    cfg = tg.TPUGenerationConfig()
    object.__setattr__(m, "_dental_tpu_generation", cfg)
    seq = iter([SimpleNamespace(to_dict=lambda: {"final_answer": []}), tg.GenerationTooLong("too long"),
                SimpleNamespace(to_dict=lambda: {"final_answer": []})])

    def fake_run_agent(**kw):
        item = next(seq)
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(G, "run_agent", fake_run_agent)
    monkeypatch.setattr(G, "combine_reward", lambda traj, gt, max_tool_calls: (1.0, {}))
    trajs, rewards, lps, masks = G.collect_grpo_group_batched_with_tools(
        model=m, processor=None, image_id=1, images_df=None, ground_truth=[], registry=object(), group_size=3,
        compute_old_log_probs=False)
    assert len(trajs) == len(rewards) == len(lps) == len(masks) == 2
    assert cfg.stats["rollouts_dropped"] == 1


def test_grpo_step_counts_group_level_refusal_and_empty_groups(monkeypatch):
    from dental_agent.training import grpo as G

    m = Dummy()
    cfg = tg.TPUGenerationConfig()
    object.__setattr__(m, "_dental_tpu_generation", cfg)
    answers = iter(["refused", "empty"])

    def fake_collect(**kw):
        a = next(answers)
        if a == "refused":
            raise tg.SlotOverflow("too many images")
        return [], [], [], []

    monkeypatch.setattr(G, "collect_grpo_group", fake_collect)
    stats, _ = G.grpo_step(
        model=m, processor=None, image_ids_and_gts=[(1, []), (2, [])], images_df=None, registry=object(), group_size=4,
        kl_beta=0.0, epochs_per_batch=1, optimizer=torch.optim.SGD(m.parameters(), lr=0.1), train_collator=None)
    assert stats["n_dropped_generation"] == 4 and stats["n_rollouts"] == 0          # one refused group of K=4


def test_cli_exposes_flags_and_preflight_checks_them():
    done = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_grpo.py"), "--help"], capture_output=True, text=True,
                          cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT)})
    assert done.returncode == 0
    for flag in ("--tpu-generation-chunk-tokens", "--tpu-generation-tiers", "--tpu-generation-warmup"):
        assert flag in done.stdout
    from dental_agent.training import preflight as pf

    def run(**kw):
        args = SimpleNamespace(
            temperature=0.7, resume_hf=False, resume=False, spmd=True, fsdp=True, pad_vision_to_slots=True,
            canonical_resize=True, canonical_letterbox=True, max_seq_len=32768, vision_slots=[8, 16, 6],
            tpu_generation_chunk_tokens=512, tpu_generation_tiers=[8192, 16384, 32768], tpu_logprob_chunk_tokens=0,
            logprob_chunk_tokens=0)
        for k, v in kw.items():
            setattr(args, k, v)
        lines = []
        rep = pf._Report(lines.append)
        pf._check_flags(args, rep)
        return rep, "\n".join(lines)

    rep, text = run()
    assert rep.n_fail == 0 and "tpu-generation" in text and "4 programs" in text
    assert run(tpu_generation_chunk_tokens=100)[0].n_fail == 1                       # not a multiple of 64
    assert run(tpu_generation_tiers=[8000])[0].n_fail == 1                           # not a multiple of the chunk
    assert run(spmd=False)[0].n_fail == 1
    rep, text = run(tpu_generation_tiers=[8192, 16384])
    assert rep.n_fail == 0 and rep.n_warn == 1 and "largest tier 16384" in text       # holds less than --max-seq-len
    assert "tpu-generation" not in run(tpu_generation_chunk_tokens=0)[1]             # off: silent


def test_summary_reports_generation_drops():
    sys.path.insert(0, str(ROOT / "scripts"))
    import summarize_grpo_log as sgl
    rows = [{"rollout_s": 5.0, "update_s": 5.0, "group_size": 4, "mean_reward": 0.5, "n_dropped_generation": n}
            for n in (2, 0, 1)]
    out = sgl.summarize(rows, pool_size=10, skip_first=0)
    assert out["dropped_generation"] == 3
    assert "TPU generation: 3 rollouts refused" in sgl.format_summary(out)
