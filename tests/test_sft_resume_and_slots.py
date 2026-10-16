"""SFT resume round trip (weights + optimizer + schedule), the loud SPMD wrapper failure, and no-tools slot budgets.

CPU only, tiny random Qwen3.5 from scripts/smoke_test_sft.py (no downloads)."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

from dental_agent.training import sft  # noqa: E402
from dental_agent.utils.canonical import grid_thw as canonical_grid_thw, patches_per_image, tokens_per_image  # noqa: E402

pytest.importorskip("peft")
peft = pytest.importorskip("peft")


# ---------------------------------------------------------------------------------------------------------------------
# Resume
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    import smoke_test_sft as sm

    d = tmp_path_factory.mktemp("tiny")
    recs = [{"messages": [{"role": "user", "content": "analyze this panoramic radiograph carefully"},
                          {"role": "assistant", "content": '{"thought": "teeth look normal", "final_answer": []}'}]}] * 4
    sm.build_tiny_model(recs, d)
    return d


def peft_model(tiny_dir, linear_attn=True):
    from dental_agent.model.backbone import get_model_classes, lora_target_modules

    base = get_model_classes().from_pretrained(str(tiny_dir), dtype=torch.float32)
    cfg = peft.LoraConfig(r=4, lora_alpha=8, task_type="CAUSAL_LM",
                          target_modules=lora_target_modules(base, linear_attention=linear_attn, vision_projector=True))
    return peft.get_peft_model(base, cfg)


def lora_state(model):
    return {k: v.detach().clone() for k, v in model.named_parameters() if "lora_" in k}


def make_trained_checkpoint(tiny_dir, folder: Path, **state):
    """A model trained for 3 steps with a real optimizer and scheduler, saved the way train_sft.py saves it."""
    from transformers import get_cosine_schedule_with_warmup

    a = peft_model(tiny_dir)
    opt = torch.optim.AdamW([p for p in a.parameters() if p.requires_grad], lr=1e-2)
    sch = get_cosine_schedule_with_warmup(opt, num_warmup_steps=1, num_training_steps=10)
    for _ in range(3):
        loss = sum(p.pow(2).sum() for p in a.parameters() if p.requires_grad) + sum(
            (p + 1.0).pow(2).sum() for n, p in a.named_parameters() if p.requires_grad and "lora_B" in n)
        loss.backward()
        opt.step()
        sch.step()
        opt.zero_grad()
    a.save_pretrained(str(folder))
    torch.save(opt.state_dict(), folder / "optimizer.pt")
    torch.save(sch.state_dict(), folder / "scheduler.pt")
    (folder / "training_state.json").write_text(json.dumps({"step": 3, "epoch": 2, "best_val_loss": 1.25, **state}))
    return a, opt, sch


def fresh_training_objects(model):
    from transformers import get_cosine_schedule_with_warmup

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    return opt, get_cosine_schedule_with_warmup(opt, num_warmup_steps=1, num_training_steps=10)


def test_resume_restores_weights_optimizer_and_schedule(tiny_dir, tmp_path):
    a, opt_a, sch_a = make_trained_checkpoint(tiny_dir, tmp_path)
    b = peft_model(tiny_dir)
    assert all(v.abs().sum() == 0 for k, v in lora_state(b).items() if "lora_B" in k)     # a fresh LoRA starts at B = 0
    opt_b, sch_b = fresh_training_objects(b)

    got = sft.restore_sft_checkpoint(tmp_path, b, opt_b, sch_b)

    assert got == {"step": 3, "epoch": 2, "best_val_loss": 1.25, "mid_epoch": False, "tensors": got["tensors"]} and got["tensors"] > 0
    for k, v in lora_state(a).items():                                       # the trained weights, not a fresh LoRA
        assert torch.equal(lora_state(b)[k], v), k
    assert any(v.abs().sum() > 0 for k, v in lora_state(b).items() if "lora_B" in k)
    assert sch_b.get_last_lr() == pytest.approx(sch_a.get_last_lr())          # schedule at step 3 of 10
    assert len(opt_b.state_dict()["state"]) == len(opt_a.state_dict()["state"]) > 0


def test_resume_mid_epoch_goes_back_to_the_step_the_epoch_began_at(tiny_dir, tmp_path):
    """Worked example: epoch 2 began at step 1; the SIGTERM checkpoint was written at step 3, mid-epoch. Resume restarts epoch 2
    from its first batch, so the step counter and the LR schedule must be back at step 1, not at 3 (else 2 steps are counted twice)."""
    make_trained_checkpoint(tiny_dir, tmp_path, mid_epoch=True, epoch_start_step=1, preempted=True)
    b = peft_model(tiny_dir)
    opt_b, sch_b = fresh_training_objects(b)
    got = sft.restore_sft_checkpoint(tmp_path, b, opt_b, sch_b)
    assert got["step"] == 1 and got["epoch"] == 2 and got["mid_epoch"] is True
    _, ref = fresh_training_objects(b)
    sft.fast_forward_scheduler(ref, 1)
    assert sch_b.get_last_lr() == pytest.approx(ref.get_last_lr())


def test_resume_legacy_checkpoint_without_epoch_start_step_keeps_saved_step(tiny_dir, tmp_path):
    make_trained_checkpoint(tiny_dir, tmp_path, preempted=True)             # written before patch 0035
    b = peft_model(tiny_dir)
    got = sft.restore_sft_checkpoint(tmp_path, b, *fresh_training_objects(b))
    assert got["step"] == 3


def test_resume_without_a_checkpoint_returns_none(tiny_dir, tmp_path):
    b = peft_model(tiny_dir)
    assert sft.restore_sft_checkpoint(tmp_path, b, *fresh_training_objects(b)) is None


def test_resume_never_continues_with_missing_or_foreign_weights(tiny_dir, tmp_path):
    # state file but no adapter weights
    (tmp_path / "training_state.json").write_text(json.dumps({"step": 3, "epoch": 2}))
    b = peft_model(tiny_dir)
    with pytest.raises(FileNotFoundError):
        sft.restore_sft_checkpoint(tmp_path, b, *fresh_training_objects(b))
    # an adapter trained with MORE LoRA targets than this run uses (--lora-linear-attn on, now off)
    full = tmp_path / "full"
    full.mkdir()
    make_trained_checkpoint(tiny_dir, full)
    narrow = peft_model(tiny_dir, linear_attn=False)
    with pytest.raises(RuntimeError, match="different LoRA targets"):
        sft.restore_sft_checkpoint(full, narrow, *fresh_training_objects(narrow))


# ---------------------------------------------------------------------------------------------------------------------
# SPMD wrapper: no silent fallback to an unsharded model
# ---------------------------------------------------------------------------------------------------------------------
def test_wrap_spmd_model_raises_when_fsdpv2_is_missing(monkeypatch):
    pkg = types.ModuleType("torch_xla")
    pkg.__path__ = []
    dist = types.ModuleType("torch_xla.distributed")
    dist.__path__ = []
    spmd = types.ModuleType("torch_xla.distributed.spmd")
    exp = types.ModuleType("torch_xla.experimental")
    exp.__path__ = []                                                          # a package without the FSDPv2 module
    for name, mod in {"torch_xla": pkg, "torch_xla.distributed": dist, "torch_xla.distributed.spmd": spmd, "torch_xla.experimental": exp}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    with pytest.raises(RuntimeError, match="SpmdFullyShardedDataParallel.*--no-spmd"):
        sft.wrap_spmd_model(torch.nn.Linear(2, 2), mesh=object())


# ---------------------------------------------------------------------------------------------------------------------
# Static vision slots for the no-tools track (1 FULL image, no CROP, no COMPARE)
# ---------------------------------------------------------------------------------------------------------------------
IMG = 151655


def fake_processor():
    tok = types.SimpleNamespace(pad_token_id=0, eos_token_id=1)
    return types.SimpleNamespace(tokenizer=tok, image_token_id=IMG)


def example(n_text=40, families=("FULL",)):
    """One sample: n_text text tokens, then one <|image_pad|> per merged patch group of every image."""
    grids = [canonical_grid_thw(f) for f in families]
    n_img = sum(tokens_per_image(f) for f in families)
    ids = torch.cat([torch.arange(10, 10 + n_text), torch.full((n_img,), IMG)]).unsqueeze(0)
    pix = torch.randn(sum(patches_per_image(f) for f in families), sft.PATCH_FEATURE_DIM)
    return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids),
            "image_grid_thw": torch.tensor(grids), "pixel_values": pix}


def test_no_tools_slots_1_0_0_pad_nothing_and_default_pads_6336_tokens():
    ex = example()
    tight = sft.BucketedQwenVLCollator(fake_processor(), track="no_tools", max_seq_len=2048, pad_vision_to_slots=True,
                                       slot_budget={"FULL": 1, "CROP": 0, "COMPARE": 0})
    default = sft.BucketedQwenVLCollator(fake_processor(), track="no_tools", max_seq_len=16384, pad_vision_to_slots=True)

    out = tight([ex, ex])
    assert out["input_ids"].shape == (2, 2048)
    assert out["pixel_values"].shape[0] == 2 * patches_per_image("FULL")          # 2 x 4608: no dummy slots added
    assert out["image_grid_thw"].shape == (2, 3)
    assert tight.padded_length(ex) == 40 + 1152                                    # text 40 + 1,152 vision tokens

    big = default([ex])
    assert big["input_ids"].shape == (1, 16384)
    assert big["pixel_values"].shape[0] == 29952                                   # 19 slots
    assert default.padded_length(ex) == 40 + 7488                                  # 6,336 more vision tokens than needed
    assert default.padded_length(ex) - tight.padded_length(ex) == 6336
    # the added tail is masked: attention 0 and no loss
    assert int(big["attention_mask"].sum()) == 40 + 1152
    assert int((big["labels"] != -100).sum()) == 40 + 1152


def test_slots_1_0_0_reject_a_trace_with_tool_images_instead_of_truncating():
    ex = example(families=("FULL", "CROP"))
    tight = sft.BucketedQwenVLCollator(fake_processor(), track="no_tools", max_seq_len=4096, pad_vision_to_slots=True,
                                       slot_budget={"FULL": 1, "CROP": 0, "COMPARE": 0})
    with pytest.raises(ValueError, match="uses 1 CROP images but the static budget is 0"):
        tight([ex])


def test_slot_padded_sequence_longer_than_max_seq_len_raises_never_truncates():
    ex = example(n_text=5000)
    col = sft.BucketedQwenVLCollator(fake_processor(), track="no_tools", max_seq_len=4096, pad_vision_to_slots=True,
                                     slot_budget={"FULL": 1, "CROP": 0, "COMPARE": 0})
    with pytest.raises(ValueError, match="exceeds target length 4096"):
        col([ex])
