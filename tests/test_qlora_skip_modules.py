"""QLoRA quantizes the language model only: the vision tower (projector included) and lm_head stay 16-bit.

CPU only. Real Qwen3.5 classes (tiny random weights) and the REAL transformers module matcher; bitsandbytes is not needed because
the 4-bit layer is replaced by a stand-in class at exactly the modules the matcher selects. What this cannot show is bitsandbytes
itself, which is what ``scripts/check_qlora_load.py`` is for (run it on the 4090)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")
from peft import PeftModel  # noqa: E402
from transformers import BitsAndBytesConfig, Qwen3_5ForConditionalGeneration  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig  # noqa: E402
from transformers.quantizers.base import HfQuantizer  # noqa: E402
from transformers.quantizers.quantizers_utils import should_convert_module  # noqa: E402

from dental_agent.config import QLORA_SKIP_MODULES  # noqa: E402
from dental_agent.model.backbone import (  # noqa: E402
    check_qlora_model, is_vision_param_name, lora_coverage, lora_target_modules, prepare_qlora_model, projector_module_names,
    qlora_bnb_config,
)
from dental_agent.training.grpo_resume import init_policy_from_reference  # noqa: E402


class FakeLinear4bit(nn.Linear):
    """Stand-in for bitsandbytes.nn.Linear4bit: an nn.Linear subclass, as the real one is."""


def tiny(seed=0):
    text = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=512,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    vision = Qwen3_5VisionConfig(
        depth=2, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=64, patch_size=16,
        spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=2304,
    )
    torch.manual_seed(seed)
    return Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=False))


def fake_quantize(model, patterns=None):
    """Replace nn.Linear by FakeLinear4bit wherever transformers' own bitsandbytes integration would (same matcher, same list)."""
    if patterns is None:
        cfg = qlora_bnb_config(torch.bfloat16)
        patterns = HfQuantizer.get_modules_to_not_convert(model, cfg.llm_int8_skip_modules, None)
    for name, module in list(model.named_modules()):
        if type(module) is nn.Linear and should_convert_module(name, patterns):
            new = FakeLinear4bit(module.in_features, module.out_features, bias=module.bias is not None)
            new.load_state_dict(module.state_dict())
            parent, _, child = name.rpartition(".")
            setattr(model.get_submodule(parent) if parent else model, child, new)
    return model


# ---------------------------------------------------------------------------------------------------------------------
# The list itself, against the installed transformers
# ---------------------------------------------------------------------------------------------------------------------
def test_one_constant_feeds_the_config_with_nf4_double_quant_and_compute_dtype():
    cfg = qlora_bnb_config(torch.bfloat16)
    assert isinstance(cfg, BitsAndBytesConfig)
    assert cfg.load_in_4bit and cfg.bnb_4bit_quant_type == "nf4" and cfg.bnb_4bit_use_double_quant
    assert cfg.bnb_4bit_compute_dtype == torch.bfloat16
    assert cfg.llm_int8_skip_modules == list(QLORA_SKIP_MODULES) == ["model.visual", "visual", "lm_head"]
    assert qlora_bnb_config(torch.float16).bnb_4bit_compute_dtype == torch.float16          # T4-class fallback is the caller's


def test_skip_list_leaves_the_whole_vision_tower_and_lm_head_unquantized_and_quantizes_the_rest():
    m = tiny()
    linears = [n for n, c in m.named_modules() if type(c) is nn.Linear]
    patterns = qlora_bnb_config(torch.bfloat16).llm_int8_skip_modules
    converted = [n for n in linears if should_convert_module(n, patterns)]
    vision = [n for n in linears if is_vision_param_name(n)]
    assert vision and not set(vision) & set(converted)                                      # blocks, merger: none converted
    assert "lm_head" in linears and "lm_head" not in converted
    assert set(converted) == set(linears) - set(vision) - {"lm_head"} and len(converted) > 20  # every language-model layer is


def test_a_bare_visual_entry_does_not_match_on_this_transformers_which_is_why_the_list_has_model_visual():
    """transformers 5.x matches from the start of the dotted name; the tower is model.visual.*, so ["visual"] alone quantizes it."""
    linears = [n for n, c in tiny().named_modules() if type(c) is nn.Linear and is_vision_param_name(n)]
    assert all(should_convert_module(n, ["visual"]) for n in linears)                       # NOT skipped
    assert not any(should_convert_module(n, ["model.visual"]) for n in linears)             # skipped


def test_giving_a_skip_list_drops_the_automatic_lm_head_skip_so_the_constant_names_it():
    m = tiny()
    patterns = HfQuantizer.get_modules_to_not_convert(m, ["model.visual"], None)
    assert "lm_head" not in patterns and should_convert_module("lm_head", patterns)         # without it lm_head WOULD be 4-bit
    patterns = HfQuantizer.get_modules_to_not_convert(m, list(QLORA_SKIP_MODULES), None)
    assert not should_convert_module("lm_head", patterns)


# ---------------------------------------------------------------------------------------------------------------------
# The load check (a)/(b)/(c)
# ---------------------------------------------------------------------------------------------------------------------
def test_check_passes_on_a_correctly_loaded_model_and_reports_the_projector():
    m = fake_quantize(tiny())
    targets = lora_target_modules(m, linear_attention=True, vision_projector=True)
    s = check_qlora_model(m, targets, linear4bit_cls=FakeLinear4bit)
    assert s["ok"] and s["vision_linear4bit"] == 0 and s["vision_linears"] == 10
    assert s["language_linears"] == s["language_linear4bit"] > 20 and s["lm_head_quantized"] is False
    assert s["merger_linears"] == ["model.visual.merger.linear_fc1", "model.visual.merger.linear_fc2"]
    assert s["lora_targets_missing"] == [] and s["vision_dtypes"] == ["float32"]


def test_check_fails_loudly_when_the_vision_tower_was_quantized():
    m = fake_quantize(tiny(), patterns=["lm_head"])                                          # the old behaviour: only lm_head skipped
    with pytest.raises(AssertionError, match=r"\(a\) 10 vision-tower layers are 4-bit"):
        check_qlora_model(m, linear4bit_cls=FakeLinear4bit)


def test_check_fails_when_language_layers_or_lm_head_are_wrong_or_a_target_is_missing():
    with pytest.raises(AssertionError, match=r"\(b\).*NOT 4-bit"):
        check_qlora_model(tiny(), linear4bit_cls=FakeLinear4bit)                              # nothing quantized: a bf16 load, not QLoRA
    with pytest.raises(AssertionError, match=r"\(b\) lm_head is 4-bit"):
        check_qlora_model(fake_quantize(tiny(), patterns=["model.visual"]), linear4bit_cls=FakeLinear4bit)
    with pytest.raises(AssertionError, match=r"\(c\).*not_a_layer"):
        check_qlora_model(fake_quantize(tiny()), ["q_proj", "not_a_layer"], linear4bit_cls=FakeLinear4bit)


# ---------------------------------------------------------------------------------------------------------------------
# Projector LoRA on plain Linear layers; adapters trained under this load reload identically; GRPO dual adapters
# ---------------------------------------------------------------------------------------------------------------------
def lora_config(base):
    return peft.LoraConfig(r=4, lora_alpha=8, lora_dropout=0.0, task_type="CAUSAL_LM",
                           target_modules=lora_target_modules(base, linear_attention=True, vision_projector=True))


def test_peft_attaches_projector_lora_to_the_plain_merger_layers():
    base = fake_quantize(tiny())
    merger = projector_module_names(base)
    assert merger and all(type(base.get_submodule(n)) is nn.Linear for n in merger)         # plain Linear, not 4-bit
    wrapped = peft.get_peft_model(base, lora_config(base))
    cov = lora_coverage(wrapped)
    assert cov["projector"] == 2 and cov["mlp"] > 0 and cov["full_attention"] > 0 and cov["linear_attention"] > 0
    names = [n for n, _ in wrapped.named_parameters() if "lora_A" in n and "merger" in n]
    assert len(names) == 2
    assert not any("lora_" in n and is_vision_param_name(n) and "merger" not in n for n, _ in wrapped.named_parameters())  # ViT blocks: none


def trained_adapter(tmp_path):
    base = fake_quantize(tiny())
    wrapped = peft.get_peft_model(base, lora_config(base))
    torch.manual_seed(1)
    with torch.no_grad():
        for n, p in wrapped.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn_like(p) * 0.3)                                           # non-zero B: a zero B would make this vacuous
    wrapped.save_pretrained(tmp_path / "sft")
    return wrapped, tmp_path / "sft"


def logits(model, ids):
    model.eval()
    with torch.no_grad():
        return model(input_ids=ids).logits


def test_adapter_trained_under_the_skip_list_reloads_onto_the_same_loading_path(tmp_path):
    trained, folder = trained_adapter(tmp_path)
    ids = torch.arange(1, 17).unsqueeze(0)
    reloaded = PeftModel.from_pretrained(fake_quantize(tiny()), str(folder))                # the evaluate_models / GRPO load order
    check_qlora_model(reloaded.base_model.model, linear4bit_cls=FakeLinear4bit)              # still quantized only where intended
    assert torch.allclose(logits(trained, ids), logits(reloaded, ids), atol=1e-6)
    for n, p in reloaded.named_parameters():
        if "lora_" in n and "merger" in n:                                                   # the projector LoRA came back too
            assert torch.equal(p, dict(trained.named_parameters())[n]) and p.abs().sum() > 0


def test_grpo_dual_adapters_on_the_skip_list_base_reference_frozen_policy_starts_at_sft(tmp_path):
    trained, folder = trained_adapter(tmp_path)
    ids = torch.arange(1, 17).unsqueeze(0)
    model = PeftModel.from_pretrained(fake_quantize(tiny()), str(folder), adapter_name="reference", is_trainable=False)
    ref = model.peft_config["reference"]
    model.add_adapter("grpo_policy", peft.LoraConfig(r=ref.r, lora_alpha=ref.lora_alpha, target_modules=ref.target_modules,
                                                      lora_dropout=ref.lora_dropout, bias="none", task_type="CAUSAL_LM"))
    n = init_policy_from_reference(model)
    model.set_adapter("grpo_policy")
    assert n > 0 and any("merger" in k and "grpo_policy" in k for k, _ in model.named_parameters())
    model.set_adapter("reference")
    ref_logits = logits(model, ids)
    model.set_adapter("grpo_policy")
    assert torch.allclose(logits(model, ids), ref_logits, atol=1e-6) and torch.allclose(ref_logits, logits(trained, ids), atol=1e-6)
    ref_params = [p for k, p in model.named_parameters() if ".reference." in k and "lora_" in k]
    pol_params = [p for k, p in model.named_parameters() if ".grpo_policy." in k and "lora_" in k]
    assert ref_params and not any(p.requires_grad for p in ref_params) and all(p.requires_grad for p in pol_params)
    assert any("merger" in k for k, p in model.named_parameters() if ".grpo_policy." in k and p.requires_grad)   # projector is trained in GRPO


# ---------------------------------------------------------------------------------------------------------------------
# prepare_model_for_kbit_training must not upcast the (now 16-bit) vision tower
# ---------------------------------------------------------------------------------------------------------------------
def test_prepare_qlora_model_keeps_the_vision_tower_dtype_and_upcasts_the_rest_as_before():
    m = tiny().to(torch.bfloat16)
    m.is_loaded_in_4bit = True                                                               # what bitsandbytes loading sets
    out = prepare_qlora_model(m, use_gradient_checkpointing=False)
    vis = {p.dtype for n, p in out.named_parameters() if is_vision_param_name(n)}
    rest = {p.dtype for n, p in out.named_parameters() if not is_vision_param_name(n) and p.ndim <= 2}
    assert vis == {torch.bfloat16}                                                           # peft alone would give float32
    assert torch.float32 in rest                                                             # embeddings / norms / lm_head: peft's upcast unchanged
    assert not any(p.requires_grad for p in out.parameters())

    m2 = tiny().to(torch.bfloat16)
    m2.is_loaded_in_4bit = True
    from peft import prepare_model_for_kbit_training
    plain = prepare_model_for_kbit_training(m2, use_gradient_checkpointing=False)
    assert {p.dtype for n, p in plain.named_parameters() if is_vision_param_name(n)} == {torch.float32}   # the behaviour being avoided


# ---------------------------------------------------------------------------------------------------------------------
# GRPO preflight compares the SFT adapter's quantization with the one GRPO will load
# ---------------------------------------------------------------------------------------------------------------------
def preflight_rows(tmp_path, meta, grpo_precision):
    import json
    from argparse import Namespace

    from dental_agent.training import preflight as pf

    (tmp_path / "sft_run_meta.json").write_text(json.dumps({"track": "with_tools", "canonical_resize": True, **meta}))
    rep = pf._Report(lambda line: None)
    args = Namespace(track="with_tools", canonical_resize=True, dev_split=None, max_seq_len=16384, pad_vision_to_slots=False,
                     vision_slots=(5, 10, 4), precision=grpo_precision)
    pf._check_sft_consistency(args, tmp_path, set(), rep)
    return {name: (level, detail) for level, name, detail in rep.rows}


def test_preflight_passes_when_sft_and_grpo_share_the_qlora_skip_list(tmp_path):
    rows = preflight_rows(tmp_path, {"precision": "qlora", "qlora_skip_modules": list(QLORA_SKIP_MODULES)}, "qlora")
    assert rows["QLoRA skip list vs SFT"][0] == "PASS" and "precision vs SFT" not in rows


def test_preflight_warns_for_a_qlora_adapter_trained_before_the_vision_tower_was_excluded(tmp_path):
    rows = preflight_rows(tmp_path, {"precision": "qlora"}, "qlora")                          # no qlora_skip_modules key: old adapter
    assert rows["QLoRA vision tower vs SFT"][0] == "WARN" and "NF4 merger" in rows["QLoRA vision tower vs SFT"][1]


def test_preflight_warns_on_a_different_skip_list_and_on_qlora_vs_bf16(tmp_path):
    assert preflight_rows(tmp_path, {"precision": "qlora", "qlora_skip_modules": ["visual"]}, "qlora")["QLoRA skip list vs SFT"][0] == "WARN"
    rows = preflight_rows(tmp_path, {"precision": "bf16"}, "qlora")
    assert rows["precision vs SFT"][0] == "WARN"
    assert "precision vs SFT" not in preflight_rows(tmp_path, {"precision": "bf16"}, "bf16")             # bf16 on both: nothing to say
