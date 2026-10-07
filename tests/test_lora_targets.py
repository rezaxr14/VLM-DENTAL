"""LoRA reaches the Gated-DeltaNet (linear-attention) layers, and only the intended modules."""

import pytest
import torch

from dental_agent.model.backbone import lora_coverage, lora_target_modules, projector_module_names
from dental_agent.training.sft import freeze_and_guard_vision_tower

peft = pytest.importorskip("peft")
from transformers import Qwen3_5ForConditionalGeneration  # noqa: E402
from transformers.models.qwen3_5.configuration_qwen3_5 import (  # noqa: E402
    Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig,
)


def _tiny():
    text = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=2,
        num_key_value_heads=1, head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16,
        linear_value_head_dim=16, linear_conv_kernel_dim=4, max_position_embeddings=512,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
    )
    vision = Qwen3_5VisionConfig(
        depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=64, patch_size=16,
        spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=2304,
    )
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vision, tie_word_embeddings=False))


def _wrap(linear_attention, vision_projector=True):
    base = _tiny()
    cfg = peft.LoraConfig(
        r=4, lora_alpha=8, task_type="CAUSAL_LM",
        target_modules=lora_target_modules(base, linear_attention=linear_attention, vision_projector=vision_projector),
    )
    return peft.get_peft_model(base, cfg)


def test_target_list_contents():
    base = lora_target_modules(linear_attention=False)
    full = lora_target_modules(_tiny(), linear_attention=True, vision_projector=True)
    assert base == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert full[len(base):len(base) + 3] == ["in_proj_qkv", "in_proj_z", "out_proj"]
    assert full[len(base) + 3:] == projector_module_names(_tiny()) and len(full) == len(base) + 3 + 2


def test_projector_lora_fails_loudly_instead_of_matching_nothing():
    with pytest.raises(ValueError, match="needs the loaded model"):
        lora_target_modules(vision_projector=True)
    with pytest.raises(ValueError, match="no vision merger"):
        lora_target_modules(torch.nn.Linear(2, 2), vision_projector=True)


def test_linear_attention_layers_are_adapted_only_when_requested():
    without = lora_coverage(_wrap(linear_attention=False))
    with_la = lora_coverage(_wrap(linear_attention=True))
    assert without["linear_attention"] == 0 and without["full_attention"] == 4  # q,k,v,o of the one full-attn layer
    assert with_la["linear_attention"] == 9      # 3 Gated-DeltaNet layers x (in_proj_qkv, in_proj_z, out_proj)
    assert with_la["full_attention"] == 4 and with_la["mlp"] == 12 and with_la["projector"] == 2


def test_gates_and_vision_tower_are_not_adapted_and_gradients_flow():
    model = _wrap(linear_attention=True)
    adapted = [n for n, m in model.named_modules() if hasattr(m, "lora_A")]
    assert not any(n.endswith(("in_proj_a", "in_proj_b")) for n in adapted)           # tiny decay/write gates untouched
    assert not any(".visual." in n and "merger" not in n for n in adapted)            # vision blocks untouched
    ids = torch.randint(2, 100, (1, 40))
    model(input_ids=ids, labels=ids).loss.backward()
    grads = {n: m.lora_B["default"].weight.grad for n, m in model.named_modules() if hasattr(m, "lora_B") and ".linear_attn." in n}
    assert len(grads) == 9 and all(g is not None and g.abs().sum() > 0 for g in grads.values())


def test_projector_trains_through_lora_and_is_saved_with_the_adapter(tmp_path):
    from safetensors.torch import load_file

    model = _wrap(linear_attention=True, vision_projector=True)
    freeze_and_guard_vision_tower(model, train_merger=True, is_master=False)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("lora_" in n for n in trainable)                       # base weights (incl. merger) stay frozen
    assert any("merger" in n for n in trainable)                                    # ...but the projector adapts via LoRA
    model.save_pretrained(tmp_path)
    saved = load_file(str(tmp_path / "adapter_model.safetensors"))
    assert any("merger" in k and "lora" in k for k in saved)                        # persisted, so inference sees it


def test_vision_guard_finds_the_tower_when_peft_wrapped_and_blocks_gradients():
    model = _wrap(linear_attention=True, vision_projector=False)
    visual = next(m for n, m in model.named_modules() if n.split(".")[-1] == "visual")
    freeze_and_guard_vision_tower(model, train_merger=False, is_master=False)
    pixels = torch.randn(64, 3 * 2 * 16 * 16, requires_grad=True)
    grid = torch.tensor([[1, 8, 8]])
    out = visual(pixels, grid_thw=grid)
    tensors = out if isinstance(out, torch.Tensor) else (out.pooler_output if hasattr(out, "pooler_output") else out[0])
    assert not tensors.requires_grad                                                # whole tower ran under no_grad
