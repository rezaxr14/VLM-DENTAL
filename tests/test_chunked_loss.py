"""--loss-chunk-tokens: the chunked, assistant-tokens-only loss equals the stock HuggingFace loss (value and gradients)."""

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dental_agent.training.chunked_loss import causal_lm_loss  # noqa: E402


def _tiny(vocab=300):
    from transformers import Qwen3_5ForConditionalGeneration
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
    text = Qwen3_5TextConfig(
        vocab_size=vocab, hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, max_position_embeddings=512, layer_types=["linear_attention", "full_attention"])
    vis = Qwen3_5VisionConfig(depth=1, hidden_size=64, intermediate_size=128, num_heads=4, out_hidden_size=64, patch_size=16,
                              spatial_merge_size=2, temporal_patch_size=2, num_position_embeddings=2304)
    torch.manual_seed(0)
    return Qwen3_5ForConditionalGeneration(Qwen3_5Config(text_config=text, vision_config=vis, tie_word_embeddings=False)).float()


def _batch(b=2, t=40, vocab=300, masked_prefix=25):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(5, vocab, (b, t), generator=g)
    labels = ids.clone()
    labels[:, :masked_prefix] = -100            # prompt / images / tool results are not supervised
    mask = torch.ones(b, t, dtype=torch.long)
    if b > 1:
        labels[1, -6:] = -100                   # padding of the shorter row
        mask[1, -6:] = 0
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


@pytest.mark.parametrize("chunk", [1, 7, 64, 100000])
def test_value_matches_stock_loss(chunk):
    model = _tiny().eval()
    inputs = _batch()
    with torch.no_grad():
        want = model(**inputs).loss
        got = causal_lm_loss(model, inputs, chunk)
    assert torch.allclose(want, got, atol=1e-5, rtol=1e-5), (want.item(), got.item())


def test_gradients_match_stock_loss_under_lora():
    from peft import LoraConfig, get_peft_model
    model = get_peft_model(_tiny(), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj", "o_proj", "gate_proj"]))
    model.train()
    inputs = _batch()

    def grads(loss_fn):
        model.zero_grad()
        loss = loss_fn()
        loss.backward()
        return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}

    l0, g0 = grads(lambda: model(**inputs).loss)
    l1, g1 = grads(lambda: causal_lm_loss(model, inputs, 7))
    assert g0 and g0.keys() == g1.keys()
    assert torch.allclose(l0, l1, atol=1e-5, rtol=1e-5)
    for n in g0:
        assert torch.allclose(g0[n], g1[n], atol=1e-5, rtol=1e-4), n


def test_zero_chunk_is_the_stock_loss_and_all_masked_is_safe():
    model = _tiny().eval()
    inputs = _batch()
    with torch.no_grad():
        assert torch.equal(causal_lm_loss(model, inputs, 0), model(**inputs).loss)
        inputs["labels"][:] = -100
        assert causal_lm_loss(model, inputs, 8).item() == 0.0


def test_only_one_chunk_of_logits_is_alive_at_a_time():
    """Peak extra memory scales with the chunk, not with the sequence: count the largest [rows, vocab] tensor the head sees."""
    model = _tiny(vocab=300).eval()
    inputs = _batch(b=1, t=200, masked_prefix=10)
    rows = []
    model.lm_head.register_forward_hook(lambda m, i, o: rows.append(o.shape[0] * o.shape[1] if o.dim() == 3 else o.shape[0]))
    with torch.no_grad():
        causal_lm_loss(model, inputs, 16)
    assert rows and max(rows) <= 16, rows


def test_training_forward_does_not_save_vocabulary_sized_activations():
    model = _tiny(vocab=300).train()
    inputs = _batch()
    shapes = []
    def pack(t):
        shapes.append(tuple(t.shape))
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        loss = causal_lm_loss(model, inputs, 7)
    assert not any(len(s) >= 2 and s[-1] == 300 for s in shapes), shapes
    loss.backward()
    assert model.lm_head.weight.grad is not None


def test_multimodal_lora_checkpointed_gradients_match_stock():
    from peft import LoraConfig, get_peft_model
    plain = _tiny()
    plain.config.image_token_id = 299
    plain.config.video_token_id = 298
    plain.config.vision_start_token_id = 297
    model = get_peft_model(plain, LoraConfig(r=4, lora_alpha=8, lora_dropout=0,
        target_modules=["q_proj", "v_proj", "linear_fc1", "linear_fc2"]))
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    inputs = _batch(b=1)
    inputs['input_ids'].remainder_(290)
    inputs['input_ids'][0, 4] = 297
    inputs['input_ids'][0, 5:9] = 299
    inputs['labels'] = inputs['input_ids'].clone()
    inputs['labels'][:, :25] = -100
    inputs['mm_token_type_ids'] = torch.zeros_like(inputs['input_ids'])
    inputs['mm_token_type_ids'][0, 5:9] = 1
    inputs['image_grid_thw'] = torch.tensor([[1, 4, 4]])
    inputs['pixel_values'] = torch.randn(16, 3*2*16*16)
    def grads(fn):
        model.zero_grad(set_to_none=True)
        loss = fn()
        loss.backward()
        return loss.detach(), {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    expected, g0 = grads(lambda: model(**inputs).loss)
    got, g1 = grads(lambda: causal_lm_loss(model, inputs, 7))
    assert g0.keys() == g1.keys()
    assert any('merger' in n for n in g1)
    torch.testing.assert_close(got, expected)
    for n in g0:
        torch.testing.assert_close(g1[n], g0[n], atol=1e-5, rtol=1e-4)


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='requires two real CUDA GPUs')
def test_body_and_head_on_different_gpus_backward():
    from accelerate import dispatch_model
    model = _tiny().to(dtype=torch.float16).train()
    model = dispatch_model(model, device_map={'model':0, 'lm_head':1})
    inputs = {k:v.cuda(0) for k,v in _batch(b=1).items()}
    loss = causal_lm_loss(model, inputs, 7)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()


@pytest.mark.parametrize('chunk', [128, 256, 512, 1024, 2048])
def test_larger_chunks_and_uneven_tail_preserve_loss_and_all_gradients(chunk):
    model = _tiny(vocab=257).train()
    inputs = _batch(b=1, t=219, vocab=257, masked_prefix=19)
    def run(fn):
        model.zero_grad(set_to_none=True)
        loss = fn()
        loss.backward()
        return loss.detach(), {n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
    expected, eg = run(lambda: model(**inputs).loss)
    actual, ag = run(lambda: causal_lm_loss(model, inputs, chunk))
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert ag.keys() == eg.keys()
    for name in eg:
        torch.testing.assert_close(ag[name], eg[name], rtol=2e-4, atol=2e-6)


def test_trainable_head_with_frozen_body_is_still_checkpointed():
    model = _tiny(vocab=257).train()
    model.model.requires_grad_(False)
    inputs = _batch(b=1, t=119, vocab=257)
    shapes = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (shapes.append(tuple(t.shape)) or t), lambda t:t):
        loss = causal_lm_loss(model, inputs, 16)
    assert not any(len(s) >= 2 and s[-1] == 257 for s in shapes), shapes
    loss.backward()
    assert model.lm_head.weight.grad is not None
    assert torch.isfinite(model.lm_head.weight.grad).all()
