"""
Inference generation and vision-token probing utilities for Qwen3-VL.
"""

from __future__ import annotations

from typing import Any, Tuple, Optional
from PIL import Image

from dental_agent.model.backbone import safe_process_vision_info


def probe_vision_tokens(
    processor: Any,
    image: Image.Image,
    prompt_text: str = "Analyze this X-ray.",
) -> int:
    """Measure how many tokens a single image consumes under Qwen3-VL's patch merger."""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = safe_process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    return inputs.input_ids.shape[1]


def generate_agent_reply(
    model: Any,
    processor: Any,
    messages: list[dict[str, Any]],
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    return_ids: bool = False,
    past_key_values: Any | None = None,
    cache_state: dict[str, Any] | None = None,
    return_cache: bool = False,
) -> str | tuple[str, int, list[int]] | tuple[str, int, list[int], Any, dict[str, Any]]:
    """Execute one forward generation step for multi-turn assistant messages.

    Supports both stateless prefill and stateful cross-turn KV-cache reuse with
    automatic 3D MRoPE coordinate alignment and self-healing fallback.

    Parameters
    ----------
    return_ids : bool
        If True, returns (reply_text, prompt_len, generated_token_ids) for GRPO span tracking.
    past_key_values : Any, optional
        Pre-existing KV-cache (e.g. DynamicCache) from prior turns to bypass quadratic prefill.
    cache_state : dict[str, Any], optional
        State tracker containing 'total_len', 'last_msg_idx', etc.
    return_cache : bool
        If True, returns updated (past_key_values, cache_state) alongside reply and IDs.
    """
    import torch
    from dental_agent.model.local_runtime import generate_local, input_device

    # Build the exact full prompt first. Cross-turn reuse is safe only if its
    # token prefix matches the cached sequence; raw chat-template deltas can
    # shift BPE boundaries, omit a final sampled token or alter M-RoPE positions.
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = safe_process_vision_info(messages)
    inputs = dict(processor(text=[text], images=image_inputs, videos=video_inputs,
                            padding=True, return_tensors="pt"))
    device = input_device(model)
    inputs = {k:v.to(device) if hasattr(v,'to') else v for k,v in inputs.items()}
    prompt_len = inputs['input_ids'].shape[1]
    gen_kwargs = {
        'max_new_tokens': max_new_tokens,
        'pad_token_id': (getattr(processor.tokenizer,'pad_token_id',None)
                         if getattr(processor.tokenizer,'pad_token_id',None) is not None
                         else getattr(processor.tokenizer,'eos_token_id',None)),
        'return_dict_in_generate': True,
        'do_sample': temperature > 0,
    }
    if temperature > 0:
        gen_kwargs['temperature'] = temperature
    reused = False
    generation_inputs = inputs
    generated_prompt_len = prompt_len
    if past_key_values is not None and cache_state and 'cached_input_ids' in cache_state:
        last_count = cache_state.get('last_msg_count', len(messages))
        delta = messages[last_count:]
        has_new_images = any(isinstance(m.get('content'),list) and any(
            isinstance(c,dict) and c.get('type') in {'image','video'} for c in m['content']) for m in delta)
        length_fn = getattr(past_key_values,'get_seq_length',None)
        if delta and not has_new_images and callable(length_fn):
            cached_len = int(length_fn())
            previous = cache_state['cached_input_ids']
            prefix_matches = (0 < cached_len < prompt_len and previous.shape[1] >= cached_len and
                              torch.equal(inputs['input_ids'][:, :cached_len].cpu(), previous[:, :cached_len]))
            if prefix_matches:
                generation_inputs = {'input_ids':inputs['input_ids'][:,cached_len:],
                                     'attention_mask':inputs['attention_mask'], 'past_key_values':past_key_values}
                if 'mm_token_type_ids' in inputs:
                    generation_inputs['mm_token_type_ids'] = inputs['mm_token_type_ids'][:,cached_len:]
                generated_prompt_len = prompt_len - cached_len
                reused = True
    try:
        gen_out = generate_local(model, **generation_inputs, **gen_kwargs)
    except (ValueError, RuntimeError) as exc:
        if not reused or 'out of memory' in str(exc).lower():
            raise
        # Cache/multimodal incompatibility: retry the exact full prompt, never
        # remove its images or lower its generation budget.
        import logging
        logging.getLogger(__name__).warning('Validated cache reuse failed; full prefill: %s', exc)
        generated_prompt_len = prompt_len
        gen_out = generate_local(model, **inputs, **gen_kwargs)
    generated_ids = gen_out.sequences if hasattr(gen_out,'sequences') else gen_out
    new_ids = generated_ids[:, generated_prompt_len:]
    new_pkv = getattr(gen_out,'past_key_values',None)
    reply = processor.batch_decode(new_ids,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()
    gen_ids_list = new_ids[0].tolist()
    new_cache_state = {
        'total_len':prompt_len+new_ids.shape[1], 'last_msg_count':len(messages),
        'cached_input_ids':torch.cat([inputs['input_ids'],new_ids.to(device)],dim=1).detach().cpu(),
    }
    # Full prompt offsets are essential for GRPO assistant-token span labels,
    # even if generate() was called with a shorter cache delta.
    if return_cache:
        return reply,prompt_len,gen_ids_list,new_pkv,new_cache_state
    if return_ids:
        return reply,prompt_len,gen_ids_list
    return reply
