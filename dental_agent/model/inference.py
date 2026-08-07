"""
Inference generation and vision-token probing utilities for Qwen2.5-VL.
"""

from __future__ import annotations

from typing import Any, Tuple, Optional
from PIL import Image


def probe_vision_tokens(
    processor: Any,
    image: Image.Image,
    prompt_text: str = "Analyze this X-ray.",
) -> int:
    """Measure how many tokens a single image consumes under Qwen2.5-VL's patch merger."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        def process_vision_info(msgs):
            return None, None

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
    image_inputs, video_inputs = process_vision_info(messages)
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
) -> str | tuple[str, int, list[int]]:
    """Execute one forward generation step for multi-turn assistant messages.

    Parameters
    ----------
    return_ids : bool
        If True, returns (reply_text, prompt_len, generated_token_ids) for GRPO span tracking.
    """
    import torch
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        def process_vision_info(msgs):
            return None, None

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    prompt_len = inputs["input_ids"].shape[1]

    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": getattr(processor.tokenizer, "pad_token_id", None) or getattr(processor.tokenizer, "eos_token_id", None),
    }
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        generated_ids = model.generate(**inputs, **gen_kwargs)

    new_ids = generated_ids[:, prompt_len:]
    reply = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    if return_ids:
        gen_ids_list = new_ids[0].tolist()
        return reply.strip(), prompt_len, gen_ids_list
    return reply.strip()
