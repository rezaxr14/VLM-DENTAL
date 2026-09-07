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

    # Determine if we can attempt delta KV-cache generation
    attempt_cache_reuse = (
        past_key_values is not None
        and cache_state is not None
        and "total_len" in cache_state
        and "last_msg_count" in cache_state
        and len(messages) > cache_state["last_msg_count"]
    )

    if attempt_cache_reuse:
        try:
            last_count = cache_state["last_msg_count"]
            delta_messages = messages[last_count:]

            # Check if delta messages contain any new images
            has_new_images = any(
                isinstance(m.get("content"), list)
                and any(c.get("type") == "image" for c in m["content"] if isinstance(c, dict))
                for m in delta_messages
            )

            # Format delta text
            delta_text = processor.apply_chat_template(delta_messages, tokenize=False, add_generation_prompt=True)

            if not has_new_images:
                # Text-only delta: Bypass vision encoder entirely!
                delta_inputs = processor.tokenizer(delta_text, return_tensors="pt")
                delta_inputs = {k: v.to(model.device) for k, v in delta_inputs.items()}
                delta_len = delta_inputs["input_ids"].shape[1]
                total_len = cache_state["total_len"] + delta_len

                # Build full attention mask covering cached + delta tokens
                attention_mask = torch.ones((1, total_len), dtype=torch.long, device=model.device)

                gen_kwargs: dict[str, Any] = {
                    "max_new_tokens": max_new_tokens,
                    "pad_token_id": getattr(processor.tokenizer, "pad_token_id", None) or getattr(processor.tokenizer, "eos_token_id", None),
                    "use_cache": True,
                    "return_dict_in_generate": True,
                }
                if temperature > 0:
                    gen_kwargs["do_sample"] = True
                    gen_kwargs["temperature"] = temperature
                else:
                    gen_kwargs["do_sample"] = False

                with torch.no_grad():
                    gen_out = model.generate(
                        input_ids=delta_inputs["input_ids"],
                        attention_mask=attention_mask,
                        past_key_values=past_key_values,
                        **gen_kwargs,
                    )

                new_ids = gen_out.sequences[:, delta_len:]
                new_pkv = getattr(gen_out, "past_key_values", None)
                reply = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

                updated_cache_state = {
                    "total_len": total_len + new_ids.shape[1],
                    "last_msg_count": len(messages),
                }

                gen_ids_list = new_ids[0].tolist()
                if return_cache:
                    return reply.strip(), delta_len, gen_ids_list, new_pkv, updated_cache_state
                if return_ids:
                    return reply.strip(), delta_len, gen_ids_list
                return reply.strip()

        except Exception as e:
            # Self-healing fallback: smoothly revert to full prefill on any mismatch
            import logging
            logging.getLogger(__name__).debug("KV-cache delta prefill fallback triggered: %s", e)

    # Standard full prefill (Turn 1 or cache fallback)
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = safe_process_vision_info(messages)

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
        "use_cache": True,
        "return_dict_in_generate": True,
    }
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        gen_out = model.generate(**inputs, **gen_kwargs)

    if hasattr(gen_out, "sequences"):
        generated_ids = gen_out.sequences
        new_pkv = getattr(gen_out, "past_key_values", None)
    else:
        generated_ids = gen_out
        new_pkv = None

    new_ids = generated_ids[:, prompt_len:]
    reply = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    gen_ids_list = new_ids[0].tolist()

    new_cache_state = {
        "total_len": prompt_len + new_ids.shape[1],
        "last_msg_count": len(messages),
    }

    if return_cache:
        return reply.strip(), prompt_len, gen_ids_list, new_pkv, new_cache_state
    if return_ids:
        return reply.strip(), prompt_len, gen_ids_list
    return reply.strip()

