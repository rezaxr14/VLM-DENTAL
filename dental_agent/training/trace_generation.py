"""
Aim 1: Synthetic Expert Diagnostic Demonstration Trace Generation & Cross-Family Verification (§15, §16).

Implements an Interactive Teacher Loop: the teacher VLM explicitly interacts with the environment
turn-by-turn to build a realistic SFT trajectory that perfectly matches the `loop.py` inference contract.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable
from PIL import Image
import pandas as pd

from dental_agent.agent.parsing import parse_agent_json
from dental_agent.agent.prompts import build_agent_system_prompt
from dental_agent.tools.registry import ToolRegistry
from dental_agent.training.api_pool import (
    call_llm,
    get_gemini_pool,
    AllKeysExhaustedToday,
)
from dental_agent.utils.serialization import to_jsonable


def _is_valid_key(val: str | None) -> bool:
    if not val:
        return False
    v = val.strip().lower()
    return bool(v and not v.startswith("your_") and not v.startswith("placeholder") and v != "none")


_has_anthropic = _is_valid_key(os.environ.get("ANTHROPIC_API_KEY"))

GENERATOR_PROVIDER = os.environ.get("GENERATOR_PROVIDER", "gemini")
VERIFIER_PROVIDER = os.environ.get(
    "VERIFIER_PROVIDER",
    "anthropic" if _has_anthropic else "gemini",
)
GENERATOR_MODEL = os.environ.get("GEMINI_PRIMARY_MODEL", "gemini-3.6-flash")
VERIFIER_MODEL = os.environ.get(
    "VERIFIER_MODEL",
    "claude-3-5-sonnet-20241022" if _has_anthropic else os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash"),
)

VERIFIER_SYSTEM_PROMPT = (
    "You are a strict verifier, not a rewriter. Given an X-ray image, the KNOWN correct "
    "ground truth, and a candidate multi-turn reasoning trace, judge ONLY whether every claim "
    "in the trace is actually supported by the image and the tools used. Reject any trace asserting "
    "things that cannot be seen in the visual evidence, even if the final answer is technically correct.\n"
    'Respond with exactly one JSON object: {"grounded": true/false, "reason": "..."}.'
)


def _format_ground_truth(anns: pd.DataFrame, cat_lookup: dict[int, str], diag_col: str) -> list[dict[str, Any]]:
    """Format all findings for an image, safely handling missing diagnoses."""
    findings = []
    for _, ann in anns.iterrows():
        diag_id = ann.get(diag_col)
        # Fix: fallback to "unknown" instead of guessing "Caries"
        diag_name = cat_lookup.get(diag_id, "unknown")
        
        findings.append({
            "quadrant": int(ann.get("category_id_1", 1)),
            "tooth_position": int(ann.get("category_id_2", 1)),
            "diagnosis": diag_name,
            "bbox": list(ann.get("bbox", [0, 0, 50, 50])),
        })
    return findings


def generate_interactive_trajectory(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    registry: ToolRegistry,
    max_turns: int = 5,
    provider: str = GENERATOR_PROVIDER,
    model: str = GENERATOR_MODEL,
    call_llm_fn: Callable[..., str] = call_llm,
) -> dict[str, Any] | None:
    """
    Interactive Teacher Loop: Generates a realistic multi-turn reasoning trajectory by
    actually executing tools turn-by-turn.
    """
    system_prompt = build_agent_system_prompt(registry.format_tool_descriptions())
    
    # We add a hidden instruction to the teacher telling it the ground truth it needs to reach.
    teacher_directive = (
        f"TEACHER DIRECTIVE: You are generating an expert demonstration trace for SFT. "
        f"You MUST eventually reach this exact diagnosis: {json.dumps(ground_truth)}\n"
        f"Use tools to discover and verify these findings, then output them in your final answer."
    )
    
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Analyze this panoramic X-ray. Identify any abnormal teeth and determine the diagnosis.\n\n" + teacher_directive},
            ],
        },
    ]

    current_image = image
    turns: list[dict[str, Any]] = []
    final_answer = None
    
    for turn_idx in range(max_turns):
        try:
            # We must pass the conversation history to the LLM. 
            # `call_llm` currently expects system_prompt + user_content. 
            # To support multi-turn in the API pool, we pass the raw messages list.
            raw_output = call_llm_fn(
                provider, 
                model, 
                system_prompt="", 
                user_content=messages, 
                image=None, 
                temperature=0.7 # Add diversity for self-consistency if k>1
            )
        except Exception as e:
            print(f"LLM call failed during interactive loop: {e}")
            return None

        parsed = parse_agent_json(raw_output)
        
        turn_record = {
            "turn": turn_idx,
            "raw_output": raw_output,
            "parsed": parsed,
        }

        if not parsed:
            turn_record["status"] = "unparseable"
            turns.append(turn_record)
            break

        messages.append({"role": "assistant", "content": raw_output})

        if "final_answer" in parsed:
            final_answer = parsed["final_answer"]
            turn_record["status"] = "final_answer"
            turns.append(turn_record)
            break

        tool_name = parsed.get("tool")
        tool_args = parsed.get("args", {})

        if not tool_name or not registry.get(tool_name):
            turn_record["status"] = "invalid_tool"
            turns.append(turn_record)
            messages.append({"role": "user", "content": f"Error: Tool '{tool_name}' is not recognized."})
            continue

        turn_record["tool_name"] = tool_name
        turn_record["tool_args"] = tool_args

        # Execute tool interactively
        try:
            if tool_name in ["zoom_crop", "window_level", "denoise", "contralateral_compare"]:
                tool_out = registry.execute(tool_name, image=current_image, **tool_args)
                current_image = tool_out
                obs = [
                    {"type": "image", "image": tool_out},
                    {"type": "text", "text": f"Result of {tool_name}:"},
                ]
            else:
                tool_out = registry.execute(tool_name, **tool_args)
                obs = [{"type": "text", "text": f"Tool output: {json.dumps(tool_out)}"}]
                
            turn_record["tool_ok"] = True
            messages.append({"role": "user", "content": obs})
            
        except Exception as e:
            turn_record["tool_ok"] = False
            turn_record["tool_error"] = str(e)
            messages.append({"role": "user", "content": f"Tool execution failed: {e}"})

        turns.append(turn_record)

    if final_answer is None:
        return None

    # Strip the teacher directive from the final output messages to prevent data leakage in SFT
    clean_messages = list(messages)
    first_user_content = clean_messages[1]["content"]
    clean_messages[1] = {
        "role": "user",
        "content": [
            first_user_content[0], 
            {"type": "text", "text": "Analyze this panoramic X-ray. Identify any abnormal teeth and determine the diagnosis."}
        ]
    }

    return {
        "turns": turns,
        "tool_calls": len([t for t in turns if "tool_name" in t]),
        "final_answer": final_answer,
        "messages": clean_messages,
        "format_ok": True,
    }


def verify_trace(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    trajectory: dict[str, Any],
    provider: str = VERIFIER_PROVIDER,
    model: str = VERIFIER_MODEL,
    call_llm_fn: Callable[..., str] = call_llm,
) -> dict[str, Any]:
    """Verify trace using the Verifier model."""
    trace_text = json.dumps(trajectory.get("turns", []), indent=2)
    user_content = f"Ground Truth: {json.dumps(ground_truth)}\n\nCandidate Trace:\n{trace_text}"
    
    raw = call_llm_fn(provider, model, VERIFIER_SYSTEM_PROMPT, user_content, image=image, temperature=0.0)
    parsed = parse_agent_json(raw)
    
    if parsed and "grounded" in parsed:
        return parsed
    return {"grounded": False, "reason": "verifier output unparseable"}


def build_trace_example(
    image_id: int,
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame | None = None,
    k: int = 1, # Default to 1 for interactive loop to save cost
    diag_col: str = "category_id_3",
    call_llm_fn: Callable[..., str] = call_llm,
) -> dict[str, Any] | None:
    """Canonical Aim 1 pipeline for generating and verifying traces for an image."""
    matches = images_df[images_df["id"] == image_id]
    if matches.empty:
        return None
    row = matches.iloc[0]
    image_path = row.get("local_path")
    if not image_path or not os.path.exists(str(image_path)):
        return None

    image = Image.open(image_path).convert("RGB")
    anns = annots_df[annots_df["image_id"] == image_id]
    if anns.empty:
        return None

    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )
    
    # Format ALL findings, not just iloc[0]
    ground_truth = _format_ground_truth(anns, cat_lookup, diag_col)
    
    registry = ToolRegistry.create_default()

    candidates = []
    for _ in range(k):
        traj = generate_interactive_trajectory(image, ground_truth, registry, call_llm_fn=call_llm_fn)
        if traj:
            candidates.append(traj)

    verified = []
    for t in candidates:
        v_result = verify_trace(image, ground_truth, t, call_llm_fn=call_llm_fn)
        if v_result.get("grounded"):
            t["verifier_reason"] = v_result.get("reason")
            verified.append(t)

    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "ground_truth": ground_truth,
        "n_candidates": len(candidates),
        "n_verified": len(verified),
        "verified_traces": verified,
    }


def run_aim1_batch(
    image_ids: list[int],
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame | None = None,
    k: int = 1,
    cache_path: str | Path | None = None,
    resume: bool = True,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    diag_col: str = "category_id_3",
) -> list[dict[str, Any]]:
    """Production batch generation with disk caching and retry."""
    results: list[dict[str, Any]] = []
    done_ids: set[int] = set()

    if cache_path and resume and os.path.exists(str(cache_path)):
        # Load from jsonl
        with open(cache_path, "r") as f:
            for line in f:
                if not line.strip(): continue
                data = json.loads(line)
                results.append(data)
                done_ids.add(data["image_id"])
        print(f"Resuming: {len(done_ids)} image(s) already processed in {cache_path}")

    todo = [i for i in image_ids if i not in done_ids]
    total_candidates = total_verified = 0

    if cache_path:
        out_f = open(cache_path, "a" if resume else "w")
    else:
        out_f = None

    try:
        for idx, image_id in enumerate(todo):
            result = None
            try:
                for attempt in range(max_retries):
                    try:
                        result = build_trace_example(
                            image_id=image_id,
                            images_df=images_df,
                            annots_df=annots_df,
                            categories_df=categories_df,
                            k=k,
                            diag_col=diag_col,
                        )
                        break
                    except AllKeysExhaustedToday:
                        raise
                    except Exception as e:
                        wait = retry_delay * (2 ** attempt)
                        print(f"  image_id={image_id}: attempt {attempt + 1}/{max_retries} failed ({e}); retrying in {wait:.0f}s")
                        time.sleep(wait)
                else:
                    print(f"  image_id={image_id}: giving up after {max_retries} attempts, skipping")
            except AllKeysExhaustedToday as e:
                print(f"\n{e}")
                print(f"Stopped after {idx}/{len(todo)} image(s) from this run.")
                break

            if result:
                results.append(to_jsonable(result))
                total_candidates += result.get("n_candidates", 0)
                total_verified += result.get("n_verified", 0)
                
                if out_f and result.get("n_verified", 0) > 0:
                    for vt in result["verified_traces"]:
                        # Unpack verified traces as individual examples for SFT
                        vt["image_id"] = image_id
                        vt["image_path"] = result["image_path"]
                        vt["ground_truth"] = result["ground_truth"]
                        out_f.write(json.dumps(to_jsonable(vt)) + "\n")
                    out_f.flush()

            if (idx + 1) % 10 == 0 or idx == len(todo) - 1:
                rate = total_verified / max(total_candidates, 1)
                print(f"  {idx + 1}/{len(todo)} done — verified rate so far: {rate:.1%}")

    finally:
        if out_f:
            out_f.close()

    print(f"\nBatch run finished.")
    return results
