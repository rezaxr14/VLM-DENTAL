"""
Aim 1: Synthetic Expert Diagnostic Demonstration Trace Generation & Cross-Family Verification (§15, §16).

Implements an Interactive Teacher Loop: the teacher VLM explicitly interacts with the environment
turn-by-turn to build a realistic SFT trajectory that perfectly matches the `loop.py` inference contract.
"""

from __future__ import annotations

import json
import os
import re
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
    'Respond with EXACTLY ONE JSON object and NO OTHER TEXT: {"grounded": true, "reason": "..."}.'
)


def _format_ground_truth(anns: pd.DataFrame, cat_lookup: dict[int, str], diag_col: str) -> list[dict[str, Any]]:
    """Format all findings for an image, safely handling missing diagnoses."""
    findings = []
    for _, ann in anns.iterrows():
        diag_id = ann.get(diag_col)
        # Dentex hardcoded mapping if JSON doesn't provide it
        fallback_map = {0: "Impacted", 1: "Caries", 2: "Periapical Lesion", 3: "Deep Caries"}
        diag_name = cat_lookup.get(diag_id) or fallback_map.get(diag_id, "unknown")
        
        # Convert DENTEX 0-indexed categories to standard FDI notation
        # DENTEX quadrant (0-3) -> FDI (1-4)
        # DENTEX position (0-7) -> FDI (1-8)
        fdi_quadrant = int(ann.get("category_id_1", 0)) + 1
        fdi_position = int(ann.get("category_id_2", 0)) + 1
        
        findings.append({
            "quadrant": fdi_quadrant,
            "tooth_position": fdi_position,
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
) -> tuple[dict[str, Any] | None, str | None]:
    """
    Hybrid Interactive Teacher Loop: 
    Pre-computes common tools to save API costs, sends them upfront, and allows the LLM 
    to hallucinate standard tool usage using `<fake_tool_call>`. If the LLM needs a tool 
    that wasn't pre-computed, it can still request it dynamically.
    The final trace is stitched back into a perfect multi-turn sequence for SFT.
    """
    system_prompt = build_agent_system_prompt(registry.format_tool_descriptions())
    
    # 1. Pre-compute all deterministic tool outputs so the AI has the full suite available instantly
    precomputed = {}
    
    # Always try to include zoom_crop and contralateral if we have a GT bbox
    has_gt = ground_truth and "bbox" in ground_truth[0]
    
    try:
        # Pre-compute all window level presets
        precomputed["window_level_bone"] = registry.execute("window_level", image=image, preset="bone")
        precomputed["window_level_enamel"] = registry.execute("window_level", image=image, preset="enamel")
        precomputed["window_level_soft_tissue"] = registry.execute("window_level", image=image, preset="soft_tissue")
        
        # Pre-compute all denoise methods
        precomputed["denoise_bilateral"] = registry.execute("denoise", image=image, method="bilateral")
        precomputed["denoise_median"] = registry.execute("denoise", image=image, method="median")
            
        if ground_truth:
            for i, gt in enumerate(ground_truth):
                quad = gt.get("quadrant")
                bbox = gt.get("bbox")
                if quad and bbox:
                    precomputed[f"contralateral_compare_gt_{i}"] = registry.execute("contralateral_compare", image=image, bbox=bbox, quadrant=quad)
                if bbox:
                    precomputed[f"zoom_crop_gt_{i}"] = registry.execute("zoom_crop", image=image, bbox=bbox)
            
    except Exception as e:
        print(f"Pre-computation failed: {e}")
    
    # 2. Build the initial prompt with pre-computed images
    initial_content = [
        {"type": "image", "image": image},
        {"type": "text", "text": "Analyze this panoramic X-ray. Identify any abnormal teeth and determine the diagnosis.\n\n"}
    ]
    
    # Hidden teacher directive
    directive = (
        f"TEACHER DIRECTIVE: You are generating an expert demonstration trace for SFT.\n"
        f"You MUST eventually reach this exact diagnosis: {json.dumps(ground_truth)}\n\n"
        f"To save API calls, I have already pre-computed ALL standard tool outputs for you:\n"
    )
    
    for key, img_result in precomputed.items():
        if key.startswith("window_level_"):
            preset = key.split("window_level_")[-1]
            initial_content.extend([{"type": "text", "text": f"Pre-computed: window_level(preset='{preset}'):"}, {"type": "image", "image": img_result}])
            directive += f"- window_level(preset='{preset}')\n"
        elif key.startswith("denoise_"):
            method = key.split("denoise_")[-1]
            initial_content.extend([{"type": "text", "text": f"Pre-computed: denoise(method='{method}'):"}, {"type": "image", "image": img_result}])
            directive += f"- denoise(method='{method}')\n"
        elif key.startswith("contralateral_compare_gt_"):
            idx = int(key.split("_gt_")[-1])
            quad = ground_truth[idx].get("quadrant")
            initial_content.extend([{"type": "text", "text": f"Pre-computed: contralateral_compare(target_quadrant={quad}):"}, {"type": "image", "image": img_result}])
            directive += f"- contralateral_compare(target_quadrant={quad})\n"
        elif key.startswith("zoom_crop_gt_"):
            idx = int(key.split("_gt_")[-1])
            bbox = ground_truth[idx].get("bbox")
            initial_content.extend([{"type": "text", "text": f"Pre-computed: zoom_crop(bbox={bbox}):"}, {"type": "image", "image": img_result}])
            directive += f"- zoom_crop(bbox={bbox})\n"
            
    directive += (
        f"\nCRITICAL INSTRUCTIONS:\n"
        f"1. You MUST NOT provide the final answer immediately!\n"
        f"2. Review the pre-computed images. You must pick the ones most useful for this diagnosis and write a fake tool call when you use them using this exact XML format:\n"
        f"<fake_tool_call>{{\"tool\": \"<tool_name>\", \"args\": {{<args>}}}}</fake_tool_call>\n"
        f"3. You must use several tools in your reasoning chain to arrive at the answer.\n"
        f"4. If you need a tool that was NOT pre-computed, output a standard JSON tool call (WITHOUT XML tags) and stop. I will provide the result in the next turn.\n"
        f"5. Once you have used the tools to verify the findings, output your final_answer JSON."
    )
    
    initial_content.append({"type": "text", "text": directive})
    
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": initial_content},
    ]

    turns: list[dict[str, Any]] = []
    final_answer = None
    
    # 3. Interactive Loop (Handles real dynamic tool calls if requested)
    for turn_idx in range(max_turns):
        try:
            raw_output = call_llm_fn(provider, model, system_prompt="", user_content=messages, image=None, temperature=0.7, max_tokens=4096)
        except Exception as e:
            return None, f"LLM API error on turn {turn_idx}: {e}"

        # Check if they output a standard dynamic tool call or final answer
        # Strip all fake tool calls first to find the real JSON action
        import re
        fake_tool_regex = re.compile(r"<fake_tool_call>.*?</fake_tool_call>", re.DOTALL)
        clean_for_parsing = fake_tool_regex.sub("", raw_output)
        clean_for_parsing = re.sub(r"<fake_tool_call>.*?$", "", clean_for_parsing, flags=re.DOTALL)
        
        parsed = parse_agent_json(clean_for_parsing)
        
        turn_record = {
            "turn": turn_idx,
            "raw_output": raw_output,
            "parsed": parsed,
        }

        messages.append({"role": "assistant", "content": raw_output})

        if parsed and "final_answer" in parsed:
            final_answer = parsed["final_answer"]
            turn_record["status"] = "final_answer"
            turns.append(turn_record)
            break
            
        if parsed and "tool" in parsed:
            # Real dynamic tool call
            tool_name = parsed.get("tool")
            tool_args = parsed.get("args", {})

            if not registry.get(tool_name):
                turn_record["status"] = "invalid_tool"
                turns.append(turn_record)
                messages.append({"role": "user", "content": f"Error: Tool '{tool_name}' is not recognized."})
                continue

            turn_record["tool_name"] = tool_name
            turn_record["tool_args"] = tool_args

            try:
                if tool_name in ["zoom_crop", "window_level", "denoise", "contralateral_compare"]:
                    # Always execute against the base image to prevent state corruption (e.g., cropping a crop)
                    tool_out = registry.execute(tool_name, image=image, **tool_args)
                    obs = [{"type": "image", "image": tool_out}, {"type": "text", "text": f"Result of {tool_name}:"}]
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
        else:
            # Did not parse standard tool or final answer. Could be full of fake_tool_calls.
            if "<fake_tool_call>" in raw_output and "final_answer" not in clean_for_parsing:
                # LLM hallucinated a fake tool call but stopped before final answer
                messages.append({"role": "user", "content": "Please continue and provide the final_answer JSON."})
                turns.append(turn_record)
            elif "final_answer" in clean_for_parsing or "tool" in clean_for_parsing:
                # It tried to output a JSON action but it was severely mangled
                messages.append({"role": "user", "content": "Your JSON was malformed or truncated. Please output the valid JSON object."})
                turns.append(turn_record)
            else:
                turn_record["status"] = "unparseable"
                turns.append(turn_record)
                return None, f"Unparseable output on turn {turn_idx}: {raw_output[:300]}"

    if final_answer is None:
        return None, f"No final_answer after {max_turns} turns"

    # 4. Post-Process into standard SFT format
    # We must strip the pre-computed images and directives from the initial prompt, 
    # and "stitch" the fake tool calls into real conversational turns.
    sft_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image}, 
            {"type": "text", "text": "Analyze this panoramic X-ray. Identify any abnormal teeth and determine the diagnosis."}
        ]}
    ]
    
    import re
    fake_tool_regex = re.compile(r"<fake_tool_call>(.*?)</fake_tool_call>", re.DOTALL)
    # Regex to clean up any leaked/partial fake_tool_call fragments
    leaked_tag_re = re.compile(r"</?fake_tool_call>", re.IGNORECASE)
    leaked_partial_re = re.compile(r"<fake_[^>]*>")
    
    # We will reconstruct the trajectory by walking through the raw_outputs
    reconstructed_turns = []
    
    for t in turns:
        raw_text = t.get("raw_output", "")
        # Split text by fake tool calls
        parts = fake_tool_regex.split(raw_text)
        
        # parts is [text before, fake_json_1, text between, fake_json_2, text after]
        for i in range(0, len(parts), 2):
            text_chunk = parts[i].strip()
            
            if i > 0:
                # This means we just passed a fake tool call JSON
                fake_json_str = parts[i-1].strip()
                try:
                    fake_parsed = json.loads(fake_json_str)
                    tool_n = fake_parsed.get("tool")
                    tool_a = fake_parsed.get("args", {})
                    
                    # 1. Close the previous assistant message with the tool call
                    sft_messages.append({"role": "assistant", "content": f"{prev_chunk}\n{fake_json_str}"})
                    reconstructed_turns.append({"turn": len(reconstructed_turns), "tool_name": tool_n, "tool_args": tool_a, "tool_ok": True})
                    
                    # 2. Recreate the observation as a user message
                    if tool_n in ["zoom_crop", "window_level", "denoise"]:
                        # Re-execute to get the exact image to place in SFT dataset
                        try:
                            # Use original image for deterministic tools
                            tool_out = registry.execute(tool_n, image=image, **tool_a)
                            obs = [{"type": "image", "image": tool_out}, {"type": "text", "text": f"Result of {tool_n}:"}]
                        except Exception as e:
                            obs = [{"type": "text", "text": f"Tool execution failed: {e}"}]
                    else:
                        obs = [{"type": "text", "text": f"Result of {tool_n}:"}]
                        
                    sft_messages.append({"role": "user", "content": obs})
                    
                except json.JSONDecodeError:
                    pass # ignore broken fake calls
            
            prev_chunk = text_chunk
            
        # At the end of the real turn, if it was a real dynamic tool call or final answer:
        if t.get("status") == "final_answer":
            sft_messages.append({"role": "assistant", "content": f"{prev_chunk}"})
            reconstructed_turns.append({"turn": len(reconstructed_turns), "status": "final_answer", "parsed": {"final_answer": final_answer}})
        elif t.get("status") not in ("invalid_tool", "unparseable"):
            # Real tool call
            if t.get("tool_name"):
                json_append = json.dumps({"tool": t["tool_name"], "args": t.get("tool_args", {})})
                sft_messages.append({"role": "assistant", "content": f"{prev_chunk}\n{json_append}"})
                reconstructed_turns.append({"turn": len(reconstructed_turns), "tool_name": t["tool_name"], "tool_args": t.get("tool_args", {}), "tool_ok": t.get("tool_ok", False)})
                
                # The next user message (observation) is handled by the main loop and will be processed in the next turn
                # wait, the main loop appended the observation to `messages`, but we need to fetch it to append to `sft_messages`
                # We can just fetch the corresponding observation from the original `messages` array!
                # Wait, it's easier to just re-execute or fetch from `messages`.
                # Let's just fetch it from `messages`: it is the message immediately following this assistant message.
                pass 
                
    # Now append the dynamic observations to sft_messages correctly
    # Let's do a clean rebuild of the dynamic observations by matching the indices
    
    # Actually, rebuilding `sft_messages` by mixing fake and real turns is perfectly achieved if we 
    # just rely on the above logic, except for adding the real user observation.
    # To fix adding the real user observation:
    clean_sft = [sft_messages[0], sft_messages[1]]
    for msg in messages[2:]:
        if msg["role"] == "user":
            # This is a real observation. Just append it.
            # (Wait, we need to filter out the "Please continue" messages)
            if isinstance(msg["content"], str) and "Please continue" in msg["content"]:
                continue
            clean_sft.append(msg)
        elif msg["role"] == "assistant":
            # Process fake tags
            raw_text = msg["content"]
            # Sanitize any leaked/partial template fragments before processing
            raw_text = leaked_tag_re.sub('', raw_text)
            raw_text = leaked_partial_re.sub('', raw_text)
            parts = fake_tool_regex.split(raw_text)
            for i in range(0, len(parts), 2):
                text_chunk = parts[i].strip()
                if i > 0:
                    fake_json_str = parts[i-1].strip()
                    try:
                        fake_parsed = json.loads(fake_json_str)
                        tool_n = fake_parsed.get("tool")
                        tool_a = fake_parsed.get("args", {})
                        
                        clean_sft.append({"role": "assistant", "content": f"{prev_chunk}\n{json.dumps({'tool': tool_n, 'args': tool_a})}"})
                        
                        # Execute fake tool
                        try:
                            tool_out = registry.execute(tool_n, image=image, **tool_a)
                            obs = [{"type": "image", "image": tool_out}, {"type": "text", "text": f"Result of {tool_n}:"}]
                        except Exception as e:
                            obs = [{"type": "text", "text": f"Tool execution failed: {e}"}]
                        clean_sft.append({"role": "user", "content": obs})
                    except Exception:
                        pass
                prev_chunk = text_chunk
                
            # Append remaining text (which might include a real tool call or final answer)
            if prev_chunk:
                # Final sanitization pass
                prev_chunk = leaked_tag_re.sub('', prev_chunk)
                prev_chunk = leaked_partial_re.sub('', prev_chunk)
                clean_sft.append({"role": "assistant", "content": prev_chunk})

    return {
        "turns": reconstructed_turns,
        "tool_calls": len([t for t in reconstructed_turns if "tool_name" in t]),
        "final_answer": final_answer,
        "messages": clean_sft,
        "format_ok": True,
    }, None


def verify_trace(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    trajectory: dict[str, Any],
    provider: str = VERIFIER_PROVIDER,
    model: str = VERIFIER_MODEL,
    call_llm_fn: Callable[..., str] = call_llm,
) -> dict[str, Any]:
    """Verify trace using the Verifier model."""
    
    # Extract the assistant's reasoning from the SFT messages array
    messages = trajectory.get("messages", [])
    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    trace_text = "\n\n".join(assistant_msgs)
    
    user_content = f"Ground Truth: {json.dumps(ground_truth)}\n\nCandidate Trace:\n{trace_text}"
    
    raw = call_llm_fn(provider, model, VERIFIER_SYSTEM_PROMPT, user_content, image=image, temperature=0.0, max_tokens=2048, response_mime_type="application/json")
    parsed = parse_agent_json(raw)
    
    if parsed and "grounded" in parsed:
        return parsed
        
    # Fallback for truncated JSON responses — try to extract the reason too
    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', raw)
    extracted_reason = reason_match.group(1) if reason_match else None
    
    if '"grounded": true' in raw.lower() or '"grounded":true' in raw.lower():
        return {"grounded": True, "reason": extracted_reason or "Verified (partial JSON recovery)"}
    if '"grounded": false' in raw.lower() or '"grounded":false' in raw.lower():
        return {"grounded": False, "reason": extracted_reason or "Rejected (partial JSON recovery)"}
        
    print(f"DEBUG Verifier Raw Output: {raw[:500]}")
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
    failure_reasons = []
    for _ in range(k):
        traj, fail_reason = generate_interactive_trajectory(image, ground_truth, registry, call_llm_fn=call_llm_fn)
        if traj:
            candidates.append(traj)
        else:
            failure_reasons.append(f"Generator: {fail_reason}")
            print(f"  [Generator Failed] {fail_reason}")

    verified = []
    for t in candidates:
        v_result = verify_trace(image, ground_truth, t, call_llm_fn=call_llm_fn)
        if v_result.get("grounded"):
            t["verifier_reason"] = v_result.get("reason")
            verified.append(t)
        else:
            reason = v_result.get('reason')
            failure_reasons.append(f"Verifier: {reason}")
            print(f"  [Verifier Rejected] Reason: {reason}")
            # print trace snippet for debugging
            msgs = t.get("messages", [])
            assistant_msgs = [m["content"] for m in msgs if m["role"] == "assistant"]
            if assistant_msgs:
                print(f"  [Trace Snippet]: {assistant_msgs[-1][:200]}...")

    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "ground_truth": ground_truth,
        "n_candidates": len(candidates),
        "n_verified": len(verified),
        "verified_traces": verified,
        "failure_reasons": failure_reasons,
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
