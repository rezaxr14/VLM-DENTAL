"""
Aim 1: Synthetic Expert Diagnostic Demonstration Trace Generation & Cross-Family Verification (§15, §16).

Generation is ground-truth-directed -- the model is told which finding(s) exist
(quadrant, tooth position, diagnosis) as a directive, but NOT given ground-truth
bounding boxes directly: langgraph_loop.py's run_trace_gen deliberately drops
bbox coordinates from that initial directive ("now that locate_tooth actually
works, handing over exact coordinates just teaches copy-the-hint instead of
genuine tool-mediated localization" -- see that function's own docstring), so
the model still has to call locate_tooth (and verify/correct via nudge_crop) for
real. Separately, locate_tooth's OWN tool-call-time behavior does use ground
truth internally when it exists for the requested tooth (guaranteeing the box
is found), but what the model is actually *shown* is independently
tier-perturbed rather than handed over exact -- see AGENT_HANDOVER.md's tool
section and TRACE_GEN_CONFIG.md for that separate mechanism's full numbers.

This runs through a real LangGraph agent loop
(dental_agent/agent/langgraph_loop.py) against a frontier LLM -- in practice
primarily Gemini 3.5 Flash Lite or an NVIDIA NIM-hosted model, routed through
api_pool.py's provider pool; a self-hosted Qwen/Qwen3.5-9B via local vLLM is
also a supported provider but not the primary one in practice. Every tool call
the model makes executes for real against the source image — this replaces
the earlier scheme where tool outputs were pre-computed and the model was told
to narrate a fake tool call (`<fake_tool_call>`) around them.

generate_no_tools_trajectory (below) is a separate, single-turn, tool-free
sibling of generate_interactive_trajectory -- for training baseline #3 in the
proposal's evaluation plan (dentex-agentic-vlm-proposal.md §6), not a variant
of the main generation path above. See its own docstring for why ground-truth
conditioning has to work differently there (no grounding tool to narrow a
search region through).
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

from dental_agent.agent.langgraph_loop import run_trace_gen
from dental_agent.agent.parsing import parse_agent_json
from dental_agent.agent.prompts import build_agent_system_prompt, NO_TOOLS_COT_TEACHER_PROMPT
from dental_agent.data.dentex import dentex_row_to_fdi
from dental_agent.tools.registry import ToolRegistry
from dental_agent.training.api_pool import (
    call_llm,
    verify_local_server_health,
)
from dental_agent.utils.serialization import to_jsonable


def _is_valid_key(val: str | None) -> bool:
    if not val:
        return False
    v = val.strip().lower()
    return bool(v and not v.startswith("your_") and not v.startswith("placeholder") and v != "none")


# ---------------------------------------------------------------------------
# Generator: locally-hosted Qwen/Qwen3.5-9B via vLLM (see api_pool.py's
# "local" OpenAI-compatible provider). Override via GENERATOR_PROVIDER/
# GENERATOR_MODEL in .env if you're pointing at a different setup.
# When GENERATOR_PROVIDER is an external API (not 'local'), the GeneratorPool
# handles rate limiting via 'auto_generator' routing.
# ---------------------------------------------------------------------------
GENERATOR_PROVIDER = os.environ.get("GENERATOR_PROVIDER", "local")
GENERATOR_MODEL = os.environ.get("GENERATOR_MODEL", "QuantTrio/Qwen3.5-9B-AWQ")


def _resolve_generator() -> tuple[str, str]:
    """Pick (provider, model) for the generator.
    
    Returns the explicit provider (e.g., 'local', 'groq') or 'auto_generator' if pooling is desired.
    """
    return GENERATOR_PROVIDER, GENERATOR_MODEL


# ---------------------------------------------------------------------------
# Verifier: The trace_generation handles dispatching to external APIs
# (NVIDIA, Groq, OpenRouter, Gemini) and enforces strict 5-minute cooldowns.
# ---------------------------------------------------------------------------

VERIFIER_PROVIDER = os.environ.get("VERIFIER_PROVIDER", "local")
VERIFIER_MODEL = os.environ.get("VERIFIER_MODEL", "QuantTrio/Qwen3.5-9B-AWQ")

def _resolve_verifier() -> tuple[str, str]:
    """Pick (provider, model) for the verifier."""
    return VERIFIER_PROVIDER, VERIFIER_MODEL


VERIFIER_SYSTEM_PROMPT = (
    "You are a strict, expert dental radiologist verifier. Given a panoramic dental radiograph, "
    "the KNOWN ground-truth findings, and a candidate diagnostic reasoning trace:\n"
    "1. Judge whether the reasoning, tooth localization, and diagnostic claims are clinically accurate and supported by the visual evidence.\n"
    "2. NUDGE AWARENESS: Note that `nudge_crop` is an intentional self-correction tool used to refine bounding box alignment. Using nudge_crop or recovering from a slightly offset initial crop is valid expert radiologist behavior and must NOT be rejected.\n"
    "3. MULTI-LINE REASONING: Detailed multi-line clinical thought processes (analyzing radiolucency, enamel margins, alveolar bone, etc.) are valid and encouraged.\n"
    "4. MULTI-BLOB / FORMATTING REJECTION: Reject traces that contain unparsed XML artifacts (e.g. `<fake_tool_call>`), fabricate multiple action outputs in a single turn without execution, or contradict the ground-truth pathology.\n"
    'Respond with EXACTLY ONE JSON object: {"grounded": true|false, "reason": "<concise explanation>"}.'
)

VERIFIER_REPAIR_SYSTEM_PROMPT = (
    "You are an expert dental radiology editor and clinical teacher. Your task is to repair and clean a candidate diagnostic trace that failed verification.\n\n"
    "GUIDELINES FOR REPAIR:\n"
    "1. Ground Truth Alignment: Ensure the final diagnosis and FDI tooth notation (Quadrant 1-4, Position 1-8) strictly match the verified Ground Truth findings.\n"
    "2. Preserve Chain of Thought: Retain and refine multi-line clinical reasoning (evaluating radiolucency, crown margins, bone level, pulp depth). Do NOT strip clinical reasoning into a dry single line.\n"
    "3. Clean Multi-Blob & XML Artifacts: Remove any historical pseudo-tool calls, fabricated observations in the same turn, or unparsed XML tags (`<fake_tool_call>`, `<tool_call>`). Ensure clean, valid JSON formatted turns.\n"
    "4. Nudge & Tool Consistency: Maintain valid tool sequence flow (e.g. locate_tooth -> zoom_crop -> [nudge_crop] -> final_answer).\n\n"
    "Respond with the complete, repaired trace JSON object: {\"thought\": \"<repaired clinical reasoning>\", \"final_answer\": [{\"quadrant\": ..., \"tooth_position\": ..., \"diagnosis\": ..., \"confidence\": ...}]}"
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
        fdi_quadrant, fdi_position = dentex_row_to_fdi(ann)

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
    max_turns: int = 25,
    min_turns: int = 5,
    max_tool_calls: int = 50,
    max_tokens_per_turn: int | None = None,
    context_trim_threshold: int | None = None,
    perturb_small_probability: float = 0.25,
    perturb_big_probability: float = 0.30,
    perturb_small_range: tuple[float, float] = (0.12, 0.28),
    perturb_big_range: tuple[float, float] = (0.45, 0.75),
    max_blobs_per_turn: int = 2,
    max_padding_turns: int = 3,
    max_identical_repeats: int = 3,
    provider: str = GENERATOR_PROVIDER,
    model: str = GENERATOR_MODEL,
    call_llm_fn: Callable[..., str] | None = None,
) -> tuple[dict[str, Any] | None, str | None]:
    """Ground-truth-directed trace generation with REAL tool execution via LangGraph.

    `call_llm_fn` is accepted for backward-compatible call signatures but unused —
    the LangGraph loop (dental_agent/agent/langgraph_loop.py) calls api_pool.call_llm
    internally for every model turn. If you need to inject a mock/test LLM for offline
    testing, patch dental_agent.training.api_pool.call_llm instead of passing call_llm_fn
    here (tests/ should do this via monkeypatch, not this parameter).
    """
    gen_provider, gen_model = _resolve_generator()
    system_prompt = build_agent_system_prompt(registry.format_tool_descriptions())
    
    return run_trace_gen(
        image=image,
        ground_truth=ground_truth,
        registry=registry,
        system_prompt=system_prompt,
        provider=gen_provider,
        model=gen_model,
        max_turns=max_turns,
        min_turns=min_turns,
        max_tool_calls=max_tool_calls,
        max_tokens_per_turn=max_tokens_per_turn,
        context_trim_threshold=context_trim_threshold,
        perturb_small_probability=perturb_small_probability,
        perturb_big_probability=perturb_big_probability,
        perturb_small_range=perturb_small_range,
        perturb_big_range=perturb_big_range,
        max_blobs_per_turn=max_blobs_per_turn,
        max_padding_turns=max_padding_turns,
        max_identical_repeats=max_identical_repeats,
    )



def generate_no_tools_trajectory(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    provider: str = GENERATOR_PROVIDER,
    model: str = GENERATOR_MODEL,
    max_tokens: int = 2048,
) -> tuple[dict[str, Any] | None, str | None]:
    """Ground-truth-directed, SINGLE-TURN, TOOL-FREE trace generation -- for
    training baseline #3 in the proposal's evaluation plan (dentex-agentic-vlm-
    proposal.md §6: "Full agent without tool access (RL-tuned but reasoning over
    the whole image only) — isolates the contribution of tools"). This produces
    SFT training data for that baseline's Stage 1, structurally matching
    generate_interactive_trajectory's output shape (same "messages"/"final_answer"/
    "tool_calls"/"format_ok" keys) but via one direct API call instead of a
    LangGraph tool-execution loop, since there is no tool orchestration to do.

    Do NOT confuse this with:
    - dental_agent.agent.prompts.NO_TOOLS_SYSTEM_PROMPT / dental_agent.agent.loop.py's
      GRPO rollout usage of it -- that's the no-tools policy's live RL rollout
      (Stage 2), already built, unrelated to generating SFT training data (Stage 1).
    - dental_agent.agent.prompts.ZERO_SHOT_PROMPT / evaluation/baselines.py -- that's
      baseline #1, a raw untrained model prompted at eval time, no training involved.

    Ground-truth conditioning here works differently from the tool-based path:
    generate_interactive_trajectory narrows locate_tooth's search region toward
    the true location (a real detector still has to find and the model still has
    to verify it via zoom_crop) while never revealing the bbox itself. There is no
    equivalent narrowing mechanism without a grounding tool -- so this function
    tells the model directly which finding(s) to cover (matching the same
    "TEACHER DIRECTIVE" principle already used in langgraph_loop.py's run_trace_gen,
    just applied here with no tool-verification step to preserve). This means the
    reasoning it produces is closer to hindsight rationalization ("write the
    reasoning that would justify this known answer") than genuine blind discovery
    -- an intentional, documented tradeoff (the alternative, blind single-pass
    generation with no conditioning, would tank yield the same way blind tool-based
    generation would, per generate_interactive_trajectory's own docstring/proposal
    §5.2) -- and the SAME cross-family verify_trace this module already uses for
    tool-based traces is reused unmodified to check the reasoning stays visually
    plausible and doesn't leak that it was told the answer, rather than trusting
    the generator's self-restraint alone.
    """
    gen_provider, gen_model = (provider, model) if provider and model else _resolve_generator()
    hint_text = "; ".join(
        f"Q{f['quadrant']}T{f['tooth_position']}:{f['diagnosis']}" for f in ground_truth
    )
    directive = (
        "TEACHER DIRECTIVE: You are generating an expert demonstration trace for SFT.\n"
        f"This image has {len(ground_truth)} finding(s): {hint_text}\n\n"
        "Write the clinical reasoning a radiologist would give for noticing these on "
        "direct visual inspection, then give your final answer covering all of them."
    )

    try:
        raw = call_llm(
            gen_provider, gen_model, NO_TOOLS_COT_TEACHER_PROMPT, directive,
            image=image, temperature=0.3, max_tokens=max_tokens,
            response_mime_type="application/json", label="generate_no_tools_trajectory",
            role="generator",
        )
    except Exception as e:
        return None, f"generator call failed: {e}"

    parsed = parse_agent_json(raw)
    messages = [
        {"role": "system", "content": NO_TOOLS_COT_TEACHER_PROMPT},
        {"role": "user", "content": directive},
        {"role": "assistant", "content": raw},
    ]
    # Single-item list, matching langgraph_loop.py's per-turn record schema
    # exactly ({"turn": int, "raw_output": str, "parsed": dict|None}) -- NOT
    # an int turn-count as this previously (bug) had it. Getting this wrong
    # silently breaks anything that reads trajectory["turns"] expecting a
    # list of turn dicts -- e.g. dental_agent/rewards/judge.py's
    # reward_judge, which iterates turns for "raw_output" and would raise
    # TypeError: 'int' object is not iterable the moment it's pointed at a
    # no-tools trace generated before this fix.
    turns = [{"turn": 0, "raw_output": raw, "parsed": parsed}]

    if not parsed or not parsed.get("final_answer"):
        return {
            "turns": turns,
            "tool_calls": [],
            "final_answer": None,
            "messages": messages,
            "format_ok": False,
        }, "no parseable final_answer in single-turn response"

    return {
        "turns": turns,
        "tool_calls": [],
        "final_answer": parsed["final_answer"],
        "messages": messages,
        "format_ok": True,
    }, None


def verify_trace(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    trajectory: dict[str, Any],
    provider: str | None = None,
    model: str | None = None,
    call_llm_fn: Callable[..., str] = call_llm,
    max_repairs: int = 1,
    current_repair_attempt: int = 0,
) -> dict[str, Any]:
    """Verify trace using an independent verifier model. Includes LLM-based repair on rejection."""
    if provider is None or model is None:
        provider, model = _resolve_verifier()

    # Extract the assistant's reasoning
    messages = trajectory.get("messages", [])
    assistant_msgs = [m["content"] for m in messages if m["role"] == "assistant"]
    trace_text = "\n\n".join(assistant_msgs)

    user_content = f"Ground Truth: {json.dumps(ground_truth)}\n\nCandidate Trace:\n{trace_text}"

    # Section 8: stream=True
    raw = call_llm_fn(provider, model, VERIFIER_SYSTEM_PROMPT, user_content, image=image, temperature=0.0, max_tokens=2048, response_mime_type="application/json", stream=True, label="verify_trace", role="verifier")
    parsed = parse_agent_json(raw)
    
    extracted_reason = None
    reason_match = re.search(r'"reason"\s*:\s*"([^"]+)"', raw)
    if reason_match:
        extracted_reason = reason_match.group(1)

    grounded = False
    if parsed and "grounded" in parsed:
        grounded = parsed["grounded"]
        extracted_reason = parsed.get("reason", extracted_reason)
    elif '"grounded": true' in raw.lower() or '"grounded":true' in raw.lower():
        grounded = True
        extracted_reason = extracted_reason or "Verified (partial JSON recovery)"
    elif '"grounded": false' in raw.lower() or '"grounded":false' in raw.lower():
        grounded = False
        extracted_reason = extracted_reason or "Rejected (partial JSON recovery)"
    else:
        print(f"DEBUG Verifier Raw Output: {raw[:500]}")
        extracted_reason = "verifier output unparseable"
        grounded = False

    result = {"grounded": grounded, "reason": extracted_reason}

    # Section 7: LLM-Based Repair
    if not grounded and current_repair_attempt < max_repairs:
        print(f"  [verify_trace] Trace rejected: {extracted_reason}. Attempting repair {current_repair_attempt + 1}/{max_repairs}...")
        gen_provider, gen_model = _resolve_generator()
        repair_sys_prompt = "You are a medical AI assistant. Fix the provided diagnostic trace based on the verifier's feedback. Ensure the final diagnosis remains unchanged, but correct any visual claims that were rejected."
        repair_user_content = f"The verifier rejected this trace because: {extracted_reason}\n\nOriginal Trace:\n{trace_text}\n\nRewrite the trace to fix the issue. Output the complete revised reasoning."
        
        try:
            repaired_raw = call_llm_fn(gen_provider, gen_model, repair_sys_prompt, repair_user_content, image=image, temperature=0.3, max_tokens=4096, stream=True, label="repair_trace", role="verifier")
            # Replace the last assistant message with the repaired raw
            repaired_trajectory = dict(trajectory)
            repaired_messages = list(trajectory.get("messages", []))
            for i in range(len(repaired_messages)-1, -1, -1):
                if repaired_messages[i]["role"] == "assistant":
                    repaired_messages[i]["content"] = repaired_raw
                    break
            repaired_trajectory["messages"] = repaired_messages
            
            # Re-verify the repaired trace
            return verify_trace(
                image, ground_truth, repaired_trajectory, provider, model, call_llm_fn, max_repairs, current_repair_attempt + 1
            )
        except Exception as e:
            print(f"  [verify_trace] Repair attempt failed: {e}")
            return result

    return result


def build_trace_example(
    image_id: int,
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame | None = None,
    k: int = 1,  # Default to 1 — generation is local/free now, but keep this cheap by default
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
        traj, fail_reason = generate_interactive_trajectory(image, ground_truth, registry)
        if traj and traj.get("final_answer") is not None:
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


# ---------------------------------------------------------------------------
# Decoupled pipeline: generate_only / verify_pending
# ---------------------------------------------------------------------------

def generate_only(
    image_id: int,
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame | None = None,
    diag_col: str = "category_id_3",
    max_turns: int = 25,
    max_tool_calls: int = 50,
    max_tokens_per_turn: int | None = None,
    min_turns: int = 5,
    turns_per_finding_buffer: int = 5,
    context_trim_threshold: int | None = None,
    perturb_small_probability: float = 0.25,
    perturb_big_probability: float = 0.30,
    perturb_small_range: tuple[float, float] = (0.12, 0.28),
    perturb_big_range: tuple[float, float] = (0.45, 0.75),
    max_blobs_per_turn: int = 2,
    max_padding_turns: int = 3,
    max_identical_repeats: int = 3,
) -> dict[str, Any] | None:
    """Generate a raw (unverified) trace for a single image.
    
    Does NOT call the verifier. Returns the raw trajectory dict to be
    written to ``train_cot_traces_unverified.jsonl``.
    """
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

    ground_truth = _format_ground_truth(anns, cat_lookup, diag_col)
    registry = ToolRegistry.create_default()

    # Dynamic per-image turn budget: max(min_turns, n_findings + buffer), capped at
    # max_turns. A flat budget can't serve both a 1-finding image and a 14-finding
    # image well -- this scales the room a trace gets to match how much investigation
    # it actually needs, rather than either starving complex cases or teaching
    # wandering into simple ones via an inflated global default.
    n_findings = len(ground_truth)
    print(f"  (findings={n_findings}, max_turns={max_turns})", flush=True)

    traj, fail_reason = generate_interactive_trajectory(
        image=image,
        ground_truth=ground_truth,
        registry=registry,
        max_turns=max_turns,
        min_turns=min_turns,
        max_tool_calls=max_tool_calls,
        max_tokens_per_turn=max_tokens_per_turn,
        context_trim_threshold=context_trim_threshold,
        perturb_small_probability=perturb_small_probability,
        perturb_big_probability=perturb_big_probability,
        perturb_small_range=perturb_small_range,
        perturb_big_range=perturb_big_range,
        max_blobs_per_turn=max_blobs_per_turn,
        max_padding_turns=max_padding_turns,
        max_identical_repeats=max_identical_repeats,
    )
    if traj is None or traj.get("final_answer") is None:
        return {
            "image_id": image_id,
            "image_path": str(image_path),
            "ground_truth": ground_truth,
            "status": "generation_failed",
            "failure_reason": fail_reason,
            # Preserve whatever the model actually did before failing (successful
            # tool calls, partial reasoning turns) rather than losing it outright.
            "partial_trajectory": to_jsonable(traj) if traj else None,
        }

    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "ground_truth": ground_truth,
        "status": "unverified",
        "trajectory": to_jsonable(traj),
    }


def generate_only_no_tools(
    image_id: int,
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame | None = None,
    diag_col: str = "category_id_3",
    max_tokens: int = 2048,
) -> dict[str, Any] | None:
    """Tool-free sibling of generate_only -- for baseline #3's SFT training
    data (dentex-agentic-vlm-proposal.md §6). Mirrors generate_only's exact
    return shape (image_id/image_path/ground_truth/status/trajectory or
    failure_reason) so it slots into the same append_trace/file-writing and
    downstream verify_pending/SFT-loading code paths unchanged -- only the
    generation step itself differs (one direct call via
    generate_no_tools_trajectory, no LangGraph loop, no ToolRegistry).

    Does NOT call the verifier -- same decoupled generate/verify split as
    generate_only, and the same verify_pending function verifies these
    traces too (see run_trace_gen.py's --no-tools flag, which points both
    generate and verify at their own separate _no_tools-suffixed files
    rather than mixing them into the main system's tool-based traces).
    """
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

    ground_truth = _format_ground_truth(anns, cat_lookup, diag_col)

    gen_provider, gen_model = _resolve_generator()
    traj, fail_reason = generate_no_tools_trajectory(
        image, ground_truth, provider=gen_provider, model=gen_model, max_tokens=max_tokens
    )
    if traj is None or traj.get("final_answer") is None:
        return {
            "image_id": image_id,
            "image_path": str(image_path),
            "ground_truth": ground_truth,
            "status": "generation_failed",
            "failure_reason": fail_reason,
            "partial_trajectory": to_jsonable(traj) if traj else None,
        }

    return {
        "image_id": image_id,
        "image_path": str(image_path),
        "ground_truth": ground_truth,
        "status": "unverified",
        "trajectory": to_jsonable(traj),
    }


import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def resolve_trace_image_path(
    image_path: str | Path | None,
    image_id: int,
    dataset_name: str = "dentex",
    images_df: pd.DataFrame | None = None,
    data_dir: str | Path | None = None,
) -> Path | None:
    """Resolve an image path across heterogeneous runtime environments (Kaggle -> Colab -> Local).

    If the raw image_path stored in a trace JSONL was recorded on a different filesystem
    (e.g., /kaggle/working/... when running in Colab), this helper tries:
      1. Checking if the path exists directly as-is on the current filesystem.
      2. Checking if images_df contains a valid local_path for this image_id.
      3. Searching standard local dataset folders (data/, data/dentex/, etc.) for {id}.png/jpg.
      4. Searching HF hub cache snapshots on disk.
      5. Dynamically downloading the slice image via DENTEX_IMAGES_REPO / TUFTS_IMAGES_REPO.
    """
    if image_path:
        p = Path(image_path)
        if p.exists() and p.is_file():
            return p

    # 1. Lookup in images_df if provided
    if images_df is not None and "id" in images_df.columns and "local_path" in images_df.columns:
        match = images_df[images_df["id"] == image_id]
        if not match.empty:
            df_path = match.iloc[0].get("local_path")
            if df_path and os.path.exists(str(df_path)):
                return Path(df_path)

    # 2. Search local candidate directories
    ds_norm = dataset_name.lower()
    fname = Path(image_path).name if image_path else f"{image_id}.png"
    if ds_norm == "dentex":
        candidate_names = [fname, f"{image_id}.png", f"val_{image_id}.png", f"train_{image_id}.png"]
        search_roots: list[Path] = []
        if data_dir:
            search_roots.append(Path(data_dir))
        search_roots.extend([
            Path("data/dentex"),
            Path("data/dentex/DENTEX"),
            Path("data/training_data"),
            Path("data/validation_data"),
            Path("data/dentex/training_data"),
            Path("data/dentex/validation_data"),
        ])
    elif ds_norm == "tufts":
        candidate_names = [fname, f"{image_id}.jpg", f"{image_id}.JPG", f"{image_id}.png"]
        search_roots = []
        if data_dir:
            search_roots.append(Path(data_dir))
        search_roots.extend([
            Path("data/Tufts"),
            Path("data/tufts"),
            Path("data/Tufts/Radiographs"),
            Path("data/tufts/Radiographs"),
        ])
    else:
        candidate_names = [fname, f"{image_id}.png", f"{image_id}.jpg"]
        search_roots = [Path(data_dir)] if data_dir else [Path("data")]

    for root in search_roots:
        if not root.exists():
            continue
        for name in candidate_names:
            candidate = root / name
            if candidate.exists() and candidate.is_file():
                return candidate
            candidate_nested = root / "images" / name
            if candidate_nested.exists() and candidate_nested.is_file():
                return candidate_nested

    # 3. Glob match in local dataset folders
    for root in search_roots:
        if root.exists():
            for name in candidate_names:
                for match in root.glob(f"**/{name}"):
                    if match.is_file():
                        return match

    # 4. Check HuggingFace hub cache
    repo_id = os.environ.get("DENTEX_IMAGES_REPO" if ds_norm == "dentex" else "TUFTS_IMAGES_REPO")
    hf_cache_root = Path.home() / ".cache" / "huggingface" / "hub"
    if repo_id and hf_cache_root.exists():
        hf_repo_dir = hf_cache_root / f"datasets--{repo_id.replace('/', '--')}"
        if hf_repo_dir.exists():
            for name in candidate_names:
                for match in hf_repo_dir.glob(f"**/{name}"):
                    if match.is_file():
                        return match

    # 5. Dynamic slice download if repo is configured
    if ds_norm == "dentex":
        repo_id = os.environ.get("DENTEX_IMAGES_REPO")
        if repo_id:
            try:
                from dental_agent.data.dentex import download_dentex_slice
                paths_map = download_dentex_slice([image_id], repo_id=repo_id, cache_dir=data_dir)
                if image_id in paths_map and paths_map[image_id] is not None:
                    p_down = Path(paths_map[image_id])
                    if p_down.exists():
                        return p_down
            except Exception as e:
                print(f"Warning: Failed to fetch image {image_id} from {repo_id}: {e}")
    elif ds_norm == "tufts":
        repo_id = os.environ.get("TUFTS_IMAGES_REPO")
        if repo_id:
            try:
                from dental_agent.data.tufts import download_tufts_slice
                paths_map = download_tufts_slice([image_id], repo_id=repo_id, cache_dir=data_dir)
                if image_id in paths_map and paths_map[image_id] is not None:
                    p_down = Path(paths_map[image_id])
                    if p_down.exists():
                        return p_down
            except Exception as e:
                print(f"Warning: Failed to fetch image {image_id} from {repo_id}: {e}")

    return None


def verify_pending(
    unverified_path: str | Path,
    verified_path: str | Path,
    images_df: pd.DataFrame | None = None,
    data_dir: str | Path | None = None,
    call_llm_fn: Callable[..., str] = call_llm,
    max_repairs: int = 1,
    total_slices: int = 1,
    slice_index: int = 1,
    slice_seed: int = 42,
    pacing_delay: float = 1.5,
    max_images: int | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> dict[str, int]:
    """Read unverified traces, verify each, and append passing traces to the verified file."""
    unverified_path = Path(unverified_path)
    verified_path = Path(verified_path)

    if not unverified_path.exists():
        print(f"No unverified trace file found at {unverified_path}")
        return {"pending": 0, "verified": 0, "rejected": 0}

    verified_ids: set[tuple[str, int]] = set()
    if verified_path.exists():
        with open(verified_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "image_id" in record:
                        verified_ids.add((record.get("dataset", "dentex"), int(record["image_id"])))
                except Exception:
                    pass

    pending = []
    with open(unverified_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                img_id = int(record.get("image_id", -1))
                record_dataset = record.get("dataset", "dentex")
                status = record.get("status", "")
                if (record_dataset, img_id) not in verified_ids and status != "generation_failed":
                    pending.append(record)
            except Exception:
                pass

    if total_slices > 1 and len(pending) > 0:
        all_ids = sorted(list({int(r.get("image_id", -1)) for r in pending if int(r.get("image_id", -1)) >= 0}))
        from dental_agent.data.slicing import compute_slice_assignment
        assigned_slice = compute_slice_assignment(all_ids, total_slices, slice_seed)
        slice_target_ids = {img_id for img_id, sl in assigned_slice.items() if sl == (slice_index - 1)}
        pending = [r for r in pending if int(r.get("image_id", -1)) in slice_target_ids]
        print(f"  [Slice {slice_index}/{total_slices}] Filtered to {len(pending)} pending traces for this worker.")

    if max_images is not None and max_images > 0:
        pending = pending[:max_images]
        print(f"  [Max Images] Limited verification session to {len(pending)} traces.")

    print(f"Verification: {len(pending)} pending, {len(verified_ids)} already verified")

    n_verified = 0
    n_rejected = 0
    file_lock = threading.Lock()
    max_workers = 1

    def process_record(record, idx):
        if pacing_delay > 0 and idx > 1:
            time.sleep(pacing_delay)

        image_id = int(record["image_id"])
        image_path = record.get("image_path", "")
        ground_truth = record.get("ground_truth", [])
        trajectory = record.get("trajectory", {})
        dataset_name = record.get("dataset", "dentex")

        resolved_path = resolve_trace_image_path(
            image_path=image_path,
            image_id=image_id,
            dataset_name=dataset_name,
            images_df=images_df,
            data_dir=data_dir,
        )

        if not trajectory or resolved_path is None or not resolved_path.exists():
            return False, image_id, f"Skipped (no trajectory or missing image: {image_path})"

        image_path = str(resolved_path)

        try:
            image = Image.open(image_path).convert("RGB")
            v_result = verify_trace(
                image,
                ground_truth,
                trajectory,
                provider=provider,
                model=model,
                call_llm_fn=call_llm_fn,
                max_repairs=max_repairs,
            )
            
            if v_result.get("grounded"):
                trajectory["verifier_reason"] = v_result.get("reason")
                trajectory["image_id"] = image_id
                trajectory["image_path"] = image_path
                trajectory["ground_truth"] = ground_truth
                trajectory["dataset"] = dataset_name
                
                with file_lock:
                    verified_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(verified_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(to_jsonable(trajectory)) + "\n")
                
                return True, image_id, v_result.get("reason", "")
            else:
                return False, image_id, v_result.get("reason", "")
                
        except Exception as e:
            from dental_agent.training.api_pool import RPDLimitExhausted
            if isinstance(e, RPDLimitExhausted):
                raise
            return False, image_id, f"ERROR ({e})"

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_record, r, i): (r, i) for i, r in enumerate(pending, start=1)}
            
            for future in as_completed(futures):
                record, idx = futures[future]
                try:
                    passed, img_id, reason = future.result()
                    if passed:
                        n_verified += 1
                        print(f"  [Img {img_id}] PASSED ({reason[:60]})", flush=True)
                    else:
                        n_rejected += 1
                        print(f"  [Img {img_id}] REJECTED ({reason[:60]})", flush=True)
                except Exception as e:
                    from dental_agent.training.api_pool import RPDLimitExhausted
                    if isinstance(e, RPDLimitExhausted):
                        print(f"\n[DAILY LIMIT REACHED] {e}")
                        print("Verifier API usage limit reached. Progress saved; resume later.")
                    else:
                        print(f"\n[ERROR] {e}")
                    for f in futures:
                        f.cancel()
                    break
    except KeyboardInterrupt:
        print("\nVerification interrupted.")

    return {
        "pending": len(pending),
        "verified": n_verified,
        "rejected": n_rejected,
    }


def repair_and_clean_trace(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    record: dict[str, Any],
    verifier_provider: str | None = None,
    verifier_model: str | None = None,
    call_llm_fn: Callable[..., str] = call_llm,
) -> tuple[bool, dict[str, Any] | None, str]:
    """Take an unverified/rejected trace, prompt the verifier model with VERIFIER_REPAIR_SYSTEM_PROMPT
    along with ground truth and the original trace, clean out artifacts, and verify the repaired trace."""
    v_prov, v_mod = (verifier_provider, verifier_model) if verifier_provider and verifier_model else _resolve_verifier()
    
    trajectory = record.get("trajectory") or record.get("partial_trajectory") or {}
    messages = trajectory.get("messages", [])
    assistant_msgs = [m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "assistant"]
    trace_text = "\n\n".join(assistant_msgs) if assistant_msgs else json.dumps(trajectory)
    
    gt_text = json.dumps(ground_truth, indent=2)
    user_prompt = (
        f"GROUND TRUTH FINDINGS:\n{gt_text}\n\n"
        f"CANDIDATE TRACE TO REPAIR:\n{trace_text}\n\n"
        "Please rewrite and clean this diagnostic trace so it accurately reasons through the radiograph, "
        "matches the ground truth FDI findings, and removes any multi-blob or XML artifacts."
    )
    
    try:
        raw_repaired = call_llm_fn(
            v_prov,
            v_mod,
            VERIFIER_REPAIR_SYSTEM_PROMPT,
            user_prompt,
            image=image,
            temperature=0.2,
            max_tokens=4096,
            label="repair_and_clean_trace",
            role="verifier",
        )
    except Exception as e:
        return False, None, f"Repair LLM call failed: {e}"
        
    parsed_repair = parse_agent_json(raw_repaired)
    if not parsed_repair or not parsed_repair.get("final_answer"):
        return False, None, "Repaired trace could not be parsed into valid final_answer schema"
        
    # Build updated trajectory
    repaired_trajectory = dict(trajectory)
    repaired_trajectory["final_answer"] = parsed_repair["final_answer"]
    repaired_trajectory["repaired"] = True
    
    # Update messages
    repaired_messages = list(messages) if messages else []
    if repaired_messages and isinstance(repaired_messages[-1], dict) and repaired_messages[-1].get("role") == "assistant":
        repaired_messages[-1]["content"] = raw_repaired
    else:
        repaired_messages.append({"role": "assistant", "content": raw_repaired})
    repaired_trajectory["messages"] = repaired_messages

    # Synchronize turns so downstream SFT and RL reward functions see the repaired output
    if "turns" in repaired_trajectory and isinstance(repaired_trajectory["turns"], list) and len(repaired_trajectory["turns"]) > 0:
        repaired_turns = list(repaired_trajectory["turns"])
        if isinstance(repaired_turns[-1], dict):
            last_turn = dict(repaired_turns[-1])
            last_turn["raw_output"] = raw_repaired
            last_turn["parsed"] = parsed_repair
            repaired_turns[-1] = last_turn
            repaired_trajectory["turns"] = repaired_turns
    
    # Re-verify the repaired trajectory
    v_res = verify_trace(
        image,
        ground_truth,
        repaired_trajectory,
        provider=v_prov,
        model=v_mod,
        call_llm_fn=call_llm_fn,
        max_repairs=0,
    )
    
    if v_res.get("grounded"):
        repaired_trajectory["verifier_reason"] = v_res.get("reason", "Verified after automated repair")
        return True, repaired_trajectory, v_res.get("reason", "Repaired and verified")
    else:
        return False, None, f"Repaired trace failed verification: {v_res.get('reason')}"


def repair_pending(
    unverified_path: str | Path,
    verified_path: str | Path,
    images_df: pd.DataFrame | None = None,
    data_dir: str | Path | None = None,
    provider: str | None = None,
    model: str | None = None,
    total_slices: int = 1,
    slice_index: int = 1,
    slice_seed: int = 42,
    pacing_delay: float = 1.5,
    max_images: int | None = None,
    call_llm_fn: Callable[..., str] = call_llm,
) -> dict[str, int]:
    """Phase 2: Read all traces in unverified_path that have NOT yet been verified into verified_path,
    attempt intelligent clinical repair and re-verification, and append passing traces to verified_path."""
    unverified_path = Path(unverified_path)
    verified_path = Path(verified_path)

    if not unverified_path.exists():
        print(f"No unverified trace file found at {unverified_path}")
        return {"pending_repair": 0, "repaired_and_promoted": 0, "still_unverified": 0}

    verified_ids: set[tuple[str, int]] = set()
    if verified_path.exists():
        with open(verified_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "image_id" in record:
                        verified_ids.add((record.get("dataset", "dentex"), int(record["image_id"])))
                except Exception:
                    pass

    needs_repair = []
    with open(unverified_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                img_id = int(record.get("image_id", -1))
                rec_dataset = record.get("dataset", "dentex")
                if (rec_dataset, img_id) not in verified_ids:
                    needs_repair.append(record)
            except Exception:
                pass

    if total_slices > 1 and len(needs_repair) > 0:
        all_ids = sorted(list({int(r.get("image_id", -1)) for r in needs_repair if int(r.get("image_id", -1)) >= 0}))
        from dental_agent.data.slicing import compute_slice_assignment
        assigned_slice = compute_slice_assignment(all_ids, total_slices, slice_seed)
        slice_target_ids = {img_id for img_id, sl in assigned_slice.items() if sl == (slice_index - 1)}
        needs_repair = [r for r in needs_repair if int(r.get("image_id", -1)) in slice_target_ids]
        print(f"  [Slice {slice_index}/{total_slices}] Filtered to {len(needs_repair)} traces for repair on this worker.")

    if max_images is not None and max_images > 0:
        needs_repair = needs_repair[:max_images]
        print(f"  [Max Images] Limited repair session to {len(needs_repair)} traces.")

    print(f"\n--- Phase 2: Verifier Self-Repair & Editor Pass ---")
    print(f"Targeting {len(needs_repair)} unverified/rejected traces for intelligent clinical repair...")

    n_promoted = 0
    n_failed = 0

    for idx, record in enumerate(needs_repair, start=1):
        if pacing_delay > 0 and idx > 1:
            time.sleep(pacing_delay)

        img_id = int(record.get("image_id", -1))
        img_path = record.get("image_path", "")
        ground_truth = record.get("ground_truth", [])
        dataset_name = record.get("dataset", "dentex")

        resolved_path = resolve_trace_image_path(
            image_path=img_path,
            image_id=img_id,
            dataset_name=dataset_name,
            images_df=images_df,
            data_dir=data_dir,
        )

        if resolved_path is None or not resolved_path.exists():
            print(f"[{idx}/{len(needs_repair)}] Img {img_id}: image missing ({img_path}). Skipping.")
            n_failed += 1
            continue

        img_path = str(resolved_path)

        try:
            image = Image.open(img_path).convert("RGB")
            ok, repaired_traj, reason = repair_and_clean_trace(
                image,
                ground_truth,
                record,
                verifier_provider=provider,
                verifier_model=model,
                call_llm_fn=call_llm_fn,
            )

            if ok and repaired_traj is not None:
                repaired_traj["image_id"] = img_id
                repaired_traj["image_path"] = img_path
                repaired_traj["ground_truth"] = ground_truth
                repaired_traj["dataset"] = dataset_name
                repaired_traj["status"] = "verified"

                verified_path.parent.mkdir(parents=True, exist_ok=True)
                with open(verified_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(to_jsonable(repaired_traj)) + "\n")

                verified_ids.add((dataset_name, img_id))
                n_promoted += 1
                print(f"[{idx}/{len(needs_repair)}] Img {img_id}: REPAIRED & VERIFIED ({reason[:60]})", flush=True)
            else:
                n_failed += 1
                print(f"[{idx}/{len(needs_repair)}] Img {img_id}: REPAIR FAILED ({reason[:60]})", flush=True)

        except Exception as e:
            from dental_agent.training.api_pool import RPDLimitExhausted
            if isinstance(e, RPDLimitExhausted):
                print(f"\n[RPD LIMIT REACHED] {e}")
                break
            n_failed += 1
            print(f"[{idx}/{len(needs_repair)}] Img {img_id}: ERROR ({e})", flush=True)

    print(f"\nRepair Pass Complete: {n_promoted} promoted, {n_failed} remaining unverified.")
    return {
        "pending_repair": len(needs_repair),
        "repaired_and_promoted": n_promoted,
        "still_unverified": n_failed,
    }


def clean_unverified_traces(
    unverified_path: str | Path,
    backup: bool = True,
    purge_failed: bool = True,
) -> dict[str, int]:
    """Scan and clean train_cot_traces_unverified.jsonl to purge historical multi-blob
    tool hallucinations, XML artifacts, and generation_failed entries. Deduplicates keeping
    the latest/most recent entry per image ID."""
    unverified_path = Path(unverified_path)
    if not unverified_path.exists():
        print(f"No file found at {unverified_path}")
        return {"kept": 0, "corrupted": 0, "failed": 0}

    if backup:
        backup_path = unverified_path.with_suffix(".jsonl.bak")
        try:
            import shutil
            shutil.copy2(unverified_path, backup_path)
            print(f"  [Backup] Created backup at {backup_path.name}")
        except Exception as e:
            print(f"  [WARN] Failed to create backup: {e}")

    kept_map: dict[tuple[str, int], dict[str, Any]] = {}
    n_corrupted = 0
    n_failed = 0

    with open(unverified_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                n_corrupted += 1
                continue

            img_id = rec.get("image_id")
            rec_dataset = rec.get("dataset", "dentex")
            status = rec.get("status", "")

            # 1. Purge failed generations to prevent duplicate ID collisions on retry
            if status == "generation_failed" and purge_failed:
                n_failed += 1
                continue

            # 2. Check for XML artifacts or pseudo-tool hallucinations in turns
            traj = rec.get("trajectory", {})
            turns = traj.get("turns", []) if isinstance(traj, dict) else []
            messages = traj.get("messages", []) if isinstance(traj, dict) else []

            raw_texts = []
            for t in turns:
                if isinstance(t, dict):
                    raw_texts.append(t.get("raw_output", ""))
            for m in messages:
                if isinstance(m, dict) and m.get("role") == "assistant":
                    raw_texts.append(m.get("content", ""))

            combined_text = "\n".join(raw_texts)
            has_corrupt_xml = "<fake_tool_call>" in combined_text or "<tool_call>" in combined_text
            has_multi_blob = combined_text.count('"action":') > 3 and combined_text.count('"turn":') <= 1

            if has_corrupt_xml or has_multi_blob:
                n_corrupted += 1
                continue

            key = (rec_dataset, img_id)
            # Latest entry replaces earlier entry in the file
            kept_map[key] = rec

    kept_records = list(kept_map.values())

    # Write cleaned records back
    with open(unverified_path, "w", encoding="utf-8") as f:
        for r in kept_records:
            f.write(json.dumps(to_jsonable(r)) + "\n")

    print(f"Trace Cleaner: {len(kept_records)} valid traces kept, {n_corrupted} corrupted purged, {n_failed} failed purged.")
    return {
        "kept": len(kept_records),
        "corrupted": n_corrupted,
        "failed": n_failed,
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
                if GENERATOR_PROVIDER == "local":
                    health_retries = 0
                    while not verify_local_server_health(timeout=5.0):
                        health_retries += 1
                        if health_retries > 24:
                            raise RuntimeError("Local vLLM server is unresponsive for > 2 minutes. Aborting batch.")
                        print(f"Local vLLM server unresponsive. Waiting 5s... ({health_retries}/24)")
                        time.sleep(5)
                                
                result = build_trace_example(
                    image_id=image_id,
                    images_df=images_df,
                    annots_df=annots_df,
                    categories_df=categories_df,
                    k=k,
                    diag_col=diag_col,
                )
            except RuntimeError as e:
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
