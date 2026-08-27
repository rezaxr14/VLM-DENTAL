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
    max_turns: int = 8,
    max_tool_calls: int = 50,
    max_tokens_per_turn: int | None = None,
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
    
    # Scale max_turns based on number of findings
    dynamic_max_turns = max(max_turns, len(ground_truth) + 3)
    
    return run_trace_gen(
        image=image,
        ground_truth=ground_truth,
        registry=registry,
        system_prompt=system_prompt,
        provider=gen_provider,
        model=gen_model,
        max_turns=dynamic_max_turns,
        max_tool_calls=max_tool_calls,
        max_tokens_per_turn=max_tokens_per_turn,
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
    min_turns: int = 15,
    turns_per_finding_buffer: int = 5,
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
    turns_budget = min(max_turns, max(min_turns, n_findings + turns_per_finding_buffer))
    print(f"  (findings={n_findings}, turn budget={turns_budget})", flush=True)

    traj, fail_reason = generate_interactive_trajectory(
        image, ground_truth, registry, max_turns=turns_budget, max_tool_calls=max_tool_calls, max_tokens_per_turn=max_tokens_per_turn
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

def verify_pending(
    unverified_path: str | Path,
    verified_path: str | Path,
    images_df: pd.DataFrame | None = None,
    call_llm_fn: Callable[..., str] = call_llm,
) -> dict[str, int]:
    """Read unverified traces, verify each concurrently, and append passing traces to the verified file."""
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
                        # .get("dataset", "dentex"): every trace record before this
                        # field existed was DENTEX-only, so that's the correct
                        # default for old files, not just an arbitrary fallback.
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
                if (record_dataset, img_id) not in verified_ids and status == "unverified":
                    pending.append(record)
            except Exception:
                pass

    print(f"Verification: {len(pending)} pending, {len(verified_ids)} already verified")

    n_verified = 0
    n_rejected = 0
    file_lock = threading.Lock()
    
    # Section 6: Concurrent Dispatch (Serialized to protect API keys)
    max_workers = 1
    print(f"Starting ThreadPoolExecutor with {max_workers} workers to prevent rate-limit bans.")

    def process_record(record, idx):
        image_id = int(record["image_id"])
        image_path = record.get("image_path", "")
        ground_truth = record.get("ground_truth", [])
        trajectory = record.get("trajectory", {})
        dataset_name = record.get("dataset", "dentex")

        if not trajectory or not os.path.exists(str(image_path)):
            return False, image_id, "Skipped (no trajectory or image)"

        try:
            image = Image.open(image_path).convert("RGB")
            v_result = verify_trace(image, ground_truth, trajectory, call_llm_fn=call_llm_fn)
            
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
                except Exception as e:
                    n_rejected += 1
                    print(f"  [Img {record['image_id']}] ERROR ({e})", flush=True)
    except KeyboardInterrupt:
        print("\nVerification interrupted.")

    return {
        "pending": len(pending),
        "verified": n_verified,
        "rejected": n_rejected,
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
