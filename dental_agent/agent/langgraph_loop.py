"""
LangGraph-based multi-turn agent loop for local, ground-truth-directed trace generation.

Replaces the earlier pre-computed + <fake_tool_call> scheme in trace_generation.py
(see AGENT_HANDOVER.md §3 for the history). Generation still conditions on the known
ground-truth label and seeds tool coordinates from the ground-truth bounding box —
that's kept deliberately, since blind exploration on an untuned model would tank the
yield of usable, correctly-labeled training data. What changed is that every tool call
the model makes now executes for real against the source image; the model receives the
actual resulting crop/window/contrast output at each turn, not a pre-computed stand-in
paired with a narrated tool-call tag it never actually triggered.
"""

from __future__ import annotations

import json
import random
from typing import Any, TypedDict
from PIL import Image

try:
    from langgraph.graph import StateGraph, END
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "langgraph is required for dental_agent.agent.langgraph_loop. "
        "Add `langgraph` to requirements.txt / pyproject.toml and `pip install langgraph`."
    ) from _e

from dental_agent.agent.parsing import parse_agent_json
from dental_agent.agent.tool_dispatch import execute_tool_call
from dental_agent.tools.registry import ToolRegistry
from dental_agent.tools.nudge import tool_nudge_crop
from dental_agent.training.api_pool import call_llm
from dental_agent.utils.serialization import to_jsonable

# NOTE: tool-dispatch logic (which tools need `image=`, and which image) now
# lives in tool_dispatch.py, shared with loop.py's GRPO rollout -- see that
# module's docstring for why. Don't reintroduce a local copy here.


class TraceGenState(TypedDict):
    base_image: Image.Image
    messages: list[dict[str, Any]]
    turns: list[dict[str, Any]]
    tool_calls: int
    final_answer: Any
    status: str  # "running" | "needs_tool" | "done" | "error"
    error: str | None
    consecutive_parse_errors: int
    pending_tool_calls: list[dict[str, Any]] | None
    tools_used: set[str]
    located_teeth: set[int]



import os

# Per-provider defaults for context-trim threshold (roughly 70% of that provider's
# real TPM/context ceiling) and max output tokens. Both are .env-overridable per
# provider; these are just the fallback if no env var is set. Groq's numbers are
# confirmed from the console (8,000 TPM) for qwen/qwen3.6-27b -- the others are
# generous since neither NVIDIA nor Gemini has shown an actual ceiling.
_PROVIDER_TRIM_THRESHOLD_DEFAULTS = {
    "groq": 5600, "nvidia_nim": 30000, "openrouter": 12000, "gemini": 100000, "local": 11468,
}
_PROVIDER_MAX_TOKENS_DEFAULTS = {
    "groq": 1024, "openrouter": 1536, "nvidia_nim": 16384, "gemini": 16384, "local": 8192,
}


def _provider_env(provider: str, suffix: str, code_default: int) -> int:
    prefix = provider.upper().replace("_NIM", "")
    val = os.environ.get(f"{prefix}_{suffix}")
    return int(val) if val else code_default


def _reasoning_node_factory(provider: str, model: str, max_tool_calls: int, max_tokens: int | None = None, context_trim_threshold: int | None = None):
    if max_tokens is not None:
        resolved_max_tokens = max_tokens
    else:
        resolved_max_tokens = _provider_env(provider, "MAX_TOKENS", _PROVIDER_MAX_TOKENS_DEFAULTS.get(provider, 8192))
        
    resolved_trim_threshold = context_trim_threshold or _provider_env(
        provider, "CONTEXT_TRIM_THRESHOLD", _PROVIDER_TRIM_THRESHOLD_DEFAULTS.get(provider, 11468)
    )

    def _reasoning_node(state: TraceGenState) -> TraceGenState:
        if state["tool_calls"] >= max_tool_calls and state["final_answer"] is None:
            state["status"] = "error"
            state["error"] = f"max_tool_calls ({max_tool_calls}) reached without a final_answer"
            return state

        # 4b. Sliding-window context trimming -- threshold is now provider-specific
        # (see _PROVIDER_TRIM_THRESHOLD_DEFAULTS above). A threshold computed off
        # vLLM's ~16-24K local budget never fires early enough to protect Groq's
        # 8,000 TPM ceiling -- that mismatch was the actual cause of Groq/OpenRouter
        # "getting limited at higher turn numbers" even with this trimming in place.
        est_tokens = sum(len(str(m)) for m in state["messages"]) // 4
        image_count = 0
        for m in state["messages"]:
            if isinstance(m.get("content"), list):
                for block in m["content"]:
                    if isinstance(block, dict) and block.get("type") == "image":
                        image_count += 1
        est_tokens += image_count * 600

        def _trim_images(keep_last_n: int):
            # Find all image blocks in tool responses (skip the first user turn which has the original X-ray)
            img_blocks = []
            for m in state["messages"][1:]:
                if isinstance(m.get("content"), list):
                    for block in m["content"]:
                        if isinstance(block, dict) and block.get("type") == "image":
                            img_blocks.append(block)
            
            # Trim the oldest ones
            if len(img_blocks) > keep_last_n:
                to_trim = img_blocks[:-keep_last_n] if keep_last_n > 0 else img_blocks
                for m in state["messages"][1:]:
                    if isinstance(m.get("content"), list):
                        new_content = []
                        for block in m["content"]:
                            if block in to_trim:
                                new_content.append({"type": "text", "text": "[Earlier tool result omitted to save context — already reasoned about above]"})
                            else:
                                new_content.append(block)
                        m["content"] = new_content

        if est_tokens > resolved_trim_threshold:
            _trim_images(keep_last_n=2)

        try:
            raw = call_llm(
                provider=provider,
                model=model,
                system_prompt="",
                user_content=state["messages"],
                image=None,
                temperature=0.7,
                max_tokens=resolved_max_tokens,
                stream=True,
                label=f"turn {len(state['turns'])}",
                role="generator",
            )
        except Exception as e:
            error_str = str(e).lower()
            if "maximum context length" in error_str or "context_length_exceeded" in error_str:
                # 4c. Aggressive trim and retry once
                _trim_images(keep_last_n=0)
                try:
                    raw = call_llm(
                        provider=provider,
                        model=model,
                        system_prompt="",
                        user_content=state["messages"],
                        image=None,
                        temperature=0.7,
                        max_tokens=resolved_max_tokens,
                        stream=True,
                        label=f"turn {len(state['turns'])} (retry)",
                        role="generator",
                    )
                except Exception as retry_e:
                    raise retry_e
            else:
                raise e

        parsed = parse_agent_json(raw)
        state["messages"].append({"role": "assistant", "content": raw})

        turn_record: dict[str, Any] = {
            "turn": len(state["turns"]),
            "raw_output": raw,
            "parsed": parsed,
        }

        if not parsed:
            state["consecutive_parse_errors"] += 1

            if state["consecutive_parse_errors"] == 2:
                turn_record["status"] = "unparseable_recovery_attempt"
                state["turns"].append(turn_record)
                state["messages"].append({
                    "role": "user",
                    "content": (
                        "Your last two responses were cut off before producing valid JSON "
                        "(your reasoning ran too long). Stop reasoning now. Based on "
                        "everything you've seen so far, output ONLY one JSON object: "
                        "either {\"tool\": \"...\", \"args\": {...}} for your next tool call, "
                        "or {\"final_answer\": {...}} if you already have enough to diagnose. "
                        "No <think> block, no other text — the JSON object only."
                    ),
                })
                state["status"] = "running"
                return state

            if state["consecutive_parse_errors"] >= 3:
                turn_record["status"] = "unparseable_fatal"
                state["turns"].append(turn_record)
                state["status"] = "error"
                state["error"] = f"Unparseable model output on turn {turn_record['turn']}: {raw[:300]}"
            else:
                if "{" not in raw:
                    hint = (
                        "Error: Your response was cut off before reaching a JSON action "
                        "(too much reasoning). Be more concise — briefly note your next "
                        "step, then immediately output the JSON tool call or final_answer."
                    )
                else:
                    hint = (
                        "Error: Your output was not valid JSON or could not be parsed. "
                        "Please correct the format and output a single JSON object."
                    )
                turn_record["status"] = "unparseable_retry"
                state["turns"].append(turn_record)
                state["messages"].append({"role": "user", "content": hint})
                state["status"] = "running"
            return state

        state["consecutive_parse_errors"] = 0

        if "final_answer" in parsed:
            if state["tool_calls"] == 0:
                turn_record["status"] = "rejected_final_answer"
                state["turns"].append(turn_record)
                state["messages"].append(
                    {"role": "user", "content": "Error: You MUST use at least one tool before providing a final answer."}
                )
                state["status"] = "running"
                return state

            # 1c. Tool-diversity gate. Prompting alone is a request, not a guarantee --
            # this is enforced structurally: a trace that hasn't genuinely used tools
            # gets bounced back regardless of how many turns that costs.
            if len(state["tools_used"]) < 3:
                turn_record["status"] = "rejected_final_answer"
                state["turns"].append(turn_record)
                state["messages"].append({
                    "role": "user",
                    "content": (
                        f"Error: You've only used {sorted(state['tools_used'])}. Use a wider "
                        "range of tools (locate_tooth, fdi_label, zoom_crop, window_level, "
                        "denoise, contralateral_compare) before finalizing."
                    ),
                })
                state["status"] = "running"
                return state

            proposed = parsed["final_answer"]
            if isinstance(proposed, list):
                unlocated = [
                    f for f in proposed
                    if isinstance(f, dict) and "quadrant" in f and "tooth_position" in f
                    and (int(f["quadrant"]) * 10 + int(f["tooth_position"])) not in state["located_teeth"]
                ]
                if unlocated:
                    bad = ", ".join(f"Q{f['quadrant']}T{f['tooth_position']}" for f in unlocated)
                    turn_record["status"] = "rejected_final_answer"
                    state["turns"].append(turn_record)
                    state["messages"].append({
                        "role": "user",
                        "content": (
                            f"Error: Finding(s) at {bad} were never located with locate_tooth. "
                            "Locate every tooth you're diagnosing before including it in your "
                            "final answer."
                        ),
                    })
                    state["status"] = "running"
                    return state

            state["final_answer"] = proposed
            turn_record["status"] = "final_answer"
            state["turns"].append(turn_record)
            state["status"] = "done"
            return state

        # 1d. Multi-tool-call-per-turn: parsing.py already normalizes a legacy single
        # "tool" key into a one-element "tool_calls" list, so this only ever needs to
        # handle the list form.
        tool_calls = parsed.get("tool_calls")

        if not tool_calls or not isinstance(tool_calls, list):
            turn_record["status"] = "invalid_tool_format"
            state["turns"].append(turn_record)
            state["messages"].append(
                {"role": "user", "content": "Error: Missing or empty 'tool_calls' list in JSON output."}
            )
            state["status"] = "running"
            return state

        if len(tool_calls) > 4:
            turn_record["status"] = "invalid_tool_format"
            state["turns"].append(turn_record)
            state["messages"].append(
                {"role": "user", "content": "Error: At most 4 tool calls per turn. Split this across more turns."}
            )
            state["status"] = "running"
            return state

        for tc in tool_calls:
            if not isinstance(tc, dict) or not tc.get("tool"):
                turn_record["status"] = "invalid_tool_format"
                state["turns"].append(turn_record)
                state["messages"].append(
                    {"role": "user", "content": "Error: Each entry in 'tool_calls' needs a 'tool' key."}
                )
                state["status"] = "running"
                return state

        state["pending_tool_calls"] = [
            {"tool": tc["tool"], "args": tc.get("args", {}) or {}} for tc in tool_calls
        ]
        state["_pending_turn_record"] = turn_record
        state["status"] = "needs_tool"
        return state

    return _reasoning_node


def _tool_node_factory(
    registry: ToolRegistry,
    ground_truth: list[dict[str, Any]] | None = None,
    hint_probability: float = 1.0,
    perturb_small_probability: float = 0.25,
    perturb_big_probability: float = 0.30,
    perturb_small_range: tuple[float, float] = (0.12, 0.28),
    perturb_big_range: tuple[float, float] = (0.45, 0.75),
):
    """hint_probability: chance search_region_hint is applied when ground truth
    exists for the requested tooth. Defaults to 1.0 (always) -- the hint gives
    locate_tooth's real underlying detector the best chance of a correct result.
    Even at 1.0 this is NOT literally the ground-truth box -- it's the hint
    NARROWING where the real detector searches; what gets shown is still real
    (hint-assisted) YOLO inference, which is usually very accurate within that
    narrowed window but is not synthetically guaranteed to be. The hint is not
    the mechanism that teaches accept-vs-nudge either way (see below).

    perturb_small_probability / perturb_big_probability: independent chances
    that, once locate_tooth returns a bbox, a synthetic offset is applied to
    what the model is SHOWN (never to what's logged internally as true_bbox).
    Two tiers, not one: "small" (12-28% of box size) is a genuine judgment
    call -- often still fine given zoom_crop's padding, sometimes worth a
    precise nudge, and deliberately NOT resolved either way by anything in the
    prompt, so the model has to actually decide rather than learn a fixed
    rule. "Big" (45-75%) is sized to be visually self-evident on its own --
    no verbal cue needed, because a verbal one (e.g. a "this needs
    correction" hint in the trace-gen directive) would only exist during
    trace-gen and not at GRPO/inference time, where locate_tooth's real
    errors are never flagged in advance. Teaching the model to respond to a
    directive instead of to what it actually sees would be a shortcut that
    works during data generation and silently fails at the one time it
    matters. The magnitude alone has to carry it, which is why "big" is
    deliberately large enough to.

    Both are decoupled from the real detector's actual accuracy on purpose:
    a pipeline-level teaching device, not a report of locate_tooth's true
    error rate. That matters because the grounding tool is expected to keep
    improving (better weights, eventually a multi-dataset detector) -- if
    "how often the model must verify/correct" were sourced from the current
    detector's real precision, traces generated today would go stale
    relative to tomorrow's detector and need regenerating. Fixed, configured
    rates keep teaching the same behavior regardless of how good locate_tooth
    gets, so today's traces stay valid without a re-run."""
    ground_truth = ground_truth or []

    def _hint_for_tooth(fdi_number: int) -> list[float] | None:
        """Look up this trace's ground truth for a bbox matching the requested FDI
        tooth, for locate_tooth's search_region_hint (Section 2c) -- privileged,
        pipeline-internal only, never derived from anything the model said."""
        fdi_str = str(fdi_number)
        if len(fdi_str) != 2:
            return None
        quadrant, position = int(fdi_str[0]), int(fdi_str[1])
        for f in ground_truth:
            if f.get("quadrant") == quadrant and f.get("tooth_position") == position and "bbox" in f:
                return f["bbox"]
        return None

    def _synthetic_offset(bbox: list[float], image: Image.Image, magnitude_range: tuple[float, float]) -> list[float]:
        """Nudge *bbox* by a random, bounded, tool-independent offset within
        magnitude_range -- reuses nudge_crop's own shift/clamp math so the
        perturbation always lands somewhere nudge_crop itself could plausibly
        produce or correct."""
        lo, hi = magnitude_range
        dx = random.uniform(lo, hi) * random.choice([-1, 1])
        dy = random.uniform(lo, hi) * random.choice([-1, 1])
        result = tool_nudge_crop(image, bbox, dx_frac=dx, dy_frac=dy)
        return result.get("bbox", bbox)

    def _tool_node(state: TraceGenState) -> TraceGenState:
        pending = state.get("pending_tool_calls")
        if not pending:
            state["status"] = "error"
            state["error"] = "Tool node called but no pending tool calls found."
            return state

        turn_record = state.pop("_pending_turn_record", {"turn": len(state["turns"])})
        turn_record["tool_calls_this_turn"] = []
        observation: list[dict[str, Any]] = []
        any_ok = False

        for call in pending:
            tool_name = call["tool"]
            tool_args = dict(call["args"])
            call_record: dict[str, Any] = {"tool_name": tool_name, "tool_args": dict(tool_args)}

            if not registry.get(tool_name):
                call_record["tool_ok"] = False
                call_record["tool_error"] = f"Tool '{tool_name}' is not recognized."
                observation.append({"type": "text", "text": f"Error: Tool '{tool_name}' is not recognized."})
                turn_record["tool_calls_this_turn"].append(call_record)
                continue

            try:
                if tool_name == "locate_tooth" and "tooth" in tool_args:
                    hint = _hint_for_tooth(tool_args["tooth"])
                    if hint is not None and random.random() < hint_probability:
                        tool_args["search_region_hint"] = hint

                tool_out = execute_tool_call(registry, tool_name, tool_args, state["base_image"])

                if tool_name == "locate_tooth" and isinstance(tool_out, dict) and "bbox" in tool_out:
                    roll = random.random()
                    tier = None
                    magnitude_range = None
                    if roll < perturb_big_probability:
                        tier, magnitude_range = "big", perturb_big_range
                    elif roll < perturb_big_probability + perturb_small_probability:
                        tier, magnitude_range = "small", perturb_small_range
                    if tier is not None:
                        call_record["true_bbox"] = tool_out["bbox"]  # internal only, never shown to the model
                        call_record["perturb_tier"] = tier
                        tool_out = dict(tool_out)
                        tool_out["bbox"] = _synthetic_offset(tool_out["bbox"], state["base_image"], magnitude_range)

                if isinstance(tool_out, Image.Image):
                    observation.append({"type": "image", "image": tool_out})
                    observation.append({"type": "text", "text": f"Result of {tool_name}:"})
                else:
                    observation.append({"type": "text", "text": f"Result of {tool_name}: {json.dumps(to_jsonable(tool_out))}"})

                call_record["tool_ok"] = True
                any_ok = True
                state["tool_calls"] += 1
                state["tools_used"].add(tool_name)
                if tool_name == "locate_tooth" and isinstance(tool_out, dict) and "bbox" in tool_out and "tooth" in tool_out:
                    state["located_teeth"].add(int(tool_out["tooth"]))

            except Exception as e:
                call_record["tool_ok"] = False
                call_record["tool_error"] = str(e)
                observation.append({"type": "text", "text": f"Tool '{tool_name}' execution failed: {e}"})

            turn_record["tool_calls_this_turn"].append(call_record)

        turn_record["status"] = "tool_executed" if any_ok else "tool_all_failed"
        state["turns"].append(turn_record)
        state["messages"].append({"role": "user", "content": observation})
        state["pending_tool_calls"] = None
        state["status"] = "running"
        return state

    return _tool_node


def _route(state: TraceGenState) -> str:
    if state["status"] == "needs_tool":
        return "tools"
    elif state["status"] == "running":
        return "reasoning"
    return END


def build_trace_gen_graph(
    registry: ToolRegistry,
    ground_truth: list[dict[str, Any]] | None = None,
    provider: str = "local",
    model: str = "Qwen/Qwen3.5-9B",
    max_tool_calls: int = 8,
    max_tokens: int | None = None,
    hint_probability: float = 1.0,
    perturb_small_probability: float = 0.25,
    perturb_big_probability: float = 0.30,
):
    """Build and compile the LangGraph for ground-truth-directed, real-tool-execution generation."""
    graph = StateGraph(TraceGenState)
    
    graph.add_node("reasoning", _reasoning_node_factory(provider, model, max_tool_calls, max_tokens=max_tokens))
    graph.add_node("tools", _tool_node_factory(
        registry, ground_truth=ground_truth or [],
        hint_probability=hint_probability,
        perturb_small_probability=perturb_small_probability,
        perturb_big_probability=perturb_big_probability,
    ))
    
    graph.set_entry_point("reasoning")
    
    graph.add_conditional_edges("reasoning", _route, {"reasoning": "reasoning", "tools": "tools", END: END})
    graph.add_conditional_edges("tools", _route, {"reasoning": "reasoning", END: END})
    
    return graph.compile()


def run_trace_gen(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    registry: ToolRegistry,
    system_prompt: str,
    provider: str = "local",
    model: str = "Qwen/Qwen3.5-9B",
    max_turns: int = 8,
    max_tool_calls: int = 50,
    max_tokens_per_turn: int | None = None,
    hint_probability: float = 1.0,
    perturb_small_probability: float = 0.25,
    perturb_big_probability: float = 0.30,
) -> tuple[dict[str, Any] | None, str | None]:
    """Ground-truth-directed trace generation with real tool execution via LangGraph.
    
    Returns (trajectory_dict, None) on success or (None, error_reason) on failure.
    """
    # Always drop bbox coordinates from the directive -- now that locate_tooth actually
    # works, handing over exact coordinates just teaches copy-the-hint instead of
    # genuine tool-mediated localization. Keep naming which findings exist (preserves
    # yield -- the model isn't searching blind) but never the bbox itself. This applies
    # to every provider and every finding count, not just Groq above some threshold.
    hint_text = "; ".join(
        f"Q{f['quadrant']}T{f['tooth_position']}:{f['diagnosis']}" for f in ground_truth
    )

    directive = (
        "TEACHER DIRECTIVE: You are generating an expert demonstration trace for SFT.\n"
        f"You MUST eventually reach a diagnosis covering these {len(ground_truth)} finding(s): "
        f"{hint_text}\n\n"
        "Use locate_tooth to find each tooth's position — do not guess or assert coordinates "
        "yourself. locate_tooth's box is not guaranteed to be exact or even correct: before "
        "trusting it, zoom_crop into it and check the crop actually shows the tooth you asked "
        "for. If it looks off-center, too tight, or shows the wrong tooth, use nudge_crop to "
        "shift or rescale the box, then zoom_crop again — do not silently proceed on a crop "
        "that doesn't match what you expected. Then use window_level / denoise / "
        "contralateral_compare to inspect and confirm before your final answer. You MUST use "
        "at least one tool before answering — do not output final_answer on the first turn. "
        "Never mention in your reasoning that this list, a hint, or a directive was given to "
        "you — write your thought as genuine first-look clinical analysis."
    )

    initial_content = [
        {"type": "image", "image": image},
        {
            "type": "text",
            "text": (
                "Analyze this panoramic X-ray. Identify any abnormal teeth and determine "
                "the diagnosis.\n\n" + directive
            ),
        },
    ]

    initial_state: TraceGenState = {
        "base_image": image,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": initial_content},
        ],
        "turns": [],
        "tool_calls": 0,
        "final_answer": None,
        "status": "running",
        "error": None,
        "consecutive_parse_errors": 0,
        "pending_tool_calls": None,
        "tools_used": set(),
        "located_teeth": set(),
    }

    app = build_trace_gen_graph(
        registry, ground_truth=ground_truth, provider=provider, model=model,
        max_tool_calls=max_tool_calls, max_tokens=max_tokens_per_turn,
        hint_probability=hint_probability,
        perturb_small_probability=perturb_small_probability,
        perturb_big_probability=perturb_big_probability,
    )
    # Each logical turn can now cost up to ~4 reasoning-node visits (2 retries + 1
    # recovery attempt + 1 success) plus 1 tool-node visit, so budget generously —
    # this is just a runaway-loop safety valve, not the real cost control (that's
    # max_tool_calls / consecutive_parse_errors above).
    final_state: TraceGenState = app.invoke(
        initial_state, config={"recursion_limit": max_tool_calls * 2 + max_turns * 6 + 10}
    )

    error = final_state.get("error")

    if final_state["final_answer"] is None:
        # Preserve whatever partial progress exists (successful tool calls, partial
        # reasoning turns) instead of discarding it — a failed image still returns
        # its turns/tool_calls so the caller can persist them for inspection or
        # future reuse, rather than losing that generation work entirely.
        if final_state["turns"]:
            return {
                "turns": final_state["turns"],
                "tool_calls": final_state["tool_calls"],
                "final_answer": None,
                "messages": final_state["messages"],
                "format_ok": False,
            }, error or "No final_answer produced"
        return None, error or "No final_answer produced"

    return {
        "turns": final_state["turns"],
        "tool_calls": final_state["tool_calls"],
        "final_answer": final_state["final_answer"],
        "messages": final_state["messages"],
        "format_ok": True,
    }, None
