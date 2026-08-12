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

This module talks to the generator model through api_pool.call_llm(provider="local", ...),
i.e. a vLLM OpenAI-compatible server hosting Qwen3-VL-8B-Thinking inside the same
Kaggle/Colab session — not a direct in-process HF model/processor pair. For direct
HF-transformers generation against an already-loaded model+processor (used for SFT/GRPO-
model evaluation, batch_runner.py, ablations.py), see dental_agent/agent/loop.py — that's
a different, legitimate serving mode for a different purpose and is intentionally untouched
here beyond the tool-dispatch bugfix (image-returning tools crashing the generic
json.dumps() branch, and locate_tooth's mismatched signature — see loop.py and grounding.py).
"""

from __future__ import annotations

import json
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
from dental_agent.tools.registry import ToolRegistry
from dental_agent.training.api_pool import call_llm
from dental_agent.utils.serialization import to_jsonable

# Tools that take `image` and return a PIL.Image. Always executed against the base
# image (not whatever the model most recently zoomed into) to avoid compounding crop
# drift across turns — matching the "prevent state corruption" reasoning the old
# trace_generation.py already documented for this same set of tools, rather than
# loop.py's cumulative-crop behavior. Worth reconciling the two loops on this point
# at some point; not done here to avoid changing loop.py's behavior for its existing
# eval/ablation consumers.
IMAGE_INPUT_TOOLS = {"zoom_crop", "window_level", "denoise", "contralateral_compare"}


class TraceGenState(TypedDict):
    base_image: Image.Image
    messages: list[dict[str, Any]]
    turns: list[dict[str, Any]]
    tool_calls: int
    final_answer: Any
    status: str  # "running" | "done" | "error"
    error: str | None


def _model_node_factory(registry: ToolRegistry, provider: str, model: str, max_tool_calls: int):
    def _model_node(state: TraceGenState) -> TraceGenState:
        if state["tool_calls"] >= max_tool_calls and state["final_answer"] is None:
            state["status"] = "error"
            state["error"] = f"max_tool_calls ({max_tool_calls}) reached without a final_answer"
            return state

        raw = call_llm(
            provider=provider,
            model=model,
            system_prompt="",
            user_content=state["messages"],
            image=None,
            temperature=0.7,
            max_tokens=2048,
        )
        parsed = parse_agent_json(raw)
        state["messages"].append({"role": "assistant", "content": raw})

        turn_record: dict[str, Any] = {
            "turn": len(state["turns"]),
            "raw_output": raw,
            "parsed": parsed,
        }

        if not parsed:
            turn_record["status"] = "unparseable"
            state["turns"].append(turn_record)
            state["status"] = "error"
            state["error"] = f"Unparseable model output on turn {turn_record['turn']}: {raw[:300]}"
            return state

        if "final_answer" in parsed:
            state["final_answer"] = parsed["final_answer"]
            turn_record["status"] = "final_answer"
            state["turns"].append(turn_record)
            state["status"] = "done"
            return state

        tool_name = parsed.get("tool")
        tool_args = parsed.get("args", {}) or {}

        if not tool_name or not registry.get(tool_name):
            turn_record["status"] = "invalid_tool"
            state["turns"].append(turn_record)
            state["messages"].append(
                {"role": "user", "content": f"Error: Tool '{tool_name}' is not recognized."}
            )
            state["status"] = "running"
            return state

        turn_record["tool_name"] = tool_name
        turn_record["tool_args"] = tool_args

        try:
            if tool_name in IMAGE_INPUT_TOOLS:
                tool_out = registry.execute(tool_name, image=state["base_image"], **tool_args)
            elif tool_name == "locate_tooth":
                tool_out = registry.execute(tool_name, image=state["base_image"], **tool_args)
            else:
                tool_out = registry.execute(tool_name, **tool_args)

            if isinstance(tool_out, Image.Image):
                observation = [
                    {"type": "image", "image": tool_out},
                    {"type": "text", "text": f"Result of {tool_name}:"},
                ]
            else:
                observation = [
                    {"type": "text", "text": f"Tool output: {json.dumps(to_jsonable(tool_out))}"}
                ]

            turn_record["tool_ok"] = True
            state["tool_calls"] += 1
            state["messages"].append({"role": "user", "content": observation})

        except Exception as e:
            turn_record["tool_ok"] = False
            turn_record["tool_error"] = str(e)
            state["messages"].append({"role": "user", "content": f"Tool execution failed: {e}"})

        state["turns"].append(turn_record)
        state["status"] = "running"
        return state

    return _model_node


def _route(state: TraceGenState) -> str:
    return "model" if state["status"] == "running" else END


def build_trace_gen_graph(
    registry: ToolRegistry,
    provider: str = "local",
    model: str = "Qwen/Qwen3-VL-8B-Thinking",
    max_tool_calls: int = 5,
):
    """Build and compile the LangGraph for ground-truth-directed, real-tool-execution generation."""
    graph = StateGraph(TraceGenState)
    graph.add_node("model", _model_node_factory(registry, provider, model, max_tool_calls))
    graph.set_entry_point("model")
    graph.add_conditional_edges("model", _route, {"model": "model", END: END})
    return graph.compile()


def run_trace_gen(
    image: Image.Image,
    ground_truth: list[dict[str, Any]],
    registry: ToolRegistry,
    system_prompt: str,
    provider: str = "local",
    model: str = "Qwen/Qwen3-VL-8B-Thinking",
    max_turns: int = 5,
) -> tuple[dict[str, Any] | None, str | None]:
    """Ground-truth-directed trace generation with real tool execution via LangGraph.

    The model is still told the correct diagnosis up front and given the ground-truth
    bounding boxes as a hint for where to look (kept deliberately — see module docstring
    and AGENT_HANDOVER.md §3). What's real now is that every tool call it makes actually
    executes against the source image; nothing is pre-computed and narrated for it.

    Returns (trajectory_dict, None) on success or (None, error_reason) on failure, matching
    the return shape of the old generate_interactive_trajectory() so callers in
    trace_generation.py (build_trace_example, verify_trace, run_aim1_batch) don't need to
    change beyond the swap already made in generate_interactive_trajectory() itself.
    """
    directive = (
        "TEACHER DIRECTIVE: You are generating an expert demonstration trace for SFT.\n"
        f"You MUST eventually reach this exact diagnosis: {json.dumps(ground_truth)}\n\n"
        "The ground-truth findings above tell you what's there and roughly where — use that "
        "as your starting hint for where to look, not as something to restate without checking. "
        "Use zoom_crop / window_level / denoise / contralateral_compare / locate_tooth for real "
        "to inspect each region before your final answer. You MUST use at least one tool before "
        "answering — do not output final_answer on the first turn."
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
    }

    app = build_trace_gen_graph(registry, provider=provider, model=model, max_tool_calls=max_turns)
    final_state: TraceGenState = app.invoke(
        initial_state, config={"recursion_limit": max_turns * 2 + 4}
    )

    if final_state["final_answer"] is None:
        return None, final_state.get("error") or "No final_answer produced"

    return {
        "turns": final_state["turns"],
        "tool_calls": final_state["tool_calls"],
        "final_answer": final_state["final_answer"],
        "messages": final_state["messages"],
        "format_ok": True,
    }, None
