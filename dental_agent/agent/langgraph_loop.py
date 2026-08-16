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
# drift across turns.
IMAGE_INPUT_TOOLS = {"zoom_crop", "window_level", "denoise", "contralateral_compare"}


class TraceGenState(TypedDict):
    base_image: Image.Image
    messages: list[dict[str, Any]]
    turns: list[dict[str, Any]]
    tool_calls: int
    final_answer: Any
    status: str  # "running" | "needs_tool" | "done" | "error"
    error: str | None
    consecutive_parse_errors: int
    pending_tool_call: dict[str, Any] | None



def _reasoning_node_factory(provider: str, model: str, max_tool_calls: int, max_tokens: int = 8192, context_trim_threshold: int = 11468):
    def _reasoning_node(state: TraceGenState) -> TraceGenState:
        if state["tool_calls"] >= max_tool_calls and state["final_answer"] is None:
            state["status"] = "error"
            state["error"] = f"max_tool_calls ({max_tool_calls}) reached without a final_answer"
            return state

        # 4b. Sliding-window context trimming
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

        if est_tokens > context_trim_threshold:
            _trim_images(keep_last_n=2)

        try:
            raw = call_llm(
                provider=provider,
                model=model,
                system_prompt="",
                user_content=state["messages"],
                image=None,
                temperature=0.7,
                max_tokens=max_tokens,
                stream=True,
                label=f"turn {len(state['turns'])}",
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
                        max_tokens=max_tokens,
                        stream=True,
                        label=f"turn {len(state['turns'])} (retry)",
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

            state["final_answer"] = parsed["final_answer"]
            turn_record["status"] = "final_answer"
            state["turns"].append(turn_record)
            state["status"] = "done"
            return state

        tool_name = parsed.get("tool")
        tool_args = parsed.get("args", {}) or {}

        if not tool_name:
            turn_record["status"] = "invalid_tool_format"
            state["turns"].append(turn_record)
            state["messages"].append(
                {"role": "user", "content": "Error: Missing 'tool' key in JSON output."}
            )
            state["status"] = "running"
            return state

        state["pending_tool_call"] = {"tool": tool_name, "args": tool_args, "turn_record": turn_record}
        state["status"] = "needs_tool"
        return state

    return _reasoning_node


def _tool_node_factory(registry: ToolRegistry):
    def _tool_node(state: TraceGenState) -> TraceGenState:
        pending = state.get("pending_tool_call")
        if not pending:
            state["status"] = "error"
            state["error"] = "Tool node called but no pending tool call found."
            return state

        tool_name = pending["tool"]
        tool_args = pending["args"]
        turn_record = pending["turn_record"]

        if not registry.get(tool_name):
            turn_record["status"] = "invalid_tool"
            state["turns"].append(turn_record)
            state["messages"].append(
                {"role": "user", "content": f"Error: Tool '{tool_name}' is not recognized."}
            )
            state["pending_tool_call"] = None
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
        state["pending_tool_call"] = None
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
    provider: str = "local",
    model: str = "Qwen/Qwen3.5-9B",
    max_tool_calls: int = 8,
    max_tokens: int = 4096,
):
    """Build and compile the LangGraph for ground-truth-directed, real-tool-execution generation."""
    graph = StateGraph(TraceGenState)
    
    graph.add_node("reasoning", _reasoning_node_factory(provider, model, max_tool_calls, max_tokens=max_tokens))
    graph.add_node("tools", _tool_node_factory(registry))
    
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
    max_tokens_per_turn: int = 4096,
) -> tuple[dict[str, Any] | None, str | None]:
    """Ground-truth-directed trace generation with real tool execution via LangGraph.
    
    Returns (trajectory_dict, None) on success or (None, error_reason) on failure.
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
        "consecutive_parse_errors": 0,
        "pending_tool_call": None,
    }

    app = build_trace_gen_graph(
        registry, provider=provider, model=model, max_tool_calls=max_turns, max_tokens=max_tokens_per_turn
    )
    # Each logical turn can now cost up to ~4 reasoning-node visits (2 retries + 1
    # recovery attempt + 1 success) plus 1 tool-node visit, so budget generously —
    # this is just a runaway-loop safety valve, not the real cost control (that's
    # max_tool_calls / consecutive_parse_errors above).
    final_state: TraceGenState = app.invoke(
        initial_state, config={"recursion_limit": max_turns * 6 + 10}
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
