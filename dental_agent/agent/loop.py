"""
Multi-turn agent orchestration loop for dental radiograph analysis.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional
from PIL import Image
import pandas as pd

from dental_agent.agent.prompts import build_agent_system_prompt, NO_TOOLS_SYSTEM_PROMPT
from dental_agent.agent.parsing import parse_agent_json
from dental_agent.agent.tool_dispatch import execute_tool_call
from dental_agent.model.inference import generate_agent_reply
from dental_agent.tools.registry import ToolRegistry

# NOTE: tool-dispatch logic (which tools need `image=`, and which image) now
# lives in tool_dispatch.py, shared with langgraph_loop.py's trace-gen graph --
# see that module's docstring for why. Don't reintroduce a local copy here.


@dataclass
class AgentTrajectory:
    image_id: int
    turns: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: int = 0
    final_answer: Optional[dict[str, Any]] = None
    format_ok: bool = False
    assistant_token_spans: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "final_answer": self.final_answer,
            "format_ok": self.format_ok,
            "assistant_token_spans": self.assistant_token_spans,
            "messages": self.messages,
        }


def run_agent(
    image_id: int,
    images_df: pd.DataFrame,
    model: Any,
    processor: Any,
    registry: ToolRegistry | None = None,
    max_tool_calls: int = 50,
    verbose: bool = True,
) -> AgentTrajectory:
    """Run the multi-turn agent loop on a single dental radiograph.

    Executes:
    1. Load full panoramic image.
    2. Prompt agent with system prompt and registered tool descriptions.
    3. Loop: parse response, execute tool call, inject resulting image/observation back into conversation.
    4. Terminate when final_answer is returned or max_tool_calls is reached.
    """
    registry = registry or ToolRegistry.create_default()
    row = images_df[images_df["id"] == image_id].iloc[0]
    base_image = Image.open(row["local_path"]).convert("RGB")

    system_prompt = build_agent_system_prompt(registry.format_tool_descriptions())

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base_image},
                {
                    "type": "text",
                    "text": f"Analyze this panoramic X-ray (image_id={image_id}). "
                    f"Identify any abnormal teeth and determine the diagnosis.",
                },
            ],
        },
    ]

    turns: list[dict[str, Any]] = []
    assistant_token_spans: list[dict[str, Any]] = []
    final_answer = None
    tool_call_count = 0
    registered_names = {t.name for t in registry.list_tools()}

    # max_tool_calls bounds total CALLS, not turns -- a turn can carry several
    # (the agent may request multiple tool calls at once, same as trace-gen's
    # loop), so a turn-indexed bound would silently allow up to ~4x
    # max_tool_calls actual calls once multi-call turns are in play. The outer
    # range here is a generous safety valve against a model that never
    # produces a valid response, not the real budget -- the real budget is
    # enforced by tool_call_count below, and once it's hit the model is told
    # to answer instead of silently having further calls dropped.
    for turn_idx in range(max_tool_calls + 15):
        reply, prompt_len, gen_ids = generate_agent_reply(
            model, processor, messages, return_ids=True
        )
        assistant_token_spans.append({"prompt_len": prompt_len, "token_ids": gen_ids})
        parsed = parse_agent_json(reply)

        turn_record: dict[str, Any] = {
            "turn": turn_idx,
            "raw_output": reply,
            "parsed": parsed,
        }

        if not parsed:
            turn_record["status"] = "unparseable_json"
            turns.append(turn_record)
            break

        # Check for final answer
        if "final_answer" in parsed:
            final_answer = parsed["final_answer"]
            turn_record["status"] = "final_answer"
            turns.append(turn_record)
            messages.append({"role": "assistant", "content": reply})
            break

        # parse_agent_json normalizes a legacy single "tool"/"args" pair into a
        # one-element "tool_calls" list (popping the old keys), so tool_calls
        # is the only shape to read here regardless of whether the model asked
        # for one tool or several.
        tool_calls = parsed.get("tool_calls")
        if not tool_calls or not isinstance(tool_calls, list):
            turn_record["status"] = "invalid_tool"
            turn_record["tool_calls_this_turn"] = []
            turns.append(turn_record)
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": "Error: no valid tool_calls or final_answer found in your response.",
            })
            continue

        if tool_call_count >= max_tool_calls:
            # Budget exhausted -- don't silently drop the requested calls,
            # tell the model plainly and give it a real chance to comply
            # instead of looping until the outer safety valve trips.
            turn_record["status"] = "budget_exhausted"
            turn_record["tool_calls_this_turn"] = []
            turns.append(turn_record)
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": f"You've used your tool-call budget ({max_tool_calls}). "
                f"Provide your final_answer now based on what you've already seen.",
            })
            continue

        # Dispatch logic lives in tool_dispatch.py, shared with langgraph_loop.py's
        # trace-gen graph, so the two loops can't silently diverge on which tools
        # need `image=` or which image they see again (see that module's docstring
        # for the history of why this used to be duplicated, and the bug that came
        # of it). Always base_image -- never a compounded, previously-cropped view.
        observation_content: list[dict[str, Any]] = []
        calls_this_turn: list[dict[str, Any]] = []
        any_ok = False

        for call in tool_calls:
            if tool_call_count >= max_tool_calls:
                break  # budget ran out mid-turn (a multi-call turn requested more than remained)

            tool_name = call.get("tool") if isinstance(call, dict) else None
            tool_args = call.get("args", {}) if isinstance(call, dict) else {}
            call_record: dict[str, Any] = {"tool_name": tool_name, "tool_args": tool_args}

            if not tool_name or tool_name not in registered_names:
                call_record["tool_ok"] = False
                call_record["tool_error"] = f"Tool '{tool_name}' is not recognized."
                observation_content.append(
                    {"type": "text", "text": f"Error: Tool '{tool_name}' is not recognized."}
                )
                calls_this_turn.append(call_record)
                continue

            tool_call_count += 1
            try:
                tool_out = execute_tool_call(registry, tool_name, tool_args, base_image)

                if isinstance(tool_out, Image.Image):
                    observation_content.append({"type": "image", "image": tool_out})
                    observation_content.append({"type": "text", "text": f"Result of {tool_name}:"})
                else:
                    from dental_agent.utils.serialization import to_jsonable
                    observation_content.append({
                        "type": "text",
                        "text": f"Result of {tool_name}: {json.dumps(to_jsonable(tool_out))}",
                    })

                call_record["tool_ok"] = True
                any_ok = True

            except Exception as e:
                call_record["tool_ok"] = False
                call_record["tool_error"] = str(e)
                observation_content.append(
                    {"type": "text", "text": f"Tool '{tool_name}' execution failed: {e}"}
                )

            calls_this_turn.append(call_record)

        turn_record["tool_calls_this_turn"] = calls_this_turn
        turn_record["status"] = "tool_executed" if any_ok else "tool_all_failed"
        turns.append(turn_record)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": observation_content})

    # A valid multi-finding answer is a non-empty list (see prompts.py: "A
    # patient may have multiple findings; your final answer must be a list
    # covering all of them"), not just a single dict -- this previously only
    # accepted a dict, meaning every correctly-formatted multi-finding answer
    # would have been scored as a format failure.
    format_ok = bool(
        final_answer is not None
        and (
            isinstance(final_answer, dict)
            or (isinstance(final_answer, list) and len(final_answer) > 0)
        )
    )

    trajectory = AgentTrajectory(
        image_id=image_id,
        turns=turns,
        tool_calls=tool_call_count,
        final_answer=final_answer,
        format_ok=format_ok,
        assistant_token_spans=assistant_token_spans,
        messages=messages,
    )

    if verbose:
        print(f"[Agent] image_id={image_id} complete: tool_calls={tool_call_count}, final={final_answer}")

    return trajectory


def run_agent_no_tools(
    image_id: int,
    images_df: pd.DataFrame,
    model: Any,
    processor: Any,
    verbose: bool = True,
) -> AgentTrajectory:
    """H1 Ablation condition: One-turn direct reasoning without tool access."""
    row = images_df[images_df["id"] == image_id].iloc[0]
    image = Image.open(row["local_path"]).convert("RGB")

    messages = [
        {"role": "system", "content": NO_TOOLS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": f"Analyze this panoramic X-ray (image_id={image_id})."},
            ],
        },
    ]

    reply, prompt_len, gen_ids = generate_agent_reply(
        model, processor, messages, return_ids=True
    )
    parsed = parse_agent_json(reply)

    final_ans = (parsed or {}).get("final_answer")
    format_ok = bool(parsed and "final_answer" in parsed)

    trajectory = AgentTrajectory(
        image_id=image_id,
        turns=[{"turn": 0, "raw_output": reply, "parsed": parsed}],
        tool_calls=0,
        final_answer=final_ans,
        format_ok=format_ok,
        assistant_token_spans=[{"prompt_len": prompt_len, "token_ids": gen_ids}],
        messages=messages,
    )

    if verbose:
        print(f"[No-Tools] image_id={image_id}: final_answer={final_ans}")

    return trajectory


def run_offline_self_tests(verbose: bool = True) -> bool:
    """Offline self-test suite covering deterministic tools, FDI math, parsing, and reward combinations."""
    from dental_agent.tools.zoom_crop import tool_zoom_crop
    from dental_agent.tools.contrast import tool_enhance_contrast
    from dental_agent.tools.fdi import tool_fdi_label, fdi_encode, fdi_decode
    from dental_agent.tools.synthetic import make_synthetic_dental_image
    from dental_agent.rewards.composite import combine_reward

    # 1. Synthetic image & tools
    img = make_synthetic_dental_image(findings=[{"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}])
    crop = tool_zoom_crop(img, [100, 100, 50, 50])
    assert crop.size[0] > 0 and crop.size[1] > 0, "zoom_crop failed"

    enhanced = tool_enhance_contrast(crop, 1.5)
    assert enhanced.size == crop.size, "enhance_contrast failed"

    # 2. FDI notation
    assert tool_fdi_label(1, 6) == "16", "fdi_label failed"
    assert fdi_encode(1, 6) == 16, "fdi_encode failed"
    assert fdi_decode(16) == (1, 6), "fdi_decode failed"

    # 3. Parsing
    sample_reply = '{"final_answer": {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}}'
    parsed = parse_agent_json(sample_reply)
    assert parsed and "final_answer" in parsed, "parse_agent_json failed"

    # 4. Rewards
    r, comp = combine_reward(
        {"final_answer": {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}, "turns": [], "tool_calls": 1, "format_ok": True},
        {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"},
    )
    assert r > 0.5, f"combine_reward expected high reward, got {r}"

    if verbose:
        print("All offline self-tests passed successfully (tools, FDI, parsing, rewards).")
    return True

