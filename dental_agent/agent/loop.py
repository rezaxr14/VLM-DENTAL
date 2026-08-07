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
from dental_agent.model.inference import generate_agent_reply
from dental_agent.tools.registry import ToolRegistry


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
    max_tool_calls: int = 4,
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

    current_image = base_image
    turns: list[dict[str, Any]] = []
    assistant_token_spans: list[dict[str, Any]] = []
    final_answer = None
    tool_call_count = 0

    for turn_idx in range(max_tool_calls + 1):
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

        # Check for tool call
        tool_name = parsed.get("tool")
        tool_args = parsed.get("args", {})

        if not tool_name or tool_name not in [t.name for t in registry.list_tools()]:
            turn_record["status"] = "invalid_tool"
            turn_record["tool_ok"] = False
            turns.append(turn_record)
            messages.append({"role": "assistant", "content": reply})
            messages.append({
                "role": "user",
                "content": f"Error: Tool '{tool_name}' is not recognized.",
            })
            continue

        # Execute registered tool
        tool_call_count += 1
        turn_record["tool_name"] = tool_name
        turn_record["tool_args"] = tool_args

        try:
            if tool_name == "zoom_crop":
                tool_out = registry.execute("zoom_crop", image=current_image, **tool_args)
                current_image = tool_out
                observation_content = [
                    {"type": "image", "image": tool_out},
                    {"type": "text", "text": f"Result of zoom_crop around {tool_args.get('bbox')}:"},
                ]
            elif tool_name == "enhance_contrast":
                tool_out = registry.execute("enhance_contrast", image=current_image, **tool_args)
                current_image = tool_out
                observation_content = [
                    {"type": "image", "image": tool_out},
                    {"type": "text", "text": "Result of enhance_contrast:"},
                ]
            elif tool_name == "locate_abnormal_teeth":
                tool_out = registry.execute("locate_abnormal_teeth", image_id=image_id)
                observation_content = [
                    {"type": "text", "text": f"Found candidate abnormal teeth: {json.dumps(tool_out)}"}
                ]
            else:
                tool_out = registry.execute(tool_name, **tool_args)
                observation_content = [
                    {"type": "text", "text": f"Tool output: {json.dumps(tool_out)}"}
                ]

            turn_record["tool_ok"] = True
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": observation_content})

        except Exception as e:
            turn_record["tool_ok"] = False
            turn_record["tool_error"] = str(e)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Tool execution failed: {e}"})

        turns.append(turn_record)

    format_ok = bool(final_answer is not None and isinstance(final_answer, dict))

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

