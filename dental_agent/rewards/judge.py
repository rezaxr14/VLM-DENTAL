"""
LLM-as-judge evaluation for trajectory reasoning verification (R_judge in §5.5, §22, §25).

Includes:
- Single trajectory LLM judge (`reward_judge`)
- Batch reasoning grounding evaluator with disk caching (`evaluate_reasoning_grounding`)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping
from PIL import Image
import pandas as pd

from dental_agent.data.dentex import dentex_row_to_fdi

TRAJECTORY_JUDGE_SYSTEM_PROMPT = (
    "You are a strict verifier, not a rewriter. You will see an X-ray image, the KNOWN "
    "correct answer, and an agent's full reasoning + tool-call trace (not synthetic -- "
    "this is a real trajectory from a live agent). Judge ONLY whether every claim the "
    "agent makes is actually supported by the image, its own tool outputs, and the known "
    "answer -- reject anything asserted without support, even if the final answer happens "
    'to be correct. Respond with exactly one JSON object: {"grounded": true/false, "reason": "..."}.'
)


def reward_judge(
    image: Image.Image,
    ground_truth: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    call_llm_fn: Any = None,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
) -> dict[str, Any]:
    """Score reasoning hallucination / grounding rate using an external LLM judge (§22)."""
    if call_llm_fn is None:
        from dental_agent.training.api_pool import call_llm
        call_llm_fn = call_llm

    turns = trajectory.get("turns", [])
    reasoning_text = "\n".join(
        t.get("raw_output", "") for t in turns if isinstance(t, dict)
    )
    user_content = (
        f"Known correct answer: {json.dumps(dict(ground_truth))}\n\n"
        f"Agent's full reasoning + tool-call trace:\n{reasoning_text}\n\n"
        f"Agent's final answer: {json.dumps(trajectory.get('final_answer'))}"
    )

    raw = call_llm_fn(provider, model, TRAJECTORY_JUDGE_SYSTEM_PROMPT, user_content, image=image)

    from dental_agent.agent.parsing import parse_agent_json
    parsed = parse_agent_json(raw)
    return parsed if parsed else {"grounded": False, "reason": "judge output unparseable"}


def evaluate_reasoning_grounding(
    image_ids: list[int],
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    sample_n: int = 10,
    provider: str = "anthropic",
    model: str = "claude-3-5-sonnet-20241022",
    cache_path: str | Path | None = None,
    resume: bool = True,
    agent_fn: Any = None,
    agent_model: Any = None,
    agent_processor: Any = None,
    categories_df: pd.DataFrame | None = None,
    diag_col: str = "category_id_3",
) -> dict[str, Any]:
    """Runs reward_judge() on a sample of eval trajectories to score the hallucination /
    grounding rate with an external LLM verifier."""
    from dental_agent.agent.loop import run_agent
    from dental_agent.tools.registry import ToolRegistry
    from dental_agent.utils.serialization import to_jsonable

    sampled_ids = image_ids[:sample_n]
    results: list[dict[str, Any]] = []
    done_ids: set[int] = set()

    if cache_path and resume and os.path.exists(str(cache_path)):
        with open(cache_path) as f:
            results = json.load(f)
        done_ids = {r["image_id"] for r in results}
        print(f"Resuming: {len(done_ids)} judge evaluation(s) already in {cache_path}")

    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )
    registry = ToolRegistry.create_default()

    for image_id in sampled_ids:
        if image_id in done_ids:
            continue
        anns = annots_df[annots_df["image_id"] == image_id]
        if anns.empty:
            continue
        ann0 = anns.iloc[0]
        quadrant, tooth_position = dentex_row_to_fdi(ann0)
        ground_truth = {
            "quadrant": quadrant,
            "tooth_position": tooth_position,
            "diagnosis": cat_lookup.get(ann0.get(diag_col), "Caries"),
        }

        matches = images_df[images_df["id"] == image_id]
        if matches.empty:
            continue
        row = matches.iloc[0]
        image_path = row.get("local_path")
        if not image_path or not os.path.exists(str(image_path)):
            continue
        image = Image.open(image_path).convert("RGB")

        if agent_fn is not None:
            traj = agent_fn(image_id)
        else:
            traj = run_agent(
                image_id, images_df, model=agent_model, processor=agent_processor,
                registry=registry, verbose=False,
            )
        traj_dict = traj.to_dict() if hasattr(traj, "to_dict") else traj

        judge_verdict = reward_judge(image, ground_truth, traj_dict, provider=provider, model=model)
        results.append(to_jsonable({
            "image_id": image_id,
            "grounded": judge_verdict.get("grounded", False),
            "reason": judge_verdict.get("reason", ""),
        }))

        if cache_path:
            with open(cache_path, "w") as f:
                json.dump(to_jsonable(results), f, indent=2)

    grounded_count = sum(1 for r in results if r.get("grounded"))
    rate = grounded_count / max(len(results), 1)
    print(f"LLM-as-judge reasoning grounding rate ({provider}/{model}): "
          f"{rate:.1%} ({grounded_count}/{len(results)} traces judged grounded)")
    return {"grounding_rate": rate, "n_judged": len(results), "judged_results": results}
