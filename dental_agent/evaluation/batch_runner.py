"""
Batch evaluation engine for dental diagnostic agent cohorts (§20, §21, §22, §26).

Includes:
- Batch runner with disk caching and resume (`run_agent_batch`)
- Full comparative evaluation orchestrator (`run_full_evaluation_suite`)
- Legacy cohort evaluator (`evaluate_dataset`)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable, Sequence
import pandas as pd
from tqdm import tqdm

from dental_agent.agent.loop import run_agent
from dental_agent.tools.registry import ToolRegistry
from dental_agent.model.backbone import load_model
from dental_agent.model.checkpoints import load_checkpoint
from dental_agent.rewards.composite import combine_reward
from dental_agent.evaluation.metrics import (
    compute_diagnostic_metrics,
    compute_evaluation_metrics,
    compute_ece,
    bootstrap_metric_ci,
)
from dental_agent.evaluation.baselines import majority_class_baseline_metrics
from dental_agent.evaluation.ablations import run_h1_ablation, compare_checkpoints
from dental_agent.rewards.judge import evaluate_reasoning_grounding
from dental_agent.evaluation.reporting import save_results_report
from dental_agent.utils.serialization import to_jsonable


def run_agent_batch(
    image_ids: list[int],
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    agent_fn: Callable[..., Any] | None = None,
    model: Any = None,
    processor: Any = None,
    registry: ToolRegistry | None = None,
    categories_df: pd.DataFrame | None = None,
    cache_path: str | Path | None = None,
    resume: bool = True,
    diag_col: str = "category_id_3",
) -> list[dict[str, Any]]:
    """Runs the agent across a list of image_ids, computes reward against ground truth,
    and caches each result to disk as it completes (resume=True avoids re-running)."""
    results: list[dict[str, Any]] = []
    done_ids: set[int] = set()

    if cache_path and resume and os.path.exists(str(cache_path)):
        with open(cache_path) as f:
            results = json.load(f)
        done_ids = {r["image_id"] for r in results}
        print(f"Resuming: {len(done_ids)} image(s) already in {cache_path}")

    if registry is None:
        registry = ToolRegistry.create_default()

    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )

    for image_id in tqdm(image_ids, desc="Running agent batch"):
        if image_id in done_ids:
            continue
        anns = annots_df[annots_df["image_id"] == image_id]
        if anns.empty:
            continue
        ann0 = anns.iloc[0]
        ground_truth = {
            "quadrant": int(ann0.get("category_id_1", 1)),
            "tooth_position": int(ann0.get("category_id_2", 1)),
            "diagnosis": cat_lookup.get(ann0.get(diag_col), "Caries"),
        }

        if agent_fn is not None:
            traj = agent_fn(image_id)
        else:
            traj = run_agent(
                image_id, images_df, model=model, processor=processor,
                registry=registry, verbose=False,
            )

        traj_dict = traj.to_dict() if hasattr(traj, "to_dict") else traj
        reward_val, components = combine_reward(traj_dict, ground_truth)

        results.append(to_jsonable({
            "image_id": image_id,
            "ground_truth": ground_truth,
            "final_answer": traj_dict.get("final_answer"),
            "tool_calls": traj_dict.get("tool_calls", 0),
            "format_ok": traj_dict.get("format_ok", False),
            "reward": reward_val,
            "reward_components": components,
        }))

        if cache_path:
            with open(cache_path, "w") as f:
                json.dump(to_jsonable(results), f, indent=2)

    return results


def run_full_evaluation_suite(
    image_ids: list[int],
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame | None = None,
    checkpoint_tags: Sequence[str] = ("current (no reload)",),
    run_judge: bool = True,
    judge_sample_n: int = 5,
    holdout_ids: set[int] | None = None,
    current_model: Any = None,
    current_processor: Any = None,
    diag_col: str = "category_id_3",
) -> dict[str, dict[str, Any]]:
    """Runs every evaluation in the project over the SAME holdout cohort and packages
    all numbers into a single dict ready for reporting."""
    suite_results: dict[str, dict[str, Any]] = {}

    # 1. Majority class baseline
    print("\n--- 1. Majority-class baseline ---")
    suite_results["majority_class"] = majority_class_baseline_metrics(
        image_ids, annots_df, holdout_ids=holdout_ids, categories_df=categories_df, diag_col=diag_col
    )

    # 2. Checkpoint comparisons
    print("\n--- 2. Checkpoint comparisons ---")
    comparisons = compare_checkpoints(
        list(checkpoint_tags), image_ids, images_df, annots_df,
        categories_df=categories_df, current_model=current_model,
        current_processor=current_processor, diag_col=diag_col,
    )
    suite_results.update(comparisons)

    # 3. H1 Ablation (with vs without tools)
    print("\n--- 3. H1 Ablation (with vs without tools) ---")
    m_with, m_without = run_h1_ablation(
        image_ids, annots_df, model=current_model, processor=current_processor,
        images_df=images_df, categories_df=categories_df, diag_col=diag_col,
    )
    suite_results["h1_with_tools"] = m_with
    suite_results["h1_without_tools"] = m_without

    # 4. LLM-as-judge reasoning grounding
    if run_judge:
        print("\n--- 4. LLM-as-judge reasoning grounding ---")
        judge_res = evaluate_reasoning_grounding(
            image_ids, images_df, annots_df, sample_n=judge_sample_n,
            agent_model=current_model, agent_processor=current_processor,
            categories_df=categories_df, diag_col=diag_col,
        )
        suite_results["judge_grounding"] = judge_res

    # 5. Save report
    save_results_report(suite_results)
    return suite_results


def evaluate_dataset(
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    model: Any = None,
    processor: Any = None,
    checkpoint_tag: str | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    sample_limit: int | None = None,
    diag_col: str = "category_id_3",
    registry: ToolRegistry | None = None,
) -> dict[str, Any]:
    """Evaluate agent across a cohort of dental X-rays (legacy interface)."""
    if model is None and checkpoint_tag is not None:
        model, processor = load_checkpoint(checkpoint_dir, checkpoint_tag)
    elif model is None:
        model, processor = load_model()

    registry = registry or ToolRegistry.create_default()
    valid_images = images_df.dropna(subset=["local_path"])
    if sample_limit:
        valid_images = valid_images.head(sample_limit)

    trajectories = []
    ground_truths = []
    confidences = []
    accuracies = []

    for _, row in tqdm(valid_images.iterrows(), total=len(valid_images), desc="Evaluating cohort"):
        img_id = row["id"]
        matches = annots_df[annots_df["image_id"] == img_id]
        if not matches.empty:
            ann = matches.iloc[0]
            gt = {
                "quadrant": int(ann.get("category_id_1", 1)),
                "tooth_position": int(ann.get("category_id_2", 1)),
                "diagnosis": str(ann.get(diag_col, "Caries")).lower(),
            }
        else:
            gt = {"quadrant": None, "tooth_position": None, "diagnosis": ""}

        traj = run_agent(img_id, images_df, model=model, processor=processor, registry=registry, verbose=False)
        traj_dict = traj.to_dict() if hasattr(traj, "to_dict") else traj
        trajectories.append(traj_dict)
        ground_truths.append(gt)

        # Calibration tracking
        ans = traj_dict.get("final_answer") or {}
        conf = float(ans.get("confidence", 0.5)) if isinstance(ans, dict) else 0.5
        d_ok = int(str(ans.get("diagnosis", "")).lower() == gt["diagnosis"]) if gt["diagnosis"] else 0
        confidences.append(conf)
        accuracies.append(d_ok)

    metrics = compute_diagnostic_metrics(trajectories, ground_truths)
    ece = compute_ece(confidences, accuracies)
    metrics["ece"] = ece

    point_em, em_low, em_high = bootstrap_metric_ci(
        list(zip(trajectories, ground_truths)),
        lambda pairs: compute_diagnostic_metrics([p[0] for p in pairs], [p[1] for p in pairs]).get("exact_match_accuracy", 0.0),
    )
    metrics["exact_match_ci_95"] = [em_low, em_high]

    return metrics
