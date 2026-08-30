"""
Ablation experiments:
- H1: Tool-augmented multi-turn loop vs. one-turn direct reasoning (§19, §21)
- H2: GRPO-tuned policy vs. SFT vs. Zero-shot / Checkpoint comparisons (§21, §22)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import pandas as pd
import torch
from tqdm import tqdm

from dental_agent.agent.loop import run_agent, run_agent_no_tools
from dental_agent.tools.registry import ToolRegistry
from dental_agent.model.backbone import load_model
from dental_agent.data.fdi_utils import row_to_fdi
from dental_agent.evaluation.metrics import (
    compute_diagnostic_metrics,
    compute_evaluation_metrics,
    bootstrap_paired_diff_ci,
)
from dental_agent.rewards.composite import combine_reward
from dental_agent.utils.serialization import to_jsonable


def _prepare_eval_samples(
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    sample_limit: int | None = None,
    diag_col: str = "category_id_3",
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    valid_images = images_df.dropna(subset=["local_path"])
    if sample_limit:
        valid_images = valid_images.head(sample_limit)

    ground_truths = []
    for _, row in valid_images.iterrows():
        img_id = row["id"]
        matches = annots_df[annots_df["image_id"] == img_id]
        if not matches.empty:
            ann = matches.iloc[0]
            quadrant, tooth_position = row_to_fdi(ann)
            ground_truths.append({
                "image_id": img_id,
                "quadrant": quadrant,
                "tooth_position": tooth_position,
                "diagnosis": str(ann.get(diag_col, "Caries")).lower(),
            })
        else:
            ground_truths.append({"image_id": img_id, "quadrant": None, "tooth_position": None, "diagnosis": ""})

    return valid_images, ground_truths


def run_h1_ablation(
    image_ids_or_df: list[int] | pd.DataFrame,
    annots_df: pd.DataFrame,
    model: Any = None,
    processor: Any = None,
    images_df: pd.DataFrame | None = None,
    sample_limit: int | None = None,
    cache_dir: str | Path | None = None,
    diag_col: str = "category_id_3",
    categories_df: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """H1: same model, with vs. without tool access on the same images."""
    if isinstance(image_ids_or_df, pd.DataFrame):
        images_df = image_ids_or_df
        image_ids = images_df.dropna(subset=["local_path"])["id"].tolist()
        if sample_limit:
            image_ids = image_ids[:sample_limit]
    else:
        image_ids = list(image_ids_or_df)
        if sample_limit:
            image_ids = image_ids[:sample_limit]

    if images_df is None:
        raise ValueError("images_df must be provided if image_ids is a list.")

    registry = ToolRegistry.create_default()
    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )

    with_tools_results = []
    without_tools_results = []

    print(f"--- Running H1 Ablation Study (With vs. Without Tools) on {len(image_ids)} images ---")

    for img_id in tqdm(image_ids, desc="H1 Evaluation"):
        anns = annots_df[annots_df["image_id"] == img_id]
        if anns.empty:
            continue
        ann0 = anns.iloc[0]
        quadrant, tooth_position = row_to_fdi(ann0)
        ground_truth = {
            "quadrant": quadrant,
            "tooth_position": tooth_position,
            "diagnosis": cat_lookup.get(ann0.get(diag_col), "Caries"),
        }

        # With tools
        t_traj = run_agent(img_id, images_df, model=model, processor=processor, registry=registry, verbose=False)
        t_dict = t_traj.to_dict() if hasattr(t_traj, "to_dict") else t_traj
        r_w, comp_w = combine_reward(t_dict, ground_truth)
        with_tools_results.append(to_jsonable({
            "image_id": img_id,
            "ground_truth": ground_truth,
            "final_answer": t_dict.get("final_answer"),
            "tool_calls": t_dict.get("tool_calls", 0),
            "format_ok": t_dict.get("format_ok", False),
            "reward": r_w,
            "reward_components": comp_w,
        }))

        # Without tools
        nt_traj = run_agent_no_tools(img_id, images_df, model=model, processor=processor, verbose=False)
        nt_dict = nt_traj.to_dict() if hasattr(nt_traj, "to_dict") else nt_traj
        r_wo, comp_wo = combine_reward(nt_dict, ground_truth)
        without_tools_results.append(to_jsonable({
            "image_id": img_id,
            "ground_truth": ground_truth,
            "final_answer": nt_dict.get("final_answer"),
            "tool_calls": 0,
            "format_ok": nt_dict.get("format_ok", False),
            "reward": r_wo,
            "reward_components": comp_wo,
        }))

    m_with = compute_evaluation_metrics(with_tools_results)
    m_without = compute_evaluation_metrics(without_tools_results)

    print("H1 ablation — same model, with vs. without tool access:")
    for key in ("fdi_accuracy", "diagnosis_balanced_accuracy", "format_compliance_rate", "mean_reward"):
        print(f"  {key:28s}  with_tools={m_with.get(key, 0.0):.3f}   without_tools={m_without.get(key, 0.0):.3f}")

    by_id_with = {r["image_id"]: r["reward"] for r in with_tools_results}
    by_id_without = {r["image_id"]: r["reward"] for r in without_tools_results}
    common_ids = sorted(set(by_id_with) & set(by_id_without))
    if len(common_ids) >= 5:
        mean_diff, (ci_lo, ci_hi) = bootstrap_paired_diff_ci(
            [by_id_with[i] for i in common_ids], [by_id_without[i] for i in common_ids]
        )
        verdict = (
            "CI excludes 0: difference unlikely due to chance."
            if (ci_lo > 0 or ci_hi < 0)
            else "CI includes 0: not yet distinguishable from no difference."
        )
        print(f"\nPaired reward difference (with - without), n={len(common_ids)}: "
              f"{mean_diff:.3f}  [95% CI: {ci_lo:.3f}, {ci_hi:.3f}]\n  -> {verdict}")

    return m_with, m_without


def compare_checkpoints(
    checkpoint_tags: list[str],
    image_ids: list[int],
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    checkpoint_dir: str | Path = "checkpoints",
    categories_df: pd.DataFrame | None = None,
    current_model: Any = None,
    current_processor: Any = None,
    diag_col: str = "category_id_3",
) -> dict[str, dict[str, Any]]:
    """Reload each named checkpoint and evaluate all of them on the same image_ids."""
    registry = ToolRegistry.create_default()
    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )

    comparison: dict[str, dict[str, Any]] = {}

    for tag in checkpoint_tags:
        if tag == "current (no reload)" and current_model is not None:
            active_model, active_processor = current_model, current_processor
        else:
            ckpt_path = os.path.join(checkpoint_dir, tag)
            if not os.path.exists(ckpt_path):
                print(f"Checkpoint {ckpt_path} does not exist, skipping.")
                continue
            active_model, active_processor = load_model(adapter_path=ckpt_path)

        results = []
        for img_id in image_ids:
            anns = annots_df[annots_df["image_id"] == img_id]
            if anns.empty:
                continue
            ann0 = anns.iloc[0]
            quadrant, tooth_position = row_to_fdi(ann0)
            ground_truth = {
                "quadrant": quadrant,
                "tooth_position": tooth_position,
                "diagnosis": cat_lookup.get(ann0.get(diag_col), "Caries"),
            }

            traj = run_agent(img_id, images_df, model=active_model, processor=active_processor, registry=registry, verbose=False)
            t_dict = traj.to_dict() if hasattr(traj, "to_dict") else traj
            r_val, comp = combine_reward(t_dict, ground_truth)
            results.append(to_jsonable({
                "image_id": img_id,
                "ground_truth": ground_truth,
                "final_answer": t_dict.get("final_answer"),
                "tool_calls": t_dict.get("tool_calls", 0),
                "format_ok": t_dict.get("format_ok", False),
                "reward": r_val,
                "reward_components": comp,
            }))

        comparison[tag] = compute_evaluation_metrics(results)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("Checkpoint comparison:")
    for tag, m in comparison.items():
        print(f"  {tag}: fdi_accuracy={m.get('fdi_accuracy', 0.0):.3f}  "
              f"balanced_accuracy={m.get('diagnosis_balanced_accuracy', 0.0):.3f}  "
              f"mean_reward={m.get('mean_reward', 0.0):.3f}")

    return comparison


def run_h2_evaluation(
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    grpo_model: Any,
    sft_model: Any,
    processor: Any,
    sample_limit: int | None = None,
) -> dict[str, Any]:
    """Execute H2 Hypothesis Test: GRPO Policy vs. SFT Baseline (legacy wrapper)."""
    eval_images, ground_truths = _prepare_eval_samples(images_df, annots_df, sample_limit)
    registry = ToolRegistry.create_default()

    print(f"--- Running H2 Comparison on {len(eval_images)} images ---")

    grpo_trajs = []
    sft_trajs = []

    for _, row in tqdm(eval_images.iterrows(), total=len(eval_images), desc="H2 Evaluation"):
        img_id = row["id"]
        grpo_traj = run_agent(img_id, images_df, grpo_model, processor, registry, verbose=False)
        sft_traj = run_agent(img_id, images_df, sft_model, processor, registry, verbose=False)

        grpo_trajs.append(grpo_traj.to_dict())
        sft_trajs.append(sft_traj.to_dict())

    grpo_metrics = compute_diagnostic_metrics(grpo_trajs, ground_truths)
    sft_metrics = compute_diagnostic_metrics(sft_trajs, ground_truths)

    return {
        "grpo": grpo_metrics,
        "sft": sft_metrics,
        "h2_delta_exact_match": grpo_metrics.get("exact_match_accuracy", 0.0) - sft_metrics.get("exact_match_accuracy", 0.0),
        "h2_delta_macro_f1": grpo_metrics.get("pathology_macro_f1", 0.0) - sft_metrics.get("pathology_macro_f1", 0.0),
    }
