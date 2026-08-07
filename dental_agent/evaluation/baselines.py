"""
Baseline evaluation runners: Majority-class floor and Zero-shot commercial VLMs (§20).

Includes:
- Majority class baseline (`majority_class_baseline_metrics`)
- Zero-shot API baseline with incremental caching (`run_zero_shot_baseline`, `run_zeroshot_baseline`)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from PIL import Image
import pandas as pd
from tqdm import tqdm

from dental_agent.agent.prompts import ZERO_SHOT_PROMPT
from dental_agent.agent.parsing import parse_agent_json
from dental_agent.training.api_pool import call_llm
from dental_agent.rewards.components import reward_accuracy
from dental_agent.evaluation.metrics import compute_evaluation_metrics
from dental_agent.utils.serialization import to_jsonable


def majority_class_baseline_metrics(
    holdout_image_ids: list[int],
    annots_df: pd.DataFrame,
    holdout_ids: set[int] | None = None,
    categories_df: pd.DataFrame | None = None,
    diag_col: str = "category_id_3",
) -> dict[str, Any]:
    """Always predicts the single most common quadrant, tooth position, and diagnosis
    (measured on the training pool) — a naive floor for evaluation metrics."""
    if holdout_ids is None:
        holdout_ids = set(holdout_image_ids)

    train_annots = annots_df[~annots_df["image_id"].isin(holdout_ids)]
    if train_annots.empty:
        train_annots = annots_df

    majority_quadrant = int(train_annots["category_id_1"].mode().iloc[0]) if "category_id_1" in train_annots else 1
    majority_tooth = int(train_annots["category_id_2"].mode().iloc[0]) if "category_id_2" in train_annots else 1

    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )
    if diag_col in train_annots and not train_annots[diag_col].dropna().empty:
        majority_diag_id = train_annots[diag_col].mode().iloc[0]
        majority_diag = cat_lookup.get(majority_diag_id, str(majority_diag_id))
    else:
        majority_diag = "Caries"

    fake_results = []
    for image_id in holdout_image_ids:
        anns = annots_df[annots_df["image_id"] == image_id]
        if anns.empty:
            continue
        ann0 = anns.iloc[0]
        gt = {
            "quadrant": int(ann0.get("category_id_1", 1)),
            "tooth_position": int(ann0.get("category_id_2", 1)),
            "diagnosis": cat_lookup.get(ann0.get(diag_col), "Caries"),
        }
        fake_results.append(to_jsonable({
            "image_id": image_id,
            "ground_truth": gt,
            "final_answer": {
                "quadrant": majority_quadrant,
                "tooth_position": majority_tooth,
                "diagnosis": majority_diag,
                "confidence": 1.0,
            },
            "tool_calls": 0,
            "format_ok": True,
            "reward": 0.0,
            "reward_components": {},
        }))

    metrics = compute_evaluation_metrics(fake_results)
    print(f"Majority-class baseline (always predicts quadrant={majority_quadrant}, "
          f"tooth_position={majority_tooth}, diagnosis={majority_diag!r}):")
    print(f"  fdi_accuracy={metrics.get('fdi_accuracy', 0.0):.3f}  "
          f"balanced_accuracy={metrics.get('diagnosis_balanced_accuracy', 0.0):.3f}")
    return metrics


def run_zero_shot_baseline(
    image_ids: list[int],
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame | None = None,
    provider: str = "openai",
    model: str = "gpt-4o",
    cache_path: str | Path | None = None,
    resume: bool = True,
    diag_col: str = "category_id_3",
) -> list[dict[str, Any]]:
    """No fine-tuning, no tool access, single pass — zero-shot commercial VLM evaluation."""
    results: list[dict[str, Any]] = []
    done_ids: set[int] = set()

    if cache_path and resume and os.path.exists(str(cache_path)):
        with open(cache_path) as f:
            results = json.load(f)
        done_ids = {r["image_id"] for r in results}
        print(f"Resuming: {len(done_ids)} image(s) already processed in {cache_path}")

    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )

    for image_id in image_ids:
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

        matches = images_df[images_df["id"] == image_id]
        if matches.empty:
            continue
        row = matches.iloc[0]
        image_path = row.get("local_path")
        if not image_path or not os.path.exists(str(image_path)):
            continue
        image = Image.open(image_path).convert("RGB")

        raw = call_llm(provider, model, ZERO_SHOT_PROMPT, "Analyze this X-ray.", image=image)
        parsed = parse_agent_json(raw)

        reward_val = reward_accuracy({"final_answer": parsed}, ground_truth) if parsed else 0.0
        results.append(to_jsonable({
            "image_id": image_id,
            "ground_truth": ground_truth,
            "final_answer": parsed,
            "tool_calls": 0,
            "format_ok": parsed is not None,
            "reward": reward_val,
            "reward_components": {},
        }))

        if cache_path:
            with open(cache_path, "w") as f:
                json.dump(to_jsonable(results), f, indent=2)

    return results


def run_zeroshot_baseline(
    images_df: pd.DataFrame,
    provider: str = "openai",
    model: str = "gpt-4o",
    sample_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Legacy wrapper for running zero-shot on an images dataframe."""
    eval_images = images_df.dropna(subset=["local_path"])
    if sample_limit:
        eval_images = eval_images.head(sample_limit)

    results = []
    for _, row in tqdm(eval_images.iterrows(), total=len(eval_images), desc=f"ZeroShot {model}"):
        img_id = row["id"]
        image = Image.open(row["local_path"]).convert("RGB")
        try:
            raw_reply = call_llm(
                provider=provider,
                model=model,
                system_prompt="You are a dental radiologist.",
                user_content=ZERO_SHOT_PROMPT,
                image=image,
                temperature=0.0,
            )
            parsed = parse_agent_json(raw_reply)
        except Exception as e:
            raw_reply = f"Error: {e}"
            parsed = None

        results.append({
            "image_id": img_id,
            "raw_output": raw_reply,
            "final_answer": parsed,
            "tool_calls": 0,
            "format_ok": bool(parsed and "diagnosis" in parsed),
        })

    return results
