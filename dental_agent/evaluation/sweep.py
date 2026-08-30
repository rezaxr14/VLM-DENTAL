"""
Hyperparameter sweep runner for GRPO group size, KL beta, and reward weights (§23, §26).

Includes:
- Post-hoc reward weight sweep (`sweep_reward_weights`, `DEFAULT_WEIGHT_GRID`)
- Full training hyperparameter grid sweep (`run_hyperparameter_sweep`)
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any, Callable
import pandas as pd

from dental_agent.config import ProjectConfig, TrainingConfig, RewardWeights
from dental_agent.training.grpo import train_grpo
from dental_agent.agent.loop import run_agent
from dental_agent.data.fdi_utils import row_to_fdi
from dental_agent.rewards.composite import combine_reward
from dental_agent.tools.registry import ToolRegistry


DEFAULT_WEIGHT_GRID: list[dict[str, float]] = [
    {"acc": 1.0, "fmt": 0.2, "tool": 0.2, "eff": 0.1},   # Proposal default (§5.5)
    {"acc": 1.0, "fmt": 0.0, "tool": 0.0, "eff": 0.0},   # Accuracy only
    {"acc": 1.0, "fmt": 0.1, "tool": 0.4, "eff": 0.1},   # Weight tool-use heavily
    {"acc": 1.0, "fmt": 0.4, "tool": 0.1, "eff": 0.1},   # Weight format compliance heavily
    {"acc": 0.7, "fmt": 0.2, "tool": 0.2, "eff": 0.1},   # De-emphasize accuracy slightly
]


def sweep_reward_weights(
    image_ids: list[int],
    weight_grid: list[dict[str, float]] | None = None,
    images_df: pd.DataFrame | None = None,
    annots_df: pd.DataFrame | None = None,
    categories_df: pd.DataFrame | None = None,
    agent_fn: Callable[..., Any] | None = None,
    model: Any = None,
    processor: Any = None,
    diag_col: str = "category_id_3",
) -> pd.DataFrame:
    """Collects trajectories once, then rescores them under each weights dict in
    weight_grid via combine_reward — only weighting changes per row."""
    if weight_grid is None:
        weight_grid = DEFAULT_WEIGHT_GRID

    registry = ToolRegistry.create_default()
    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )

    trajectories_and_gts = []
    for image_id in image_ids:
        if annots_df is not None:
            anns = annots_df[annots_df["image_id"] == image_id]
            if anns.empty:
                continue
            # Every row is a real finding -- .iloc[0] previously kept only the
            # first and discarded the rest (same root cause fixed in
            # train_grpo). ground_truth is now a list.
            #
            # dentex_row_to_fdi applies the DENTEX 0-index quirk conversion --
            # this eval sweep previously had the identical bug (raw 0-indexed
            # category_id_1/category_id_2 used directly against a model
            # trained on trace_gen's 1-indexed FDI convention), meaning
            # quadrant+position accuracy (0.50 of R_accuracy) would have
            # scored wrong for a correct answer in every evaluation run using
            # this function, not just training.
            ground_truth = [
                {
                    "quadrant": row_to_fdi(row)[0],
                    "tooth_position": row_to_fdi(row)[1],
                    "diagnosis": cat_lookup.get(row.get(diag_col), "Caries"),
                }
                for _, row in anns.iterrows()
            ]
        else:
            ground_truth = [{"quadrant": 1, "tooth_position": 1, "diagnosis": "Caries"}]

        if agent_fn is not None:
            traj = agent_fn(image_id)
        elif images_df is not None:
            traj = run_agent(image_id, images_df, model=model, processor=processor, registry=registry, verbose=False)
        else:
            continue

        traj_dict = traj.to_dict() if hasattr(traj, "to_dict") else traj
        trajectories_and_gts.append((traj_dict, ground_truth))

    rows = []
    for weights in weight_grid:
        rw = RewardWeights(
            accuracy=weights.get("acc", 1.0),
            format_adherence=weights.get("fmt", 0.2),
            tool_validity=weights.get("tool", 0.2),
            efficiency=weights.get("eff", 0.1),
        )
        totals = [combine_reward(traj, gt, weights=rw)[0] for traj, gt in trajectories_and_gts]
        mean_r = sum(totals) / max(len(totals), 1)
        rows.append({**weights, "mean_reward": mean_r, "n": len(totals)})

    return pd.DataFrame(rows)


def run_hyperparameter_sweep(
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    holdout_images_df: pd.DataFrame,
    group_sizes: list[int] | None = None,
    kl_betas: list[float] | None = None,
    output_dir: str | Path = "experiments/sweep",
) -> list[dict[str, Any]]:
    """Execute grid sweep over GRPO hyperparameter configurations."""
    from dental_agent.evaluation.batch_runner import evaluate_dataset

    group_sizes = group_sizes or [2, 4, 8]
    kl_betas = kl_betas or [0.01, 0.04, 0.1]
    grid = list(itertools.product(group_sizes, kl_betas))

    results = []
    print(f"--- Starting GRPO Hyperparameter Sweep ({len(grid)} configurations) ---")

    for G, beta in grid:
        tag = f"sweep_G{G}_beta{beta}"
        print(f"\n[Sweep] Running configuration: GroupSize={G}, KLBeta={beta} -> tag={tag}")

        ckpt_path = train_grpo(
            images_df=images_df,
            annots_df=annots_df,
            categories_df=categories_df,
            group_size=G,
            kl_beta=beta,
            checkpoint_dir=Path(output_dir) / "checkpoints",
        )

        metrics = evaluate_dataset(
            images_df=holdout_images_df,
            annots_df=annots_df,
            checkpoint_tag=tag,
            checkpoint_dir=Path(output_dir) / "checkpoints",
            sample_limit=20,
        )

        record = {
            "group_size": G,
            "kl_beta": beta,
            "checkpoint_path": ckpt_path,
            "metrics": metrics,
        }
        results.append(record)

    out_file = Path(output_dir) / "sweep_results.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSweep complete! Summary saved to {out_file}")
    return results
