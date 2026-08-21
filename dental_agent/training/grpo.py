"""
Stage 2: Group Relative Policy Optimization (GRPO) for Tool-Augmented VLMs (§17).

Includes the full GRPO inner loop: group advantage normalization, per-token log-prob
extraction (policy + frozen reference via LoRA disable), prompt masking,
PPO-clipped policy gradient with KL penalty, training-curve logging, and plotting.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from dental_agent.config import ProjectConfig, TrainingConfig
from dental_agent.model.backbone import load_model, apply_lora
from dental_agent.model.checkpoints import save_checkpoint
from dental_agent.agent.loop import run_agent
from dental_agent.rewards.composite import combine_reward
from dental_agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# GRPO Mathematical Core
# ---------------------------------------------------------------------------

def compute_group_advantages(rewards: list[float]) -> torch.Tensor:
    """A_i = (r_i - mean(r)) / (std(r) + eps) — GRPO's group-relative advantage,
    computed once per group of rollouts sampled for the SAME prompt/image."""
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    mean, std = rewards_t.mean(), rewards_t.std(unbiased=False)
    return (rewards_t - mean) / (std + 1e-4)


def compute_token_log_probs(
    model: Any,
    enc: dict[str, torch.Tensor],
    use_reference: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs + loss mask, for either the current policy (default) or the
    frozen reference policy (use_reference=True)."""
    labels = enc["labels"]
    model_inputs = {k: v for k, v in enc.items() if k != "labels"}

    # Toggle between trainable GRPO adapter and frozen SFT reference
    if use_reference and hasattr(model, "set_adapter"):
        model.set_adapter("reference")
    elif not use_reference and hasattr(model, "set_adapter"):
        model.set_adapter("grpo_policy")

    with torch.set_grad_enabled(not use_reference):
        outputs = model(**model_inputs)

    # Revert to grpo_policy just in case
    if use_reference and hasattr(model, "set_adapter"):
        model.set_adapter("grpo_policy")

    logits = outputs.logits[:, :-1, :]
    shift_labels = labels[:, 1:].to(logits.device)
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = torch.gather(log_probs, 2, shift_labels.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    mask = (shift_labels != -100).float()
    return token_log_probs, mask


def build_full_trajectory_labels(
    trajectory: dict[str, Any],
    processor: Any,
) -> dict[str, torch.Tensor]:
    """Re-tokenize the finished conversation and unmask only the spans this policy
    actually generated (one per assistant turn), using each turn's recorded prompt_len
    as the split point. Returns encoded inputs + labels ready for a forward pass."""
    from qwen_vl_utils import process_vision_info

    messages = trajectory["messages"]
    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    full_enc = processor(text=[full_text], images=image_inputs, videos=video_inputs, return_tensors="pt")

    labels = torch.full_like(full_enc["input_ids"], -100)
    for span in trajectory.get("assistant_token_spans", []):
        start = span["prompt_len"]
        gen_ids = span["token_ids"]
        end = start + len(gen_ids)
        if end <= labels.shape[1]:
            labels[0, start:end] = torch.tensor(gen_ids, dtype=labels.dtype)
    full_enc["labels"] = labels
    return dict(full_enc)


def validate_span_alignment(
    trajectory: dict[str, Any],
    processor: Any,
    verbose: bool = True,
) -> bool:
    """Sanity check: decode the unmasked (non-100) label tokens and compare them against
    what the assistant turns actually said. Run this on a few trajectories before trusting
    the GRPO loss."""
    enc = build_full_trajectory_labels(trajectory, processor)
    labels = enc["labels"][0]
    unmasked_ids = labels[labels != -100]
    decoded = processor.tokenizer.decode(unmasked_ids, skip_special_tokens=True)
    actual = " ".join(t["raw_output"] for t in trajectory.get("turns", []))
    if verbose:
        print("Decoded from labels: ", decoded[:300])
        print("Actual turn outputs: ", actual[:300])
    return decoded.strip() == actual.strip()


# ---------------------------------------------------------------------------
# Group Rollout Collection
# ---------------------------------------------------------------------------

def collect_grpo_group(
    model: Any,
    processor: Any,
    image_id: int,
    images_df: pd.DataFrame,
    ground_truth: dict[str, Any] | list[dict[str, Any]],
    registry: ToolRegistry,
    group_size: int = 4,
    max_tool_calls: int = 50,
) -> tuple[list[dict], list[float], list[torch.Tensor], list[torch.Tensor]]:
    """Sample `group_size` rollouts for one image. Also captures each trajectory's OLD
    (pre-update) per-token log-probs under no_grad, right after generation."""
    trajectories, rewards, old_log_probs_list, masks_list = [], [], [], []
    model.eval()
    for _ in range(group_size):
        traj = run_agent(
            image_id=image_id,
            images_df=images_df,
            model=model,
            processor=processor,
            registry=registry,
            max_tool_calls=max_tool_calls,
            verbose=False,
        )
        traj_dict = traj.to_dict() if hasattr(traj, "to_dict") else traj
        # max_tool_calls passed through explicitly -- combine_reward has its own
        # default, which previously silently diverged from whatever budget the
        # rollout above actually used (reward_efficiency's reference ceiling
        # wouldn't match the real one the rollout was bounded by).
        total_reward, _ = combine_reward(traj_dict, ground_truth, max_tool_calls=max_tool_calls)
        trajectories.append(traj_dict)
        rewards.append(total_reward)

        enc = build_full_trajectory_labels(traj_dict, processor)
        enc = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in enc.items()}
        with torch.no_grad():
            old_lp, mask = compute_token_log_probs(model, enc, use_reference=False)
        old_log_probs_list.append(old_lp.detach())
        masks_list.append(mask)
        
        # Clear VRAM after each rollout forward pass
        torch.cuda.empty_cache()

    return trajectories, rewards, old_log_probs_list, masks_list


# ---------------------------------------------------------------------------
# Core GRPO Training Step
# ---------------------------------------------------------------------------

def grpo_step(
    model: Any,
    processor: Any,
    image_ids_and_gts: list[tuple[int, list[dict[str, Any]]]],
    images_df: pd.DataFrame,
    registry: ToolRegistry | None = None,
    group_size: int = 4,
    max_tool_calls: int = 50,
    lr: float = 1e-5,
    epochs_per_batch: int = 1,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
) -> dict[str, Any]:
    """One GRPO update cycle. Rollouts are collected ONCE from the current policy (GRPO's
    on-policy design); epochs_per_batch then controls how many gradient passes are taken
    over that same fixed batch, each scored against the group-normalized advantage, a
    PPO-style clipped ratio relative to the pre-update log-probs, and a KL penalty toward
    the frozen (adapter-disabled) reference policy."""
    if registry is None:
        registry = ToolRegistry.create_default()

    if not hasattr(model, "disable_adapter") and kl_beta > 0:
        print("model has no disable_adapter() (not LoRA-wrapped) — forcing kl_beta=0.")
        kl_beta = 0.0

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    # Collect once, per GRPO's on-policy design
    groups, all_rewards = [], []
    for image_id, ground_truth in image_ids_and_gts:
        trajs, rewards, old_lps, masks = collect_grpo_group(
            model, processor, image_id, images_df, ground_truth, registry, group_size, max_tool_calls,
        )
        advantages = compute_group_advantages(rewards)
        groups.append((trajs, advantages, old_lps, masks))
        all_rewards.extend(rewards)

    n_total_rollouts = len(image_ids_and_gts) * group_size

    model.train()
    for epoch in range(epochs_per_batch):
        optimizer.zero_grad()
        for trajs, advantages, old_lps, masks in groups:
            for traj, advantage, old_lp, mask in zip(trajs, advantages, old_lps, masks):
                enc = build_full_trajectory_labels(traj, processor)
                enc = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in enc.items()}
                new_lp, _ = compute_token_log_probs(model, enc, use_reference=False)

                ratio = torch.exp(new_lp - old_lp.to(model.device))
                adv = advantage.to(model.device)
                unclipped = ratio * adv
                clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv
                per_token_loss = -torch.min(unclipped, clipped)

                if kl_beta > 0:
                    with torch.no_grad():
                        ref_lp, _ = compute_token_log_probs(model, enc, use_reference=True)
                    # k3 estimator (Schulman): always >= 0, lower variance
                    log_ratio_ref = ref_lp - new_lp
                    per_token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1
                    per_token_loss = per_token_loss + kl_beta * per_token_kl

                loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1) / n_total_rollouts
                loss.backward()
                
                # Clear VRAM after processing each trajectory in the batch
                torch.cuda.empty_cache()
                
        optimizer.step()

    model.eval()
    return {
        "mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
        "n_rollouts": len(all_rewards),
        "epochs_per_batch": epochs_per_batch,
        "clip_eps": clip_eps,
        "kl_beta": kl_beta,
    }


# ---------------------------------------------------------------------------
# Training Curve Logging & Plotting
# ---------------------------------------------------------------------------

def log_grpo_step(
    stats: dict[str, Any],
    log_path: str | Path = "data/grpo_training_log.jsonl",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one grpo_step() call's stats to a persistent JSONL log."""
    from dental_agent.utils.serialization import to_jsonable

    record = {**stats, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **(extra or {})}
    os.makedirs(os.path.dirname(str(log_path)) or ".", exist_ok=True)
    with open(log_path, "a") as f:
        f.write(json.dumps(to_jsonable(record)) + "\n")
    return record


def plot_grpo_training_curve(
    log_path: str | Path = "data/grpo_training_log.jsonl",
    save_path: str | Path | None = None,
) -> pd.DataFrame | None:
    """Reward-vs-training-step curve — the figure an RL paper needs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(log_path):
        print(f"No log found at {log_path} yet — call log_grpo_step() after grpo_step() calls.")
        return None

    records = [json.loads(line) for line in open(log_path) if line.strip()]
    if not records:
        print("Log file exists but is empty.")
        return None

    df = pd.DataFrame(records)
    df["step"] = range(1, len(df) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(df["step"], df["mean_reward"], marker="o")
    plt.xlabel("grpo_step() call number")
    plt.ylabel("mean reward")
    plt.title("GRPO training progress")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curve saved to {save_path}")
    else:
        plt.show()

    print(f"{len(df)} logged calls. Latest mean_reward: {df['mean_reward'].iloc[-1]:.3f}  "
          f"(first logged: {df['mean_reward'].iloc[0]:.3f})")
    return df


# ---------------------------------------------------------------------------
# High-level train_grpo() orchestrator (kept for CLI compatibility)
# ---------------------------------------------------------------------------

def train_grpo(
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    config: ProjectConfig | TrainingConfig | None = None,
    sft_model_dir: str | Path | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    sft_checkpoint_tag: str = "sft-final",
    group_size: int | None = None,
    epochs_per_batch: int | None = None,
    kl_beta: float | None = None,
    clip_eps: float | None = None,
    learning_rate: float | None = None,
    diag_col: str = "category_id_3",
) -> str:
    """Execute Stage 2 GRPO multi-turn policy optimization with group advantage normalization."""
    from peft import PeftModel, LoraConfig
    tr_cfg = config.training if isinstance(config, ProjectConfig) else (config or TrainingConfig())
    G = group_size or tr_cfg.grpo_group_size
    lr = learning_rate or tr_cfg.grpo_lr
    beta = kl_beta if kl_beta is not None else tr_cfg.grpo_kl_beta
    eps = clip_eps or tr_cfg.grpo_clip_eps
    n_epochs = epochs_per_batch or tr_cfg.grpo_epochs_per_batch

    print(f"--- Starting Stage 2 GRPO Training (GroupSize={G}, KLBeta={beta}, ClipEps={eps}, LR={lr}) ---")

    model, processor = load_model(config)
    
    # Dual-adapter setup for GRPO reference KL penalty
    if sft_model_dir and os.path.exists(sft_model_dir):
        print(f"Loading SFT Reference Model from {sft_model_dir}...")
        model = PeftModel.from_pretrained(model, sft_model_dir, adapter_name="reference", is_trainable=False)
        # Create a new trainable adapter mimicking the SFT one
        grpo_config = LoraConfig(
            r=model.peft_config["reference"].r,
            lora_alpha=model.peft_config["reference"].lora_alpha,
            target_modules=model.peft_config["reference"].target_modules,
            lora_dropout=model.peft_config["reference"].lora_dropout,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model.add_adapter("grpo_policy", grpo_config)
        model.set_adapter("grpo_policy")
    else:
        print(f"WARNING: No SFT model found at {sft_model_dir}. Falling back to base model reference.")
        model = apply_lora(model, config)

    cat_lookup = dict(zip(categories_df["id"], categories_df["name"])) if len(categories_df) else {}
    registry = ToolRegistry.create_default()

    valid_images = images_df.dropna(subset=["local_path"])
    total_steps = len(valid_images)

    for step, (_, img_row) in enumerate(valid_images.iterrows(), start=1):
        img_id = img_row["id"]
        img_annots = annots_df[annots_df["image_id"] == img_id]
        if img_annots.empty:
            continue

        # Every annotation row for this image is a real finding -- .iloc[0]
        # previously kept only the first and silently discarded the rest, so
        # even a perfectly multi-finding-aware reward_accuracy would never
        # have seen more than one finding per image. gt is now a list.
        gt = [
            {
                "quadrant": int(row.get("category_id_1", 1)),
                "tooth_position": int(row.get("category_id_2", 1)),
                "diagnosis": cat_lookup.get(row.get(diag_col), "Caries"),
            }
            for _, row in img_annots.iterrows()
        ]

        stats = grpo_step(
            model=model,
            processor=processor,
            image_ids_and_gts=[(img_id, gt)],
            images_df=images_df,
            registry=registry,
            group_size=G,
            lr=lr,
            epochs_per_batch=n_epochs,
            clip_eps=eps,
            kl_beta=beta,
        )
        log_grpo_step(stats, extra={"step": step, "image_id": int(img_id)})
        print(f"[GRPO Step {step}/{total_steps}] mean_reward={stats['mean_reward']:.3f}")

        if step % 50 == 0 or step == total_steps:
            save_checkpoint(
                model=model, processor=processor,
                tag=f"grpo-step-{step}", checkpoint_dir=checkpoint_dir,
                extra_metadata={"step": step, "mean_reward": stats["mean_reward"]},
            )

    final_path = save_checkpoint(
        model=model, processor=processor,
        tag="grpo-final", checkpoint_dir=checkpoint_dir,
    )
    print(f"Stage 2 GRPO complete. Checkpoint saved to: {final_path}")
    return final_path
