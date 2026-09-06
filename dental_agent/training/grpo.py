"""
Stage 2: Group Relative Policy Optimization (GRPO) for Tool-Augmented VLMs (§17).

Production implementation supporting:
- [G2] Batched Rollout Generation: One-pass batched sampling closing the tok/s decode gap
- [G3] Dual-LoRA Adapter Toggle: "reference" (frozen SFT) vs "grpo_policy" (trainable RL)
- Flexible Group Size: K in {1, 2, 4, 8, 16} with EMA fallback for K=1 and tie-breaking for K=2
- Multi-Finding Complete Ground Truth Bipartite Matching (Rule 13)
- Cross-Turn KV-Cache Reuse with 3D MRoPE coordinate slicing
- Action-Only Policy Gradients with Schulman k3 unbiased KL penalty
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import Any, Tuple, List, Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from dental_agent.config import ProjectConfig, TrainingConfig
from dental_agent.model.backbone import load_model, apply_lora
from dental_agent.model.checkpoints import save_checkpoint
from dental_agent.agent.loop import run_agent, run_agent_no_tools, AgentTrajectory
from dental_agent.agent.prompts import NO_TOOLS_SYSTEM_PROMPT, build_agent_system_prompt
from dental_agent.agent.parsing import parse_agent_json
from dental_agent.data.fdi_utils import row_to_fdi
from dental_agent.rewards.composite import combine_reward
from dental_agent.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# GRPO Mathematical Core & Group Advantage Normalization
# ---------------------------------------------------------------------------

def compute_group_advantages(
    rewards: list[float],
    running_ema_baseline: float = 0.0,
    beta: float = 0.95,
) -> tuple[torch.Tensor, float]:
    """A_i = (r_i - mean(r)) / (std(r) + eps) with numerical stabilization across K in {1, 2, 4, 8, 16}.

    Parameters
    ----------
    rewards : list[float]
        The rewards for K rollouts sampled for the same prompt/image.
    running_ema_baseline : float
        Running baseline maintained for K=1 REINFORCE updates.
    beta : float
        EMA decay rate for running baseline (default 0.95).

    Returns
    -------
    tuple[torch.Tensor, float]
        (advantages, updated_running_ema_baseline)
    """
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    k = len(rewards)

    if k <= 1:
        # K = 1 Degeneracy: Fall back to REINFORCE with EMA running baseline
        adv = rewards_t - running_ema_baseline
        new_baseline = beta * running_ema_baseline + (1.0 - beta) * float(rewards_t.mean())
        return adv, new_baseline

    mean = rewards_t.mean()
    std = rewards_t.std(unbiased=False)

    if std < 1e-6:
        # Uninformative tie: clamp advantages to 0.0 to prevent noisy gradient updates
        adv = torch.zeros_like(rewards_t)
    else:
        adv = (rewards_t - mean) / (std + 1e-4)

    new_baseline = beta * running_ema_baseline + (1.0 - beta) * float(mean)
    return adv, new_baseline


def compute_token_log_probs(
    model: Any,
    enc: dict[str, torch.Tensor],
    use_reference: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token log-probs + loss mask for current policy vs frozen reference policy.

    Toggles between 'reference' (frozen SFT adapter) and 'grpo_policy' (trainable RL adapter).
    """
    labels = enc["labels"]
    model_inputs = {k: v for k, v in enc.items() if k != "labels"}

    # Toggle dual-adapter mechanism
    if use_reference:
        if hasattr(model, "set_adapter"):
            model.set_adapter("reference")
        elif hasattr(model, "disable_adapter"):
            model.disable_adapter()
    else:
        if hasattr(model, "set_adapter"):
            model.set_adapter("grpo_policy")
        elif hasattr(model, "enable_adapter"):
            model.enable_adapter()

    with torch.set_grad_enabled(not use_reference):
        outputs = model(**model_inputs)

    # Revert to grpo_policy just in case
    if use_reference:
        if hasattr(model, "set_adapter"):
            model.set_adapter("grpo_policy")
        elif hasattr(model, "enable_adapter"):
            model.enable_adapter()

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
    """Re-tokenize finished trajectory and unmask exclusively the assistant generated spans."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        def process_vision_info(msgs):
            return None, None

    messages = trajectory.get("messages", [])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = process_vision_info(messages)
    full_enc = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")

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
    """Sanity check: decode the unmasked (non -100) label tokens and compare them against
    what the assistant turns actually said."""
    enc = build_full_trajectory_labels(trajectory, processor)
    labels = enc["labels"][0]
    unmasked_ids = labels[labels != -100]
    decoded = processor.tokenizer.decode(unmasked_ids, skip_special_tokens=True)
    actual = " ".join(t.get("raw_output", "") for t in trajectory.get("turns", []))
    if verbose:
        print("Decoded from labels: ", decoded[:300])
        print("Actual turn outputs: ", actual[:300])
    return decoded.strip() == actual.strip()



# ---------------------------------------------------------------------------
# [G2] Batched Rollout Collection
# ---------------------------------------------------------------------------

def collect_grpo_group_batched_no_tools(
    model: Any,
    processor: Any,
    image_id: int,
    images_df: pd.DataFrame,
    ground_truth: list[dict[str, Any]],
    group_size: int = 4,
    temperature: float = 0.7,
) -> tuple[list[dict], list[float], list[torch.Tensor], list[torch.Tensor]]:
    """Sample K candidate trajectories simultaneously in ONE batched forward pass (Track B)."""
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError:
        def process_vision_info(msgs):
            return None, None

    row = images_df[images_df["id"] == image_id].iloc[0]
    base_image = Image.open(row["local_path"]).convert("RGB")

    prompt_messages = [
        {"role": "system", "content": NO_TOOLS_SYSTEM_PROMPT},
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

    text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(prompt_messages)

    single_enc = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    prompt_len = single_enc["input_ids"].shape[1]

    # Batch K copies together across batch dimension
    batched_inputs = {
        k: v.repeat_interleave(group_size, dim=0).to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in single_enc.items()
    }

    gen_kwargs = {
        "max_new_tokens": 1024,
        "do_sample": True,
        "temperature": temperature,
        "pad_token_id": getattr(processor.tokenizer, "pad_token_id", None) or getattr(processor.tokenizer, "eos_token_id", None),
    }

    model.eval()
    with torch.no_grad():
        gen_out = model.generate(**batched_inputs, **gen_kwargs)

    trajectories, rewards, old_log_probs_list, masks_list = [], [], [], []

    for k_idx in range(group_size):
        seq = gen_out[k_idx : k_idx + 1]
        new_ids = seq[:, prompt_len:]
        reply = processor.batch_decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        parsed = parse_agent_json(reply)
        final_answer = parsed.get("final_answer") if parsed else None

        traj_messages = list(prompt_messages)
        traj_messages.append({"role": "assistant", "content": reply})

        traj_dict = {
            "image_id": image_id,
            "turns": [{"turn": 0, "raw_output": reply, "parsed": parsed}],
            "tool_calls": 0,
            "final_answer": final_answer,
            "format_ok": bool(final_answer is not None),
            "assistant_token_spans": [{"prompt_len": prompt_len, "token_ids": new_ids[0].tolist()}],
            "messages": traj_messages,
        }

        # Track B reward excludes tool efficiency penalties
        reward, _ = combine_reward(traj_dict, ground_truth, max_tool_calls=0)
        trajectories.append(traj_dict)
        rewards.append(reward)

        enc = build_full_trajectory_labels(traj_dict, processor)
        enc = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in enc.items()}
        with torch.no_grad():
            old_lp, mask = compute_token_log_probs(model, enc, use_reference=False)
        old_log_probs_list.append(old_lp.detach())
        masks_list.append(mask)

    return trajectories, rewards, old_log_probs_list, masks_list


def collect_grpo_group_batched_with_tools(
    model: Any,
    processor: Any,
    image_id: int,
    images_df: pd.DataFrame,
    ground_truth: list[dict[str, Any]],
    registry: ToolRegistry,
    group_size: int = 4,
    max_tool_calls: int = 50,
) -> tuple[list[dict], list[float], list[torch.Tensor], list[torch.Tensor]]:
    """Sample K candidate multi-turn trajectories with workstation tools (Track A)."""
    trajectories, rewards, old_log_probs_list, masks_list = [], [], [], []
    model.eval()

    # Roll out K trajectories with KV-cache reuse enabled
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
        reward, _ = combine_reward(traj_dict, ground_truth, max_tool_calls=max_tool_calls)

        trajectories.append(traj_dict)
        rewards.append(reward)

        enc = build_full_trajectory_labels(traj_dict, processor)
        enc = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in enc.items()}
        with torch.no_grad():
            old_lp, mask = compute_token_log_probs(model, enc, use_reference=False)
        old_log_probs_list.append(old_lp.detach())
        masks_list.append(mask)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return trajectories, rewards, old_log_probs_list, masks_list


def collect_grpo_group(
    model: Any,
    processor: Any,
    image_id: int,
    images_df: pd.DataFrame,
    ground_truth: list[dict[str, Any]],
    registry: ToolRegistry | None = None,
    group_size: int = 4,
    max_tool_calls: int = 50,
    track: str = "with_tools",
) -> tuple[list[dict], list[float], list[torch.Tensor], list[torch.Tensor]]:
    """Unified group rollout entrypoint supporting Track A and Track B."""
    if track == "no_tools":
        return collect_grpo_group_batched_no_tools(
            model=model,
            processor=processor,
            image_id=image_id,
            images_df=images_df,
            ground_truth=ground_truth,
            group_size=group_size,
        )
    else:
        if registry is None:
            registry = ToolRegistry.create_default()
        return collect_grpo_group_batched_with_tools(
            model=model,
            processor=processor,
            image_id=image_id,
            images_df=images_df,
            ground_truth=ground_truth,
            registry=registry,
            group_size=group_size,
            max_tool_calls=max_tool_calls,
        )


# ---------------------------------------------------------------------------
# Core GRPO Training Step with Schulman k3 Penalty & Advantage Normalization
# ---------------------------------------------------------------------------

def grpo_step(
    model: Any,
    processor: Any,
    image_ids_and_gts: list[tuple[int, list[dict[str, Any]]]],
    images_df: pd.DataFrame,
    registry: ToolRegistry | None = None,
    group_size: int = 4,
    max_tool_calls: int = 50,
    track: str = "with_tools",
    lr: float = 5e-6,
    epochs_per_batch: int = 2,
    clip_eps: float = 0.2,
    kl_beta: float = 0.04,
    running_ema_baseline: float = 0.0,
) -> tuple[dict[str, Any], float]:
    """One GRPO update cycle over on-policy sampled groups."""
    if registry is None:
        registry = ToolRegistry.create_default()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    groups, all_rewards = [], []
    current_baseline = running_ema_baseline

    # 1. On-policy rollout collection
    for image_id, ground_truth in image_ids_and_gts:
        trajs, rewards, old_lps, masks = collect_grpo_group(
            model=model,
            processor=processor,
            image_id=image_id,
            images_df=images_df,
            ground_truth=ground_truth,
            registry=registry,
            group_size=group_size,
            max_tool_calls=max_tool_calls,
            track=track,
        )
        advantages, current_baseline = compute_group_advantages(
            rewards=rewards,
            running_ema_baseline=current_baseline,
        )
        groups.append((trajs, advantages, old_lps, masks))
        all_rewards.extend(rewards)

    n_total_rollouts = len(image_ids_and_gts) * group_size
    model.train()

    total_kl = 0.0
    kl_count = 0

    # 2. Optimization passes over fixed rollout batch
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
                clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
                per_token_loss = -torch.min(unclipped, clipped)

                if kl_beta > 0:
                    with torch.no_grad():
                        ref_lp, _ = compute_token_log_probs(model, enc, use_reference=True)
                    # Schulman k3 estimator: strictly non-negative with lower variance
                    log_ratio_ref = ref_lp - new_lp
                    per_token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
                    per_token_loss = per_token_loss + kl_beta * per_token_kl

                    valid_kl = (per_token_kl * mask).sum() / mask.sum().clamp(min=1)
                    total_kl += float(valid_kl.item())
                    kl_count += 1

                loss = (per_token_loss * mask).sum() / mask.sum().clamp(min=1) / n_total_rollouts
                loss.backward()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        optimizer.step()

    model.eval()
    mean_kl = total_kl / max(kl_count, 1)
    stats = {
        "mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
        "kl_divergence": mean_kl,
        "n_rollouts": len(all_rewards),
        "group_size": group_size,
        "track": track,
    }
    return stats, current_baseline


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
    """Reward-vs-training-step curve for publications and reporting."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not os.path.exists(log_path):
        print(f"No log found at {log_path} yet.")
        return None

    records = [json.loads(line) for line in open(log_path) if line.strip()]
    if not records:
        return None

    df = pd.DataFrame(records)
    df["step"] = range(1, len(df) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(df["step"], df["mean_reward"], marker="o", color="#2ca02c")
    plt.xlabel("grpo_step() call number")
    plt.ylabel("mean reward")
    plt.title("GRPO Training Progress")
    plt.grid(alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Training curve saved to {save_path}")
    else:
        plt.show()

    return df


# ---------------------------------------------------------------------------
# High-Level train_grpo() Orchestrator
# ---------------------------------------------------------------------------

def train_grpo(
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    config: ProjectConfig | TrainingConfig | None = None,
    sft_model_dir: str | Path | None = None,
    checkpoint_dir: str | Path = "data/models",
    sft_checkpoint_tag: str = "sft-final",
    group_size: int | None = None,
    epochs_per_batch: int | None = None,
    kl_beta: float | None = None,
    clip_eps: float | None = None,
    learning_rate: float | None = None,
    diag_col: str = "category_id_3",
    track: str = "with_tools",
    hf_repo: str | None = None,
    push_every_steps: int = 25,
) -> str:
    """Execute Stage 2 GRPO policy optimization with dual-adapter reference and group advantage normalization."""
    from peft import PeftModel, LoraConfig
    tr_cfg = config.training if isinstance(config, ProjectConfig) else (config or TrainingConfig())
    G = group_size or tr_cfg.grpo_group_size
    lr = learning_rate or tr_cfg.grpo_lr
    beta = kl_beta if kl_beta is not None else tr_cfg.grpo_kl_beta
    eps = clip_eps or tr_cfg.grpo_clip_eps
    n_epochs = epochs_per_batch or tr_cfg.grpo_epochs_per_batch

    print(f"--- Starting Stage 2 GRPO Training (Track={track}, GroupSize={G}, KLBeta={beta}, LR={lr}) ---")

    model, processor = load_model(config)

    # Dual-adapter setup
    if sft_model_dir and os.path.exists(sft_model_dir):
        print(f"Loading SFT Reference Model from {sft_model_dir}...")
        model = PeftModel.from_pretrained(model, sft_model_dir, adapter_name="reference", is_trainable=False)
        grpo_config = LoraConfig(
            r=model.peft_config["reference"].r,
            lora_alpha=model.peft_config["reference"].lora_alpha,
            target_modules=model.peft_config["reference"].target_modules,
            lora_dropout=model.peft_config["reference"].lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.add_adapter("grpo_policy", grpo_config)
        model.set_adapter("grpo_policy")
    else:
        print(f"WARNING: No SFT model found at {sft_model_dir}. Applying fresh LoRA adapter.")
        model = apply_lora(model, config)

    cat_lookup = dict(zip(categories_df["id"], categories_df["name"])) if len(categories_df) else {}
    registry = ToolRegistry.create_default() if track == "with_tools" else None

    valid_images = images_df.dropna(subset=["local_path"])
    total_steps = len(valid_images)
    current_baseline = 0.0

    for step, (_, img_row) in enumerate(valid_images.iterrows(), start=1):
        img_id = img_row["id"]
        img_annots = annots_df[annots_df["image_id"] == img_id]
        if img_annots.empty:
            continue

        gt = [
            {
                "quadrant": row_to_fdi(row)[0],
                "tooth_position": row_to_fdi(row)[1],
                "diagnosis": cat_lookup.get(row.get(diag_col), "Caries"),
            }
            for _, row in img_annots.iterrows()
        ]

        stats, current_baseline = grpo_step(
            model=model,
            processor=processor,
            image_ids_and_gts=[(img_id, gt)],
            images_df=images_df,
            registry=registry,
            group_size=G,
            track=track,
            lr=lr,
            epochs_per_batch=n_epochs,
            clip_eps=eps,
            kl_beta=beta,
            running_ema_baseline=current_baseline,
        )
        log_grpo_step(stats, extra={"step": step, "image_id": int(img_id)})
        print(f"[GRPO Step {step}/{total_steps}] mean_reward={stats['mean_reward']:.3f} kl={stats['kl_divergence']:.4f}")

        if step % 50 == 0 or step == total_steps:
            save_checkpoint(
                model=model,
                processor=processor,
                tag=f"grpo-{track}-step-{step}",
                checkpoint_dir=checkpoint_dir,
                extra_metadata={"step": step, "mean_reward": stats["mean_reward"], "track": track},
            )

    final_path = save_checkpoint(
        model=model,
        processor=processor,
        tag=f"grpo-{track}-final",
        checkpoint_dir=checkpoint_dir,
        extra_metadata={"track": track, "group_size": G},
    )
    print(f"Stage 2 GRPO complete. Checkpoint saved to: {final_path}")
    return final_path

