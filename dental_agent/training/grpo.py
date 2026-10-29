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
from dental_agent.model.backbone import load_model, apply_lora, safe_process_vision_info
from dental_agent.model.checkpoints import save_checkpoint
from dental_agent.agent.loop import run_agent, run_agent_no_tools, AgentTrajectory
from dental_agent.agent.prompts import FIRST_USER_PROMPT, NO_TOOLS_SYSTEM_PROMPT, build_agent_system_prompt
from dental_agent.agent.parsing import parse_agent_json
from dental_agent.data.fdi_utils import row_to_fdi
from dental_agent.rewards.composite import combine_reward
from dental_agent.tools.registry import ToolRegistry
from dental_agent.training.grpo_resume import (
    find_latest_step_checkpoint,
    init_policy_from_reference,
    load_policy_state,
    pool_fingerprint,
)
from dental_agent.training.sft import (
    build_conversational_labels,
    unwrap_peft_model,
    wrap_distributed_model,
)


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
    logprob_chunk_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FP32 token log-probs; respect outer no_grad and restore policy before backward.

    Old/reference scoring must not build graphs. Policy checkpoint recomputation
    must see the same active adapter as its forward, even after reference scoring.
    """
    from dental_agent.model.local_runtime import local_context, runtime_for
    from dental_agent.training.chunked_log_probs import token_log_probs
    peft_m = unwrap_peft_model(model)
    configs = getattr(peft_m, 'peft_config', {})
    policy = 'grpo_policy' if not configs or 'grpo_policy' in configs else next(iter(configs))
    use_named_reference = not configs or 'reference' in configs
    def activate(name):
        if hasattr(peft_m, 'set_adapter'):
            peft_m.set_adapter(name)
        # Never permit the frozen reference to join optimizer parameters.
        for name, p in peft_m.named_parameters():
            if '.reference.' in name:
                p.requires_grad_(False)
    opts = runtime_for(model)
    chunk = logprob_chunk_tokens if logprob_chunk_tokens is not None else (opts.logprob_chunk_tokens if opts else 0)
    grad_enabled = torch.is_grad_enabled() and not use_reference
    try:
        if use_reference and use_named_reference:
            activate('reference')
        else:
            activate(policy)
        adapter_context = peft_m.disable_adapter() if use_reference and not use_named_reference and hasattr(peft_m,'disable_adapter') else contextlib.nullcontext()
        with adapter_context, torch.set_grad_enabled(grad_enabled), local_context(model):
            return token_log_probs(model, enc, chunk)
    finally:
        activate(policy)


def build_full_trajectory_labels(
    trajectory: dict[str, Any],
    processor: Any,
) -> dict[str, torch.Tensor]:
    """Re-tokenize finished trajectory and unmask exclusively the assistant generated spans."""
    messages = trajectory.get("messages", [])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    image_inputs, video_inputs = safe_process_vision_info(messages)
    full_enc = processor(text=[text], images=image_inputs, videos=video_inputs, return_tensors="pt")

    labels = torch.full_like(full_enc["input_ids"], -100)
    for span in trajectory.get("assistant_token_spans", []):
        start = span.get("prompt_len", 0)
        gen_ids = span.get("token_ids", [])
        end = start + len(gen_ids)
        if end <= labels.shape[1]:
            labels[0, start:end] = torch.tensor(gen_ids, dtype=labels.dtype)

    # Defensive fallback (Claude Point 4): if assistant_token_spans was empty or offset shifted,
    # use context-aware conversational label extraction
    if (labels != -100).sum() == 0:
        labels = build_conversational_labels(full_enc["input_ids"], processor.tokenizer)

    # Fail-fast assertion: ensure we never train on zero supervision
    if (labels != -100).sum() == 0:
        raise ValueError(
            f"[GRPO LABELS ERROR] Zero assistant completion tokens found in trajectory (image_id={trajectory.get('image_id')})! "
            "Cannot optimize policy on empty completion spans."
        )

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
    canonical_resize: bool = False,
    compute_old_log_probs: bool = True,
    max_new_tokens: int = 2048,
) -> tuple[list[dict], list[float], list[torch.Tensor], list[torch.Tensor]]:
    """Sample K candidate trajectories simultaneously in ONE batched forward pass (Track B)."""
    row = images_df[images_df["id"] == image_id].iloc[0]
    base_image = Image.open(row["local_path"]).convert("RGB")
    from dental_agent.utils.canonical import to_canonical
    if canonical_resize:
        base_image = to_canonical(base_image, "FULL")

    prompt_messages = [
        {"role": "system", "content": NO_TOOLS_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": base_image},
                {"type": "text", "text": FIRST_USER_PROMPT},
            ],
        },
    ]

    text = processor.apply_chat_template(prompt_messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = safe_process_vision_info(prompt_messages)

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
        k: (v.repeat((group_size,) + (1,) * (v.ndim - 1)) if k in {"pixel_values", "pixel_values_videos"} else v.repeat_interleave(group_size, dim=0)).to(model.device) if isinstance(v, torch.Tensor) else v
        for k, v in single_enc.items()
    }

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "pad_token_id": getattr(processor.tokenizer, "pad_token_id", None) or getattr(processor.tokenizer, "eos_token_id", None),
    }

    model.eval()
    from dental_agent.model.local_runtime import generate_local
    gen_out = generate_local(model, **batched_inputs, **gen_kwargs)

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
            "format_ok": isinstance(final_answer, (dict, list)),
            "assistant_token_spans": [{"prompt_len": prompt_len, "token_ids": new_ids[0].tolist()}],
            "messages": traj_messages,
        }

        # Track B reward excludes tool efficiency penalties
        reward, _ = combine_reward(traj_dict, ground_truth, max_tool_calls=0)
        trajectories.append(traj_dict)
        rewards.append(reward)

        if compute_old_log_probs:
            enc = build_full_trajectory_labels(traj_dict, processor)
            enc = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in enc.items()}
            with torch.no_grad():
                old_lp, mask = compute_token_log_probs(model, enc, use_reference=False)
            old_log_probs_list.append(old_lp.detach())
            masks_list.append(mask)
        else:
            old_log_probs_list.append(None)  # computed later in the static-shape update path
            masks_list.append(None)

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
    canonical_resize: bool = False,
    compute_old_log_probs: bool = True,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    harness: Any = None,
) -> tuple[list[dict], list[float], list[torch.Tensor], list[torch.Tensor]]:
    """Sample K candidate multi-turn trajectories with workstation tools (Track A), at ``temperature`` (> 0)."""
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
            canonical_resize=canonical_resize,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            harness=harness,
        )
        traj_dict = traj.to_dict() if hasattr(traj, "to_dict") else traj
        reward, _ = combine_reward(traj_dict, ground_truth, max_tool_calls=max_tool_calls)

        trajectories.append(traj_dict)
        rewards.append(reward)

        if compute_old_log_probs:
            enc = build_full_trajectory_labels(traj_dict, processor)
            enc = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in enc.items()}
            with torch.no_grad():
                old_lp, mask = compute_token_log_probs(model, enc, use_reference=False)
            old_log_probs_list.append(old_lp.detach())
            masks_list.append(mask)
        else:
            old_log_probs_list.append(None)  # computed later in the static-shape update path
            masks_list.append(None)

        # Retain allocator blocks between rollout candidates.

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
    canonical_resize: bool = False,
    compute_old_log_probs: bool = True,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    harness: Any = None,
) -> tuple[list[dict], list[float], list[torch.Tensor], list[torch.Tensor]]:
    """Unified group rollout entrypoint supporting Track A and Track B."""
    if track == "no_tools" and harness is not None:
        raise ValueError("Cannot attach rules/skills to no_tools rollouts")
    if track == "no_tools":
        return collect_grpo_group_batched_no_tools(
            model=model,
            processor=processor,
            image_id=image_id,
            images_df=images_df,
            ground_truth=ground_truth,
            group_size=group_size,
            canonical_resize=canonical_resize,
            compute_old_log_probs=compute_old_log_probs,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
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
            canonical_resize=canonical_resize,
            compute_old_log_probs=compute_old_log_probs,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            harness=harness,
        )


# ---------------------------------------------------------------------------
# Static-shape policy forward (TPU/SPMD): one XLA graph for every update
# ---------------------------------------------------------------------------

def collate_static_rows(
    encs: list[dict[str, torch.Tensor]],
    collator: Any,
    rows: int,
) -> dict[str, torch.Tensor]:
    """Collate trajectory encodings into ONE fixed-shape batch of exactly ``rows`` rows.

    ``collator`` is the SFT ``BucketedQwenVLCollator(pad_vision_to_slots=True)``, so every batch has the same
    ``input_ids`` / ``pixel_values`` / ``image_grid_thw`` shapes as SFT (one compiled graph, rows shardable across the
    SPMD mesh). Missing rows are inert copies of row 0 with ``labels = -100`` (zero loss, zero gradient).
    """
    if not encs or len(encs) > rows:
        raise ValueError(f"collate_static_rows needs 1..{rows} encodings, got {len(encs)}.")
    padded = list(encs)
    for _ in range(rows - len(encs)):
        dummy = dict(encs[0])
        dummy["labels"] = torch.full_like(encs[0]["labels"], -100)
        padded.append(dummy)
    return collator(padded)


def _split_rows(token_lp: torch.Tensor, mask: torch.Tensor, n_real: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [(token_lp[i : i + 1].detach(), mask[i : i + 1]) for i in range(n_real)]


def compute_static_old_log_probs(
    model: Any,
    encs: list[dict[str, torch.Tensor]],
    collator: Any,
    rows: int,
    to_device: Any,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """No-grad behaviour-policy log-probs for every trajectory, computed with the SAME static shapes as the update."""
    old_lps: list[torch.Tensor] = []
    masks: list[torch.Tensor] = []
    for start in range(0, len(encs), rows):
        chunk = encs[start : start + rows]
        batch = to_device(collate_static_rows(chunk, collator, rows))
        with torch.no_grad():
            lp, mask = compute_token_log_probs(model, batch, use_reference=False)
        for a, b in _split_rows(lp, mask, len(chunk)):
            old_lps.append(a)
            masks.append(b)
    return old_lps, masks


def static_policy_epoch(
    model: Any,
    collator: Any,
    rows: int,
    groups_static: list[tuple[list[dict[str, torch.Tensor]], torch.Tensor, list[torch.Tensor]]],
    clip_eps: float,
    kl_beta: float,
    n_total_rollouts: int,
    to_device: Any,
    grad_scaler: Any = None,
) -> tuple[float, int]:
    """One PPO-clip/k3-KL pass over micro-batches of ``rows`` trajectories with static shapes.

    Math is identical to the per-trajectory path (per-trajectory masked mean, divided by ``n_total_rollouts``);
    only the batching differs. Returns (sum of per-trajectory mean KL, number of trajectories).
    """
    kl_sum, kl_n = 0.0, 0
    for encs, advs, old_lps in groups_static:
        for start in range(0, len(encs), rows):
            chunk = encs[start : start + rows]
            n = len(chunk)
            batch = to_device(collate_static_rows(chunk, collator, rows))
            new_lp, mask = compute_token_log_probs(model, batch, use_reference=False)

            old = torch.cat(old_lps[start : start + n], dim=0).to(new_lp.device)
            adv = new_lp.new_zeros((rows, 1))
            adv[:n, 0] = advs[start : start + n].to(new_lp.device)
            if n < rows:
                old = torch.cat([old, old.new_zeros((rows - n, old.shape[1]))], dim=0)

            ratio = torch.exp(new_lp - old)
            per_token_loss = -torch.min(ratio * adv, torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv)

            if kl_beta > 0:
                with torch.no_grad():
                    ref_lp, _ = compute_token_log_probs(model, batch, use_reference=True)
                log_ratio_ref = ref_lp - new_lp
                per_token_kl = torch.exp(log_ratio_ref) - log_ratio_ref - 1.0
                per_token_loss = per_token_loss + kl_beta * per_token_kl
                row_kl = (per_token_kl * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                kl_sum += float(row_kl[:n].sum().item())
                kl_n += n

            per_row = (per_token_loss * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            from dental_agent.model.local_runtime import local_context
            loss = per_row.sum() / n_total_rollouts
            with local_context(model, autocast=False):
                (grad_scaler.scale(loss) if grad_scaler is not None else loss).backward()
    return kl_sum, kl_n


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
    canonical_resize: bool = False,
    optimizer: torch.optim.Optimizer | None = None,
    train_collator: Any = None,
    rows_per_forward: int = 1,
    spmd_mesh: Any = None,
    use_spmd: bool = False,
    temperature: float = 0.7,
    max_new_tokens: int = 2048,
    grad_scaler: Any = None,
    harness: Any = None,
) -> tuple[dict[str, Any], float]:
    """One GRPO update cycle over on-policy sampled groups.

    ``optimizer`` should be created ONCE by the caller and reused: building a fresh AdamW per call reset the Adam
    moments on every image, turning each update into a sign-SGD step. (If omitted, a fresh one is created for
    backwards compatibility.)

    ``train_collator`` (a ``BucketedQwenVLCollator`` with ``pad_vision_to_slots=True``) switches the update to the
    static-shape path: ``rows_per_forward`` trajectories per forward (use the number of SPMD cores so each core owns
    one sequence), identical shapes to SFT, old log-probs recomputed with the same shapes.
    """
    if registry is None:
        registry = ToolRegistry.create_default()

    if optimizer is None:
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    gen_free_model = model  # policy forward goes through the (possibly FSDPv2-wrapped) model
    # Rollouts call .generate(), which the SPMD FSDP wrapper does not expose; the inner module shares the very same
    # sharded parameters, so GSPMD partitions it identically.
    rollout_model = unwrap_peft_model(model) if use_spmd else model

    def _to_device(batch: dict[str, Any]) -> dict[str, Any]:
        if use_spmd:
            import torch_xla.core.xla_model as xm
            dev = xm.xla_device()
        else:
            dev = model.device
        moved = {k: v.to(dev) if hasattr(v, "to") else v for k, v in batch.items()}
        if use_spmd and spmd_mesh is not None:
            from dental_agent.training.sft import apply_spmd_input_sharding
            apply_spmd_input_sharding(moved, spmd_mesh, num_cores=rows_per_forward)
        return moved

    groups, all_rewards = [], []
    n_tied_groups = 0
    current_baseline = running_ema_baseline
    t_step_start = time.perf_counter()

    # 1. On-policy rollout collection
    for image_id, ground_truth in image_ids_and_gts:
        trajs, rewards, old_lps, masks = collect_grpo_group(
            model=rollout_model,
            processor=processor,
            image_id=image_id,
            images_df=images_df,
            ground_truth=ground_truth,
            registry=registry,
            group_size=group_size,
            max_tool_calls=max_tool_calls,
            track=track,
            canonical_resize=canonical_resize,
            compute_old_log_probs=train_collator is None,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            harness=harness,
        )
        advantages, current_baseline = compute_group_advantages(
            rewards=rewards,
            running_ema_baseline=current_baseline,
        )
        # A tie (all K rewards equal, K >= 2) gives every advantage 0: its K rollouts carry no learning signal.
        if group_size >= 2 and float(advantages.abs().max()) == 0.0:
            n_tied_groups += 1
        groups.append((trajs, advantages, old_lps, masks))
        all_rewards.extend(rewards)

    n_total_rollouts = len(image_ids_and_gts) * group_size
    t_rollout_end = time.perf_counter()

    # Static-shape path: encode each trajectory once, drop those that cannot fit the static length, and compute the
    # behaviour-policy log-probs with exactly the shapes used by the update (one compiled graph, no per-length recompiles).
    groups_static: list[tuple[list[dict[str, torch.Tensor]], torch.Tensor, list[torch.Tensor]]] = []
    n_dropped_overlength = 0
    if train_collator is not None:
        for trajs, advantages, _, _ in groups:
            kept_idx, encs = [], []
            for i, traj in enumerate(trajs):
                enc = build_full_trajectory_labels(traj, processor)
                if train_collator.padded_length(enc) > train_collator.max_seq_len:
                    n_dropped_overlength += 1
                    continue
                kept_idx.append(i)
                encs.append(enc)
            if not encs:
                continue
            old_lps, _ = compute_static_old_log_probs(
                gen_free_model, encs, train_collator, rows_per_forward, _to_device
            )
            groups_static.append((encs, advantages[torch.tensor(kept_idx, dtype=torch.long)], old_lps))
        if n_dropped_overlength:
            import warnings
            warnings.warn(
                f"[GRPO] {n_dropped_overlength} rollout(s) exceeded the static length "
                f"{train_collator.max_seq_len} and were excluded from this update (rewards/advantages unchanged).",
                UserWarning,
                stacklevel=2,
            )
    model.train()

    total_kl = 0.0
    kl_count = 0

    # 2. Optimization passes over fixed rollout batch
    for epoch in range(epochs_per_batch):
        optimizer.zero_grad()
        if train_collator is not None:
            ks, kn = static_policy_epoch(
                model, train_collator, rows_per_forward, groups_static,
                clip_eps, kl_beta, n_total_rollouts, _to_device, grad_scaler,
            )
            total_kl += ks
            kl_count += kn
        else:
            for trajs, advantages, old_lps, masks in groups:
                for traj, advantage, old_lp, mask in zip(trajs, advantages, old_lps, masks):
                    enc = build_full_trajectory_labels(traj, processor)
                    enc = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in enc.items()}
                    new_lp, _ = compute_token_log_probs(model, enc, use_reference=False)

                    ratio = torch.exp(new_lp - old_lp.to(new_lp.device))
                    adv = advantage.to(new_lp.device)
                    mask = mask.to(new_lp.device)
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
                    from dental_agent.model.local_runtime import local_context
                    if not torch.isfinite(loss).item():
                        raise RuntimeError('Nonfinite GRPO loss; policy/reference log probabilities must be finite')
                    with local_context(model, autocast=False):
                        (grad_scaler.scale(loss) if grad_scaler is not None else loss).backward()

        is_tpu = False
        try:
            import torch_xla.core.xla_model as xm
            is_tpu = True
        except Exception:
            pass

        if is_tpu and use_spmd:
            optimizer.step()  # SPMD: gradients are already globally consistent; no cross-replica all-reduce
            xm.mark_step()
        elif is_tpu:
            xm.optimizer_step(optimizer)
        else:
            if grad_scaler is not None:
                grad_scaler.step(optimizer)
                grad_scaler.update()
            else:
                optimizer.step()

    model.eval()
    if is_tpu:
        try:
            xm.wait_device_ops()  # XLA is lazy: without this the update time would be charged to the next rollout
        except Exception:
            pass
    t_step_end = time.perf_counter()
    mean_kl = total_kl / max(kl_count, 1)
    all_trajs = [t for trajs, _, _, _ in groups for t in trajs]
    n_traj = max(len(all_trajs), 1)
    stats = {
        "mean_reward": sum(all_rewards) / max(len(all_rewards), 1),
        "kl_divergence": mean_kl,
        "n_rollouts": len(all_rewards),
        "group_size": group_size,
        "track": track,
        "n_dropped_overlength": n_dropped_overlength,
        "n_groups": len(groups),
        "n_tied_groups": n_tied_groups,
        # Cost and behaviour diagnostics. Rollout = sampling the K trajectories (incl. tool calls); update = static
        # old-log-prob pass + the policy epochs. summarize_grpo_log.py projects total run time from these.
        "rollout_s": t_rollout_end - t_step_start,
        "update_s": t_step_end - t_rollout_end,
        "mean_turns": sum(len(t.get("turns", [])) for t in all_trajs) / n_traj,
        "mean_tool_calls": sum(int(t.get("tool_calls", 0)) for t in all_trajs) / n_traj,
        "format_ok_rate": sum(1 for t in all_trajs if t.get("format_ok")) / n_traj,
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

def upload_checkpoint_to_hf(
    checkpoint_dir: str | Path,
    hf_repo: str,
    step: int,
    path_in_repo: str | None = None,
) -> None:
    """Upload lightweight LoRA checkpoint (~760 MB) to Hugging Face Hub under structured subfolder."""
    try:
        from dental_agent.model.hub_artifacts import upload_adapter_folder
        target_path = path_in_repo or f"grpo/{Path(checkpoint_dir).name}"
        commit_msg = f"VLM-DENTAL GRPO Checkpoint: {Path(checkpoint_dir).name} (Step {step})"
        print(f"[HF-HUB] Uploading GRPO checkpoint from {checkpoint_dir} to {hf_repo} ({target_path})...")
        upload_adapter_folder(
            checkpoint_dir, hf_repo,
            path_in_repo=target_path,
            commit_message=commit_msg,
        )
        print(f"[HF-HUB] GRPO checkpoint successfully uploaded to {hf_repo} under {target_path}.")
    except Exception as e:
        print(f"[HF-HUB WARNING] Failed to upload checkpoint to {hf_repo}: {e}")


def plot_grpo_training_curve(
    log_path: str | Path = "data/grpo_training_log.jsonl",
    save_path: str | Path | None = None,
) -> pd.DataFrame | None:
    """Publication-ready dual-axis training curve showing Mean Reward (left) and KL Divergence (right)."""
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

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color_reward = "#2ca02c"
    ax1.set_xlabel("GRPO Rollout Step", fontsize=11)
    ax1.set_ylabel("Composite Reward", color=color_reward, fontsize=11)
    ax1.plot(df["step"], df["mean_reward"], color=color_reward, lw=2.0, marker="o", markersize=4, label="Mean Reward")
    ax1.tick_params(axis="y", labelcolor=color_reward)
    ax1.grid(True, alpha=0.3)

    if "kl_divergence" in df.columns:
        ax2 = ax1.twinx()
        color_kl = "#d62728"
        ax2.set_ylabel("KL Divergence (Reference vs Policy)", color=color_kl, fontsize=11)
        ax2.plot(df["step"], df["kl_divergence"], color=color_kl, lw=1.8, linestyle="--", label="KL Divergence")
        ax2.tick_params(axis="y", labelcolor=color_kl)

    plt.title("VLM-DENTAL Stage 2 GRPO Convergence: Dual-Axis Dynamics", fontsize=12, fontweight="bold")
    fig.tight_layout()

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
    path_in_repo_prefix: str | None = None,
    num_cores: int = 1,
    use_fsdp: bool = True,
    canonical_resize: bool = False,
    temperature: float = 0.7,
    use_spmd: bool = False,
    max_seq_len: int | None = None,
    pad_vision_to_slots: bool = False,
    slot_budget: dict[str, int] | None = None,
    resume: bool = False,
    max_new_tokens: int = 2048,
    log_path: str | Path = "data/grpo_training_log.jsonl",
    runtime_options: Any = None,
    gradient_checkpointing: bool = True,
    harness_mode: str = "off",
    seed: int | None = None,
) -> str:
    """Execute Stage 2 GRPO policy optimization with dual-adapter reference and group advantage normalization."""
    from peft import PeftModel, LoraConfig
    from dental_agent.harness.configuration import harness_options, harness_identity, validate_harness_mode, assert_harness_identity
    validate_harness_mode(harness_mode, track)
    harness = harness_options(harness_mode)
    harness_record = harness_identity(harness_mode)
    if seed is not None:
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    tr_cfg = config.training if isinstance(config, ProjectConfig) else (config or TrainingConfig())
    G = group_size or tr_cfg.grpo_group_size
    lr = learning_rate or tr_cfg.grpo_lr
    beta = kl_beta if kl_beta is not None else tr_cfg.grpo_kl_beta
    eps = clip_eps or tr_cfg.grpo_clip_eps
    n_epochs = epochs_per_batch or tr_cfg.grpo_epochs_per_batch

    print(f"--- Starting Stage 2 GRPO Training (Track={track}, GroupSize={G}, KLBeta={beta}, LR={lr}) ---")

    import importlib.util
    is_tpu = importlib.util.find_spec('torch_xla') is not None and not torch.cuda.is_available()
    if runtime_options is None:
        model, processor = load_model(config)  # preserve legacy stand-in loaders in tests/callers
    else:
        model, processor = load_model(config, runtime_options=runtime_options, is_tpu=is_tpu)
    from dental_agent.model.local_runtime import configure_model, runtime_for, runtime_record
    opts = runtime_for(model)
    runtime_dtype = getattr(model, '_dental_compute_dtype', None)
    runtime_backend = getattr(model, '_dental_attention_backend', None)
    if opts and opts.allow_cpu_offload:
        raise ValueError('CPU/disk offload is inference-only and forbidden for GRPO training')
    if getattr(model, 'is_loaded_in_4bit', False):
        from dental_agent.model.backbone import prepare_qlora_model
        model = prepare_qlora_model(model, use_gradient_checkpointing=False)

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
        # add_adapter makes a zero-init LoRA; start the policy AT the SFT weights (KL = 0 at step 1, rollouts from SFT).
        n_copied = init_policy_from_reference(model)
        print(f"[GRPO] Policy adapter initialised from the SFT adapter ({n_copied} LoRA tensors copied).")
        model.set_adapter("grpo_policy")
    else:
        print(f"WARNING: No SFT model found at {sft_model_dir}. Applying fresh LoRA adapter.")
        model = apply_lora(model, config)

    if opts:
        configure_model(model, opts, runtime_dtype, runtime_backend)
    if torch.cuda.is_available():
        from dental_agent.training.sft import freeze_and_guard_vision_tower
        train_merger = any(p.requires_grad and '.visual.' in n and 'merger' in n for n,p in model.named_parameters())
        freeze_and_guard_vision_tower(model, train_merger=train_merger)
        if gradient_checkpointing:
            model.enable_input_require_grads()
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            print('[GRPO] Non-reentrant gradient checkpointing enabled; rollout generate explicitly uses cache', flush=True)
    # Training scores request use_cache=False; rollout generation requests True.
    cuda_fp16 = torch.cuda.is_available() and runtime_dtype == torch.float16
    grad_scaler = torch.amp.GradScaler('cuda', enabled=cuda_fp16, init_scale=128.0)
    if opts:
        from dental_agent.training.gpu_runtime import runtime_metadata
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(checkpoint_dir) / 'grpo_runtime.json', 'w') as f:
            json.dump({'local_runtime': runtime_record(model), 'environment': runtime_metadata()}, f, indent=2)

    # Resume: reload the trained policy adapter and skip the images already trained on (before any TPU/FSDP wrapping).
    pool_fp = pool_fingerprint(images_df.dropna(subset=["local_path"])["id"])
    start_step, resume_baseline = 0, 0.0
    if resume:
        if not (sft_model_dir and os.path.exists(sft_model_dir)):
            raise ValueError("resume needs the SFT reference adapter (sft_model_dir); none was found.")
        final_meta = Path(checkpoint_dir) / f"grpo-{track}-final" / "checkpoint_meta.json"
        if final_meta.is_file():
            assert_harness_identity(json.loads(final_meta.read_text()).get("extra", {}).get("harness"), harness_mode)
            print(f"[RESUME] {final_meta.parent} already exists: this run is complete; nothing to resume.")
            return str(final_meta.parent)
        found = find_latest_step_checkpoint(checkpoint_dir, track)
        if found is None:
            raise FileNotFoundError(f"--resume given but no 'grpo-{track}-step-N' checkpoint was found under {checkpoint_dir}.")
        ckpt_path, meta = found
        assert_harness_identity(meta.get("harness"), harness_mode)
        if meta.get("pool_fingerprint") != pool_fp or int(meta.get("group_size", G)) != G:
            raise ValueError(
                f"Checkpoint {ckpt_path.name} was trained on a different image pool or group size "
                f"(pool {meta.get('pool_fingerprint')} vs {pool_fp}, K {meta.get('group_size')} vs {G}). Resume with the same "
                "--dev-split, --seed, --max-images, --include-healthy and --group-size, or start a new run without --resume."
            )
        load_policy_state(model, ckpt_path)
        start_step = int(meta["step"])
        resume_baseline = float(meta.get("baseline", 0.0))
        print(f"[RESUME] Policy adapter restored from {ckpt_path.name}; continuing at step {start_step + 1}. "
              "Optimizer moments restart cold (not stored in checkpoints).")

    is_tpu = False
    try:
        import torch_xla.core.xla_model as xm
        is_tpu = True
    except Exception:
        pass

    spmd_mesh = None
    train_collator = None
    rows_per_forward = 1
    if pad_vision_to_slots and not is_tpu:
        raise ValueError("pad_vision_to_slots is TPU-only (static shapes avoid XLA recompilation); GPU/CPU pad dynamically.")
    if pad_vision_to_slots and not canonical_resize:
        raise ValueError("pad_vision_to_slots requires canonical_resize (static slots assume canonical image sizes).")
    from dental_agent.utils.canonical import check_gpu_image_policy, letterbox_enabled
    if pad_vision_to_slots and not letterbox_enabled():
        raise ValueError("pad_vision_to_slots requires canonical_letterbox (static slots assume the fixed canonical canvas).")
    check_gpu_image_policy(canonical_resize, letterbox_enabled(), cuda=torch.cuda.is_available(), tpu=is_tpu, what="GRPO")
    if use_spmd and is_tpu:
        if not use_fsdp:
            raise ValueError("SPMD shards weights through FSDPv2; use_fsdp=False would replicate 18 GB on every core.")
        from dental_agent.training.sft import freeze_and_guard_vision_tower, setup_spmd_mesh, wrap_spmd_model
        freeze_and_guard_vision_tower(model, train_merger=False)
        spmd_mesh = setup_spmd_mesh(num_cores=num_cores)
        model = wrap_spmd_model(model, mesh=spmd_mesh, is_master=True)
        rows_per_forward = num_cores  # one trajectory per core in every policy forward
    else:
        model = wrap_distributed_model(model, is_tpu=is_tpu, num_cores=num_cores, use_fsdp=use_fsdp)

    if pad_vision_to_slots:
        if max_seq_len is None:
            raise ValueError("pad_vision_to_slots requires max_seq_len.")
        from dental_agent.training.sft import BucketedQwenVLCollator
        train_collator = BucketedQwenVLCollator(
            processor=processor, track=track, max_seq_len=max_seq_len, dynamic_padding=False,
            pad_vision_to_slots=True, slot_budget=slot_budget,
        )
    elif use_spmd and is_tpu:
        print("[GRPO WARNING] SPMD without --pad-vision-to-slots: every distinct trajectory length/image mix compiles "
              "a new XLA graph. Use --canonical-resize --pad-vision-to-slots --max-seq-len N.")
    print(f"[CONFIG] spmd={use_spmd and is_tpu} fsdp={use_fsdp} canonical_resize={canonical_resize} "
          + (f"canonical_letterbox={letterbox_enabled()} " if canonical_resize else "")
          + f"pad_vision_to_slots={pad_vision_to_slots} max_seq_len={max_seq_len} rows_per_forward={rows_per_forward} "
          f"num_cores={num_cores}")

    # One optimizer for the whole run: Adam moments must persist across GRPO steps.
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)

    cat_lookup = dict(zip(categories_df["id"], categories_df["name"])) if len(categories_df) else {}
    registry = ToolRegistry.create_default() if track == "with_tools" else None

    valid_images = images_df.dropna(subset=["local_path"])
    total_steps = len(valid_images)
    current_baseline = resume_baseline

    repo_prefix = path_in_repo_prefix or f"grpo/{Path(checkpoint_dir).name}"

    for step, (_, img_row) in enumerate(valid_images.iterrows(), start=1):
        if step <= start_step:
            continue
        img_id = img_row["id"]
        img_annots = annots_df[annots_df["image_id"] == img_id]
        # An image without annotation rows is a HEALTHY image: ground truth [] (reward 1.0 only for an empty
        # answer). It used to be skipped, so the policy never practised saying "no findings". Callers that want
        # pathological images only filter the pool (dental_agent.data.dev_split.select_training_pool).

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
            canonical_resize=canonical_resize,
            optimizer=optimizer,
            train_collator=train_collator,
            rows_per_forward=rows_per_forward,
            spmd_mesh=spmd_mesh,
            use_spmd=bool(use_spmd and is_tpu),
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            grad_scaler=grad_scaler,
            harness=harness,
        )
        log_grpo_step(stats, log_path=log_path, extra={"step": step, "image_id": int(img_id), "harness_mode": harness_mode})
        print(f"[GRPO Step {step}/{total_steps}] mean_reward={stats['mean_reward']:.3f} kl={stats['kl_divergence']:.4f}")

        if (push_every_steps > 0 and step % push_every_steps == 0) or step == total_steps:
            step_tag = f"grpo-{track}-step-{step}"
            ckpt_path = save_checkpoint(
                model=unwrap_peft_model(model),
                processor=processor,
                tag=step_tag,
                checkpoint_dir=checkpoint_dir,
                extra_metadata={
                    "step": step, "mean_reward": stats["mean_reward"], "track": track, "baseline": float(current_baseline),
                    "group_size": G, "pool_fingerprint": pool_fp, "pool_size": total_steps, "harness": harness_record,
                },
            )
            if hf_repo:
                upload_checkpoint_to_hf(
                    checkpoint_dir=ckpt_path,
                    hf_repo=hf_repo,
                    step=step,
                    path_in_repo=f"{repo_prefix}/{step_tag}",
                )

    final_tag = f"grpo-{track}-final"
    final_path = save_checkpoint(
        model=unwrap_peft_model(model),
        processor=processor,
        tag=final_tag,
        checkpoint_dir=checkpoint_dir,
        extra_metadata={"track": track, "group_size": G, "harness": harness_record, "step": total_steps},
    )
    if hf_repo:
        upload_checkpoint_to_hf(
            checkpoint_dir=final_path,
            hf_repo=hf_repo,
            step=total_steps,
            path_in_repo=f"{repo_prefix}/{final_tag}",
        )
    print(f"Stage 2 GRPO complete. Checkpoint saved to: {final_path}")
    return final_path
