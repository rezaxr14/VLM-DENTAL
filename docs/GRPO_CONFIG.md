# Stage 2: Group Relative Policy Optimization (GRPO) Configuration & Architecture Reference

This document serves as the master technical specification for Stage 2 Group Relative Policy Optimization (GRPO) of **VLM-DENTAL** (`Qwen/Qwen3.5-9B`). It details the dual-adapter reference/policy mechanism (G3), batched rollout sampling (G2), group advantage normalization across group sizes $K \in \{1, 2, 4, 8, 16\}$, multi-finding objective reward formulations (Rule 13), cross-turn KV-cache reuse, and Cloud TPU v5e-8 cluster execution.

---

## 1. Algorithmic Overview & Invariants

GRPO eliminates the dedicated critic/value model required by standard PPO, reducing memory overhead and training instability. For each prompt panoramic image $x$, the model samples a group of $K$ candidate diagnostic trajectories:
$$\{y_1, y_2, \dots, y_K\} \sim \pi_{\theta_{\text{old}}}(\cdot \mid x)$$

A rule-based objective reward function evaluates each trajectory against multi-finding clinical ground truth:
$$\{R_1, R_2, \dots, R_K\}$$

The policy parameters $\theta$ are optimized using a clipped surrogate objective with an unbiased KL penalty against the frozen Stage 1 SFT reference policy:
$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{K} \sum_{i=1}^K \frac{1}{|y_i|} \sum_{t=1}^{|y_i|} \left[ \min\left( \frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})} A_i, \; \text{clip}\left(\frac{\pi_\theta(y_{i,t} \mid x, y_{i,<t})}{\pi_{\theta_{\text{old}}}(y_{i,t} \mid x, y_{i,<t})}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta D_{KL}(\pi_\theta \parallel \pi_{\text{ref}}) \right]$$

---

## 2. Policy Architecture: Dual-LoRA Adapter Toggle (G3)

Loading two separate 18.4 GB models into TPU v5e-8 HBM would cause catastrophic out-of-memory errors. Instead, the base `Qwen/Qwen3.5-9B` model is loaded once with two active PEFT LoRA adapters:

| Adapter Name | State | Purpose |
| :--- | :--- | :--- |
| `"reference"` | Frozen | Stage 1 SFT checkpoint weights. Computes reference log-probs $\pi_{\text{ref}}$ for the KL divergence term. Zero gradient updates. |
| `"grpo_policy"` | Trainable | Active RL policy updated by policy gradients. |

- **Adapter Switching**: Dynamically toggled via `model.set_adapter("reference")` and `model.set_adapter("grpo_policy")`.
- **Unit Verification Test (`tests/test_dual_adapter_grpo.py`)**: Mathematically validates that log-probs diverge under updated policy weights, switching back to reference reproduces baseline log-probs to 6 decimal places, and the Schulman $k_3$ KL divergence is strictly non-negative.

---

## 3. Group Advantage Normalization Across $K \in \{1, 2, 4, 8, 16\}$

The normalized advantage $A_i$ assesses the relative quality of trajectory $i$ within its peer group:

### 3.1 Standard Formulation ($K \ge 2$)
$$A_i = \frac{R_i - \bar{R}}{\text{std}(R) + 1e-4}, \quad \text{where } \bar{R} = \frac{1}{K}\sum_{j=1}^K R_j$$

- **Tie-Breaking Stabilization ($K=2$)**: If both trajectories obtain identical rewards ($\text{std}(R) < 1e-6$), all advantages $A_i$ are explicitly clamped to $0.0$, eliminating gradient noise on uninformative ties.

### 3.2 $K=1$ Degeneracy & Running EMA Baseline
When $K=1$, group variance is mathematically zero (causing division by zero in standard GRPO). The algorithm automatically transitions to REINFORCE with an Exponential Moving Average (EMA) baseline:
$$A_1 = R_1 - \bar{R}_{\text{EMA}}$$
$$\bar{R}_{\text{EMA}} \leftarrow \beta \bar{R}_{\text{EMA}} + (1 - \beta) R_1, \quad (\beta = 0.95)$$

---

## 4. [G2] Batched Rollout Generation (Closing the Tok/s Gap)

Replacing sequential `for _ in range(group_size):` loops with batched inference closes the gap between theoretical FLOPs and actual hardware throughput:

- **Track B (No-Tools)**: The initial prompt is replicated $K$ times: `prompt.repeat_interleave(K, dim=0)`. All $K$ candidate trajectories generate simultaneously in a single forward pass with `temperature = 0.7, do_sample = True`.
- **Track A (With-Tools)**:
  - **Turn 1 (Prefill + First Action)**: Generated concurrently across all $K$ candidates.
  - **Turns $t > 1$**: Vectorized rollout manager batches generation across all active trajectories at each turn.
- **$K=16$ Micro-Batching**: Generates candidates in $2 \times 8$ or $4 \times 4$ micro-batches to stay safely within the 16 GB per-chip TPU HBM ceiling on long sequences.

---

## 5. Cross-Turn KV-Cache Reuse Engine

To eliminate quadratic $O(N^2)$ prefill latency across multi-turn tool trajectories:
1. **DynamicCache Preservation**: `past_key_values` are preserved across turns in `run_agent()`.
2. **Vision Encoder Bypass**: When tools return metadata/bounding boxes (`locate_tooth`, `nudge_crop`), the vision encoder is completely bypassed. Only the new observation tokens are encoded.
3. **Global 3D-MRoPE Coordinate Slicing**: Positional IDs are sliced globally (`delta_position_ids = full_position_ids[:, :, past_len:]`), ensuring temporal and spatial rotary coordinates remain continuous across turns.
4. **Self-Healing Fallback**: Defensively catches any tensor shape mismatch and falls back to full-history prefill for that turn if needed.

---

## 6. Multi-Finding Clinical Reward Formulations (Rule 13)

Dental panoramic radiographs contain 1 to 7 labeled pathologies per image. Ground truth is never truncated with `.iloc[0]`. Rewards are computed via set-level bipartite matching (`match_multi_findings`):

### 6.1 Track A: With-Tools Composite Reward
$$R_{\text{Track A}} = w_{\text{FDI}} R_{\text{FDI}} + w_{\text{Diag}} R_{\text{Diag}} + w_{\text{Format}} R_{\text{Format}} + w_{\text{Eff}} R_{\text{Eff}}$$

| Component | Weight ($w$) | Metric & Formulation |
| :--- | :--- | :--- |
| **FDI Localization ($R_{\text{FDI}}$)** | $0.40$ | Precision, Recall, and F1 over ground-truth tooth numbers. |
| **Diagnostic Match ($R_{\text{Diag}}$)** | $0.40$ | Exact and hierarchical clinical match over matched findings. |
| **Format Adherence ($R_{\text{Format}}$)** | $0.10$ | Strict adherence to JSON schema, valid FDI digits (11–48), and tool formats. |
| **Tool Efficiency ($R_{\text{Eff}}$)** | $0.10$ | Bounded efficiency score: $1.0 - \frac{\text{tool\_calls}}{\text{max\_tool\_calls}}$, penalizing redundant crops. |

### 6.2 Track B: Without-Tools Direct Reward
$$R_{\text{Track B}} = 0.45 R_{\text{FDI}} + 0.45 R_{\text{Diag}} + 0.10 R_{\text{Format}}$$
*(Zero tool calls permitted; tool efficiency penalty is strictly excluded).*

---

## 7. Hyperparameter Specifications & Optimization

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Group Size ($K$)** | $1, 2, 4, 8, 16$ | Evaluated across experimental sweep (`scripts/run_grpo_sweep.py`). Default: $4$. |
| **PPO Clip Ratio ($\epsilon$)** | $0.20$ | Standard PPO clipping range preventing policy collapse. |
| **KL Penalty Factor ($\beta$)** | $0.04$ | Schulman $k_3$ penalty weight preventing policy divergence from Stage 1 SFT. |
| **Learning Rate** | $5.0 \times 10^{-6}$ | Low learning rate suited for policy gradient fine-tuning on high-dimensional vision inputs. |
| **Optimizer** | `AdamW` | Weight decay: $0.01$. |
| **Temperature** | $0.70$ | Sampling temperature for exploratory rollout diversity. |
| **Max Tool Calls** | $50$ (Track A) / $0$ (Track B) | Workstation tool execution budget. |
| **Epochs** | $2$ | RL policy update epochs per trajectory batch. |

---

## 8. Multi-Core Cloud TPU v5e-8 Distributed Execution & Continuity

1. **PyTorch/XLA FSDP Parameter Sharding (`XlaFullyShardedDataParallel`)**:
   - On Cloud TPU v5e-8, each chip has 16 GB of HBM2e. To prevent out-of-memory errors on the 18.4 GB `Qwen/Qwen3.5-9B` base model, the policy model is sharded across all 8 cores via `torch_xla.distributed.fsdp.XlaFullyShardedDataParallel`.
   - Shards base model parameters down to ~2.3 GB per chip, leaving $>10\text{ GB}$ of free HBM for dynamic agent rollouts.
   - Dual-adapter switching (`reference` vs `grpo_policy`) operates cleanly through `unwrap_peft_model()`.
   - Controlled via `--fsdp` (default: enabled on multi-core TPU) or `--no-fsdp`.
2. **Multi-Core Distributed Execution Topology (`xmp.spawn`)**:
   - `scripts/run_grpo.py` and `scripts/run_grpo_sweep.py` support `--num-cores 8` on Cloud TPU v5e-8 via `torch_xla.distributed.xmp.spawn(run_worker, nprocs=8)`.
   - Datasets are partitioned across the 8 cores without overlap (`images_df.iloc[rank::world_size]`), and policy gradients are synchronized via `xm.optimizer_step(optimizer)` over the 2D Torus Inter-Chip Interconnect.
   - All I/O, terminal progress, and Hugging Face checkpoint uploads are strictly gated to the master ordinal (`xm.is_master_ordinal()`).
3. **Context-Aware Completion Masking & Zero-Supervision Guard**:
   - `build_full_trajectory_labels()` unmasks exclusively the assistant-generated reasoning and tool actions.
   - If turn offsets shift due to BPE tokenization differences, the pipeline automatically falls back to context-aware `build_conversational_labels()`.
   - A fail-fast assertion (`assert (labels != -100).sum() > 0`) prevents the policy from optimizing on empty completion spans.
3. **Unified Models Repository Checkpoint Sync**: Checkpoints store lightweight LoRA adapter + optimizer states (~760 MB), uploaded to `--hf-repo Reza-Nadimi/vlm-dental-models` under structured subfolders:
   - SFT References: `sft/qwen3_5_9b_sft_{track}_{sft_stage}/`
   - GRPO Checkpoints: `grpo/qwen3_5_9b_grpo_{track}_k{group_size}_{sft_stage}/`
4. **Kaggle 9h Timeout & Preemption Handling**: Python `SIGTERM` handler automatically captures session termination and uploads the latest checkpoint to HF Hub.
5. **Curriculum-Aware Execution & Seamless Resume**:
   ```bash
   # Launch Stage 2 GRPO with Stage 1a SFT reference
   python scripts/run_grpo.py --track with_tools --sft-stage dentex_alone --group-size 4 --hf-repo Reza-Nadimi/vlm-dental-models

   # Resume from latest checkpoint across Kaggle accounts
   python scripts/run_grpo.py --track with_tools --sft-stage dentex_alone --group-size 4 --resume-hf Reza-Nadimi/vlm-dental-models
   ```
