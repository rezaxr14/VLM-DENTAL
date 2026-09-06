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

## 8. TPU v5e-8 Execution & Multi-Account Kaggle Continuity

1. **8-Way FSDP Mesh**: Shards the 9B model base weights (~2.3 GB per chip), leaving ~12.95 GB HBM free per chip.
2. **Hugging Face Hub Checkpoint Sync**: Checkpoints store only the trainable LoRA adapter + optimizer states (~760 MB), uploaded in <15 seconds to `--hf-repo Reza-Nadimi/vlm-dental-checkpoints`.
3. **Kaggle 9h Timeout & Preemption Handling**: Python `SIGTERM` handler automatically captures session termination and uploads the latest checkpoint to HF Hub.
4. **Seamless Resume**:
   ```bash
   python scripts/run_grpo.py --track with_tools --group-size 4 --resume-hf Reza-Nadimi/vlm-dental-checkpoints
   ```
