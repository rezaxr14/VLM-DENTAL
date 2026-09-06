# Stage 1: Supervised Fine-Tuning (SFT) Configuration & Architecture Reference

This document serves as the master technical specification for Stage 1 Supervised Fine-Tuning (SFT) of **VLM-DENTAL** (`Qwen/Qwen3.5-9B`). It outlines model architectures, hardware invariants (specifically **Google Cloud TPU v5e-8** and multi-GPU clusters), sequence length bucketing, track segregation, conversational loss masking, and checkpoint synchronization.

---

## 1. Model Backbone & PEFT Configuration

| Parameter | Specification | Rationale & Invariants |
| :--- | :--- | :--- |
| **Base Backbone** | `Qwen/Qwen3.5-9B` | Unified multimodal vision-language backbone across SFT and GRPO (§14). |
| **Precision** | Native **BF16** (`bfloat16`) | Default on Cloud TPU v5e-8 and Ampere+ GPUs. Eliminates quantization artifacts and maintains full numerical dynamic range. |
| **Quantization (Optional)** | 4-bit NF4 (`--precision qlora`) | Available **only** for local memory-constrained GPUs via `bitsandbytes`. Strictly disabled on TPU/XLA (incompatible kernels). |
| **LoRA Rank ($r$)** | $32$ | Maximizes expressive adaptation capacity for clinical dental reasoning. |
| **LoRA Alpha ($\alpha$)** | $64$ | Standard scaling ratio $\alpha / r = 2.0$. |
| **LoRA Dropout** | $0.05$ | Prevents clinical feature co-adaptation and overfitting on synthetic reasoning. |
| **Target Modules** | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Comprehensive adaptation across all self-attention and MLP feed-forward projections. |

---

## 2. Hardware Architecture & Cloud TPU v5e-8 Optimization

### 2.1 The 16 GB Per-Chip HBM Ceiling & 8-Way FSDP
A Cloud TPU v5e-8 slice consists of 8 chips, each with **16 GB of HBM2e** (128 GB total node memory).
- `Qwen/Qwen3.5-9B` in BF16 consumes **~18.4 GB**, exceeding single-chip memory.
- SFT utilizes **8-way FSDPv2 / SPMD** across the 8-chip 2D Torus ICI network:
  $$\text{Base Model Weights per Chip} = \frac{18.4\text{ GB}}{8} \approx 2.30\text{ GB}$$
  $$\text{LoRA Adapter Parameters } (r=32, \alpha=64) = \sim 150\text{ MB}$$
  $$\text{AdamW Optimizer States for LoRA} = \sim 600\text{ MB}$$
  $$\mathbf{\text{Total Dedicated Memory per Chip}} \approx \mathbf{3.05\text{ GB}}$$
- This leaves **~12.95 GB of unencumbered HBM per chip** for static sequence bucketing, visual patch tokens, and gradient buffers.

---

## 3. Strict Track Segregation & Dataset Manifests

Training is strictly segregated into two isolated pipelines to eliminate behavioral bleed:

### 3.1 Track A: With-Tools Multi-Turn Diagnostic Agent (`with_tools`)
- **Target Policy**: Multi-turn clinical diagnostic agent equipped with 8 real workstation tools.
- **Trace Ingestion**:
  1. `data/traces/train_cot_traces.jsonl` (Canonical verified traces: 678 DENTEX + 202 Tufts = 880 traces)
  2. `data/traces/train_cot_traces_healthy_tufts.jsonl` (400 healthy verified traces)
- **Output Adapter**: `data/models/qwen3_5_9b_sft_tools`

### 3.2 Track B: Without-Tools Direct Radiologist (`no_tools`)
- **Target Policy**: Direct zero-tool clinical radiologist generating structured CoT diagnosis from the full panoramic X-ray.
- **Trace Ingestion**:
  1. `data/traces/train_cot_traces_no_tools.jsonl` (Canonical verified tool-free traces: 678 DENTEX + 202 Tufts = 880 traces)
  2. `data/traces/train_cot_traces_healthy_tufts.jsonl` (400 healthy verified traces)
- **Output Adapter**: `data/models/qwen3_5_9b_sft_no_tools`

---

## 4. Sequence Length Bucketing & Collator Invariants

### 4.1 Static Discrete Buckets
Dynamic sequence lengths cause continuous XLA graph recompilations (30–120s stalls per shape). `BucketedQwenVLCollator` rounds sequences up to the nearest static boundary:

- **Track A (`with_tools`)**: `[4096, 6144, 8192, 10240]`
- **Track B (`no_tools`)**: `[1536, 2048, 2560, 3072]`

### 4.2 Right-Padding Invariant for 3D MRoPE
Qwen2.5/3.5-VL incorporates 3D Rotary Position Embeddings (temporal, vertical, horizontal). Left-padding shifts token positions, shifting the temporal origin $t=0$ for visual patches and corrupting spatial reasoning.
- **Collator Invariant**: Strictly enforce `padding_side = "right"` using `tokenizer.pad_token_id`.
- Padding positions are assigned `labels = -100` and masked out of attention.

---

## 5. Conversational Assistant-Only Loss Masking

To prevent the model from penalizing or memorizing system prompts, user instructions, or environment tool observation returns, loss is computed strictly on assistant generations.

- **Token-Level Identification**:
  1. Full multi-turn conversation is formatted via `processor.apply_chat_template()`.
  2. Assistant turn boundaries are detected between `<|im_start|>assistant\n` and `<|im_end|>`.
  3. Tokens strictly inside assistant spans (clinical reasoning, tool call JSON, final diagnostic synthesis) and the closing `<|im_end|>` token retain their true `input_ids`.
  4. All tokens outside assistant spans (system prompt, user query, tool returns, crop metadata) receive `labels = -100`.

---

## 6. Hyperparameter Specifications & Optimization

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Optimizer** | `AdamW` | Standard decoupled weight decay optimizer. |
| **Peak Learning Rate** | $2.0 \times 10^{-5}$ | Conservative learning rate preserving pretrained visual features. |
| **Learning Rate Schedule** | Cosine with 10% Warmup | Smooth decay to 10% of peak learning rate at final step. |
| **Weight Decay** | $0.01$ | Regularization applied to LoRA adapter weights. |
| **Per-Device Batch Size** | $1$ | Maximizes available memory for high-resolution visual tokens. |
| **Gradient Accumulation** | $16$ (Dual GPU) / $4$ (8-TPU) | Enforces an effective batch size of $32$. |
| **Epochs** | $3$ | Optimal convergence across verified synthetic traces without overfitting. |
| **Gradient Clipping** | $1.0$ | Mitigates exploding gradients on high-loss multi-turn transitions. |

---

## 7. Hugging Face Hub Checkpoint Sync & Kaggle Continuity

To survive Kaggle's 9-hour session timeout and 20-hour weekly quota per account:
1. **Lightweight Checkpoints**: Checkpoints store only LoRA adapter weights (`adapter_model.safetensors`), optimizer state (`optimizer.pt`), scheduler state (`scheduler.pt`), and training state metadata (`training_state.json`). Total size is **~760 MB** (uploaded in <15 seconds).
2. **Automated Step Uploads**: Checkpoints automatically upload to Hugging Face Hub (`--hf-repo Reza-Nadimi/vlm-dental-checkpoints`) every 25 steps (~30 mins) and on epoch completion.
3. **Emergency Preemption Hook**: Python `SIGTERM` signal handler immediately flushes an emergency checkpoint before Kaggle terminates the session.
4. **Zero-Waste Multi-Account Resume**:
   ```bash
   python scripts/train_sft.py --track with_tools --resume-hf Reza-Nadimi/vlm-dental-checkpoints
   ```
   Account 2 pulls the checkpoint in seconds, restores optimizer states, skips completed samples, and resumes seamlessly.
