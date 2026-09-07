# Stage 1: Supervised Fine-Tuning (SFT) Configuration & Architecture Reference

This document serves as the master technical specification for Stage 1 Supervised Fine-Tuning (SFT) of **VLM-DENTAL** (`Qwen/Qwen3.5-9B`). It outlines model architectures, hardware invariants (specifically **Google Cloud TPU v5e-8** and multi-GPU clusters), sequence length bucketing, track segregation, conversational loss masking, curriculum stages with negative control calibration, multimodal vision projector LoRA, and checkpoint synchronization.

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
| **LLM Target Modules** | `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | Comprehensive adaptation across all self-attention and MLP feed-forward projections. |
| **Vision Target Modules** | `["merger.mlp.0", "merger.mlp.2"]` (`--lora-target-vision projector`) | Adapts the multimodal patch projector mapping visual tokens to language embeddings without disturbing early ViT representations. |

---

## 2. Hardware Architecture & Cloud TPU v5e-8 Multi-Core Optimization

### 2.1 PyTorch/XLA FSDP Parameter Sharding & Multi-Core Execution Topology
A Cloud TPU v5e-8 slice consists of 8 chips, each with **16 GB of HBM2e** (128 GB total node memory).
Because a 9B parameter model in BF16 requires **~18.4 GB** just for the frozen base weights, running in standard DDP (full model replication per core) would exceed the 16 GB per-chip limit and cause an immediate Out-Of-Memory (OOM) error before step 0.

To solve this, VLM-DENTAL implements **PyTorch/XLA Fully Sharded Data Parallelism (`torch_xla.distributed.fsdp.XlaFullyShardedDataParallel`)**:
- **8-Way Parameter Sharding**:
  $$\text{Sharded Base Weights per Chip} = \frac{18.4\text{ GB}}{8\text{ cores}} \approx 2.30\text{ GB}$$
  $$\text{Trainable LoRA Parameters } (r=32, \alpha=64) \approx 60\text{ MB}$$
  $$\text{AdamW Optimizer States for LoRA} \approx 120\text{ MB}$$
  $$\text{Total Memory Footprint per Chip} \approx 4\text{--}5\text{ GB (comfortably within 16 GB HBM)}$$
- **Immediate Resharding (`reshard_after_forward=True`)**: Discards all-gathered layer weights immediately after forward and backward passes, preventing memory accumulation.
- **Generous Headroom**: Leaves $>10\text{ GB}$ of free HBM per chip to accommodate large multi-turn vision token sequences up to 16,384 tokens without risk of OOM.
- **CLI Flag**: Controlled via `--fsdp` (default: enabled on multi-core TPU) or `--no-fsdp` (for standard single-device/CUDA placement).
- **Process Orchestration (`xmp.spawn`)**:
  - `xmp.spawn(run_training, args=(args,), nprocs=8)` spawns 8 worker processes, each binding to its respective device (`xla:0` through `xla:7`).
  - `torch.utils.data.distributed.DistributedSampler` partitions the SFT curriculum shards across all 8 chips.
  - Cross-replica gradient synchronization is performed via `xm.optimizer_step(optimizer)` utilizing the 2D Torus Inter-Chip Interconnect (ICI).
- All logging, evaluation, and Hugging Face checkpoint uploads are strictly gated to the master ordinal (`xm.is_master_ordinal()`), preventing race conditions.

### 2.2 Mathematical Token-Weighted Gradient Accumulation
Standard Hugging Face CausalLM models return `outputs.loss` as an internal mean over that specific microbatch's valid non-$(-100)$ tokens. In conversational multi-turn data where sequence lengths and assistant turn counts vary widely across microbatches, dividing `outputs.loss / grad_accum` causes short assistant turns to exert disproportionately high gradient magnitude ("mean-of-means" error).
- **Exact Token-Weighted Solution**:
  $$\mathcal{L}_{\text{unreduced}}^{(k)} = \mathcal{L}_{\text{mean}}^{(k)} \times N_k, \quad \text{where } N_k = \sum \mathbb{I}(\text{labels}_{i, j}^{(k)} \neq -100)$$
  $$\text{Total Window Tokens } N_{\text{total}} = \sum_{k=1}^M N_k$$
  $$\nabla_\theta \mathcal{L}_{\text{true}} = \frac{1}{N_{\text{total}}} \sum_{k=1}^M \nabla_\theta \mathcal{L}_{\text{unreduced}}^{(k)}$$
- Gradients are accumulated unnormalized across the accumulation window and scaled once by $1 / N_{\text{total}}$ at the step boundary, ensuring exact mathematical equivalence to full-batch training.

---

## 3. Multi-Stage SFT Curriculum & Negative Control Calibration

To evaluate in-domain performance, cross-institution transfer, and complete multi-cohort pathology coverage without over-diagnosis confirmation bias, training is structured across 3 progressive stages. **Healthy negative control traces are included in every stage.**

```
+--------------------------------------------------------------------------------------------------------------------------+
|                                             Stage 1 SFT Curriculum & Manifest                                            |
+--------------------------------------------------------------------------------------------------------------------------+
|  Stage 1a: DENTEX Alone + Healthy Controls (dentex_alone)                                                                |
|  - Traces (Track A with_tools):                                                                                          |
|      * train_cot_traces_dentex.jsonl (678 disease traces)                                                                |
|      * train_cot_traces_healthy_dentex.jsonl (27 healthy control traces)                                                 |
|  - Traces (Track B no_tools):                                                                                            |
|      * train_cot_traces_dentex_no_tools.jsonl                                                                            |
|      * train_cot_traces_healthy_dentex_no_tools.jsonl                                                                     |
|  - Checkpoint Directory: data/models/qwen3_5_9b_sft_{track}_dentex                                                       |
|  - Resuming: Intermediate step checkpoints (step-25, step-50, ...) pushed to HF Hub & local                              |
|  - Purpose: Pure in-domain baseline on primary panoramic dataset with negative control calibration                      |
+--------------------------------------------------------------------------------------------------------------------------+
|  Stage 1b: DENTEX + Tufts Overlap + Healthy Controls (dentex_tufts_overlap)                                              |
|  - Traces (Track A with_tools):                                                                                          |
|      * train_cot_traces_dentex.jsonl (678 DENTEX disease)                                                                |
|      * train_cot_traces_tufts.jsonl (202 Tufts overlapping disease: caries & periapical)                                 |
|      * train_cot_traces_healthy_dentex.jsonl + train_cot_traces_healthy_tufts.jsonl (healthy controls)                    |
|  - Traces (Track B no_tools):                                                                                            |
|      * train_cot_traces_dentex_no_tools.jsonl                                                                            |
|      * train_cot_traces_tufts_no_tools.jsonl                                                                             |
|      * train_cot_traces_healthy_dentex_no_tools.jsonl + train_cot_traces_healthy_tufts_no_tools.jsonl                    |
|  - Checkpoint Directory: data/models/qwen3_5_9b_sft_{track}_dentex_tufts_overlap                                         |
|  - Resuming: Intermediate step checkpoints pushed to HF Hub & local                                                      |
|  - Purpose: Cross-institution domain shift across shared overlapping pathologies                                         |
+--------------------------------------------------------------------------------------------------------------------------+
|  Stage 1c: Full Multi-Cohort: DENTEX + Tufts All 4 Findings + Full Healthy Controls (multicohort_all)                   |
|  - Traces (Track A with_tools):                                                                                          |
|      * train_cot_traces_dentex.jsonl (678 DENTEX disease: 4 DENTEX findings)                                             |
|      * train_cot_traces_tufts_all.jsonl (Full Tufts disease: all 4 Tufts findings on their own)                          |
|      * train_cot_traces_healthy_dentex.jsonl + train_cot_traces_healthy_tufts.jsonl (full negative controls)             |
|  - Traces (Track B no_tools):                                                                                            |
|      * train_cot_traces_dentex_no_tools.jsonl                                                                            |
|      * train_cot_traces_tufts_all_no_tools.jsonl                                                                         |
|      * train_cot_traces_healthy_dentex_no_tools.jsonl + train_cot_traces_healthy_tufts_no_tools.jsonl                    |
|  - Checkpoint Directory: data/models/qwen3_5_9b_sft_{track}_multicohort_all                                              |
|  - Resuming: Intermediate step checkpoints pushed to HF Hub & local                                                      |
|  - Purpose: Complete joint multi-cohort clinical model trained across all pathologies with full negative controls        |
+--------------------------------------------------------------------------------------------------------------------------+
```

---

## 4. Multimodal Vision Projector LoRA (`merger.mlp`)

Rather than freezing the entire vision stack or fine-tuning early ViT blocks:
1. **ViT Transformer Blocks (`visual.blocks`)**: Kept **frozen**. Preserves foundational edge, texture, and spatial detectors pre-trained on millions of images.
2. **Patch Merger Projector (`visual.merger.mlp`)**: LoRA-adapted via `--lora-target-vision projector`.
   - Targets linear projection layers `merger.mlp.0` and `merger.mlp.2`.
   - Trains specialized alignment between subtle radiographic densities (radiolucencies, bone trabeculae, pulp chambers) and clinical language representations.
   - Total parameter overhead is $<1.5\text{ MB}$ of adapter weights.

---

## 5. Sequence Length Bucketing & Collator Invariants

### 5.1 Static Discrete Buckets & 16,384 Headroom
Dynamic sequence lengths cause continuous XLA graph recompilations (30–120s stalls per shape). `BucketedQwenVLCollator` rounds sequences up to the nearest static boundary:

- **Track A (`with_tools`)**: `[4096, 6144, 8192, 12288, 16384]` (accommodating full multi-turn tool observation reasoning traces with real vision patch tokens up to Rule 19's 16,384 limit, with active overlength warning diagnostics)
- **Track B (`no_tools`)**: `[1536, 2048, 2560, 3072]`

### 5.2 Right-Padding Invariant for 3D MRoPE
Qwen2.5/3.5-VL incorporates 3D Rotary Position Embeddings (temporal, vertical, horizontal). Left-padding shifts token positions, shifting the temporal origin $t=0$ for visual patches and corrupting spatial reasoning.
- **Collator Invariant**: Strictly enforce `padding_side = "right"` using `tokenizer.pad_token_id`.
- Padding positions are assigned `labels = -100` and masked out of attention.

---

## 6. Conversational Assistant-Only Loss Masking

To prevent the model from penalizing or memorizing system prompts, user instructions, or environment tool observation returns, loss is computed strictly on assistant generations.

- **Token-Level Identification**:
  1. Full multi-turn conversation is formatted via `processor.apply_chat_template()`.
  2. Assistant turn boundaries are detected between `<|im_start|>assistant\n` and `<|im_end|>`.
  3. Tokens strictly inside assistant spans (clinical reasoning, tool call JSON, final diagnostic synthesis) and the closing `<|im_end|>` token retain their true `input_ids`.
  4. All tokens outside assistant spans (system prompt, user query, tool returns, crop metadata) receive `labels = -100`.

---

## 7. Hyperparameter Specifications & Optimization

| Hyperparameter | Value | Description |
| :--- | :--- | :--- |
| **Optimizer** | `AdamW` | Standard decoupled weight decay optimizer. |
| **Peak Learning Rate** | $2.0 \times 10^{-5}$ | Conservative learning rate preserving pretrained visual features. |
| **Learning Rate Schedule** | Cosine with 5% Warmup | Smooth decay to $1\times 10^{-6}$ at final step. |
| **Weight Decay** | $0.01$ | Regularization applied to LoRA adapter weights. |
| **Per-Device Batch Size** | $1$ | Maximizes available memory for high-resolution visual tokens. |
| **Gradient Accumulation** | $16$ (Dual GPU) / $4$ (8-TPU) | Enforces an effective batch size of $16$ to $32$. |
| **Epochs** | $3$ | Optimal convergence across verified synthetic traces without overfitting. |
| **Gradient Clipping** | $1.0$ | Mitigates exploding gradients on high-loss multi-turn transitions. |
| **Validation Split** | $5\%$ held-out | Evaluated every 25 steps; best validation loss triggers `best_adapter/` saving. |

---

## 8. Hugging Face Hub Checkpoint Sync & Kaggle Continuity

To survive Kaggle's 9-hour session timeout and 20-hour weekly quota per account:
1. **Lightweight Checkpoints**: Checkpoints store only LoRA adapter weights (`adapter_model.safetensors`), optimizer state (`optimizer.pt`), scheduler state (`scheduler.pt`), and training state metadata (`training_state.json`). Total size is **~760 MB** (uploaded in <15 seconds).
2. **Unified Models Repository Uploads**: Checkpoints automatically upload to the canonical models repository (`--hf-repo Reza-Nadimi/vlm-dental-models`) under structured folders (`sft/qwen3_5_9b_sft_{track}_{stage}/`) every 25 steps (~30 mins) and on epoch completion.
3. **Emergency Preemption Hook**: Python `SIGTERM` signal handler immediately flushes an emergency checkpoint before Kaggle terminates the session.
4. **Zero-Waste Multi-Account Resume**:
   ```bash
   python scripts/train_sft.py --track with_tools --stage dentex_alone --resume-hf Reza-Nadimi/vlm-dental-models
   ```
   Account 2 pulls the checkpoint in seconds, restores optimizer and scheduler states, skips completed samples, and resumes seamlessly.
