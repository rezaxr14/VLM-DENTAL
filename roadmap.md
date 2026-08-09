# VLM-Dental Agent: Project Roadmap

This document outlines the end-to-end trajectory for building our autonomous dental diagnostic agent, derived directly from the **Think, Look, Measure** research proposal.

---

## ✅ Phase 1: Setup & Data Curation (Completed / In Progress)
- [x] Scaffolded the project structure (Agent, Tools, Training, Utils).
- [x] Configured environment-aware settings (local/Kaggle/Colab) and persistent directories.
- [x] Integrated automated downloading and extraction for the HuggingFace DENTEX dataset.
- [x] Implemented COCO-JSON annotation parsing and robust caching using Parquet DataFrames.
- [x] Engineered the **Hybrid Interactive Teacher Loop** for Chain-of-Thought (CoT) generation.
- [x] Built the **Strict Verifier** that evaluates the generated clinical reasoning and rejects ungrounded hallucinations via LLM-as-judge.
- [ ] Complete the download and extraction of the full 10GB DENTEX `training` split (Currently downloading).
- [ ] Run the automated `run_daily_trace_generator.py` pipeline across the full dataset to generate thousands of verified CoT traces.

---

## 🚀 Phase 2: Tool Suite + Stage 0 (Active Phase)
- [x] Built a dynamic `ToolRegistry` to expose Python functions to the VLM.
- [x] Implemented **Window Leveling** (`window_level`) with clinical presets: Bone, Enamel, and Soft Tissue.
- [x] Implemented **Denoising** (`denoise`) using Median and Bilateral filters.
- [x] Implemented **Zoom & Crop** (`zoom_crop`) for high-resolution inspection.
- [x] Implemented **Contralateral Compare** (`contralateral_compare`) for symmetry assessment.
- [x] Implemented **FDI numbering function** mapping tooth positions.
- [ ] **Stage 0 (Grounding Tool)**: Pretrain/fine-tune a segmentation/grounding tool (Faster R-CNN, Grounding DINO, or SAM/MedSAM variant) on DENTEX's quadrant and quadrant-enumeration subsets. This supervised detector allows the VLM to request bounding boxes/masks for specific teeth.

---

## 📅 Phase 3: Supervised Fine-Tuning (SFT) - Stage 1
- [ ] Wire the completed tool suite (including the R-CNN/Grounding tool) into the VLM's tool-calling interface.
- [ ] Set up `unsloth` or `trl` for efficient LoRA/QLoRA parameter-efficient fine-tuning (PEFT).
- [ ] Load the open-weight VLM backbone (starting with `Qwen2.5-VL-3B-Instruct` on free tiers).
- [ ] Train the VLM on the generated CoT dataset so it reliably produces well-formed reasoning + tool calls (learning the *shape* of the behavior).

---

## 📅 Phase 4: GRPO Smoke Test (3B)
- [ ] Implement the customized reward functions (Accuracy, Format, Tool Validity, Efficiency, plus optional LLM-judge grounding reward).
- [ ] Get the GRPO (Group Relative Policy Optimization) loop running end-to-end at small scale (group size G≈4) on free-tier constraints to validate rollout mechanics.

---

## 📅 Phase 5: First Real GRPO Run (3B → 7B)
- [ ] Run full-group (G≈8) GRPO on the 3B model on a dedicated RTX 4090.
- [ ] Once the 3B pipeline is validated, initialize the first 7B QLoRA SFT/GRPO attempts.

---

## 📅 Phase 6: Full-Scale Training & Reward Sweep
- [ ] Full 7B GRPO training on dedicated compute.
- [ ] Complete the reward-weight sweep to analyze the impact of different reward components.
- [ ] (Optional) Port to EasyR1 if the 4090 throughput becomes a bottleneck for the ablation matrix.

---

## 📅 Phase 7: Full Evaluation
- [ ] **Zero-shot baselines**: Evaluate GPT-4o / Qwen2.5-VL without tools.
- [ ] **SFT-only baseline**: Evaluate the Stage 1 agent without RL to isolate the contribution of RL (Hypothesis 2).
- [ ] **No-tools RL agent**: Evaluate the full agent reasoning over the whole image without tool access to isolate the contribution of tools (Hypothesis 1).
- [ ] **Full agent**: Evaluate the proposed SFT + tools + RL system.
- [ ] **Specialist detector**: Compare against a prior supervised specialist detector (e.g., YOLO/Faster R-CNN pipeline) to quantify the accuracy/interpretability trade-off (Hypothesis 3).
- [ ] **Detailed Error Analysis**: Disaggregate prediction errors into specific failure modes to understand agent behavior:
  - *Localization Errors*: Mistaken quadrant or FDI tooth position.
  - *Classification Errors*: Correct tooth located, but wrong disease diagnosed.
  - *Tool Misuse*: Agent failed to invoke the correct tool for the suspected lesion.
- [ ] **Reasoning-grounding check**: Run the LLM-judge protocol to check if final reasoning stays evidentially grounded in cited tool outputs.
- [ ] **Cross-dataset generalization**: Zero-shot evaluation on secondary datasets to check out-of-distribution performance.
