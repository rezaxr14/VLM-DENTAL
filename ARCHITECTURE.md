# VLM-DENTAL: Master Architecture & Wiring Guide

This document maps out every file in the `VLM-DENTAL` repository, explaining its role in the overarching system architecture. 

* **Terminology Note**: In standard software architecture, a `dental_agent/` folder is called the **Source Package** (or Root Module), the `scripts/` folder contains **CLI Entrypoints** (or Runners), and the `tests/` folder contains the **Test Suite**.

---

## 1. Core Source Package (`dental_agent/`)
This is the heart of the project containing all reusable logic. It is imported by scripts and notebooks.

### `dental_agent/agent/` (Agent Loop & Parsing)
*Logic for the autonomous agent's decision-making cycle.*
- `langgraph_loop.py`: **The primary execution loop (LangGraph-orchestrated).** Takes an input X-ray image, a system prompt, and a `ToolRegistry`, builds a `StateGraph` with `reasoning` → `tools` conditional edges, runs the model through real tool executions turn-by-turn, and outputs a complete multi-turn trajectory dict (`turns`, `tool_calls`, `final_answer`, `messages`).
- `loop.py`: **GRPO rollout loop.** Not legacy — this is the real, unhinted policy-rollout loop `grpo.py`'s `collect_grpo_group` actually calls (`run_agent`). `langgraph_loop.py` is trace-gen's loop, not a replacement for this one; the two exist for genuinely different jobs (see `tool_dispatch.py`'s docstring).
- `tool_dispatch.py`: **Shared tool-call dispatcher**, used by both `loop.py` and `langgraph_loop.py` so they can't independently diverge on which tools need the source image and which image to pass — a real bug of exactly that kind (GRPO compounding crops turn-over-turn instead of always using the original image) was fixed by introducing this. Add new image-consuming tools to `IMAGE_CONSUMING_TOOLS` here, not inside either loop.
- `parsing.py`: **The JSON extractor.** Takes raw LLM text outputs (including mixed XML/Markdown/truncated text), safely isolates the JSON, and outputs a clean Python dictionary representing the chosen action or final answer.
- `prompts.py`: **The instruction sets.** Takes a list of registered tools, dynamically formats them, and outputs the final text system prompts injected into the LLM context. Also contains `NO_TOOLS_SYSTEM_PROMPT` and `ZERO_SHOT_PROMPT` for baseline evaluation.
- `visualization.py`: **The rendering utility.** Takes trajectory data and coordinates, and outputs annotated images with bounding boxes and tool results drawn on them for visual debugging.

### `dental_agent/tools/` (Agent Capabilities)
*The individual tools the VLM can invoke. 8 registered by `ToolRegistry.create_default()`.*
- `registry.py`: **The tool manager.** `ToolRegistry.create_default()` registers all built-in tools. Takes a tool name string and dictionary of arguments from the agent, routes it to the correct python function below, and outputs the result back to the agent loop. Any new tool MUST be registered here.
- `zoom_crop.py`: **Cropping tool.** Takes an input image, a bounding box, and an optional `padding_frac` (how much surrounding context to include), and outputs a cropped, high-resolution image of that region.
- `windowing.py`: **Contrast mapping tool.** Takes an input image and a tissue preset string (e.g., "bone"), or explicit `center`/`width` values to override a preset exactly, and outputs a contrast-adjusted image mimicking a CT scan.
- `denoise.py`: **Filtering tool.** Takes an input image, a method string ("bilateral" or "median"), and a continuous `strength` (0.0-1.0), and outputs a smoothed image with reduced grain/noise.
- `contralateral.py`: **Comparison tool.** Takes an input image, a bounding box, and a jaw quadrant integer (which now actually constrains the mirror search to the same jaw half, upper 1-2 / lower 3-4), calculates the opposite side, and outputs a side-by-side composite image for symmetry comparison.
- `nudge.py`: **Correction tool (nudge_crop).** Takes a bbox the agent was already given plus a shift (`dx_frac`/`dy_frac`) and/or rescale (`scale`), and outputs the adjusted coordinates — data only, not an image; pair with `zoom_crop` to view the result. Lets the agent correct `locate_tooth`'s output instead of just trusting it.
- `grounding.py`: **AI detection tool (locate_tooth).** Takes an input image and an FDI tooth number, passes it through our trained YOLOv8m model (5-fold cross-validation, Multi-Dataset val mAP50 ≈ 0.9593, baseline DENTEX-only ≈ 0.8729), and outputs bounding box coordinates locating the tooth. Requires `GROUNDING_MODEL_PATH` or auto-detects from `data/models/yolo_cv_best/weights/best.pt`.
- `fdi.py`: **Dental logic helper.** Takes quadrant and tooth position integers, handles the math for FDI two-digit tooth numbering, and outputs standardized positional data.
- `contrast.py`: **Contrast tool (enhance_contrast).** Takes an input image and a multiplicative `factor` (not alpha/beta — the function signature is factor-only), and outputs a contrast-adjusted image. Existed as a function for a while but wasn't actually registered until recently — was silently unreachable.
- `synthetic.py`: **Mock tools.** Takes mock arguments, used exclusively for testing the agent loop without real models, and outputs dummy responses.

### `dental_agent/training/` (Pipelines & RL)
*The heavy-lifting logic for fine-tuning and reinforcement learning.*
- `api_pool.py`: **The dual-pool LLM client router.** Manages two independent pool singletons:
  - **`APIUsageTracker`** (verifier): Paces external APIs (NVIDIA NIM, Groq, OpenRouter, Gemini) with per-provider cooldowns and daily caps. Fails fast on exhaustion. State persists to `data/provider_pool_state.json`.
  - **`GeneratorPool`**: Same architecture but for external-API generation when no local GPU is available. Uses separate env vars (`GENERATOR_COOLDOWN_SECONDS`, default 60s; `GENERATOR_RPD_LIMIT`, default 50). State persists to `data/generator_pool_state.json`. Bypassed entirely when `GENERATOR_PROVIDER=local`.
  - Also contains `APISessionPool` (cached OpenAI-compatible clients), `call_llm()` (universal caller supporting `auto_verifier` and `auto_generator` routing), and `verify_local_server_health()`.
- `trace_generation.py`: **The dataset synthesizer (decoupled pipeline).** Two operational modes:
  - **Generate**: `generate_interactive_trajectory()` → drives a frontier LLM (primarily Gemini 3.5 Flash Lite or an NVIDIA NIM-hosted model via `api_pool.py`'s provider pool; a self-hosted Qwen/Qwen3.5-9B via local vLLM is also a supported provider but not the primary one in practice) through a real LangGraph tool-execution loop (ground-truth-directed), outputs raw trajectory dicts to `train_cot_traces_unverified.jsonl`.
  - **Verify**: `verify_pending()` → reads unverified traces, runs cross-family verification via `ProviderPool`, promotes passing traces to `train_cot_traces.jsonl`.
  - Also contains `build_trace_example()` (canonical per-image pipeline) and `run_aim1_batch()` (batch wrapper with retry).
- `sft.py`: **The supervised trainer.** Takes the generated JSONL traces and base model architecture, formats them using a multi-modal collator, and outputs fine-tuned Qwen model weights.
- `grpo.py`: **The RL algorithm.** Takes the SFT model weights and new training data, implements Group Relative Policy Optimization (computing KL-divergence penalties and dual-adapter memory swapping), and outputs highly-optimized RL model weights.
- `detector.py`: **Faster R-CNN detector architecture (torchvision, not YOLO/Ultralytics).** Originally built both to train an FDI-position grounding detector (backing a `locate_abnormal_teeth` tool) and to supply a reusable detector architecture for the diagnosis-baseline comparison. The former use was removed entirely -- the project decided the agent finds and corrects abnormal-tooth grounding via `locate_tooth` (the actual, live YOLO detector -- trained by `scripts/train_grounding_tool.py`, a *different* file despite the similar subject matter) + `nudge_crop`'s self-correction loop, not a second learned detector backend. What remains here (`build_stage0_detector`, `detection_collate_fn`, the dataset classes, `compute_iou`) is kept because `dental_agent/evaluation/diagnosis_baseline.py` reuses it for the paper's "prior supervised detector" comparison baseline. See this module's own docstring and `roadmap.md`'s changelog for the full removal reasoning.
- `rewards.py`: **Training feedback connector.** Takes the current policy outputs during RL training, routes them through the reward functions, and outputs the loss gradients.

### `dental_agent/model/` (VLM Backbone)
*Loading and inferencing the base Qwen multimodal model.*
- `backbone.py`: **The model loader.** Takes model configuration settings, initializes the Qwen/Qwen3.5-9B multimodal model with 4-bit quantization and LoRA adapters, and outputs the PyTorch model object.
- `inference.py`: **The generation engine.** Takes tokenized inputs and image arrays, runs the PyTorch forward pass, and outputs generated text strings and token IDs.
- `checkpoints.py`: **The save manager.** Takes trained model states in memory, and outputs saved LoRA weights to disk/Drive (or vice-versa for loading).

### `dental_agent/rewards/` (RL Feedback Systems)
*Functions that score the agent's behavior during GRPO.*
- `components.py`: **Individual rubrics.** Takes an agent trajectory and ground-truth labels, and outputs a scalar score (e.g., +1 for correct format, +2 for correct diagnosis).
- `composite.py`: **The final grader.** Takes all individual component scores from a trajectory, mathematically combines them, and outputs a final unified scalar reward for the RL update step.
- `judge.py`: **LLM-as-a-judge.** Takes a trajectory's textual reasoning, prompts a powerful API model to grade its subjective clinical quality, and outputs a qualitative score.

### `dental_agent/data/` (Dataset Handling)
- `dentex.py`: **The DENTEX loader.** Takes the raw DENTEX JSON/image dataset files from disk, and outputs clean pandas DataFrames ready for training. Also owns `dentex_row_to_fdi()` -- the single source of truth for DENTEX's 0-indexed-to-FDI conversion (see `roadmap.md` and `.agents/rules/vlm_dental.md` Rule 1 -- this function existing and being centrally called is a direct fix for a real, previously-silent scoring bug).
- `dataset_catalog.py`: **The prioritization reference.** A transcription of a peer-reviewed systematic review (Uribe et al. 2024) of 16 public dental imaging datasets, each flagged with `has_diagnosis_labels` -- the load-bearing distinction between "feeds diagnosis trace-gen" (DENTEX, so far) and "grounding-only, expands `locate_tooth`'s training corpus" (everything else). Read this module's docstring before adding a new dataset.
- `tufts.py`: **Tufts Panoramic Dataset loader.** Image discovery and tooth bounding box parser implemented (`load_tufts_tooth_boxes`), parsing 25,184 tooth annotations from 1,000 radiograph images. All 1,000 images, annotations, and polygon segmentations uploaded to Hugging Face Hub (`Reza-Nadimi/tufts-train-images`). Integrated into multi-dataset YOLO 5-fold cross-validation (2,339 combined images, 46,808 boxes; mAP50 = 0.8695 / Target mAP50 = 0.93).
- `tunisia_panoramic.py`: **Tunisia Panoramic Dental Xray Dataset loader (in progress).** Image discovery, VIA2 JSON parsing, and bbox-from-region geometry implemented and tested; FDI-position mapping (`_region_to_fdi`) is a deliberate `NotImplementedError` hard stop -- see `roadmap.md`'s Datasets section.
- `hf_dataset_utils.py`: **Shared per-image HF upload/download layer.** Dataset-agnostic "upload once, download only the image IDs a given slice needs" mechanism used by DENTEX, Tufts, and Tunisia alike, so each new dataset loader doesn't reimplement this.
- `slicing.py`: **Cross-dataset seeding/slicing utilities** for building consistent train/eval subsets across multiple combined datasets.
- `preprocessing.py`: **The image cleaner.** Takes raw X-ray images, applies resizing and normalization, and outputs standardized image arrays.
- `splits.py`: **The divider.** Takes the full dataset DataFrames, ensures images are safely split without data leakage, and outputs isolated train, validation, and test subsets.
- `statistics.py`: **The analyzer.** Takes dataset DataFrames, calculates class distributions (e.g., Caries vs Impacted teeth), and outputs statistical summaries.

### `dental_agent/evaluation/` (Benchmarking)
- `batch_runner.py`: **The concurrent tester.** Takes a test dataset and model, executes tests concurrently across hundreds of images, and outputs raw result logs.
- `metrics.py`: **The math engine.** Takes raw result logs, compares predictions to ground truth, and outputs standard ML metrics (Precision, Recall, F1).
- `baselines.py` & `diagnosis_baseline.py`: **The zero-shot evaluators.** Takes standard non-agentic VLMs, runs them on the dataset without tools, and outputs baseline accuracy scores.
- `ablations.py`: **The tool tester.** Takes the agent, iteratively disables one tool at a time, runs evaluations, and outputs how much accuracy is lost (measuring tool importance).
- `failure_analysis.py`: **The debugger.** Takes failed predictions, logs where and why the agent failed, and outputs categorized error reports for human review.
- `reporting.py`: **The visualizer.** Takes all calculated metrics, and outputs formatted markdown/CSV evaluation tables.
- `sweep.py`: **The optimizer.** Takes hyperparameter ranges, runs multiple evaluations, and outputs the optimal configuration settings.

### `dental_agent/paper/` (Documentation Generation)
*Modular scripts for building the massive methods/results appendix and extracting high-quality trace demonstrations.*
- `builder.py`: **The compiler.** Reads outputted markdown traces, dynamically injects verbatim prompts and charts, and outputs the fully-rendered `paper_notes.md` file.
- `prompts.py`: **Prompt extractor.** Programmatically grabs the exact string literals for system prompts from the core modules.
- `figures.py`: **Case study generator.** Extracts intermediate tool crops (e.g., `turn2_locate.png`, `turn5_zoom_nudged.png`) to visually prove self-correction in the paper.

### `dental_agent/utils/` (Shared Helpers)
- `serialization.py`: **JSON Encoder.** Takes complex Python objects (like PIL Images or numpy arrays), safely encodes them, and outputs standard JSON-compatible strings.
- `persistence.py`: **The cacher.** Takes intermediate pipeline results, saves them to disk to survive Colab crashes, and outputs loaded data upon restart.
- `environment.py`: **The config loader.** Takes `.env` files, parses them, and outputs secure environment variables for the system.
- `export.py`: **The formatter.** Takes internal dataset representations, and outputs them to different formats (like YOLO text files or COCO JSON).
- `reproducibility.py`: **The seed setter.** Takes integer seeds, injects them into PyTorch/Numpy/Python random states, and outputs a deterministically constrained environment.

---

## 2. CLI Entrypoints (`scripts/`)
These are the executable scripts you run from the terminal. They wire the core package logic to command-line arguments.

### Trace Generation & Verification
- **`run_trace_gen.py`**: **(Phase 1)** The primary trace generation script with two operational modes:
  - `--mode generate` (default): Runs the LangGraph loop for each DENTEX image, writes raw traces to `data/traces/train_cot_traces_unverified.jsonl`. No rate limit when `GENERATOR_PROVIDER=local`; uses `GeneratorPool` for external APIs.
  - `--mode verify`: Reads unverified traces, verifies each via `api_pool.py` (external API with strict pacing limits), promotes passing traces to `data/traces/train_cot_traces.jsonl`.
  - `--status-only`: Prints pool capacity and dataset progress without generating.
  - `--split train|validation`: DENTEX split to process.
  - `--max-images N`: Cap for the session.
  - `--pacing-delay 1.5`: Inter-request delay (seconds).
  - `--k 1`: Candidate traces per image.
  - `--max-tokens N`: Tokens per generator turn (CLI argument takes strict precedence over `.env`).
  - `--min-turns N`: Floor on the turn budget.
  - `--turns-per-finding-buffer N`: Buffer added per finding.
  - `--total-slices N` and `--slice-index N`: Distributed generation controls.
  - `--output PATH`: Override output file path.
- **`test_langgraph_loop.py`**: Quick smoke test — runs a single image through the LangGraph loop to verify tool calling works.
  - `--image PATH`: Path to test image.
  - `--model MODEL`: Model name for vLLM.
- **`test_aim1_trace.py`**: Integration test for the full Aim 1 pipeline (generate + verify) on a single example.

### Dataset & Model Training
- **`download_dataset.py`**: Takes hardcoded URLs, automates downloading, and outputs extracted multi-GB datasets to disk.
- **`upload_dataset_images_to_hf.py`**: Bundles one dataset's images into a COCO-shaped `train.json` + image files and uploads them to a lightweight per-image HF repo (via a `DATASET_BUNDLERS` registry -- `dentex`/`tufts`/`tunisia`), so downstream training can `hf_dataset_utils.py`-download only the image IDs a given slice actually needs instead of the full multi-GB archive.
- **`upload_tufts_polygons.py`**: Uploads Tufts polygon segmentations (`teeth_polygon.json`), expert/student annotations, and bounding boxes to `Reza-Nadimi/tufts-train-images`.
- **`prepare_yolo_dataset.py`**: Converts one or more datasets' annotations (via a `DATASET_LOADERS` registry -- DENTEX and Tufts live) into YOLO-formatted text files for Ultralytics training. Trains a 32-class detector, one class per FDI (quadrant, position) pair (`class_idx = (quadrant-1)*8 + (position-1)`).
- **`train_grounding_tool.py`**: **(Phase 2)** Takes the YOLO dataset, supports multi-fold cross-validation training, automated remote checkpoint hydration from Hugging Face Hub, dual-table evaluation (both internal 5-fold CV splits and held-out test benchmark), and saves/syncs top-performing weights to `data/models/yolo_cv_best/` and Hugging Face Hub `yolo_cv`.
- **`train_sft.py`** / **`run_sft.py`**: **(Phase 3)** Takes the verified JSONL traces (`train_cot_traces.jsonl`) and base Qwen model, runs the supervised training loop with `QwenVLDataCollator`, and outputs a fine-tuned LoRA adapter.
  - `--dataset_path PATH`: Path to traces JSONL (default: `data/traces/train_cot_traces.jsonl`).
  - `--output_dir PATH`: Where to save weights.
  - `--batch_size N`, `--epochs N`: Training hyperparameters.
- **`run_grpo.py`**: **(Phase 5)** Takes the SFT model adapter and training dataset, runs GRPO reinforcement learning, and outputs the final optimized agent weights.
- **`run_eval.py`**: **(Phase 4)** Takes a trained model and test set, runs the evaluation pipelines, and outputs final accuracy metrics.

### Utilities
- **`verify_tools_on_real_data.py`**: Takes local images, runs tool functions, and outputs visualized results to manually verify tools are working.
- **`export_agent.py`**: Takes trained LoRA weights and the base model, merges them into a single structure, and outputs deployment-ready weights.
- **`export_prompt_demo.py`**: Interactively (via CLI or `input()`) extracts highly detailed, formatted markdown traces for specific image IDs, including visual walk-throughs of tool outputs.

---

## 3. Workspaces (`notebooks/`)
Interactive environments for Colab/Kaggle execution.

- **`VLM_Dental_Colab_TraceGen.ipynb`**: **The Data Engine.** Handles vLLM server startup, model caching, and the decoupled generate/verify trace pipeline. Sections:
  - §1 Mount & Clone (+ rename legacy traces to `.old`)
  - §2 Install dependencies
  - §3 Credentials (Colab Secrets tab)
  - §4 Model cache setup (`HF_HOME` → `data/models/vllm_cache/`)
  - §5 vLLM server startup (health-check polling)
  - §6a Generate traces (`--mode generate`)
  - §6b Verify traces (`--mode verify`)
  - §7 Status dashboard
  - §8 Download verified traces
- **`VLM_Dental_Colab_YOLO.ipynb`**: **The Grounding Trainer.** Handles YOLO dataset preparation, 5-fold cross-validation training, and optional trace syncing.
- **`VLM_Dental_Colab_SFT.ipynb`**: **The Teacher.** Isolated environment specifically for Phase 3 (SFT) using `train_cot_traces.jsonl`.
- **`VLM_Dental_Colab_GRPO.ipynb`**: **The Optimizer.** Isolated environment specifically for Phase 5 (RL/GRPO).

---

## 4. Data Files & Conventions (`data/`)

### Trace Files (`data/traces/`)
| File | Purpose | Written by |
|---|---|---|
| `train_cot_traces_unverified.jsonl` | Raw LangGraph-generated traces (not yet verified) | `run_trace_gen.py --mode generate` |
| `train_cot_traces.jsonl` | **Canonical** verified traces used by SFT/GRPO | `run_trace_gen.py --mode verify` |
| `train_cot_traces.jsonl.old` | Backup of legacy traces (built with external API keys) | One-time rename at setup |

### Model Artifacts (`data/models/`)
| Directory | Contents |
|---|---|
| `yolo_cv_best/weights/best.pt` | Best-fold YOLOv8m weights for `locate_tooth` |
| `vllm_cache/` | HuggingFace model cache for vLLM (`HF_HOME` override) |
| `qwen3_5_9b_sft/` | SFT LoRA adapter output |

### Rate-Limit State Files (`data/`)
| File | Pool | Persists |
|---|---|---|
| `provider_pool_state.json` | `ProviderPool` (verifier) | Daily call counts + cooldown timestamps |
| `generator_pool_state.json` | `GeneratorPool` (external API gen) | Daily call counts + cooldown timestamps |

---

## 5. Configuration & Setup (Root Files)
- `pyproject.toml` / `dental_agent.egg-info/`: **Package Installers.** Tells pip how to install the `dental_agent` package so it can be imported anywhere.
- `requirements.txt`: Pinned dependencies including `langgraph`, `vllm` (Colab-only), and API clients.
- `configs/`: YAML config files for `ProjectConfig` (loaded by `dental_agent/config.py`).
- `.env`: Private API keys and rate-limit settings (git-ignored). See `.env.example` for documentation.
- `.env.example`: Template with all env vars, default model names, and explanatory comments.
- `tests/`: Contains `conftest.py` and `test_*.py` files which use `pytest` to automatically verify code.
- `scripts/download_and_cleanup.py`: Script-level dataset download helper (also callable from notebooks).
