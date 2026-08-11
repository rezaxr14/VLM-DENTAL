# VLM-DENTAL: Master Architecture & Wiring Guide

This document maps out every file in the `VLM-DENTAL` repository, explaining its role in the overarching system architecture. 

* **Terminology Note**: In standard software architecture, a `dental_agent/` folder is called the **Source Package** (or Root Module), the `scripts/` folder contains **CLI Entrypoints** (or Runners), and the `tests/` folder contains the **Test Suite**.

---

## 1. Core Source Package (`dental_agent/`)
This is the heart of the project containing all reusable logic. It is imported by scripts and notebooks.

### `dental_agent/agent/` (Agent Loop & Parsing)
*Logic for the autonomous agent's decision-making cycle.*
- `loop.py`: **The main execution loop (LangGraph-orchestrated).** Takes an input X-ray image and a user prompt, loops through LLM generations and real tool executions, and outputs a complete multi-turn trajectory object.
- `parsing.py`: **The JSON extractor.** Takes raw LLM text outputs (including mixed XML/Markdown/truncated text), safely isolates the JSON, and outputs a clean Python dictionary representing the chosen action or final answer.
- `prompts.py`: **The instruction sets.** Takes a list of registered tools, dynamically formats them, and outputs the final text system prompts injected into the LLM context.
- `visualization.py`: **The rendering utility.** Takes trajectory data and coordinates, and outputs annotated images with bounding boxes and tool results drawn on them for visual debugging.

### `dental_agent/tools/` (Agent Capabilities)
*The individual tools the VLM can invoke.*
- `registry.py`: **The tool manager.** Takes a tool name string and dictionary of arguments from the agent, routes it to the correct python function below, and outputs the result back to the agent loop.
- `zoom_crop.py`: **Cropping tool.** Takes an input image and a bounding box coordinate array, and outputs a cropped, high-resolution image of that specific region.
- `windowing.py`: **Contrast mapping tool.** Takes an input image and a tissue preset string (e.g., "bone"), and outputs a contrast-adjusted image mimicking a CT scan.
- `denoise.py`: **Filtering tool.** Takes an input image and a method string ("bilateral" or "median"), and outputs a smoothed image with reduced grain/noise.
- `contralateral.py`: **Comparison tool.** Takes an input image and a jaw quadrant integer, calculates the opposite side, and outputs a cropped image of the opposing side of the jaw for symmetry comparison.
- `grounding.py`: **AI detection tool (WIP — gated behind a detection-quality threshold, e.g. val mAP50 > 0.5, before use in the live agent loop).** Takes an input image, passes it through our trained YOLOv8m model (trained with 5-fold cross-validation), and outputs an array of bounding boxes locating teeth and pathologies.
- `fdi.py`: **Dental logic helper.** Takes quadrant and tooth position integers, handles the math for FDI two-digit tooth numbering, and outputs standardized positional data.
- `contrast.py`: **Basic contrast tool.** Takes an input image and a float alpha/beta value, and outputs a manually brightened or darkened image.
- `synthetic.py`: **Mock tools.** Takes mock arguments, used exclusively for testing the agent loop without real models, and outputs dummy responses.

### `dental_agent/training/` (Pipelines & RL)
*The heavy-lifting logic for fine-tuning and reinforcement learning.*
- `api_pool.py`: **The LLM client router.** Takes raw LLM requests and routes them to a locally-hosted Qwen3-VL-8B-Thinking vLLM endpoint (OpenAI-compatible, running inside the same Kaggle/Colab session) for trace generation, with the original Gemini/Anthropic API routing retained as a fallback/verifier path, and outputs the final LLM text responses.
- `trace_generation.py`: **The dataset synthesizer.** Takes raw dataset images and ground-truth annotations, drives a locally-hosted Qwen3-VL-8B-Thinking through a real LangGraph tool-execution loop (ground-truth-directed, not blind) to solve them interactively, and outputs a JSONL file of verified diagnostic trajectories.
- `sft.py`: **The supervised trainer.** Takes the generated JSONL traces and base model architecture, formats them using a multi-modal collator, and outputs fine-tuned Qwen-VL model weights.
- `grpo.py`: **The RL algorithm.** Takes the SFT model weights and new training data, implements Group Relative Policy Optimization (computing KL-divergence penalties and dual-adapter memory swapping), and outputs highly-optimized RL model weights.
- `detector.py`: **The YOLO trainer.** Takes COCO/YOLO formatted datasets, runs the Ultralytics training loop, and outputs a trained `.pt` bounding-box model.
- `rewards.py`: **Training feedback connector.** Takes the current policy outputs during RL training, routes them through the reward functions, and outputs the loss gradients.

### `dental_agent/model/` (VLM Backbone)
*Loading and inferencing the base Qwen-VL model.*
- `backbone.py`: **The model loader.** Takes model configuration settings, initializes the 3B/7B Qwen-VL model with 4-bit quantization and LoRA adapters, and outputs the PyTorch model object.
- `inference.py`: **The generation engine.** Takes tokenized inputs and image arrays, runs the PyTorch forward pass, and outputs generated text strings and token IDs.
- `checkpoints.py`: **The save manager.** Takes trained model states in memory, and outputs saved LoRA weights to disk/Drive (or vice-versa for loading).

### `dental_agent/rewards/` (RL Feedback Systems)
*Functions that score the agent's behavior during GRPO.*
- `components.py`: **Individual rubrics.** Takes an agent trajectory and ground-truth labels, and outputs a scalar score (e.g., +1 for correct format, +2 for correct diagnosis).
- `composite.py`: **The final grader.** Takes all individual component scores from a trajectory, mathematically combines them, and outputs a final unified scalar reward for the RL update step.
- `judge.py`: **LLM-as-a-judge.** Takes a trajectory's textual reasoning, prompts a powerful API model to grade its subjective clinical quality, and outputs a qualitative score.

### `dental_agent/data/` (Dataset Handling)
- `dentex.py`: **The DENTEX loader.** Takes the raw DENTEX JSON/image dataset files from disk, and outputs clean pandas DataFrames ready for training.
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

### `dental_agent/utils/` (Shared Helpers)
- `serialization.py`: **JSON Encoder.** Takes complex Python objects (like PIL Images or numpy arrays), safely encodes them, and outputs standard JSON-compatible strings.
- `persistence.py`: **The cacher.** Takes intermediate pipeline results, saves them to disk to survive Colab crashes, and outputs loaded data upon restart.
- `environment.py`: **The config loader.** Takes `.env` files, parses them, and outputs secure environment variables for the system.
- `export.py`: **The formatter.** Takes internal dataset representations, and outputs them to different formats (like YOLO text files or COCO JSON).
- `reproducibility.py`: **The seed setter.** Takes integer seeds, injects them into PyTorch/Numpy/Python random states, and outputs a deterministically constrained environment.

---

## 2. CLI Entrypoints (`scripts/`)
These are the executable scripts you run from the terminal (e.g. `python scripts/run_grpo.py`). They wire the core package logic to command-line arguments.

- `run_daily_trace_generator.py`: **(Phase 1)** Takes API keys and DENTEX datasets, runs the Teacher Loop, and outputs generated synthetic reasoning traces (JSONL).
- `prepare_yolo_dataset.py`: Takes DENTEX COCO annotations, converts bounding box coordinates, and outputs YOLO-formatted text files.
- `train_grounding_tool.py`: **(Phase 2)** Takes the YOLO dataset, triggers the YOLOv8 training loop, and outputs the trained model weights.
- `run_sft.py` / `train_sft.py`: **(Phase 3)** Takes the generated JSONL traces and base Qwen model, runs the supervised training loop, and outputs a Supervised Fine-Tuned (SFT) LoRA adapter.
- `run_grpo.py`: **(Phase 5)** Takes the SFT model adapter and training dataset, runs reinforcement learning, and outputs the final highly-optimized agent weights.
- `run_eval.py`: **(Phase 4)** Takes a trained model and test set, runs the evaluation pipelines, and outputs final accuracy metrics.
- `download_dataset.py`: Takes hardcoded URLs, automates downloading, and outputs extracted multi-GB datasets to disk.
- `run_detector.py` & `verify_tools_on_real_data.py`: Takes local images, runs tool functions, and outputs visualized results to manually verify tools are working.
- `export_agent.py`: Takes trained LoRA weights and the base model, merges them into a single structure, and outputs deployment-ready weights.

---

## 3. Workspaces (`notebooks/`)
Interactive environments for Colab/Kaggle execution.

- `VLM_Dental_Colab_Master.ipynb`: **The Data Engine.** Handles dataset downloading, YOLO training, and Trace Generation.
- `VLM_Dental_Colab_SFT.ipynb`: **The Teacher.** Isolated environment specifically for Phase 3 (SFT).
- `VLM_Dental_Colab_GRPO.ipynb`: **The Optimizer.** Isolated environment specifically for Phase 5 (RL).

---

## 4. Configuration & Setup (Root Files)
- `setup.py` / `pyproject.toml` (or `dental_agent.egg-info/`): **Package Installers.** Tells pip how to install the `dental_agent` package so it can be imported anywhere.
- `tests/`: Contains `conftest.py` and `test_*.py` files which use `pytest` to automatically verify code isn't broken during development.
- `.env`: Stores private API keys (never pushed to GitHub).
