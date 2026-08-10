# VLM-DENTAL: Agentic AI Handover Document

Welcome, fellow Agent! This document contains everything you need to know to get up to speed with the **VLM-DENTAL** project. Read this carefully before modifying any code.

## 🎯 Project Overview
VLM-DENTAL is an autonomous, agentic system designed to train a Vision-Language Model (VLM) for expert-level dental radiology analysis on panoramic X-rays (OPGs). The system generates its own synthetic training data through a Teacher-Verifier loop, and then fine-tunes the VLM using Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).

## 📂 Codebase Architecture

### 1. `dental_agent/agent/` (Core Agent Logic)
- **`prompts.py`**: Contains the system prompts, tool schemas, and few-shot examples. 
  - *Note:* The final output schema requires the agent to return a **list** of dictionaries to support multiple pathological findings per image.
- **`parsing.py`**: Robust JSON parsing to extract tool calls and final answers from noisy LLM outputs.

### 2. `dental_agent/tools/` (Diagnostic Tool Suite)
We simulate a real radiologist's workstation. These tools are provided to the LLM during trace generation:
- `zoom_crop`: Crops to a bounding box.
- `window_level`: Applies medical intensity windowing (e.g., `bone`, `enamel`).
- `denoise`: Edge-preserving noise reduction.
- `contralateral_compare`: Side-by-side symmetry comparison.
- `fdi_label`: Helper tool to convert raw numbers to FDI notation.
- `locate_tooth`: Grounding backend powered by YOLOv8.

### 3. `dental_agent/training/` (Data & Training Pipeline)
- **`trace_generation.py`**: The heart of Aim 1. This script runs the **Teacher-Verifier Loop**:
  1. A Teacher LLM (Gemini) generates a multi-turn reasoning trace (Chain of Thought).
  2. The script executes the requested tools dynamically or pulls from pre-computed results.
  3. A strict Verifier LLM checks the final trace against the absolute ground truth. If the Teacher hallucinates, the trace is rejected.
  4. Verified traces are saved to `data/traces/train_cot_traces.jsonl`.
- **`api_pool.py`**: Manages a pool of API keys to bypass rate limits during mass generation.

### 4. `scripts/` (Execution Entrypoints)
- **`run_daily_trace_generator.py`**: The main CLI script used to generate synthetic data.

## 🦷 Datasets and the "0-Index" Quirk (CRITICAL)
We primarily use the **DENTEX** and **Tufts** datasets. 
**CRITICAL BUG FIX WE ALREADY SOLVED:** DENTEX internally uses a 0-indexed coordinate system for quadrants (0=UR, 1=UL, 2=LL, 3=LR) and positions (0 to 7). However, the medical standard is the 1-indexed **FDI Notation** (Quadrants 1-4, Positions 1-8). 
- In `trace_generation.py`, the `_format_ground_truth()` function explicitly adds `+1` to convert DENTEX annotations to standard FDI *before* they are sent to the Verifier. 
- Do NOT alter this logic, or the Verifier will falsely reject correct FDI traces due to a mismatch with the 0-indexed ground truth!

## 🚀 Current State & Yield
- The YOLOv8 model (`grounding_tool`) is currently being trained to locate teeth.
- The `run_daily_trace_generator.py` script is currently running and generating the `train_cot_traces.jsonl` dataset. 
- Our recent prompt engineering and list-schema upgrades pushed the generation yield rate from ~21% to ~80%. 

## 🗺️ Next Steps (What you might be asked to do)
According to the user's `ROADMAP.md`, the upcoming tasks are:
1. **Phase 4 (Evaluation):** Build `scripts/run_eval.py` to evaluate the VLM against standard medical benchmarks.
2. **Phase 6 (Web UI):** Build a Gradio/Streamlit interface so the user can upload a panoramic X-ray and watch the VLM agent interact with tools in real-time.

Good luck! Ensure you always use the exact bounding boxes and maintain the list-format for `final_answer` if you touch the prompts again.
