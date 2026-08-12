# VLM-DENTAL: Agentic AI Master Handover Document

Welcome, fellow Agent. If you are reading this, you have been instantiated to continue development on the **VLM-DENTAL** project. This is a highly complex, multi-phase autonomous agent project designed to train a Vision-Language Model (VLM) for expert-level dental radiology analysis. 

Do not rely on your generic pre-training for this codebase. Read this document thoroughly to understand the intricate architectures, custom tool pipelines, dataset quirks, and training loops that have already been established.

---

## 🎯 1. Core Problem Statement & Architecture
Dental panoramic radiographs (OPGs) are complex images requiring spatial awareness and clinical reasoning. Standard VLMs fail at this because they lack the ability to "zoom in" on tiny pathologies or compare bilateral symmetry. 

**VLM-DENTAL** solves this by equipping the VLM with a suite of simulated radiologist tools. Instead of answering in a single shot, the VLM is trained to interact with the image through a "Chain of Thought" (CoT) agent loop, invoking tools turn-by-turn until it reaches a diagnosis.

### The 6-Phase Roadmap
The project is structured into 6 phases (documented in `ROADMAP.md`):
1. **Agent Tooling & Architecture (Completed):** Building the Python tool registry and prompts.
2. **Aim 1 - Synthetic Trace Generation (In Progress):** Using a locally-hosted Qwen3-VL-8B-Thinking (LangGraph agent loop, run on Kaggle/Colab) to generate high-quality diagnostic traces on the DENTEX dataset.
3. **Aim 2/3 - VLM Training (Pending):** Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) to teach an open-source VLM to mimic the Teacher LLM.
4. **Aim 4 - Evaluation (Pending):** Benchmarking against clinical baselines.
5. **Aim 5 - Grounding Integration (Done, feeding the VLM):** Trained YOLOv8m with 5-fold cross-validation to locate teeth. Validation mAP50 ≈ 0.647 (R ≈ 0.90, P ≈ 0.588) — past the quality bar, `locate_tooth` is live in the agent loop.
6. **Phase 6 - Web UI (Pending):** A Gradio/Streamlit app for real-time inference.

---

## 📂 2. Directory Structure & Key Files

### `/dental_agent/agent/` (Core Reasoning)
- **`prompts.py`**: The central repository for all system prompts. 
  - **CRITICAL NOTE:** The agent uses an iterative JSON schema. The `final_answer` MUST be an array of dictionaries (e.g., `[{"quadrant": 4, "tooth_position": 8, "diagnosis": "Caries", "confidence": 0.95}]`) to support multiple pathologies per image.
- **`parsing.py`**: A custom bracket-counting JSON parser (`parse_agent_json`). Standard `json.loads` fails because LLMs inject markdown blocks and unstructured text. This module extracts valid JSON from noisy outputs and attempts to repair truncated JSON during trace generation.

### `/dental_agent/tools/` (The Diagnostic Suite)
These functions simulate a radiologist's workstation. 
- **`registry.py`**: The master `ToolRegistry` that handles dynamic execution and generates the tool descriptions injected into the system prompt.
- **`zoom_crop.py`**: Extracts a bounding box with context padding.
- **`windowing.py`**: Applies non-linear intensity transforms (bone window, enamel window) to enhance radiopacity.
- **`denoise.py`**: Edge-preserving bilateral filtering to remove sensor noise without blurring the enamel-dentin junction.
- **`contralateral.py`**: Extremely advanced tool that crops a pathology in one quadrant, computes its anatomical mirror in the opposite quadrant, flips it, and stitches them side-by-side so the LLM can compare bilateral symmetry.
- **`fdi.py`**: Converts raw numbering to standard 2-digit FDI notation.
- **`grounding.py`**: Uses YOLOv8m (5-fold CV, val mAP50 ≈ 0.647) to locate a specific tooth — live in the agent loop.

### `/dental_agent/training/` (Data Pipelines)
- **`api_pool.py`**: Manages a round-robin pool of Gemini API keys (`API_KEYS.json`) to bypass rate limits during massive generation runs.
- **`trace_generation.py`**: **The most important script in the repository.** This handles the Teacher-Verifier loop (detailed below).

---

## 🤖 3. The Teacher-Verifier Loop (`trace_generation.py`)
To train our final VLM, we need a dataset of an expert using the tools. We synthesize this data using a locally-hosted **Qwen3-VL-8B-Thinking** (served via vLLM inside the Kaggle/Colab session — not a remote API, and not the laptop, since this needs real GPU time), orchestrated as a real agent loop in LangGraph.

### Step 1: Ground-Truth-Directed, Real Tool Execution
`generate_interactive_trajectory()` still conditions the Teacher on the known ground-truth label and seeds crop coordinates from the ground-truth bounding box for *every ground truth pathology in the image* — we keep doing this deliberately: blind exploration on an untuned 8B model would tank the yield of usable, correctly-labeled training data. What changed is that tool calls now **execute for real** against the source image inside the LangGraph loop (`zoom_crop`, `window_level`, etc.) instead of being pre-computed and narrated via `<fake_tool_call>` tags. The model receives the actual resulting image at each turn, so its stated visual evidence is checkable against what it was really shown, not just plausible-sounding.

### Step 2: Dynamic Execution
If the LLM needs a tool that wasn't pre-computed, it outputs a standard JSON tool call. The script executes it dynamically against the **base image** and feeds the result back.

### Step 3: The Strict Verifier
LLMs hallucinate. To prevent poisoned training data, the generated trace is passed to a **Verifier LLM**. The Verifier compares the trace's claims against the absolute ground truth. If the trace hallucinates (e.g., claiming a tooth exists in an empty jaw), the Verifier rejects the trace.
- Only traces that pass the Verifier are saved to `data/traces/train_cot_traces.jsonl`.
- The current prompt engineering yields an ~80% success rate.

---

## 🦷 4. Datasets & The "0-Index" Quirk (CRITICAL)
The project relies on the **DENTEX** dataset for training.
DENTEX uses a highly non-standard coordinate mapping that **WILL break the Verifier** if you do not handle it correctly.

### The Quirk
DENTEX JSON labels map `category_id_1` to quadrants and `category_id_2` to tooth positions using a **0-indexed system**:
- **Quadrant:** 0 = Upper Right, 1 = Upper Left, 2 = Lower Left, 3 = Lower Right.
- **Position:** 0 to 7 (corresponding to central incisor to third molar).

### The Fix
The Agent Prompt explicitly demands that the LLM use **FDI Two-Digit Notation** (Quadrants 1-4, Positions 1-8). 
To prevent the Verifier from falsely rejecting a trace due to a numbering mismatch, `trace_generation.py` intercepts the ground truth and translates it into FDI notation using this math:
```python
fdi_quadrant = int(ann.get("category_id_1", 0)) + 1
fdi_position = int(ann.get("category_id_2", 0)) + 1
```
**DO NOT REMOVE THIS LOGIC.** The LLM only understands FDI notation. If you pass 0-indexed quadrants to the Verifier, it will reject perfectly valid traces.

---

## 🧠 5. Aim 2 & 3: Model Training Strategy
When Trace Generation completes, you will move to Notebooks to train the VLM — **Qwen3-VL-8B-Thinking** (decided; see the proposal §5.3), the same backbone used as the trace-generation teacher.
The system uses a highly optimized memory architecture for RLHF:
1. **SFT Adapter:** The base model is trained on the JSONL traces using LoRA.
2. **GRPO Adapter (Policy):** A secondary LoRA adapter is attached to the same base model for Reinforcement Learning. By keeping the SFT adapter loaded in memory as the "reference model", we can compute KL-divergence instantly by just swapping active adapters, saving massive amounts of VRAM on Kaggle/Colab environments.
*(See `ROADMAP.md` for full implementation details).*

---

## 🛠️ 6. How to Contribute
If the User asks you to build a new feature or fix a bug:
1. **Always read this file and `ROADMAP.md` first.**
2. **Check your Tool Imports:** All AI diagnostic tools must be registered in `registry.py` and take `image: Image.Image` as an argument if they manipulate pixels.
3. **Trace Generation Modifications:** If you modify `trace_generation.py`, test it by running `python scripts/run_daily_trace_generator.py` locally to ensure the Verifier doesn't start rejecting everything.
4. **Notebooks:** Use `Master_Notebook.ipynb` for dataset prep and YOLO training. Use `SFT_Notebook.ipynb` and `GRPO_Notebook.ipynb` strictly for Phase 3 VLM training.

Godspeed, Agent.
