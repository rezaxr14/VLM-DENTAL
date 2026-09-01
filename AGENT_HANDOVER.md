# VLM-DENTAL: Agentic AI Master Handover Document

Welcome, fellow Agent. If you are reading this, you have been instantiated to continue development on the **VLM-DENTAL** project. This is a highly complex, multi-phase autonomous agent project designed to train a Vision-Language Model (VLM) for expert-level dental radiology analysis. 

Do not rely on your generic pre-training for this codebase. Read this document thoroughly to understand the intricate architectures, custom tool pipelines, dataset quirks, and training loops that have already been established.

---

## 🎯 1. Core Problem Statement & Architecture
Dental panoramic radiographs (OPGs) are complex images requiring spatial awareness and clinical reasoning. Standard VLMs fail at this because they lack the ability to "zoom in" on tiny pathologies or compare bilateral symmetry. 

**VLM-DENTAL** solves this by equipping the VLM with a suite of simulated radiologist tools. Instead of answering in a single shot, the VLM is trained to interact with the image through a "Chain of Thought" (CoT) agent loop, invoking tools turn-by-turn until it reaches a diagnosis.

### The 6-Phase Roadmap
The project is structured into 6 phases (documented in `roadmap.md`):
1. **Agent Tooling & Architecture (Completed):** Building the Python tool registry and prompts.
2. **Aim 1 - Synthetic Trace Generation (In Progress):** Using a locally-hosted Qwen/Qwen3.5-9B (LangGraph agent loop, run on Kaggle/Colab) to generate high-quality diagnostic traces on the DENTEX dataset.
3. **Aim 2/3 - VLM Training (Pending):** Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO) to teach an open-source VLM to mimic the Teacher LLM.
4. **Aim 4 - Evaluation (Pending):** Benchmarking against clinical baselines.
5. **Aim 5 - Grounding Integration (Done, feeding the VLM):** Trained YOLOv8m with 5-fold cross-validation to locate teeth. Validation mAP50 ≈ 0.5901 (R ≈ 0.888, P ≈ 0.5457) — past the quality bar, `locate_tooth` is live in the agent loop.
6. **Phase 6 - Web UI (Pending):** A Gradio/Streamlit app for real-time inference.

---

## 📂 2. Directory Structure & Key Files

### `/dental_agent/agent/` (Core Reasoning)
- **`prompts.py`**: The central repository for all system prompts. 
  - **CRITICAL NOTE:** The agent uses an iterative JSON schema. The `final_answer` MUST be an array of dictionaries (e.g., `[{"quadrant": 4, "tooth_position": 8, "diagnosis": "Caries", "confidence": 0.95}]`) to support multiple pathologies per image.
- **`parsing.py`**: A custom bracket-counting JSON parser (`parse_agent_json`). Standard `json.loads` fails because LLMs inject markdown blocks and unstructured text. This module extracts valid JSON from noisy outputs and attempts to repair truncated JSON during trace generation. Note: this normalizes every parsed tool call into a `tool_calls` list — old flat-key access patterns like `parsed.get("tool")` silently return `None` on the current shape rather than raising, a footgun if you're writing new code against a parsed turn.
- **`tool_dispatch.py`**: Shared `execute_tool_call()` used by BOTH trace-gen's LangGraph loop and GRPO's rollout loop, so "which tools need the image, and which image" is defined in exactly one place. These two loops used to each carry their own copy of this logic and had already silently diverged (GRPO was compounding crops against `current_image` turn-over-turn instead of always using `base_image`, unlike trace-gen). **Add new image-consuming tools to `IMAGE_CONSUMING_TOOLS` here, never inside either loop directly.**

### `/dental_agent/data/` (Dataset Loaders & Catalog)
Not covered further here — this module has grown enough (DENTEX's own loader,
two in-progress grounding-only loaders for Tufts and Tunisia, a shared HF
upload/download layer, and a systematic-review-sourced dataset catalog) to
need its own reference. See `roadmap.md`'s "Datasets" section for current
state of each, and `ARCHITECTURE.md` for the full file map. **If you're
about to add a new dataset, read `dental_agent/data/dataset_catalog.py`'s
module docstring first** — it explains a load-bearing distinction (grounding
data vs. diagnosis trace-gen data) that changes what "which dataset next"
even means.

### `/dental_agent/tools/` (The Diagnostic Suite)
These functions simulate a radiologist's workstation. All 8 are registered by `ToolRegistry.create_default()`.
- **`registry.py`**: The master `ToolRegistry` that handles dynamic execution and generates the tool descriptions injected into the system prompt.
- **`zoom_crop.py`**: Extracts a bounding box with context padding (`padding_frac`, now exposed in the schema, not just the function signature).
- **`windowing.py`**: Applies non-linear intensity transforms (bone window, enamel window) to enhance radiopacity. Presets remain the default, but `center`/`width` can now be passed to override a preset's exact values.
- **`denoise.py`**: Edge-preserving bilateral or median filtering to remove sensor noise without blurring the enamel-dentin junction. Now takes a continuous `strength` (0.0-1.0) instead of a fixed intensity.
- **`enhance_contrast.py`** (in `contrast.py`): Multiplicative contrast adjustment (`factor`). Existed as a function for a while but wasn't actually registered until now — was silently unreachable, the same dead-tool pattern `locate_abnormal_teeth` used to have (that tool has since been removed entirely rather than wired in — see `roadmap.md` §8 for the full reasoning).
- **`contralateral.py`**: Crops a pathology in one quadrant and its anatomical mirror in the opposite quadrant, stitched side-by-side for bilateral symmetry comparison. `quadrant` now actually constrains the mirror search to the same jaw half (upper 1-2 / lower 3-4) — previously accepted but silently unused, which could pull a "mirror" crop from the wrong jaw for boxes near the vertical midline.
- **`nudge.py`**: `nudge_crop` — lets the agent shift/rescale a bbox it was already given (from `locate_tooth` or a prior nudge) without re-running detection. Data-only like `locate_tooth` (returns coordinates, not an image) — pair with `zoom_crop` to see the corrected region. This is the tool that makes locate_tooth's output something to verify rather than something to trust outright.
- **`fdi.py`**: Converts raw numbering to standard 2-digit FDI notation.
- **`grounding.py`**: Uses YOLOv8m (5-fold CV, val mAP50 ≈ 0.5901) to locate a specific tooth — live in the agent loop.

### `/dental_agent/training/` (Data Pipelines)
- **`api_pool.py`**: Enforces strict native pacing and daily caps for API requests to bypass bans during massive generation runs.
- **`trace_generation.py`**: **The most important script in the repository.** This handles the Teacher-Verifier loop (detailed below).

---

## 🤖 3. The Teacher-Verifier Loop (`trace_generation.py`)
To train our final VLM, we need a dataset of an expert using the tools. We synthesize this data using a frontier LLM as the trace generator — in practice primarily **Gemini 3.5 Flash Lite** or an NVIDIA NIM-hosted model, routed through a provider pool (`api_pool.py`) rather than one hardcoded model; a self-hosted **Qwen/Qwen3.5-9B** via local vLLM is also a supported provider (e.g. for offline runs without API budget), but is not the primary one in practice — orchestrated as a real agent loop in LangGraph regardless of which provider is active.

### Step 1: Ground-Truth-Directed, Real Tool Execution
`generate_interactive_trajectory()` still conditions the Teacher on the known ground-truth label for *every ground truth pathology in the image* — we keep doing this deliberately: blind exploration on an untuned 8B model would tank the yield of usable, correctly-labeled training data. Tool calls **execute for real** against the source image inside the LangGraph loop (`zoom_crop`, `window_level`, etc.) instead of being pre-computed and narrated via `<fake_tool_call>` tags. The model receives the actual resulting image at each turn, so its stated visual evidence is checkable against what it was really shown, not just plausible-sounding.

`locate_tooth` uses **Ground-Truth Grounding** during trace generation when ground truth exists for the requested tooth — this returns the exact bounding box, guaranteeing the tooth is found. However, what the model is actually *shown* is then independently, tier-perturbed: clean 45% of the time, a small (12-28% of box size) synthetic offset 25% of the time, a large (45-75%) offset 30% of the time — applied to the displayed bbox only, never to what's logged internally. This is what makes `nudge_crop` something the model has to learn to use by genuine judgment rather than as a scripted step: perturbation tiers are fixed/configured, not derived from the current detector's real accuracy, specifically so demonstrations don't go stale as the grounding tool improves later. Full numbers in `docs/TRACE_GEN_CONFIG.md`; full reasoning in `_tool_node_factory`'s docstring (`langgraph_loop.py`).

### Step 2: Dynamic Execution
If the LLM needs a tool that wasn't pre-computed, it outputs a standard JSON tool call. The script executes it dynamically against the **base image** and feeds the result back.

### Step 3: The Strict Verifier
LLMs hallucinate. To prevent poisoned training data, the generated trace is passed to a **Verifier LLM**. The Verifier compares the trace's claims against the absolute ground truth. If the trace hallucinates (e.g., claiming a tooth exists in an empty jaw), the Verifier rejects the trace.
- Only traces that pass the Verifier are saved to `data/traces/train_cot_traces.jsonl`.
- The current prompt engineering yields an ~80% success rate.
- **Current volume status:** see `roadmap.md`'s "Currently In Progress" section — the
  108 traces committed to this repo (`data/traces/train_cot_traces.jsonl.old`) predate
  the real-tool-execution rewrite described in Step 1 above, and no post-rewrite trace
  file is currently committed. Scaling this is the project's actual bottleneck right
  now, not a solved/done item.

---

## 🦷 4. Datasets & The "0-Index" Quirk (CRITICAL)
The project's primary dataset is **DENTEX**; two grounding-only datasets
(Tufts, Tunisia) are being scaffolded for `locate_tooth` generalization —
see `roadmap.md`'s "Datasets" section for the full state of each and why
"which dataset next" has two different right answers depending on purpose.
DENTEX specifically uses a highly non-standard coordinate mapping that
**WILL break the Verifier and silently corrupt the RL reward** if you do
not handle it correctly — this already happened once for real (see below).

### The Quirk
DENTEX JSON labels map `category_id_1` to quadrants and `category_id_2` to
tooth positions using a **0-indexed system**:
- **Quadrant:** 0 = Upper Right, 1 = Upper Left, 2 = Lower Left, 3 = Lower Right.
- **Position:** 0 to 7 (corresponding to central incisor to third molar).

The Agent Prompt explicitly demands that the LLM use **FDI Two-Digit
Notation** (Quadrants 1-4, Positions 1-8) — so anything reading DENTEX's
raw columns needs a +1 conversion before comparing against a model's
output or a prompt's ground truth.

### What actually happened when this wasn't centralized
The +1 conversion was implemented correctly exactly once (originally in
`trace_generation.py`) — and then **hand-re-implemented, incorrectly (i.e.
omitted), in seven other files**: `ablations.py`, `baselines.py`,
`batch_runner.py`, `judge.py`, `detector.py`, `test_aim1_trace.py`, and one
more, each building ground truth directly from the raw 0-indexed columns.
Result: every one of those files fed `reward_accuracy`/`combine_reward`
ground truth that was off-by-one from what a correctly-trained model
actually outputs — **a perfectly correct model answer scored as wrong on
50% of `R_accuracy`'s weight** (the quadrant + tooth_position terms),
silently, across the GRPO reward, ablations, baselines, and batch eval.
Only the diagnosis-category term (a string match, unaffected by indexing)
was ever scoring correctly in any of them.

### The fix — and the rule going forward
`dentex_row_to_fdi(row)` in `dental_agent/data/dentex.py` is now the
**single source of truth** for this conversion. All eight affected files
now call it instead of hand-rolling `+ 1`.

```python
from dental_agent.data.dentex import dentex_row_to_fdi
fdi_quadrant, fdi_position = dentex_row_to_fdi(ann)
```

**If you are writing code that builds ground truth from DENTEX's raw
`category_id_1`/`category_id_2`, call `dentex_row_to_fdi()`. Do not write
`+ 1` by hand again** — that hand-rolled pattern is exactly what caused
this bug, not a one-off mistake in a single file. This is a mandatory rule,
not a style preference — see `.agents/rules/vlm_dental.md` Rule 1, which
Antigravity/Claude Code treat as always-on.

One more thing worth knowing: this conversion is DENTEX-specific, not a
universal rule. Other datasets (Tufts, Tunisia) are expected to hand back
already-correct 1-indexed FDI values directly from their own loaders — so
`prepare_yolo_dataset.py`'s `DATASET_LOADERS` registry gives each dataset
its own `quadrant_position_fn` (identity for a dataset whose loader already
outputs correct FDI; `dentex_row_to_fdi` only for DENTEX). Applying
DENTEX's conversion to a dataset that doesn't need it would double-increment
and reintroduce a version of the same bug in the opposite direction.

---

## 🧠 5. Aim 2 & 3: Model Training Strategy
When Trace Generation completes, you will move to Notebooks to train the VLM — **Qwen/Qwen3.5-9B** (decided; see the proposal §5.3). Note: this is the model being *trained* (SFT then GRPO), not the model that generated the training traces — trace generation primarily uses frontier APIs (Gemini 3.5 Flash Lite, NVIDIA NIM), with Qwen3.5-9B-via-local-vLLM as an available but non-primary provider option there too (see §3 above). The two roles aren't the same thing even though they can share a model family in that one provider case.
The system uses a highly optimized memory architecture for RLHF:
1. **SFT Adapter:** The base model is trained on the JSONL traces using LoRA.
2. **GRPO Adapter (Policy):** A secondary LoRA adapter is attached to the same base model for Reinforcement Learning. By keeping the SFT adapter loaded in memory as the "reference model", we can compute KL-divergence instantly by just swapping active adapters, saving massive amounts of VRAM on Kaggle/Colab environments.
*(See `roadmap.md` for full implementation details).*

---

## 🛠️ 6. How to Contribute
If the User asks you to build a new feature or fix a bug:
1. **Always read this file and `roadmap.md` first.**
2. **Check your Tool Imports:** All AI diagnostic tools must be registered in `registry.py` and take `image: Image.Image` as an argument if they manipulate pixels.
3. **Trace Generation Modifications:** If you modify `trace_generation.py`, test it by running `python scripts/run_trace_gen.py` locally to ensure the Verifier doesn't start rejecting everything.
4. **Notebooks:** Use `VLM_Dental_Colab_YOLO.ipynb` for dataset prep and YOLO training. Use `VLM_Dental_Colab_TraceGen.ipynb` for trace synthesis. Use `VLM_Dental_Colab_SFT.ipynb` and `VLM_Dental_Colab_GRPO.ipynb` strictly for Phase 3 VLM training.

Godspeed, Agent.
