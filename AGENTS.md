# VLM-DENTAL: Agent Handover Brief & Architecture Guide

Welcome to the **VLM-DENTAL** project, an Agentic AI system designed to train a Vision-Language Model (VLM) for expert-level dental radiology analysis.

This document serves as the master entry point for Antigravity IDE agents.

## Core Problem Statement & Architecture
Dental panoramic radiographs (OPGs) are complex images requiring spatial awareness and clinical reasoning. Standard VLMs fail at this because they lack the ability to "zoom in" on tiny pathologies or compare bilateral symmetry. 

**VLM-DENTAL** solves this by equipping the VLM with a suite of simulated radiologist tools. The VLM is trained to interact with the image through a "Chain of Thought" (CoT) agent loop, invoking tools turn-by-turn until it reaches a diagnosis.

### The LangGraph Trace-Gen Architecture
The generation of synthetic training data is now orchestrated by a real LangGraph loop (`dental_agent/agent/langgraph_loop.py`).
1. **Dynamic Tool Execution**: Every tool call runs for real against the source image. Tool outputs are appended dynamically. `<fake_tool_call>` is deprecated.
2. **Strict Verifier**: Uses `api_pool.py` to route to a verifier model dynamically (NVIDIA NIM, Groq, OpenRouter, Anthropic, or Gemini) to ensure the agent's diagnostic trace matches ground truth and doesn't hallucinate.
3. **Generation Provider Pool**: Generation is routed through a provider pool (`api_pool.py`), not a single hardcoded model — in practice primarily **Gemini 3.5 Flash Lite** or an NVIDIA NIM-hosted model. A self-hosted `Qwen/Qwen3.5-9B` served via local vLLM inside the Colab session is also a supported provider (e.g. for offline runs without API budget), but is not the primary path in practice.
4. **Configuration Precedence**: Parameters follow a strict priority order: **CLI Arguments > `.env` variables > Provider Defaults**. This allows safe fallbacks in `.env` while supporting dynamic scale-up via CLI flags (e.g., `--max-tokens`).
5. **Self-Correcting Grounding**: `locate_tooth` uses **Ground-Truth Grounding** during trace generation: if the requested tooth exists in the ground truth, its exact bounding box is returned, guaranteeing it is found. However, what the model is *shown* is then tiered-perturbed (small/big/none) independently of this perfect box — so the demonstrated accept-vs-nudge behavior stays valid even as the grounding tool improves, instead of needing traces regenerated. See `TRACE_GEN_CONFIG.md` for exact numbers and `_tool_node_factory`'s docstring for the full reasoning.

## Important Guidelines
- Read `AGENT_HANDOVER.md` for in-depth explanations of the agent's reasoning loop and tool suite.
- Read `ARCHITECTURE.md` to map out the exact files and directories.
- Read `roadmap.md` for current project status.
- Read `TRACE_GEN_CONFIG.md` for the exact hint/perturbation probabilities and offset ranges currently in use.
- **Rules File**: Check `.agents/rules/vlm_dental.md` for mandatory coding and logic rules (like FDI notation and the 0-Index quirk).

## Migration Notes (From previous versions)
- Tool dispatch (which tools need `image=`, and which image) now lives in **`dental_agent/agent/tool_dispatch.py`**, shared by both `langgraph_loop.py` (trace-gen) and `loop.py` (GRPO rollout) — it used to be duplicated per-loop, which is exactly how a real bug got in (GRPO was executing against a compounding `current_image` instead of `base_image`, diverging from trace-gen). Add new tools to `IMAGE_CONSUMING_TOOLS` in that one file, not to either loop directly.
- Grounding Tool (`locate_tooth`) uses YOLOv8m (5-fold cross-validation) and is **live** in the agent loop. Ensure `GROUNDING_MODEL_PATH` is set in your `.env`. Current validation mAP50 ≈ 0.5901 — see `TRACE_GEN_CONFIG.md`, not the older 0.647 figure still lingering in a couple of doc/figure files.
- `nudge_crop` (8th and last registered tool) lets the agent correct a bbox it was already given, shift+rescale, without re-running detection — data-only like `locate_tooth`, pair with `zoom_crop` to view the result. A separate, learned `locate_abnormal_teeth` tool (a never-trained Faster R-CNN specialist detector) was removed entirely rather than ever wired in: the project decided abnormal-tooth grounding is handled via `locate_tooth` + `nudge_crop`'s self-correction loop, not a second detector backend.
- Tools that previously exposed only a fixed preset/method now take continuous control parameters: `denoise(strength=0.0-1.0)`, `window_level(center=..., width=...)` as an override on top of `preset`. `enhance_contrast` (factor-controlled) is now actually registered — it existed as a function before but `create_default()` never wired it in.
- `api_pool.py` handles strict rate pacing and fail-fast exhaustion across OpenAI-compatible endpoints (no rotating pools).
- **Multi-dataset infrastructure**: `prepare_yolo_dataset.py` and `upload_dataset_images_to_hf.py` were generalized from DENTEX-only scripts into `DATASET_LOADERS`/`DATASET_BUNDLERS`-registry-driven ones. `dental_agent/data/dataset_catalog.py` catalogs 16 public dental datasets (from a systematic review) flagged by whether they carry diagnosis labels or are grounding-only -- this distinction changes what "add a new dataset" even means; read that module's docstring before adding one. Two grounding-only loaders are in progress (`tufts.py`, `tunisia_panoramic.py`), each intentionally hard-stopping at one unverified annotation-semantics question rather than guessing -- see Rule 12 in `.agents/rules/vlm_dental.md` and `roadmap.md`'s Datasets section for current state of each.
- **Multi-Finding Ground Truth Completeness (Rule 13)**: Never truncate an image's annotations DataFrame with `.iloc[0]`. Panoramic X-rays carry multiple findings per image ($1$ to $7$ findings). All evaluation, verification, and reward passes must evaluate the full ground-truth set via `match_multi_findings()` in `dental_agent/evaluation/metrics.py`.

If you are asked to build a new feature, always check `registry.py` to ensure your new tools are correctly mapped, and `tool_dispatch.py` if the tool needs the source image.

