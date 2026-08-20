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
3. **Local Generation Server**: Generation uses `Qwen/Qwen3.5-9B` served via a local vLLM endpoint inside the Colab session for zero-latency, free tool execution loops.
4. **Configuration Precedence**: Parameters follow a strict priority order: **CLI Arguments > `.env` variables > Provider Defaults**. This allows safe fallbacks in `.env` while supporting dynamic scale-up via CLI flags (e.g., `--max-tokens`).
5. **Self-Correcting Grounding**: `locate_tooth`'s search_region_hint is always applied (`hint_probability=1.0`) for the best real detection, but what the model is *shown* is then tiered-perturbed (small/big/none) independently of the real detector's accuracy — so the demonstrated accept-vs-nudge behavior stays valid even as the grounding tool improves, instead of needing traces regenerated. See `TRACE_GEN_CONFIG.md` for exact numbers and `_tool_node_factory`'s docstring for the full reasoning.

## Important Guidelines
- Read `AGENT_HANDOVER.md` for in-depth explanations of the agent's reasoning loop and tool suite.
- Read `ARCHITECTURE.md` to map out the exact files and directories.
- Read `roadmap.md` for current project status.
- Read `TRACE_GEN_CONFIG.md` for the exact hint/perturbation probabilities and offset ranges currently in use.
- **Rules File**: Check `.agents/rules/vlm_dental.md` for mandatory coding and logic rules (like FDI notation and the 0-Index quirk).

## Migration Notes (From previous versions)
- Tool dispatch (which tools need `image=`, and which image) now lives in **`dental_agent/agent/tool_dispatch.py`**, shared by both `langgraph_loop.py` (trace-gen) and `loop.py` (GRPO rollout) — it used to be duplicated per-loop, which is exactly how a real bug got in (GRPO was executing against a compounding `current_image` instead of `base_image`, diverging from trace-gen). Add new tools to `IMAGE_CONSUMING_TOOLS` in that one file, not to either loop directly.
- Grounding Tool (`locate_tooth`) uses YOLOv8m (5-fold cross-validation) and is **live** in the agent loop. Ensure `GROUNDING_MODEL_PATH` is set in your `.env`. Current validation mAP50 ≈ 0.5901 — see `TRACE_GEN_CONFIG.md`, not the older 0.647 figure still lingering in a couple of doc/figure files.
- `nudge_crop` (9th... actually 8th registered tool) lets the agent correct a bbox it was already given, shift+rescale, without re-running detection — data-only like `locate_tooth`, pair with `zoom_crop` to view the result.
- Tools that previously exposed only a fixed preset/method now take continuous control parameters: `denoise(strength=0.0-1.0)`, `window_level(center=..., width=...)` as an override on top of `preset`. `enhance_contrast` (factor-controlled) is now actually registered — it existed as a function before but `create_default()` never wired it in.
- `api_pool.py` handles strict rate pacing and fail-fast exhaustion across OpenAI-compatible endpoints (no rotating pools).

If you are asked to build a new feature, always check `registry.py` to ensure your new tools are correctly mapped, and `tool_dispatch.py` if the tool needs the source image.
