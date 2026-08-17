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

## Important Guidelines
- Read `AGENT_HANDOVER.md` for in-depth explanations of the agent's reasoning loop and tool suite.
- Read `ARCHITECTURE.md` to map out the exact files and directories.
- Read `roadmap.md` for current project status.
- **Rules File**: Check `.agents/rules/vlm_dental.md` for mandatory coding and logic rules (like FDI notation and the 0-Index quirk).

## Migration Notes (From previous versions)
- `dental_agent/agent/loop.py` handles tool dispatch generically by return type, ensuring tools like `window_level`, `denoise`, `locate_tooth`, and `contralateral_compare` execute correctly.
- Grounding Tool (`locate_tooth`) uses YOLOv8m (5-fold cross-validation) and is **live** in the agent loop. Ensure `GROUNDING_MODEL_PATH` is set in your `.env`.
- `api_pool.py` handles strict rate pacing and fail-fast exhaustion across OpenAI-compatible endpoints (no rotating pools).

If you are asked to build a new feature, always check `registry.py` to ensure your new tools are correctly mapped.
