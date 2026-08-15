---
trigger: always_on
---

# VLM-DENTAL Custom Rules & Guidelines

When working on the `VLM-DENTAL` repository, adhere strictly to the following rules to ensure compatibility with the existing training pipelines and verification systems.

## 1. DENTEX "0-Index" Quirk (CRITICAL)
The DENTEX JSON labels map `category_id_1` to quadrants and `category_id_2` to tooth positions using a **0-indexed system** (Quadrant: 0=Upper Right ... 3=Lower Right. Position: 0 to 7).

**Rule:** The LLM and all Agent Prompts explicitly demand the use of **FDI Two-Digit Notation** (Quadrants 1-4, Positions 1-8). 
- **DO NOT** pass 0-indexed quadrants to the Verifier or LLM. It will reject perfectly valid traces.
- `trace_generation.py` handles the translation automatically. DO NOT remove that logic.

## 2. Tool Registration & Image Arguments
All AI diagnostic tools simulate a radiologist's workstation. 
- **Rule:** Any new tool created must be registered in `dental_agent/tools/registry.py`.
- **Rule:** If a tool manipulates pixels, it MUST take `image: Image.Image` as an argument.

## 3. Tool Execution & LangGraph
The LangGraph loop (`langgraph_loop.py`) runs tool calls dynamically and for real against the base image.
- **Rule:** Do not revert to or reintroduce the `<fake_tool_call>` or pre-computed-then-narrated tool output paradigms.
- **Rule:** Always enforce that the model uses tools (e.g., `locate_tooth`, `zoom_crop`) before issuing a final diagnosis.

## 4. No Hardcoded Verifier/Generator Models
- **Rule:** Do not hardcode a specific verifier or generator model name anywhere in Python code. All model defaults MUST be defined in `.env` and `.env.example` so they are discoverable and changeable without reading the codebase.
- **Rule:** The `GeneratorPool` and `ProviderPool` (verifier) are independent singletons with separate rate-limit state files, cooldown timers, and RPD caps. They must NEVER share state.
- **Rule:** `api_pool.py` reads model names from env vars (e.g. `NVIDIA_VERIFIER_MODEL`, `GROQ_GENERATOR_MODEL`). The `.env.example` file documents the sensible defaults.


## 5. Trace File Naming Convention (CRITICAL)
- **Rule:** `data/traces/train_cot_traces.jsonl` is the CANONICAL verified trace file used by all downstream pipelines (SFT, GRPO, YOLO notebooks).
- The legacy version (built with external API keys in earlier project versions) has been renamed to `train_cot_traces.jsonl.old` and must not be deleted — it serves as a backup.
- Raw/unverified traces from the LangGraph generator are written to `data/traces/train_cot_traces_unverified.jsonl`.
- The verification pass reads from `_unverified.jsonl` and promotes passing traces to `train_cot_traces.jsonl`.
- **DO NOT** rename or create alternative output filenames (e.g. `cot_traces_aim1.jsonl`). All notebooks expect `train_cot_traces.jsonl`.

## 6. Decoupled Generation & Verification Pipeline
- **Rule:** Trace generation and verification are two independent phases that run at different speeds.
  - **Generation** writes raw traces to `train_cot_traces_unverified.jsonl` as fast as hardware allows (no rate limit when using local vLLM).
  - **Verification** reads the unverified file, verifies via external APIs (rate-limited by `ProviderPool`), and promotes passing traces to `train_cot_traces.jsonl`.
- **Rule:** Both phases must support resume — tracking processed image IDs so they can be interrupted and restarted without data loss.
- **Rule:** When `GENERATOR_PROVIDER=local`, the generator must NOT be stalled by verifier rate limits.

## 7. Unified Backbone & Modular Notebook Architecture
- **Rule:** `Qwen/Qwen3.5-9B` is the standard, unified backbone for all pipeline stages: Aim 1 Trace Generation (local vLLM teacher), Stage 1 SFT (QLoRA student), and Stage 2 GRPO (dual-adapter RL policy).
- **Rule:** The monolithic `dentex_agentic_vlm_starter.ipynb` is deprecated (`deprecated_dentex_agentic_vlm_starter.ipynb`) and preserved solely for historical reference. All new workflows and experiments must use the modular `dental_agent/` library and dedicated notebooks in `notebooks/` (`VLM_Dental_Colab_TraceGen.ipynb`, `VLM_Dental_Colab_SFT.ipynb`, `VLM_Dental_Colab_GRPO.ipynb`).

## 8. Multi-Provider Dynamic Verifier Pool
- **Rule:** The verifier is ALWAYS a dynamic pool (`ProviderPool`), never a single pinned provider or model.
- **Rule:** The pool automatically activates all providers configured in `.env` (NVIDIA NIM, Groq, OpenRouter, Gemini) and enforces independent 300s cooldowns and daily RPD quotas.
- **Rule:** `APIConfig` and `trace_generation.py` must always use `verifier_provider = "auto_verifier"` and `verifier_model = "auto_model"`. Never single out or hardcode a single provider (like Gemini alone).
