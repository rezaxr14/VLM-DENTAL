---
trigger: always_on
---

# Agent Tools, Runtime Execution & Model Architecture Rules

## 1. Tool Registration & Image Arguments
All AI diagnostic tools simulate a radiologist's workstation.
- **Rule:** Any new tool created must be registered in `dental_agent/tools/registry.py`.
- **Rule:** If a tool manipulates pixels, it MUST take `image: Image.Image` as an argument.
- **Rule:** Image-consuming tools must be registered in `IMAGE_CONSUMING_TOOLS` in `dental_agent/agent/tool_dispatch.py`.

## 2. Tool Execution & LangGraph
The LangGraph loop (`langgraph_loop.py`) runs tool calls dynamically and for real against the base image.
- **Rule:** Do not revert to or reintroduce the `<fake_tool_call>` or pre-computed-then-narrated tool output paradigms.
- **Rule:** Always enforce that the model uses tools (e.g., `locate_tooth`, `zoom_crop`, `enhance_contrast`) before issuing a final diagnosis.

## 3. No Hardcoded Verifier / Generator Models
- **Rule:** Do not hardcode a specific verifier or generator model name anywhere in Python code. All model defaults MUST be defined in `.env` and `.env.example`.
- **Rule:** `GeneratorPool` and `ProviderPool` (verifier) are independent singletons with separate rate-limit state files, cooldown timers, and RPD caps. They must NEVER share state.
- **Rule:** `api_pool.py` reads model names from env vars (`NVIDIA_VERIFIER_MODEL`, `GROQ_GENERATOR_MODEL`, `GEMINI_VERIFIER_MODEL`).

## 4. Unified Backbone & Modular Notebook Architecture
- **Rule:** `Qwen/Qwen3.5-9B` is the standard backbone TRAINED across Stage 1 SFT (QLoRA student) and Stage 2 GRPO (dual-adapter RL policy). It is separate from frontier teacher LLMs (Gemini / NVIDIA NIM).
- **Rule:** The monolithic `dentex_agentic_vlm_starter.ipynb` is deprecated (`deprecated_dentex_agentic_vlm_starter.ipynb`). All workflows must use `dental_agent/` and dedicated notebooks in `notebooks/` (`VLM_Dental_Colab_TraceGen.ipynb`, `VLM_Dental_Colab_SFT.ipynb`, `VLM_Dental_Colab_GRPO.ipynb`, `VLM_Dental_Colab_YOLO.ipynb`).
