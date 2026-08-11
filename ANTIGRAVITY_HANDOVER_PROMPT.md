# Handover Brief: LangGraph Trace-Gen Migration — VLM-DENTAL

Repo: https://github.com/rezaxr14/VLM-DENTAL

Before doing anything, read `AGENT_HANDOVER.md`, `ARCHITECTURE.md`, `roadmap.md`, and `dentex-agentic-vlm-proposal.md` in this repo — they were just updated to reflect the decisions below and are the current source of truth. If anything in existing code comments or notebooks contradicts these docs, the docs win; the code hasn't caught up yet.

## What changed (reflected in the docs, not yet in code)

1. **Trace generation moves off the Gemini API onto a local model.** `Qwen3-VL-8B-Thinking` is now the single backbone used both as the trace-generation teacher and the trained/deployed agent — the earlier staged 2B/3B-then-8B plan is dropped. It runs via vLLM inside a Kaggle or Colab notebook session specifically, not the developer's main PC.

2. **The trace-gen loop is rebuilt on LangGraph with real tool execution.** The previous `trace_generation.py` pre-computed tool outputs for every ground-truth pathology and had the model write `<fake_tool_call>` XML tags narrating a tool use it never actually performed. That's being replaced: tool calls now execute for real (real crop, real windowing, etc.) against the source image, and the real result — not a pre-computed stand-in — is what gets appended to the trajectory.

3. **Ground-truth direction is intentionally kept, not removed.** Generation still conditions on the known ground-truth label and seeds tool coordinates from the ground-truth bounding box. This is a deliberate yield/quality decision — blind exploration on an untuned model would produce too much wrong/unusable data for the dataset being built — not something to "fix away." The only thing that changed is that the tool calls are now real.

4. **Grounding tool is YOLOv8m (not YOLOv8x), trained with 5-fold cross-validation.** The CV code already exists but has not been tested yet — needs a real test pass before being trusted. `locate_tooth` stays gated out of the live agent loop until the detector clears a quality bar (suggested: val mAP50 > 0.5).

5. **The verifier stays model-agnostic everywhere** — in docs, code, and config. Don't hardcode a specific model name as the verifier; describe/configure it as "a different model family than the generator." This should be a config value, never a hardcoded string.

## What to actually build

1. Stand up local serving: vLLM (or an equivalent OpenAI-compatible server) hosting Qwen3-VL-8B-Thinking (4-bit) inside the notebook session, reachable on `localhost` — no tunneling needed since generation and serving are in the same session.
2. Rebuild `dental_agent/agent/loop.py` as a LangGraph graph: a model node pointed at the local vLLM endpoint, plus tool nodes wired to the existing functions in `dental_agent/tools/` (`zoom_crop.py`, `windowing.py`, `denoise.py`, `contralateral.py`, `fdi.py`) via `dental_agent/tools/registry.py`. Reuse `dental_agent/agent/prompts.py` and `dental_agent/agent/parsing.py` where possible — double-check `parsing.py` correctly handles Qwen3-VL-Thinking's `<think>...</think>` output, since it was originally built against Gemini's response format.
3. Update `dental_agent/training/api_pool.py` to route generation calls to the local vLLM endpoint, keeping the existing Gemini/Anthropic routing available only for the verifier step.
4. Rewrite `generate_interactive_trajectory()` in `dental_agent/training/trace_generation.py`: drop the pre-computation + `<fake_tool_call>` insertion step; drive the new LangGraph loop instead, still seeded with the ground-truth label and coordinates. Keep the final-answer-vs-ground-truth rejection sampling. Keep the LLM-judge verifier pass, but consider making it a sampled check rather than running it on every trace, now that generation itself is cheap and local.
5. Update `scripts/run_daily_trace_generator.py` and both `configs/default.yaml` / `configs/rtx4090.yaml` for the new local-serving setup (model path, vLLM port, quantization settings). Both configs should reference the same Qwen3-VL-8B-Thinking backbone — they should differ only in GRPO group size (4 vs. 8), not model size.
6. Test the 5-fold cross-validation code for the YOLOv8m grounding tool — it's written but unverified.
7. Check `gemini_key_state.json` at the repo root before any further commits — confirm it holds only rotation bookkeeping, not actual key material, since this repo is public.
8. Add `langgraph` and an OpenAI-compatible client (e.g. `langchain-openai`, pointed at the local vLLM `base_url`) to `requirements.txt` / `pyproject.toml`.

## Don't do

- Don't reintroduce `<fake_tool_call>` or any pre-computed-then-narrated tool output.
- Don't remove ground-truth conditioning from trace generation — that stays.
- Don't hardcode a specific verifier model name anywhere in code, configs, or docs.
- Don't wire `locate_tooth` into the live agent loop before the YOLOv8m detector clears its quality gate.

## Not in scope for this pass

SFT (`sft.py`), GRPO (`grpo.py`), and the evaluation harness (`dental_agent/evaluation/`) don't need changes yet — this pass is trace-generation only. The one thing worth flagging ahead for the GRPO stage: `rewards/components.py`'s tool-validity term should eventually check that a claimed finding was preceded by a real supporting tool call, not just that the format is valid — but that's for later, not this handover.
