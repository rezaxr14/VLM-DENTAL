# Handover Brief: LangGraph Trace-Gen Migration — VLM-DENTAL

Repo: https://github.com/rezaxr14/VLM-DENTAL

Read `AGENT_HANDOVER.md`, `ARCHITECTURE.md`, `roadmap.md`, and `dentex-agentic-vlm-proposal.md` in this repo first — they were just updated and are the current source of truth. Most of the LangGraph migration described below has already been **implemented** (code attached / already committed to this branch), not just planned — your job is mostly to test it in the real Kaggle/Colab + vLLM environment and finish the pieces that need a live GPU to verify.

## Branching

Current work-in-progress branch has been renamed to `api-trace-gen` (the old Gemini/fake-tool-call implementation, preserved). All of the work below happened on a new `langgraph` branch created off it, so nothing on `api-trace-gen` was touched or lost.

## What's already implemented

1. **`dental_agent/agent/langgraph_loop.py` (new file).** A real LangGraph `StateGraph` that drives ground-truth-directed generation with actual tool execution — no more `<fake_tool_call>`. The model is still told the correct diagnosis and given ground-truth bounding boxes as a hint (kept deliberately), but every tool call now runs for real against the source image.

2. **`dental_agent/training/trace_generation.py` (rewritten).** `generate_interactive_trajectory()` now delegates to the LangGraph loop above instead of the old pre-compute-then-narrate scheme (~280 lines of fake-tool-call regex-stitching logic removed). `build_trace_example()` and `run_aim1_batch()` are unchanged in structure. The verifier provider/model is now resolved generically — see `_resolve_verifier()`: no model is hardcoded, it picks the first configured candidate from NVIDIA NIM / Groq / OpenRouter / Anthropic / Gemini (whichever has both an API key and a `*_VERIFIER_MODEL` set in `.env`), or honors an explicit `VERIFIER_PROVIDER`/`VERIFIER_MODEL` override.

3. **`dental_agent/training/api_pool.py` (extended).** Added NVIDIA NIM, Groq, OpenRouter, and a `"local"` provider (your vLLM server) — all four are OpenAI-compatible, so this is one new `call_llm()` branch and one cached-client factory (`APISessionPool.get_openai_compatible`), not four separate SDK integrations. Verified current base URLs: Groq `https://api.groq.com/openai/v1`, NVIDIA NIM `https://integrate.api.nvidia.com/v1`, OpenRouter `https://openrouter.ai/api/v1`, local vLLM `http://localhost:8000/v1` (configurable via `LOCAL_VLLM_BASE_URL`). **Known gotcha:** there are reports of NVIDIA NIM returning 403 with some OpenAI-compatible clients even with a valid key — if you hit that, try setting a custom `User-Agent` header on the client before assuming the key is bad.

4. **Two real bugs found and fixed** while reading the actual code (these were invisible before because the fake-tool-call scheme never really executed tools, so they never surfaced):
   - `dental_agent/agent/loop.py` special-cased tool dispatch by hardcoded name (`zoom_crop`, `enhance_contrast`, `locate_abnormal_teeth`) — but `enhance_contrast` and `locate_abnormal_teeth` aren't actually registered by `ToolRegistry.create_default()`, and `window_level`/`denoise`/`contralateral_compare`/`locate_tooth` all fell into the generic branch, which did `json.dumps()` on a `PIL.Image` and crashed. Fixed to dispatch generically by the tool's actual return type — this affects **eval and ablations too**, not just trace-gen, since they also go through `loop.py`.
   - `dental_agent/tools/grounding.py`'s `tool_locate_tooth(image, args: Dict)` didn't match how `ToolRegistry.execute(name, **kwargs)` actually calls tools (flat kwargs, not a nested dict) — every real call to `locate_tooth` would have raised a `TypeError`. Fixed to `tool_locate_tooth(image, tooth)`, matching every other tool's convention.

5. **Grounding tool is done, not gated.** YOLOv8m, 5-fold cross-validation, validation mAP50 ≈ 0.647 (R ≈ 0.90, P ≈ 0.588) — past the quality bar we'd set, so `locate_tooth` is live in the agent loop, not held back. `dental_agent/tools/grounding.py`'s model path is now read from `GROUNDING_MODEL_PATH` (env var) instead of hardcoded — point it at wherever your best fold's `best.pt` actually lives. **Open decision, not resolved by this pass:** whether to use a single best fold or ensemble predictions across all 5 — currently defaults to a single model path.

6. **Configs and `.env.example` updated**: single `Qwen3-VL-8B-Thinking` backbone in `configs/default.yaml` (was `Qwen3-VL-2B-Instruct`); `configs/rtx4090.yaml` no longer overrides model name (was silently diverging from default.yaml — now both configs use the same backbone, differing only in GRPO group size); `requirements.txt` has `langgraph==1.2.11` added; `.env.example` has new sections for local vLLM serving and the three new verifier candidates.

## What's NOT done — needs a real environment to finish

1. **Actually stand up vLLM** hosting Qwen3-VL-8B-Thinking inside the Kaggle/Colab session and confirm `LOCAL_VLLM_BASE_URL` reaches it. Nothing here was tested against a live model — I don't have GPU access to verify the LangGraph loop actually round-trips correctly with real generations.
2. **Check `dental_agent/agent/parsing.py`** handles Qwen3-VL-Thinking's `<think>...</think>` output correctly — it wasn't built with that in mind (it was built against Gemini's response format), and I didn't have a live model to test against. It's probably fine since it brace-matches for standalone JSON blocks, but verify with a real transcript.
3. **`scripts/run_daily_trace_generator.py` and `dental_agent/cli.py`** haven't been touched — check whether they need updating for the new default provider/model.
4. **Test the actual generation quality** — genuine tool-grounded trace yield may be lower than the old fake-tool-call scheme's, since the model now actually has to call tools correctly to see anything. Worth checking the verified-trace yield rate isn't collapsing before running this at scale.
5. **Check `gemini_key_state.json`** at the repo root before any further commits — confirm it holds only rotation bookkeeping, not real key material, since this repo is public.

## Don't do

- Don't reintroduce `<fake_tool_call>` or any pre-computed-then-narrated tool output.
- Don't remove ground-truth conditioning from trace generation — that stays.
- Don't hardcode a specific verifier model name anywhere in code, configs, or docs.
- Don't revert `loop.py`'s generic image-tool dispatch back to the hardcoded elif chain — that reintroduces the confirmed bug.

## Not in scope for this pass

SFT (`sft.py`), GRPO (`grpo.py`), and the evaluation harness (`dental_agent/evaluation/`) weren't touched — this pass is trace-generation only. Ahead for GRPO: `rewards/components.py`'s tool-validity term should eventually check that a claimed finding was preceded by a real supporting tool call, not just that the format is valid.
