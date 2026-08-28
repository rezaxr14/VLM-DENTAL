---
trigger: always_on
---

# VLM-DENTAL Custom Rules & Guidelines

When working on the `VLM-DENTAL` repository, adhere strictly to the following rules to ensure compatibility with the existing training pipelines and verification systems.

## 1. DENTEX "0-Index" Quirk (CRITICAL)
The DENTEX JSON labels map `category_id_1` to quadrants and `category_id_2` to tooth positions using a **0-indexed system** (Quadrant: 0=Upper Right ... 3=Lower Right. Position: 0 to 7).

**Rule:** The LLM and all Agent Prompts explicitly demand the use of **FDI Two-Digit Notation** (Quadrants 1-4, Positions 1-8).
- **DO NOT** pass 0-indexed quadrants to the Verifier or LLM. It will reject perfectly valid traces.
- **`dentex_row_to_fdi(row)` in `dental_agent/data/dentex.py` is the single source of truth for this conversion.** It used to be implemented once (in `trace_generation.py`) and then hand-re-implemented, incorrectly, in seven other files (`ablations.py`, `baselines.py`, `batch_runner.py`, `judge.py`, `detector.py`, `test_aim1_trace.py`, and one more) that built ground truth straight from the raw 0-indexed columns without the +1. That silently scored a correct model answer as wrong on 50% of `R_accuracy`'s weight across the GRPO reward, ablations, baselines, and batch eval, for an unknown period, because the code ran fine — it just computed something quietly wrong. All eight files now call `dentex_row_to_fdi()` instead.
- **DO NOT hand-write `+ 1` (or any other index-shift arithmetic) against `category_id_1`/`category_id_2` anywhere in this codebase.** Import and call `dentex_row_to_fdi()`. This is exactly the pattern that caused the bug above — a second hand-rolled copy of "the same" logic that silently drifted from the original.
- This conversion is DENTEX-specific, not universal. Other dataset loaders (Tufts, Tunisia, and any future one) are expected to hand back already-correct 1-indexed FDI values directly — `dentex_row_to_fdi()` must NOT be applied to their output, or it will double-increment and reintroduce a version of the same bug in the opposite direction. This is why `prepare_yolo_dataset.py`'s `DATASET_LOADERS` registry gives each dataset its own `quadrant_position_fn` rather than hardcoding one conversion for all of them.

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
  - **Generation** writes raw traces to `train_cot_traces_unverified.jsonl` as fast as the active provider allows -- in practice primarily **Gemini 3.5 Flash Lite** or an NVIDIA NIM-hosted model via `api_pool.py`'s provider pool (rate-limited per that provider's RPM/RPD caps), or, when `GENERATOR_PROVIDER=local`, a self-hosted Qwen/Qwen3.5-9B via vLLM with no rate limit -- but local is an available option, not the primary path in practice.
  - **Verification** reads the unverified file, verifies via external APIs (rate-limited by `ProviderPool`), and promotes passing traces to `train_cot_traces.jsonl`.
- **Rule:** Both phases must support resume — tracking processed image IDs so they can be interrupted and restarted without data loss.
- **Rule:** When `GENERATOR_PROVIDER=local`, the generator must NOT be stalled by verifier rate limits.
- **Rule:** For running multiple parallel Colab/Kaggle generate/verify workers (`--total-slices`/`--slice-index`/`--slice-seed` plus `--git-sync-every`), see `dental_agent/training/git_sync.py`'s module docstring and `.gitattributes` -- do not build a second, different parallel-sync mechanism without reading why that one works (union-merge, not a custom conflict resolver) and its one known gap (doesn't catch a duplicate image_id from a worker-configuration mistake -- see `check_for_duplicate_ids`).

## 7. Unified Backbone & Modular Notebook Architecture
- **Rule:** `Qwen/Qwen3.5-9B` is the standard, unified backbone that is actually TRAINED across Stage 1 SFT (QLoRA student) and Stage 2 GRPO (dual-adapter RL policy). It is a SEPARATE role from the frontier-LLM trace-generation teacher pool (primarily Gemini 3.5 Flash Lite / NVIDIA NIM via `api_pool.py`, see Rule 6) -- Qwen3.5-9B via local vLLM is one available generation provider too, but not the primary one in practice, and the two roles sharing a model family in that case is incidental, not a design requirement. Do not assume or write code that assumes Qwen3.5-9B is "the" trace-generation model.
- **Rule:** The monolithic `dentex_agentic_vlm_starter.ipynb` is deprecated (`deprecated_dentex_agentic_vlm_starter.ipynb`) and preserved solely for historical reference. All new workflows and experiments must use the modular `dental_agent/` library and dedicated notebooks in `notebooks/` (`VLM_Dental_Colab_TraceGen.ipynb`, `VLM_Dental_Colab_SFT.ipynb`, `VLM_Dental_Colab_GRPO.ipynb`).

## 8. No Retries on API Errors, Except an Explicit 429 Opt-In (CRITICAL)
- **Rule:** By default, if the generator or verifier hits a 429 Rate Limit, or ANY API error, we DO NOT retry. We stop immediately and exit. Retrying on 429s risks getting our API keys banned. `call_llm` must fail fast on these errors.
- **Exception:** setting `IGNORE_429=true` in the environment opts into up to 10 retries specifically for 429 errors, 5s apart, before hard-stopping (see `dental_agent/training/api_pool.py`'s `call_llm`). This is an explicit, deliberate opt-in for situations where a temporary rate limit is expected and tolerable (e.g. a long, paced batch run) -- it is NOT a default, and does not change the no-retry rule for any other error type (5xx still gets exactly one retry as before; everything else, including a 429 with `IGNORE_429` unset, still hard-stops immediately).

## 9. Version Control Etiquette (CRITICAL)
- **Rule:** NEVER automatically `git commit` or `git push` changes unless the user explicitly commands it. Always assume the user wants to review the code locally or in an implementation plan first before writing commits to the repository's history.

## 10. No API Calls for Testing (CRITICAL)
- **Rule:** NEVER waste the user's API call budget for testing or verification under ANY circumstances. If you need to verify a script or pipeline, you MUST use mock API calls, fake stubs, or local inference. Do not execute real calls to Groq, NVIDIA NIM, OpenRouter, or Gemini just to "see if it works."


## 11. STRICT NO GIT PUSH RULE (CRITICAL)
- **Rule:** NEVER UNDER ANY CIRCUMSTANCES run `git push` or `git commit`. DO NOT assume you should push changes even if you generated a model or wrote a script. The user must manually handle all git pushes. Your job is only to write code locally.
- **Scope note:** this rule governs IDE coding agents (Claude Code, Antigravity, or this rules file's reader generally) editing this repo directly -- it means an agent must never push code changes on the user's behalf without being asked. It does NOT prohibit `dental_agent/training/git_sync.py`, which is a pipeline feature the user explicitly opts into (via `GITHUB_TOKEN` + `--git-sync-every`) that runs under the user's own Colab session and only ever touches the exact trace-data paths it's told to sync. Do not refuse to help build or maintain that module citing this rule, and do not use that module's existence as precedent to justify pushing unrelated repo changes yourself -- both would be a misreading of this rule's actual scope.

## 12. Dataset Annotation Semantics: Honest Stop, Not a Guess (CRITICAL)
This project trains a medical diagnostic pipeline. A wrong label that looks
plausible is worse than no label, because nothing downstream catches it —
the code runs, the model trains, the numbers look reasonable, and the error
only surfaces (if ever) as unexplained underperformance much later. Rule 1
above is the concrete example of exactly this failure mode already having
happened once for real.

- **Rule:** When writing or extending a dataset loader (`dental_agent/data/*.py`), if a label's meaning, numbering convention, or category mapping is not confirmed directly against the real annotation file or its published documentation, do NOT guess or infer it from a plausible-sounding secondary source. Raise `NotImplementedError` with a docstring explaining exactly what to check and why, and implement everything else in the loader that IS independently verifiable (file discovery, parsing a published/standard annotation format, geometric bbox computation from a polygon, etc.).
- **Rule:** Keep the verifiable and unverifiable parts of a loader separable. Don't let one unresolved semantic question block code that doesn't actually depend on it — see `dental_agent/data/tunisia_panoramic.py` for the pattern: VIA JSON parsing and bbox-from-region geometry are fully implemented and tested, because they follow a published, dataset-independent format; only `_region_to_fdi` (a dataset-specific semantic mapping) hard-stops.
- **Rule:** A dataset with `has_diagnosis_labels=False` in `dental_agent/data/dataset_catalog.py` must never be wired into anything that expects a diagnosis category to exist (e.g. `category_id_3`, `R_accuracy`'s diagnosis term). It can only ever expand `locate_tooth`'s grounding training data. Check this flag before assuming a new dataset feeds trace-gen.
- Established precedent: `dental_agent/data/tufts.py` (mask-instance-to-FDI-position and abnormality-to-diagnosis-category mappings) and `dental_agent/data/tunisia_panoramic.py` (`_region_to_fdi`) both follow this pattern deliberately. If you're asked to "just fill in" one of these NotImplementedError stops, verify against the real file first — see each function's docstring for exactly what to check.

## 13. Multi-Finding Completeness: Never Truncate Ground Truth with `.iloc[0]` (CRITICAL)
Dental panoramic radiographs frequently contain multiple labeled abnormalities per image (e.g. 2 to 7 distinct teeth with Caries, Impacted teeth, or Periapical Lesions).
- **Rule:** NEVER take `.iloc[0]` on an image's annotations DataFrame (`annots_df[annots_df["image_id"] == id]`) or assume an image only carries one ground truth finding. Doing so throws away all other real clinical findings on that patient, causing correct model detections of other abnormal teeth to be falsely scored as errors (0% precision/recall).
- **Rule:** Every evaluation, reward calculation, and verification pass MUST process the **full list of ground truth findings** for that image using set-level matching (`match_multi_findings` in `dental_agent.evaluation.metrics`).
- **Rule:** When evaluating model predictions against multi-finding ground truths, compute full **FDI Localization Precision/Recall/F1** and **Exact Diagnostic Match Precision/Recall/F1** over the complete ground-truth set.

## 14. Preservation of Working Functionality (CRITICAL)
When refactoring, fixing bugs, or adding new features:
- **Rule:** Altering, modifying, enhancing, and combining existing functions is encouraged, but **deleting working functionality or dropping existing capabilities/flags/notebook cells is strictly forbidden**.
- **Rule:** Always check what existing scripts, notebooks, or tests depend on before modifying a component. Ensure all working pipelines (e.g., Option 7b LangGraph trace generation, baseline zero-shot runs) remain fully operational with backward-compatible defaults.

## 15. Mandatory Adversarial Code Self-Critique (CRITICAL)
Before concluding any coding task or marking an implementation complete, the agent MUST proactively critique and attack its own code across 5 core defensive vectors:
1. **Cross-Platform Path Normalization**: Normalize paths across Windows backslashes (`\`) and POSIX slashes (`/`). Never rely on raw `Path().name` or `os.path.basename` without slash normalization when parsing cross-environment paths.
2. **Null & Key Safety**: Guard against missing dictionary keys and `None` returns. Use `(val or "")[:N]` for string slicing to prevent `TypeError: 'NoneType' object is not subscriptable`.
3. **I/O & Search Bottlenecks**: Implement in-memory memoization caches (e.g. `_RESOLVED_PATH_CACHE`) to avoid repeating $O(N)$ filesystem scans and globbing across large datasets.
4. **Mutability Side-Effects**: Deep copy (`dict()`, `list()`) mutable nested structures before modifying values in-place (e.g., trajectory messages).
5. **Concurrency & Determinism**: Sort collections converted from sets (`sorted(set(...))`) before chunking/slicing to guarantee deterministic behavior across parallel workers.


