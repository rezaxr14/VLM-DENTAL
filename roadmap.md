# VLM-DENTAL: Agentic Radiologist Project Roadmap

This document is the **single source of truth for project state** — what's
built, what's verified working, what's deliberately left unfinished (and
why), and what's next. If you are an AI agent (Claude Code, Antigravity, or
otherwise) picking this project up cold, read this document fully before
touching code. It is written to be sufficient on its own for orientation;
`AGENT_HANDOVER.md` goes deeper on the reasoning-loop internals,
`ARCHITECTURE.md` maps every file, and `.agents/rules/vlm_dental.md`
carries mandatory, always-on coding rules — but this file tells you where
the project actually stands right now, and *why* it stands there, which is
what most often gets a well-intentioned agent to implement something
plausible-looking but wrong.

**The core hypothesis being tested:** an agentic VLM — one that plans in
natural language and calls real tools (zoom, denoise, windowing,
contralateral comparison, a learned grounding tool, a self-correction tool)
before committing to a finding — trained with RL against a
ground-truth-anchored (not LLM-judge) composite reward, will outperform
zero-shot VLMs and prior supervised detectors on the DENTEX benchmark. See
`dentex-agentic-vlm-proposal.md` for the full, currently-up-to-date framing
against concurrent work (OralGPT-Plus, OralAgent).

**A general note on why this document has so much "why," not just "what":**
this codebase has already been bitten twice by code that *looked* correct
and silently wasn't — a 0-indexed vs. 1-indexed FDI mismatch that scored
50% of the RL accuracy reward wrong on every single training sample for
months before it was caught (§ below), and a reward function that ignored
its own `max_calls` parameter for an unknown period. Both were exactly the
kind of bug that passes a casual code review because the code *runs* — it
just computes something quietly wrong. The fix in both cases wasn't "be
more careful," it was "make the correct behavior impossible to bypass by
accident" (a single conversion function everything must call; a
self-documenting reward that explains its own design in its docstring).
Treat that as the house style: when you touch reward logic, dataset
loaders, or anything crossing a numbering-convention boundary, prefer a
loud, centralized, well-commented single source of truth over a second
hand-rolled copy of the same logic.

---

## 🟢 Completed Milestones (What We've Built)

### 1. Data Pipeline & Environment
- **DENTEX loading, preprocessing, and splitting** — built and tested
  (`dental_agent/data/dentex.py`). This is the project's only dataset with
  real, working diagnosis-labeled training data end-to-end.
- **Multi-dataset infrastructure** (new since the last major roadmap pass):
  `prepare_yolo_dataset.py` was generalized from a DENTEX-only script into
  a `DATASET_LOADERS`-driven one that can combine multiple datasets'
  annotations into a single YOLO training run, each tagged for
  collision-free filenames (`dataset_tag`). `scripts/upload_dataset_images_to_hf.py`
  was similarly generalized from two near-duplicate per-dataset upload
  scripts into one script with a `DATASET_BUNDLERS` registry, and
  `dental_agent/data/hf_dataset_utils.py` pulls the "upload once, download
  only the images a given slice needs" mechanism out into dataset-agnostic
  shared code. **Adding a new dataset is now: one loader module + one
  `DATASET_BUNDLERS` entry + one `DATASET_LOADERS` entry — not three
  parallel scripts that can independently drift.**
- **Dataset catalog** (`dental_agent/data/dataset_catalog.py`) — a
  transcription of a peer-reviewed systematic review (Uribe et al. 2024,
  J Dent Res, PMC11633071) of 16 public dental imaging datasets, plus
  targeted follow-up verification on specific candidates. This exists
  because "which dataset should we add next" turned out to have **two
  different right answers** depending on what you're trying to improve —
  see the "Datasets" section below, this is important enough to not bury.
- **Two additional dataset loaders scaffolded** (Tufts, Tunisia) — see
  "Datasets" section below for exactly what's implemented vs. blocked in
  each, and why.
- **Colab/Kaggle Architecture:** Modular, memory-efficient notebooks
  (`VLM_Dental_Colab_TraceGen.ipynb`, `VLM_Dental_Colab_YOLO.ipynb`,
  `VLM_Dental_Colab_SFT.ipynb`, `VLM_Dental_Colab_GRPO.ipynb`) with smart
  storage routing (ephemeral disk for heavy datasets, Google Drive for
  outputs/weights).
- **API Key Management:** `api_pool.py` with strict rate pacing, daily
  limits, and fail-fast exhaustion. **No retry on 429 or any API error by
  default** — see `.agents/rules/vlm_dental.md` Rule 8 — because retrying a
  429 risks getting the API key banned outright, which is worse than one
  failed run. `IGNORE_429=true` opts into up to 10 retries specifically for
  429s (5s apart) for situations where a temporary rate limit is expected
  and tolerable; this is an explicit opt-in, not a default, and doesn't
  change the no-retry behavior for any other error type.
- **Parallel Colab/Kaggle trace-gen workers:** `--total-slices`/
  `--slice-index`/`--slice-seed` (`scripts/run_trace_gen.py`, pre-existing)
  give each worker a disjoint, non-overlapping set of image IDs to process.
  New: `--git-sync-every` + `dental_agent/training/git_sync.py` let each
  worker periodically pull in the other workers' already-pushed traces and
  push its own, so N Colab instances can genuinely run in parallel and
  converge on one shared trace file instead of each accumulating an
  isolated, never-synced local copy. This is NOT git's default merge
  behavior for concurrent appends to the same file — verified by hand (see
  `git_sync.py`'s module docstring) that a plain 3-way merge reports a
  spurious conflict even for two workers' genuinely disjoint appends, since
  both sides modify the same "end of file" region relative to the same
  common ancestor. `.gitattributes`' `merge=union` rule (git's own built-in
  named merge driver for exactly this shape of file) is what actually makes
  this safe, not a custom conflict resolver — `git_sync.py` still detects
  and aborts on a REAL conflict (e.g. from editing a non-union-attributed
  path) rather than ever attempting automatic resolution on training data.
  One real gap union merge doesn't cover: a duplicate image_id ending up in
  the file twice from operator error (e.g. two workers accidentally given
  the same `--slice-index`) merges silently, since union merge has no
  concept of what counts as a semantic duplicate for this file's content —
  `check_for_duplicate_ids` runs automatically after every successful sync
  and warns (non-fatally) if this happens, but does not fix it. See
  `run_trace_gen.py`'s module docstring for a full parallel-worker usage
  example. Not yet exercised at real multi-worker scale on actual Colab
  infrastructure — validated so far via a local two-worker simulation
  against a throwaway bare repo (both the intended disjoint-append case and
  a deliberately-forced duplicate-id case), not a live multi-Colab run.

### 2. Autonomous Trace Generation (Phase 1)
- **Interactive Teacher Loop:** A real LangGraph agent loop
  (`langgraph_loop.py`) where a Teacher VLM sequentially invokes tools to
  hunt for pathologies, mimicking a radiologist's actual workflow, rather
  than narrating a pre-computed answer.
- **Cross-Family Verification:** A strict Verifier (a different model
  family than the generator) rejects hallucinated reasoning that isn't
  strictly supported by what the model was actually shown.
- **Bulletproof Parsing Engine** (`parsing.py`): Intelligently parses mixed
  XML/JSON outputs, repairs truncated API responses, and recovers from
  broken outputs without killing the trajectory loop.
- **Self-Correcting Grounding**: We can't let traces succeed by blindly trusting `locate_tooth`. `locate_tooth` uses **Ground-Truth Grounding** during trace generation when ground truth exists for the requested tooth (guaranteeing it is found) — but what the model is actually *shown* is intentionally perturbed independently of this (45% clean, 25% small offset, 30% large offset). This forces the LLM to use `nudge_crop` based on genuine visual judgment rather than as a scripted step, preventing trace demonstrations from going stale as the grounding tool improves. Full numbers in `docs/TRACE_GEN_CONFIG.md`; full reasoning in `_tool_node_factory`'s docstring (`langgraph_loop.py`).
- **Shared Tool Dispatch** (`dental_agent/agent/tool_dispatch.py`):
  Trace-gen's LangGraph loop and GRPO's rollout loop (`loop.py`) each used
  to carry their own copy of "which tools need the image, and which image"
  — and had already silently diverged (GRPO was compounding crops
  turn-over-turn against `current_image` instead of always using
  `base_image`, unlike trace-gen). Both loops now call one shared
  `execute_tool_call()`. **Add new image-consuming tools to
  `IMAGE_CONSUMING_TOOLS` here, never inside either loop directly** — that
  divergence is exactly how the bug got in the first time.
- **Tool Parameter Control:** Tools that used to expose only *which* tool
  to call now also expose *how much* effect to apply — `denoise(strength:
  0.0-1.0)`, `window_level(center=..., width=...)` as an override on top of
  a preset, `zoom_crop`'s `padding_frac` actually exposed in its schema,
  `contralateral_compare`'s `quadrant` argument now actually constrains the
  mirror search to the same jaw half (previously accepted but silently
  unused — could pull a "mirror" crop from the wrong jaw for boxes near the
  vertical midline), and `enhance_contrast` (existed as a working function
  for a while but was never actually registered — the same
  built-but-unreachable pattern `locate_abnormal_teeth` used to have, before
  it was removed entirely rather than wired in, see §8 below) is now
  wired into the registry.
- **No-tools SFT trace generation** (`--no-tools` flag, `generate_no_tools_trajectory`/
  `generate_only_no_tools` in `trace_generation.py`): generates SFT training
  data for baseline #3 in the proposal's evaluation plan
  (dentex-agentic-vlm-proposal.md §6: "Full agent without tool access...
  isolates the contribution of tools"). Baseline #3 needs the SAME SFT+RL
  recipe as the main system, just with tools removed from the environment —
  otherwise "no SFT warm-start" and "no tools" would be confounded variables
  and the ablation wouldn't cleanly isolate tool contribution. This is a
  single-turn, ground-truth-directed generation (a new
  `NO_TOOLS_COT_TEACHER_PROMPT` in `prompts.py`, with a `"thought"` field
  matching the tool-based traces' schema) via one direct API call — no
  LangGraph loop, no `ToolRegistry`, since there's no tool orchestration to
  do. Do not confuse this with the two OTHER "no tools" prompts already in
  `prompts.py`: `ZERO_SHOT_PROMPT` (baseline #1, a raw untrained model
  prompted at eval time, no training involved) and the existing
  `NO_TOOLS_SYSTEM_PROMPT` (baseline #3's own GRPO-rollout-time prompt in
  `agent/loop.py`, final_answer only, no `"thought"` field — a different,
  already-built piece for the RL stage, not the SFT-data-generation gap this
  fills). Ground-truth conditioning here necessarily works differently from
  the tool-based path (no grounding tool to narrow a search region through
  without revealing the answer) — the model is told directly which
  finding(s) to cover and asked to write the reasoning a radiologist would
  give for noticing them on inspection, which is closer to hindsight
  rationalization than genuine blind discovery. That's an intentional,
  documented tradeoff (see `generate_no_tools_trajectory`'s docstring) — the
  alternative, blind single-pass generation with no conditioning at all,
  would tank yield the same way blind tool-based generation would. The same
  cross-family `verify_trace`/`verify_pending` already used for tool-based
  traces verifies these too, completely unmodified — it only ever inspected
  `trajectory["messages"]` against `ground_truth`, nothing tool-specific.
  Reads/writes separate `_no_tools`-suffixed files (both unverified and
  verified) so these never mix into the main system's SFT training set.

### 3. The FDI 0-Index Bug Fix (CRITICAL — read this even if you skip everything else)
DENTEX's raw JSON labels `category_id_1`/`category_id_2` as **0-indexed**
(quadrant 0-3, position 0-7), but every prompt, tool, and reward in this
codebase is written against **1-indexed FDI notation** (quadrant 1-4,
position 1-8). The +1 conversion was implemented correctly exactly once
(in `trace_generation.py`) — and then **re-implemented by hand, incorrectly
(i.e. omitted), in seven other files**: `ablations.py`, `baselines.py`,
`batch_runner.py`, `judge.py`, `detector.py`, `test_aim1_trace.py`, and one
more, each building its own ground-truth dict directly from the raw
columns without knowing the +1 was needed. The result: **every one of
those files' ground truth fed into `reward_accuracy`/`combine_reward` with
quadrant and tooth_position off by one from what a correctly-trained model
actually outputs — meaning a perfectly correct model answer scored as
wrong on 50% of `R_accuracy`'s weight**, silently, across the GRPO reward,
the ablation studies, the baseline comparisons, and the batch evaluator.
Only the diagnosis-category term (a string lookup, unaffected by indexing)
was ever scoring correctly in any of them.

**The fix:** `dentex_row_to_fdi(row)` in `dental_agent/data/dentex.py` is
now the **single source of truth** for this conversion. All eight affected
files now call it instead of hand-rolling `+1`. **This is a rule, not a
suggestion — see `.agents/rules/vlm_dental.md` Rule 1.** If you are adding
a ninth file that builds ground truth from DENTEX's raw
`category_id_1`/`category_id_2`, call `dentex_row_to_fdi()`. Do not write
`+ 1` by hand again. Other datasets (Tufts, Tunisia) are NOT subject to
this specific quirk — it's a DENTEX JSON encoding artifact, not a universal
convention — so their own loaders are expected to hand back already-correct
1-indexed FDI values directly, and `prepare_yolo_dataset.py`'s
`DATASET_LOADERS` registry gives each dataset its own `quadrant_position_fn`
for exactly this reason (identity for a dataset whose loader already
outputs correct FDI, `dentex_row_to_fdi` only for DENTEX). Applying
DENTEX's conversion function to a dataset that doesn't need it would
double-increment and reintroduce a version of the same bug in the opposite
direction.

### 4. Reward Redesign (RL/GRPO)
- **`R_efficiency` now scales with case complexity** instead of a flat
  per-call penalty. The old design deducted a fixed penalty per tool call
  regardless of how many findings were actually being investigated — a
  trajectory thoroughly investigating 5 findings (easily 20+ legitimate
  calls: locate + zoom + occasional nudge + contralateral compare, per
  finding) scored far worse than one that superficially investigated 1,
  and it actively fought against the tiered-perturbation/`nudge_crop`
  teaching signal, since a legitimately-needed corrective nudge cost
  exactly the same as aimless spam. The reference budget now scales with
  how many distinct teeth were located in the trajectory (a proxy for
  "genuinely needed investigation"); `locate_tooth` and `nudge_crop` are
  fully exempt from the per-call cost (finding a tooth and correcting a bad
  detection are exactly the behaviors this reward should not fight
  against); only cost *beyond* the reference budget, and exact
  back-to-back repeat calls (no new information gained), reduce the score,
  with a smooth drop-off rather than a hard cliff. See
  `reward_efficiency`'s docstring (`dental_agent/rewards/components.py`)
  for the exact math.
- **Multi-finding accuracy via F1-style greedy matching.** `reward_accuracy`
  used to assume exactly one ground-truth finding and one predicted
  finding. It now handles multiple findings on both sides: predictions are
  greedily matched to ground-truth findings by highest-pair-score-first
  (one-to-one), scored per-pair on the same 0.25 quadrant / 0.25 position /
  0.50 diagnosis rule, then combined into an F1-style harmonic mean of
  recall (missed findings score 0) and precision (hallucinated
  extra findings score 0 too — a pure-recall average would reward spamming
  guesses). With exactly one ground-truth and one predicted finding this
  reduces to exactly the original single-pair score, so existing
  single-finding call sites are unaffected.
- **`parse_agent_json` normalizes every tool call into a `tool_calls`
  list.** Old flat-key access patterns like `parsed.get("tool")` silently
  return `None` on the new normalized shape rather than raising — a
  footgun worth knowing about if you're writing new code that inspects a
  parsed agent turn.
- **Removed a dead, misleadingly-named config field.** `config.py` used to
  define `grpo_max_tool_calls: int = 4`, but nothing else in the codebase
  ever read it (confirmed via grep). Every actual `max_tool_calls` default
  live at runtime (`grpo.py`, `agent/loop.py`, `training/trace_generation.py`,
  `rewards/composite.py`, `run_trace_gen`) is consistently **50** — the
  number `R_efficiency`'s complexity-scaled reference budget above was
  designed against, not 4. Same built-but-unreachable shape as
  `enhance_contrast` before it got wired in above, and `locate_abnormal_teeth`
  (removed entirely — see §8 below). Removed rather than left stale; if
  GRPO's tool-call budget ever needs to be config-driven, re-add it wired
  to an actual call site, with 50 as the default.

### 5. SFT Training Pipeline (Phase 3)
- **Multi-Modal Collator** (`QwenVLDataCollator`): parses complex,
  multi-turn trajectories with dynamically generated image crops directly
  into Qwen-VL's processor.
- **4-Bit QLoRA Optimization:** high-efficiency LoRA training for 3B+
  parameter models on consumer GPUs (e.g. Colab T4).

### 6. RL/GRPO Implementation (Phase 5)
- **Dual-Adapter Memory Architecture:** SFT weights loaded as a frozen
  `"reference"` adapter, a trainable `"grpo_policy"` adapter on the same
  base model — rapidly toggling between them computes KL-divergence
  penalties without a second 3B model in VRAM.
- **VRAM Protections:** strict cache-clearing at the rollout-step level to
  prevent OOM during heavy multi-turn trajectory sampling.

### 7. YOLO Grounding Tool (`locate_tooth`)
Trained on Multi-Dataset (DENTEX + Tufts Dental Database - 2,339 Images, 46,808 Boxes): `yolov8m.pt`, 5-fold cross-validation,
**validation mAP50 = 0.8695 ± 0.0298** (mAP50-95 = 0.5895 ± 0.0346, best fold 4 mAP50 = 0.9226),
representing a **+28.75% absolute gain** in raw full-universe mAP over the baseline DENTEX-only model (`0.5820 ± 0.0076`).
On the **Held-Out Target Grounding Benchmark** (`validation_triple.json` - 46 Images, 182 Targets), both models achieve **>93% Target mAP50** (DENTEX-Only: 0.9319 mAP50 / 93.97% Precision; DENTEX+Tufts: 0.9296 mAP50 / 96.47% Precision) via greedy 1-to-1 bipartite target matching.
**Live in the agent loop**, loaded automatically from `data/models/dentex_tufts_grounding_tool_cv_best/weights/best.pt` (or `GROUNDING_MODEL_PATH`).
It's a **32-class detector** — one class per FDI (quadrant, position) pair, `class_idx = (quadrant-1)*8 + (position-1)`
in `convert_single_image` (`prepare_yolo_dataset.py`) — not a single-class "is this a tooth" detector. This specific design detail means any dataset feeding this tool's training data must carry a real per-tooth position/identity label, not just an anonymous "here's a tooth" box. Full cross-validation and benchmark details are documented in `docs/YOLO_CV_RESULTS.md` and `docs/TRACE_GEN_CONFIG.md`.

### 8. `locate_abnormal_teeth` Removed Entirely
This tool never actually ran in practice — it was a conditionally-registered
8th tool wrapping a learned Faster R-CNN specialist detector
(`tool_locate_abnormal_teeth_learned` in `dental_agent/training/detector.py`)
whose backing checkpoint was never trained (no `train_stage0_detector` run
ever produced one), so every `ToolRegistry.create_default()` call site in
the codebase passed no `grounding_tool` and the tool silently never
registered. **Decision: it's gone, not finished.** The agent locates and
corrects abnormal-tooth grounding via `locate_tooth` (the real, live,
trained YOLO detector — §7 above) plus `nudge_crop`'s self-correction loop
(§2 above), not a second, separate learned detector backend. Removed:
`tool_locate_abnormal_teeth_learned`, `train_stage0_detector`, and
`visualize_detector_predictions` from `detector.py`; the `grounding_tool`
parameter and conditional registration block from `registry.py`;
`scripts/run_detector.py` and `dental_agent/cli.py`'s `train_detector`
command (both existed solely to call the now-removed training function);
and `evaluate_stage0_detector`, which turned out to call
`tool_locate_abnormal_teeth_learned` directly and hardcode its
quadrant/tooth_position output shape into its correctness check — so it
wasn't the reusable generic evaluator it looked like, and removing the tool
without also removing this would have left a broken import.

**What's deliberately kept in `detector.py` despite this**:
`build_stage0_detector`, `detection_collate_fn`, the dataset classes, and
`compute_iou` — because `dental_agent/evaluation/diagnosis_baseline.py`
genuinely reuses them to train and evaluate a plain supervised object
detector directly on diagnosis labels, which is the paper's **"prior
supervised detector"** comparison baseline (a real, still-needed part of
the evaluation plan — see the core hypothesis at the top of this document).
That baseline has nothing to do with `locate_abnormal_teeth` beyond sharing
some Faster R-CNN plumbing; don't confuse the two if you're asked to touch
either again. See `detector.py`'s own module docstring for the fully
detailed breakdown of what's kept vs. removed and exactly why.

### 9. Proposal Positioning — DONE
`dentex-agentic-vlm-proposal.md`'s §3.5/§3.7/§9 have been rewritten to
directly address the two concurrent systems that most threaten an
unqualified "first" claim: **OralGPT-Plus** (CVPR 2026, code public) and
**OralAgent** (April 2026). Neither actually undercuts this project's core
claims — OralGPT-Plus evaluates via LLM-judge-scored open-ended QA on its
own benchmark rather than DENTEX's official leaderboard metrics, and
doesn't cleanly separate tool-use from RL in its own ablation; OralAgent is
a prompted ReAct orchestrator around an off-the-shelf model, not
RL-trained, and also not DENTEX-evaluated — but both are now cited
directly rather than omitted, and the proposal's positioning was narrowed
from "first" to **"among the first, and specifically the first with
DENTEX-native leaderboard-comparable evaluation and a clean tool/RL
ablation."** If you're asked to touch novelty framing again, read
`dentex-agentic-vlm-proposal.md` §3.5 first — the reasoning for exactly
which claims survive contact with these two systems and which needed
narrowing is already worked out there in detail; don't re-derive it from
scratch or drift back toward an unqualified "first" claim without
re-checking it against those same two papers.

---

## 🟡 Currently In Progress

- **Dataset Trace Generation — the actual bottleneck.** `scripts/run_trace_gen.py`
  on Colab/Kaggle builds the synthetic dataset of expert demonstrations,
  driven by a frontier LLM (primarily **Gemini 3.5 Flash Lite** or an
  NVIDIA NIM-hosted model, routed through `api_pool.py`'s provider pool —
  not a single hardcoded model; a self-hosted Qwen/Qwen3.5-9B via local
  vLLM is also a supported provider option but not the primary one in
  practice) as generator, with a different-family API model as verifier,
  via the real LangGraph tool-execution loop (including self-correcting
  grounding, tiered perturbation, all of §2 above). **This is a distinct
  role from Qwen/Qwen3.5-9B's role elsewhere in this project** — Qwen3.5-9B
  is the model actually being trained (SFT then GRPO, see Phase 5/6 below),
  not (primarily) the model generating its own training data; don't
  conflate the two when reading or writing about this pipeline. The trace
  file committed to this repo (`data/traces/train_cot_traces.jsonl.old`)
  has 108 traces, but those **predate the real-tool-execution rewrite** —
  they were generated under the old `<fake_tool_call>`-narration paradigm
  this codebase has since moved away from (see `.agents/rules/vlm_dental.md`
  Rule 3: don't reintroduce that paradigm). No post-rewrite
  `train_cot_traces.jsonl` is currently committed to the repo. This is the
  real bottleneck right now — not the training code, which is built out
  and tested ahead of having volume to run it on.
- **Tunisia dataset loader — Phase 1 done, Phase 2 blocked on one
  verification step.** See "Datasets" below for full detail; short version:
  image discovery + VIA parsing + bbox geometry work today, FDI-position
  labeling of each region is the one open question.

---

## 🦷 Datasets: Current State and Why "Which One's Next" Has Two Answers

**Read `dental_agent/data/dataset_catalog.py`'s module docstring before
adding a new dataset** — it's short and it explains the load-bearing
distinction this whole section is built on. Short version: this project
has exactly ONE dataset with real, working diagnosis-labeled training data
(**DENTEX**). Every other dataset in the catalog with a `has_diagnosis_labels=False`
flag is a **tooth-identification** dataset (segmentation, instance masks,
FDI numbering) with no pathology signal at all — useful for a **different**
purpose: generalizing `locate_tooth`'s grounding accuracy across more
images and imaging equipment, not generating more diagnosis traces. This
split-purpose pattern isn't a guess — it mirrors a real precedent (Merlin
et al., BMC Oral Health 2024, 10.1186/s12903-024-04129-5), who combined a
tooth-instance-segmentation dataset with DENTEX for exactly this reason.
**So "add a new dataset" always needs a follow-up question: for grounding,
or for diagnosis trace-gen?** — because the answer is almost never "either."

There is technically a **second** `has_diagnosis_labels=True` entry in the
catalog beyond DENTEX: **Panoramic-Caries-Segmentation** (China, 75 images,
pixel-level, caries-specific). It has not been investigated further —
license is "unspecified" (a real risk to check before building anything on
it) and 75 images is small, and the annotation is likely binary
(caries/not) rather than DENTEX's 4-class taxonomy. Worth a look if
diagnosis-side data becomes the bottleneck again after Tufts, but not
started.

### DENTEX (primary — done)
The only dataset with a fully working loader end-to-end, feeding both
diagnosis trace-gen and `locate_tooth`'s current (DENTEX-only) training
data. `dental_agent/data/dentex.py`.

### Tufts Panoramic Dataset (Integrated for Grounding)
`dental_agent/data/tufts.py`. Full dataset integrated and live:
- **1,000 Radiograph Images** and **25,184 tooth bounding box annotations** parsed via `load_tufts_tooth_boxes`.
- All images, bounding boxes, annotations, and polygon segmentations (`Segmentation/teeth_polygon.json`, 271 MB) uploaded to Hugging Face Hub dataset repository `Reza-Nadimi/tufts-train-images`.
- Active in `scripts/prepare_yolo_dataset.py` via `DATASET_LOADERS["tufts"]`.
- Co-trained with DENTEX in Multi-Dataset YOLO 5-fold cross-validation, achieving **86.95% mAP50**.

### Tunisia — Panoramic Dental Xray Dataset (in progress — Phase 1 done)
`dental_agent/data/tunisia_panoramic.py`. CC BY 4.0, **no registration
required** (this is why it was picked over Tufts, DNS, and TL-pano for the
next grounding-only loader — see the module's own docstring for the full
three-way tradeoff reasoning against those other candidates: DNS has the
right annotation shape but the same access-gating problem as Tufts, and
TL-pano is both access-gated AND non-commercial-licensed).
`has_diagnosis_labels=False` — this can only ever expand `locate_tooth`'s
grounding training corpus, **never** feed diagnosis trace-gen.

**What's implemented and verified working today:** local-directory
discovery, VIA2 JSON parsing (handles both the `_via_img_metadata`-wrapped
and bare export shapes), and bounding-box computation from
`rect`/`polygon`/`polyline`/`circle`/`ellipse` region shapes — all standard
VIA2 schema (Dutta & Zisserman, ACM MM 2019), so none of this required
guessing at dataset-specific semantics. Tested end-to-end against a
synthetic fixture archive during development; correctly discovers images,
parses regions, computes bboxes, and populates `annots_df` with real
`bbox` values plus each region's raw `region_attributes` dict.

**What's blocked, and exactly how to unblock it (`_region_to_fdi` in
`tunisia_panoramic.py`):** whether the 107-image tooth-instance-segmentation
subset's `region_attributes` carry a per-tooth FDI (or Universal Numbering)
position label, or are anonymous per-instance masks with no semantic
numbering. This is NOT a minor gap — recall from §7 above that
`locate_tooth` is a 32-class (per-FDI-position) detector, not a
single-class one, so anonymous instances literally cannot feed
`DATASET_LOADERS` as currently designed. **To resolve, once the archive
(https://data.mendeley.com/datasets/73n3kz2k4k/3) is downloaded and
extracted:** open the VIA JSON and inspect one region's
`region_attributes` dict (a ready-made one-liner is in the function's
docstring). The paper's own title — "...Instance Segmentation **and
Identification**..." — is a genuine positive signal (that phrasing is the
field's standard term for per-tooth numbering, not just telling instances
apart — see ToothNet, CVPR 2019, using identical phrasing for exactly
that), but it's not confirmed: the paper is paywalled past its abstract and
the Mendeley page blocks automated fetching, so this needs a human (or an
agent with real file access) to actually look. **This should take about
thirty seconds once someone has the file open** — see the function's
docstring for the exact three things to check. If it turns out there's no
identity field at all, that's not a dead end either — it would mean this
dataset instead suits a *different*, not-yet-built use (a single-class
"is this a tooth" pretraining stage for `locate_tooth`), which is a real
design conversation to have, not something to decide by guessing in the
loader.

Once `_region_to_fdi` is filled in: uncomment the `"tunisia"` entry in
`prepare_yolo_dataset.py`'s `DATASET_LOADERS` and `--datasets
dentex,tunisia` becomes usable immediately — the loader, upload bundler
(`_prepare_tunisia_bundle`), and download slicer
(`download_tunisia_slice`) are all already wired and waiting on that one
function.

### DNS and TL-pano (catalogued, not started)
Both in `dataset_catalog.py`, both access-gated (registration required,
same unpredictable-wait risk as Tufts). TL-pano additionally carries a
non-commercial-research-only license — a real encumbrance to check against
how this project's outputs (trained weights, published traces, the paper
itself) will actually be distributed before investing in a loader for it.
Lower priority than Tunisia for exactly these reasons; not started.

---

## 🔴 Left To Do (Future Milestones)

### Immediate next step: resolve `_region_to_fdi` for Tunisia
See "Datasets" above. This is a ~30-second file inspection, not an
engineering task — the engineering (loader, bundler, slicer) is already
done and waiting on it.

### Scale Trace Generation (the real bottleneck — see "Currently In Progress")
Run `run_trace_gen.py` at real volume on Colab/Kaggle to replace the 108
pre-rewrite legacy traces with real, post-rewrite ones. Nothing else in
Phase 3/4/5 can meaningfully proceed without this.

### Dataset Expansion Beyond Tunisia
Once Tunisia's loader is fully unblocked: DNS and TL-pano remain
lower-priority grounding-only candidates (see "Datasets" above for why).
Tufts remains the only other *potential* diagnosis-labeled dataset,
blocked on registration + the two open verification questions in
`tufts.py`. Panoramic-Caries-Segmentation is uninvestigated but worth a
look if diagnosis-side data becomes the bottleneck again.

### Phase 3: Execute Supervised Fine-Tuning
Once trace generation has real volume, run `VLM_Dental_Colab_SFT.ipynb` to
teach the base Qwen-VL model how to use tools and reason like the frontier
LLM teacher that generated its training traces (see "Currently In
Progress" above for which models that actually is).

### Phase 4: Baseline Agent Evaluation
- Create `scripts/run_eval.py` to test the SFT model.
- Compare accuracy in a standard "Zero-Shot" setting vs. an "Agentic
  Tool-Use" setting to quantify the tools' actual contribution.

### Phase 5: Execute GRPO Reinforcement Learning
Run `VLM_Dental_Colab_GRPO.ipynb` to apply Group Relative Policy
Optimization — penalizing hallucination, rewarding accurate diagnoses, and
optimizing tool-usage efficiency (including learning when a `nudge_crop`
correction is actually worth the extra call, per the redesigned
`R_efficiency`).

### Phase 6: Interactive Clinical UI
- Build a web interface (Gradio or Streamlit).
- Let dentists upload panoramic X-rays and watch the agent visually zoom,
  enhance, and reason through the image step-by-step in real time.
