# Canonical Vision Inputs, Static Vision-Slot Padding & the TPU Training Path

**Status:** implemented in the working tree, unit-tested on CPU (176 tests pass), checked against the real 880-trace corpus, **not yet run on TPU hardware** (§9). Hardware: **Cloud TPU v5e-8** (8 chips x 16 GB HBM, 330 GiB host RAM). **Revision 2** (this document supersedes the first draft where they differ: letterbox, no hidden defaults, GRPO on SPMD, duplicate-image fix).
**Scope:** Stage 1 SFT, Stage 2 GRPO rollouts, evaluation, AOT XLA warmup, token-length manifest.
**Companion docs:** `docs/SFT_CONFIG.md` (§5 collator), `docs/GRPO_CONFIG.md`, `AGENTS.md`, `roadmap.md`.

This is the single source of truth for *what we changed, why, and what we decided*. Every claim below is tagged
**[verified]** (checked against source, a test, or arithmetic in this session), **[plan-reported]** (stated in the
implementation plan but not reproducible from the repo — the `scratch/` scripts it cites are not committed), or
**[open]** (needs hardware or data we did not have).

---

## 1. Problem

1. **Canonicalization was planned but never wired.** At commit `8572362`, nothing resized images:
   `DentalSFTDataset.__getitem__` passed native-resolution panoramics and native-resolution tool outputs to the
   processor, so every earlier OOM/budget experiment ran against native images. **[verified]**
2. **Vision-shape diversity on XLA.** Under canonical resizing the number of vision patches per sample still varies
   with the tool-call mix; each distinct `(pixel_values, image_grid_thw)` shape/content is a separate XLA graph
   (30–120 s compile stall each). The plan reports **91 distinct shapes** in `train_cot_traces_dentex.jsonl`.
   **[plan-reported]** — reproduce with `scripts/census_vision_slots.py`.
3. **`torch.linalg.solve_triangular` in Qwen3.5's Gated-DeltaNet layers** has no GSPMD partitioning rule, forcing
   replication and an HBM-exhausting workspace. Upstream's matmul-only alternative exists but is gated behind
   `is_torchdynamo_exporting()` (never true on PyTorch/XLA). **[verified]** against `transformers` main
   (`modeling_qwen3_5.py`: `solve_triangular` at the `if not is_torchdynamo_exporting()` branch).

## 2. Canonical geometry (settled project spec)

Vision tower: `patch_size=16`, 2×2 spatial merge, patch feature dim 1536. **[verified]** (`canonical.py` asserts it.)

| Family  | Size (W×H) | `grid_thw` | Patches | LLM tokens | Produced by |
|---------|-----------:|-----------:|--------:|-----------:|-------------|
| FULL    | 1536×768   | (1,48,96)  | 4,608   | 1,152      | base image, `denoise`, `window_level`, `enhance_contrast` |
| CROP    | 256×384    | (1,24,16)  | 384     | 96         | `zoom_crop` |
| COMPARE | 512×384    | (1,24,32)  | 768     | 192        | `contralateral_compare` |

Static slot budget `[5 FULL, 10 CROP, 4 COMPARE]` = **19 slots, 29,952 patches, 7,488 vision tokens**
(5·4608 + 10·384 + 4·768 = 29,952 ✓). The 100 % corpus-coverage claim is **[plan-reported]**; `census_vision_slots.py`
exits non-zero and lists offenders if any trace exceeds the budget.

All of this lives in one torch-free module, `dental_agent/utils/canonical.py`, imported by SFT, GRPO, evaluation,
the collator and the census. Arithmetic invariants are `assert`ed at import time.

## 3. Critique of the implementation plan

| # | Finding | Severity | Resolution |
|---|---------|----------|------------|
| 1 | **Token budget contradiction.** Static padding makes every sample `text + 7,488` tokens. With the plan's own authentic trace (4,009 text tokens) that is **11,497 > 10,240**, yet the plan keeps the 10,240 bucket as the primary target. Text capacity at 10,240 is only 2,752 tokens. | High | Padded mode requires `--max-seq-len 16384` (auto-raised from the 10,240 default with a log line; §4 D3). |
| 2 | **Silent truncation.** The collator truncated overlength sequences. In padded mode that cuts `<\|image_pad\|>` tokens out from under their vision features → shape-mismatch crash (or, unpadded, a truncated *assistant answer* trained as ground truth). | High | Padded mode **raises** on overlength; length filtering now accounts for the static tail (§5). |
| 3 | **`qwen_vl_utils` double-resizes.** Its default `image_patch_size=14` ⇒ resize factor 28: 1536×768 → 1540×756, 256×384 → 252×392, 512×384 → 504×392 (computed from its `smart_resize`). The HF processor then resamples again. The plan only proposed PIL extraction as a *fallback*. | High | PIL extraction is now the **primary** path; the `qwen_vl_utils` fallback passes `image_patch_size=16`. |
| 4 | **Ambiguity about which image tools run on.** Tool `bbox` arguments are native pixel coordinates. | High | Invariant: tools always execute on the **native** image; only the *view* is canonicalized (tested with a white-box pixel test). |
| 5 | **Stale length manifest.** `trace_token_lengths.json` was measured at native resolution. Reusing it under canonical resize filters the wrong traces. | High | Manifest records `canonical_resize`; a mismatched manifest is ignored with a warning (§5). |
| 6 | **AOT warmup compiled the wrong graph** (512×512 dummy image, no slot padding) — the cache would never be hit by training. | High | Warmup builds the identical canonical + 19-slot batch. |
| 7 | **GRPO / evaluation not covered.** SFT on canonical views then GRPO/eval on native views is a train/test distribution shift that would invalidate any paper comparison. | High | `canonical_resize` plumbed through rollouts, `run_grpo*.py`, and `evaluate_models.py` (§4 D7). |
| 8 | **"Zero-compromise GPU path" via `is_tpu` toggles** conflicts with the standing decision that a GPU path would be a *separate* clean build with no shared backend-switching. | Medium | Revision 2 removes every TPU-dependent default: all behaviour is chosen by explicit flags, and the notebooks pass each value verbatim (§4 D8). |
| 9 | **Hardware naming.** Notes said Kaggle v3-8. | Medium | Confirmed **v5e-8**; docs and memory corrected. |
| 10 | **Unverifiable evidence.** The 67.6 % / 54.5 % / 224-token reconciliation is arithmetically correct (see §8) but its scripts (`scratch/*.py`) are not in the repo, and the 91-shape census was not reproducible. | Low | Added `scripts/census_vision_slots.py` so the numbers regenerate from the committed traces. |

## 4. Decisions

Decided in this session on the "do what is best, authentic, paper-ready" instruction. **D3, D5 and D8 are the ones to
confirm or overrule.**

- **D1 — Views, not data.** The model sees canonical views; tools run on native images; ground-truth coordinates stay
  in native pixel space. The model never emits pixel coordinates itself (every bbox comes from a tool), so the
  view/coordinate scale difference is harmless and consistent between SFT, GRPO and eval.
- **D2 — One module for geometry** (`canonical.py`); sizes live only there, the slot budget is a CLI value (`--vision-slots`) that defaults to it.
- **D3 — `--max-seq-len 16384`, set explicitly, never adjusted.** Padded mode makes every sample `text + static vision
  tokens` (7,488 for `--vision-slots 5 10 4`), so the notebook passes 16384 (text capacity 8,896). The earlier
  "auto-raise 10240 -> 16384" was removed: a hidden rewrite of a user-visible number. The only check left is a hard
  error when the length cannot hold the static vision tokens. *Measured on the 880 with-tools traces (uploaded
  manifest + census):* a single 16,384 graph keeps 812 and drops 68 (7.7 %); 18,432 keeps 864; 20,480 keeps 877. The
  dropped traces are not random: 7.8 findings/trace vs 4.5, 22.9 turns vs 11.9, so 16k removes the hardest
  multi-finding cases (evaluate stratified by finding count). Only ~51 % of each 16,384 sequence is real tokens;
  tiered budgets (e.g. 3 graphs at 10,240/14,336/20,480: 3 drops, ~84 % of the compute) are the fallback if
  HBM/throughput forces it. Kept at 16k per instruction until the first hardware measurement.
- **D4 — Fail loudly instead of degrading.** Padded mode raises on: non-canonical grids, over-budget slots,
  overlength sequences. `safe_process_vision_info` warns (instead of silently returning `(None, None)`) when images
  are present but cannot be loaded.
- **D5 — Aspect-preserving letterbox (replaces the stretch).** `to_canonical` scales uniformly (LANCZOS) to the
  largest size that fits the canvas and centres it on a constant black canvas (`LETTERBOX_FILL`); nothing is cropped
  and token counts are unchanged because the canvas is fixed. The previous stretch distorted near-square `zoom_crop`
  boxes up to 1.73x (vertical vs horizontal) when mapped onto the 2:3 CROP canvas. Canvas occupied by content:
  FULL 95 % (2872x1504), CROP 58-92 % (252x248 -> 66 %, 300x260 -> 58 %), COMPARE ~76 % (420x240). No information is
  lost: a small crop is upscaled either way, so letterboxing only spends otherwise-redundant tokens on black bars.
  Not measurable without GPU: whether the black bars change accuracy (expected neutral; compare against a stretch
  ablation if the paper needs the claim). Nothing was trained with the old stretch, so there is no legacy to keep.
- **D6 — XLA shim patches `torch.linalg.solve_triangular`, not the model function.** Copying upstream's function body
  would drift with `transformers`. The shim intercepts only `left=True, upper=False, unitriangular=True` on `xla`
  tensors and uses a log-depth matmul product expansion (6 factors for chunk size 64) instead of upstream's 63-step
  in-place loop. CPU/CUDA behaviour is untouched.
- **D7 — No hidden state.** The `DENTAL_CANONICAL_RESIZE` environment variable and `resolve_canonical_resize()` were
  removed. `canonical_resize` is an explicit parameter everywhere (`run_agent`, GRPO rollouts, `evaluate_models.py`,
  launchers). Library callers that do not pass it (ablations, sweep helpers, judge, batch runner) stay at native
  resolution; wire them explicitly before using them to evaluate a canonical-trained checkpoint.
- **D8 — Defaults and flags (one flag per behaviour, no overlapping pairs).** Plain defaults, identical on every
  backend: `--spmd` **on** (legacy `xmp.spawn` holds one full model copy per process and exhausts the 330 GiB host
  RAM), `--fsdp` on, `--canonical-resize` off, `--pad-vision-to-slots` off, `--vision-slots 5 10 4`,
  `--triangular-shim` on, `--max-seq-len 10240`. The notebooks set every one of them as a literal at the top of the
  cell and pass it verbatim, so the notebook command line is the complete description of the run. Contradictory
  combinations fail fast with a message (`--pad-vision-to-slots` without `--canonical-resize`; `--spmd --no-fsdp`).
- **D9 — Fix, don't preserve, the manifest-key bug** (§5): correctness of the length filter outranks backward
  compatibility with a filter that never matched.

## 5. Implementation map

| File | Change |
|------|--------|
| `dental_agent/utils/canonical.py` *(new)* | Geometry, slot budget, tool→family map, `to_canonical`, `resolve_canonical_resize`. |
| `dental_agent/model/backbone.py` | `safe_process_vision_info`: PIL pass-through primary path; `image_patch_size=16` fallback; warns on failure. |
| `dental_agent/training/sft.py` | `DentalSFTDataset(canonical_resize, pad_vision_to_slots)`; `BucketedQwenVLCollator(pad_vision_to_slots)` + `_pad_vision_slots`; manifest-mode guard; fixed manifest lookup (below); slot-aware length filter; "unmatched manifest entries" warning. |
| `dental_agent/training/xla_patches.py` *(new)* | `solve_triangular` shim (`install_/uninstall_xla_solve_triangular_shim`). No import-time side effects. |
| `dental_agent/agent/loop.py`, `dental_agent/training/grpo.py` | `canonical_resize` through `run_agent`, `run_agent_no_tools`, batched no-tools rollout, `collect_grpo_group*`, `grpo_step`, `train_grpo`. |
| `scripts/train_sft.py` | `--canonical-resize`, `--pad-vision-to-slots` (TPU-default on), shim install on TPU, sequence-length auto-raise, validation. |
| `scripts/warmup_xla_cache.py` | `--pad-vision-to-slots`; canonical FULL dummy image + 19-slot batch; same length auto-raise; shim install. |
| `scripts/compute_exact_trace_lengths.py` | `--canonical-resize`; records `canonical_resize` and `vision_tokens_by_file_and_id`; resume only when the mode matches. |
| `scripts/census_vision_slots.py` *(new)* | Distinct shapes, family mixes, budget coverage, slot-padded length fit. No model needed. |
| `scripts/run_grpo.py`, `scripts/run_grpo_sweep.py`, `scripts/evaluate_models.py` | `--canonical-resize` (eval sets the env var for the whole run). |
| `notebooks/VLM_Dental_Colab_SFT.ipynb` | Defined the missing `TARGET_SEQ_LEN`; `CANONICAL_RESIZE`/`PAD_VISION_TO_SLOTS` flags passed to warmup+train; new §5a cell (recompute manifest + census); stale "native resolution, no downsampling" text corrected. |
| `notebooks/VLM_Dental_Colab_GRPO.ipynb` | `CANONICAL_RESIZE` flag (must match SFT) passed to `run_grpo.py` / sweep. |
| `tests/test_canonical_pipeline.py` *(new)* | 15 tests (see §7). |

**Revision 2 additions**

| File | Change |
|------|--------|
| `dental_agent/utils/canonical.py` | Letterbox `to_canonical`, `content_fraction`, `parse_slot_budget`, `slot_totals`; env resolver removed. |
| `dental_agent/training/sft.py` | k-th image of a tool <-> k-th call of that tool (see §5.3); `slot_budget` parameter on dataset/collator; bounded LRU view cache; `padded_length()`; budget-independent shape invariant in the collator. |
| `scripts/train_sft.py`, `scripts/warmup_xla_cache.py` | `--spmd` default on; `--vision-slots`; `--triangular-shim/--no-triangular-shim` (A/B for the OOM question); no auto-raise; startup `[CONFIG]` line echoing every setting; hardcoded 29952/19 check removed. |
| `dental_agent/training/grpo.py`, `scripts/run_grpo.py`, `scripts/run_grpo_sweep.py` | SPMD single-process GRPO (§5.4); static-shape policy update; persistent optimizer; sweep accepts `--model-id` (the notebook already passed it, the launcher rejected it). |
| `notebooks/*` | Literal config cells; every flag passed verbatim; the generated commands are parsed by the real CLIs in the test-suite harness. |
| `tests/test_grpo_static_path.py` *(new)* | Static update == per-trajectory update (grads match to 1e-5); filler rows inert; one shape for any length; CLI flags. |

### 5.1 Static slot padding — exact mechanics

For each example the collator (TPU mode only): keeps real images first in order; appends zero-valued dummy patches and
`image_grid_thw` rows for every unused slot; appends exactly `Σ tokens(unused slots)` `<|image_pad|>` ids to the
**tail** of `input_ids` with `attention_mask=0`, `labels=-100`, `mm_token_type_ids=1`; then right-pads to the static
length. Resulting invariants per sample: `pixel_values == [29952,1536]`, `image_grid_thw == [19,3]`,
`#<|image_pad|> == 7,488 == pixel_values.shape[0]/4`.

Why this is safe **[verified against upstream source]**: HF checks `#image placeholders == #image features`
(`get_placeholder_mask`) — hence the tail tokens are *required*, not optional; tail tokens are causally after every
real token (full attention and the linear-attention/conv layers are both causal), so they cannot influence earlier
positions; `get_rope_index` drops `attention_mask==0` tokens before consuming grids, so dummy grids never perturb
real M-RoPE positions; `labels=-100` gives zero loss/gradient; the vision tower is frozen under `torch.no_grad()`.

### 5.2 Length filter and the manifest — two real bugs fixed

* **Key mismatch (pre-existing, silent).** `compute_exact_trace_lengths.py` writes `<file>::<dataset>::<id>` and
  `<dataset>::<id>`; `DentalSFTDataset` looked up only `<file>::<id>` and bare `<id>`. A freshly generated manifest
  therefore matched *nothing* and **no trace was ever filtered**, so overlength traces reached the collator and had
  their tails (the assistant's final answer) truncated and trained on. Lookup now tries all four key forms, most
  specific first, and warns about traces with no manifest entry. Regression test:
  `test_manifest_written_by_compute_script_actually_filters`. **Consequence for the paper:** any SFT run made before
  this fix may have trained on truncated targets; check the old runs' `[DATASET MASK]` log lines.
* **Padded-length semantics.** In padded mode a trace occupies `text + 7,488` tokens, not its raw length. The
  manifest now stores each trace's vision-token count so the filter computes `raw − vision + 7,488`.

### 5.3 Duplicate tool images (pre-existing data bug, fixed)

When one assistant turn called a tool several times (three `zoom_crop`s), the dataset looked up the *first* call named
`zoom_crop` for every image, rendering N identical crops while the assistant text discussed N different teeth. Counted
on the 880 uploaded traces (raw JSON, no images needed): **679 of 3,490 observation images (19.5 %) in 341 of 880
traces (38.8 %)** were built from the wrong call (545 `zoom_crop`, 134 `contralateral_compare`). Result order equals
call order in 2,406 of 2,450 image-bearing turns. Fix: the k-th image of a tool uses the k-th call of that tool
(`test_n_zoom_crops_in_one_turn_render_n_distinct_images`). The remaining 50 images sit behind assistant messages whose
JSON does not parse; they fall back to a default crop and now raise a counted warning (`unresolved_tool_images`).
**Consequence:** every earlier SFT result trained on these duplicated crops; re-run before quoting any number.

### 5.4 GRPO on SPMD

* **Launch/wrap:** `--spmd` (default) = one process, `xr.use_spmd()` before the first device touch, FSDPv2 over the
  8-core mesh, no data partitioning across processes. `--no-spmd` keeps the legacy path.
* **Rollouts:** `model.generate` runs on the inner module (shares the same sharded parameters).
* **Policy update at static shapes:** with `--pad-vision-to-slots --max-seq-len N` each update forward uses the SFT
  collator, so one compiled graph serves every trajectory; `rows_per_forward = num_cores` (one rollout per core);
  missing rows are inert copies with `labels = -100`. Old log-probs are recomputed with the same shapes (no
  variable-shape forward during collection). Rollouts that do not fit `N` are excluded from the update and counted in
  `n_dropped_overlength`. Numerical equivalence with the per-trajectory loop is tested.
* **Optimizer bug fixed:** `grpo_step` built a fresh AdamW on every image, wiping the Adam moments each step. It is now
  created once per run. This changes GRPO dynamics; earlier GRPO runs are not comparable.
* **Not solved:** `generate()` itself still produces dynamically shaped graphs on XLA (growing KV cache, varying prompt
  lengths). Static *update* shapes remove the training recompiles, not the decode recompiles; this needs a
  static-cache/bucketed decoding design and hardware to validate (§9).

## 6. Runbook

Every command below is what the notebooks generate; nothing is implied by a default.

```bash
# 0. (once per resolution mode) lengths + census -- CPU only
python scripts/compute_exact_trace_lengths.py --canonical-resize --recompute
python scripts/census_vision_slots.py data/traces/train_cot_traces_dentex.jsonl \
    --manifest data/traces/trace_token_lengths.json --seq-lens 10240 16384 --vision-slots 5 10 4

# 1. AOT warmup (TPU) -- must equal the training graph; run twice to A/B the solver (HBM / OOM)
python scripts/warmup_xla_cache.py --spmd --fsdp --max-seq-len 16384 --pad-vision-to-slots \
    --vision-slots 5 10 4 --triangular-shim          # then again with --no-triangular-shim

# 2. SFT
python scripts/train_sft.py --spmd --fsdp --max-seq-len 16384 --canonical-resize --pad-vision-to-slots \
    --vision-slots 5 10 4 --triangular-shim ...

# 3. GRPO -- same view/shape flags as the SFT run that produced --sft-stage
python scripts/run_grpo.py --spmd --fsdp --canonical-resize --pad-vision-to-slots --max-seq-len 16384 \
    --vision-slots 5 10 4 --triangular-shim ...

# 4. Evaluation -- same view flag as the evaluated checkpoint
python scripts/evaluate_models.py --canonical-resize ...
```

The warmup prints HBM before/after forward and backward (`[XLA MEMORY | ...]`); the A/B is the difference between the
two runs. Each script echoes a `[CONFIG] ...` line at startup.

## 7. Verification evidence

All on CPU, `transformers 5.18.0`, `torch 2.14`:

* **New tests (15, `tests/test_canonical_pipeline.py`) — pass:** geometry; PIL pass-through and failure warning;
  dataset native default; canonical views **with tools on the native image** (white-box pixel check); stale-manifest
  guard; exactly one static shape for four different image mixes; no loss/attention leakage in the tail; rejection of
  non-canonical grid / over-budget / overlength; 16,384 accepted where 10,240 is rejected; GPU path untouched;
  manifest-key regression; slot-padded length filter; env resolver; census family attribution; shim math and
  idempotence/scope.
* **Shim vs the real upstream `torch_chunk_gated_delta_rule`** (200-token random batch, `use_qk_l2norm_in_kernel=True`):
  forward max abs error 1.5e-8; gradient max relative error ≤ 3.8e-7 for q, k, v, g, beta.
* **Regression suite:** 166 pass on the working tree (5 fail). Failures (all outside this change): 5 pre-existing on a pristine
  `8572362` clone (`test_api_pool` gemini, `test_dentex_normal_loader`, `test_prepare_yolo_dataset`,
  `test_sync_traces_hf` file list, `test_trace_generation` healthy scan [langgraph `StopIteration`]);
  `test_target_grounding_eval` needs `ultralytics` (not installable in the sandbox). One regression we introduced
  (`--max-seq-len` default pinned at 10,240 by `test_sft_cli_fsdp_flags`) was fixed by restoring the default (D3).
* **Arithmetic (plan §1, verified):** (62,884−20,352)/62,884 = 67.6 % visual-patch reduction; (19,506−8,873)/19,506 =
  54.5 % total-token reduction; 4,009−3,785 = 224-token prompt gap; 5,088+4,009 = 9,097 fits the unpadded 10,240
  bucket — **but** 4,009+7,488 = 11,497 does not fit it padded.

## 8. Reconciliation of the numbers in the plan

The "68 %" is the *visual patch* reduction, the "54.5 %" the *total sequence* reduction (text tokens are identical across
modes: 3,785). The 224-token difference is the synthetic one-line prompt in `test_image5_canonical_pipeline.py` vs the
authentic system+user text loaded via `DentalSFTDataset`; the image also differs (synthetic 2780×1410 vs on-disk
2872×1504). Different scripts, different inputs — not a discrepancy in the pipeline. **[plan-reported inputs,
verified arithmetic]**

**Addendum (native token counts do not reconcile).** The plan's *native* figures (62,884 patches / 15,721 tokens for a
2780x1410 image; 69,316 patches / 17,329 tokens for 2872x1504) are about 4x what the HF Qwen3-VL image processor
produces at patch 16 / merge 2: measured **2872x1504 -> grid (1,94,180) -> 4,230 tokens; 1615x840 -> 1,300 tokens**. The
canonical figures *are* confirmed (the uploaded manifest's per-trace vision-token counts match 1,152 / 96 / 192 per
image on 880 of 880 traces). So canonical resizing cuts a native DENTEX base image from ~4.2k to 1.15k tokens (3.7x), not
the ~15x the plan implies. Do not quote the plan's native numbers until they are re-measured with the real processor.

## 9. Open items (honest list)

1. **No TPU run yet.** Unverified: GSPMD sharding of the shim, HBM fit at 16,384, single compilation, `save_pretrained`
   under SPMD. First check: warmup with and without `--triangular-shim`; the HBM difference answers whether removing the
   native triangular solver is what fixes the OOM. If 16,384 does not fit, use tiers (D3), not fewer cores.
2. **GRPO decode shapes.** Update shapes are static; `generate()` shapes are not (§5.4). Needs a static-cache design
   plus hardware. Until then GRPO will compile per decode shape.
3. **68 of 880 traces (7.7 %) drop at 16,384**, biased toward multi-finding cases (D3). Report per-finding-count recall.
4. **Re-run SFT/GRPO baselines:** the duplicate-image bug (§5.3), the manifest-key bug (§5.2) and the GRPO optimizer
   reset (§5.4) invalidate any earlier numbers.
5. **Library eval call sites** (ablations, sweep helpers, judge, batch runner) are still native-only (D7).
6. **`docs/SFT_RESULTS.md` memory figures** are pre-canonical; re-measure.
7. **Letterbox accuracy claim** is an expectation, not a measurement; run the stretch-vs-letterbox ablation if needed.
8. Still outstanding from before: regenerate the 18 Tufts no-tools traces; zero-shot notebook; Tunisia loader.

## 10. Paper-ready statements and limitations

**Methods (suggested text).** *Radiographs are presented to the model as fixed-size views: the base image and
whole-image operators at 1536×768, `zoom_crop` outputs at 256×384 and `contralateral_compare` composites at 512×384
(aspect-preserving Lanczos resampling onto a constant black canvas; 1,152 / 96 / 192 visual tokens). Tools execute on the original-resolution image and all
coordinates are in original pixel space. The identical views are used for SFT, GRPO rollouts and evaluation.*

**Limitations to state.**
* **Letterboxed canvases.** CROP/COMPARE views are aspect-preserving, so non-matching crops carry black borders (CROP
  canvas 58-92 % occupied for typical boxes). Geometry is preserved; effective resolution of a crop is capped by its
  source pixels, not by the canvas.
* **Resolution loss.** FULL downsamples a 2872×1504 scan by ≈1.9× per axis; fine findings are expected to be inspected
  via CROP (which re-samples from the *native* image, so zoomed detail is not lost).
* **Baseline fairness.** Base-model zero-shot baselines should be reported under both native and canonical views when
  compared against canonical-trained checkpoints.
* **Static padding is an engineering device**, with no effect on the loss (`labels=-100`, causal tail); report the
  length filter outcome (`[DATASET MASK]` line) per stage so excluded traces are accounted for.

## 11. Rule 15 adversarial self-critique (summary)

* *Paths:* manifest keys use `Path.name` (no separators); no new absolute paths.
* *Null safety:* collator handles examples with no images (zero-length base + full dummy slots) and missing
  `mm_token_type_ids`; unmatched manifest entries are kept and warned about, never dropped silently.
* *Caching:* `_crop_cache` stores **native** tool outputs; canonicalization happens after the cache lookup, so cache
  entries are mode-independent. `to_canonical` returns the input object when the size already matches (never mutated
  downstream).
* *Deep copies:* the shim captures the original `solve_triangular` once and is idempotent; `uninstall` restores it.
* *Determinism:* LANCZOS resampling and the matmul shim are deterministic; no randomness introduced.

## 12. Findings from the real-corpus checks and the pre-GPU smoke test

**12.1 Duplicate crops, precisely.** Zooming always uses the original image -- that part was right. The bug was *which
call's bounding box* was used when the dataset rebuilt an observation turn: for N calls to the same tool in one turn it took
the arguments of the first call for all N images. Reproduced on the original commit (`8572362`) with three differently
coloured target regions: the three rendered crops had centre pixels (255,0,0), (255,0,0), (255,0,0) instead of red, green,
blue. So the model saw three copies of tooth A beside text about teeth A, B and C. Scale on the 880 traces: 679 of 3,490
observation images (19.5 %), 341 traces (38.8 %). Fixed by matching the k-th image of a tool to the k-th call.

**12.2 Does static padding change the result? No (measured).** On the real HF Qwen3.5 code (hybrid linear/full attention,
fp32, CPU, tiny random weights, real traces and real processor) the same sample was run unpadded and slot-padded
(`--vision-slots 2 4 1`, 2,784 static vision tokens): loss identical to 6 decimals (|dloss| = 0), max logit difference at real
positions <= 2.4e-7, lm_head gradient relative difference <= 9.7e-8, over four traces with 4->7 and 3->7 images. This is
the empirical counterpart of the causal-masking argument in §5.1. What padding *does* change is the length ceiling (§4 D3).

**12.3 `--spmd` default exposed a crash on every non-TPU machine (fixed).** After `--spmd` became the default, a GPU/CPU run
with default flags reached `wrap_spmd_model` and died with `ModuleNotFoundError: torch_xla` (reproduced), and would also
have multiplied the dataloader batch by `--num-cores`. SPMD is now gated on the real backend
(`use_spmd = args.xla_spmd and is_tpu`), so `--spmd` is inert off-TPU. Verified by a default-flags CPU run.

**12.4 `--pad-vision-to-slots` was silently ignored off-TPU (fixed).** The non-TPU collator did not receive the flag while the
dataset still filtered by padded length. The flag and `--vision-slots` now apply on every backend (sequences still pad
dynamically to the longest sample in the batch).

**12.5 LoRA does not cover the linear-attention layers (not changed; decision needed).** `train_sft.py` hardcodes the LoRA
targets `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`. Qwen3.5 is hybrid: 3 of every 4 layers are
Gated-DeltaNet layers whose projections are `in_proj_qkv, in_proj_z, in_proj_a, in_proj_b, out_proj`, so attention LoRA
reaches only 1 layer in 4 (the MLP projections are adapted in every layer). This may limit what SFT can learn and is a
hardcoded list, against the "no hardcoding" rule. Proposed: an explicit `--lora-targets` flag defaulting to the current
list, with the linear-attention projections available behind it. Not applied, because it changes trainable parameters and
VRAM on the first GPU run.

**12.6 GPU dependencies.** Without `flash-linear-attention` and `causal-conv1d`, `transformers` falls back to a pure-PyTorch
reference for the Gated-DeltaNet layers ("correct but much slower"); neither is in `requirements.txt`. For a 24 GB card
install them (`pip install flash-linear-attention`; `causal-conv1d` needs a CUDA build). The fallback is also the likelier
memory hog at 10k tokens. Untested here (no GPU).

**12.7 Pre-GPU smoke test.** `scripts/smoke_test_sft.py` builds a ~2M-parameter random Qwen3.5 (hybrid layers, real ViT,
BPE tokenizer trained on your traces, real Qwen3-VL image processor), writes noise radiographs at the real resolutions,
and drives the real `train_sft.py` end to end (dataset, tools, collator, LoRA, gradient checkpointing, loop, save). Run on
the CPU sandbox, all three configurations exited 0 and saved an adapter: GPU-style canonical + dynamic padding at 8,192;
slot-padded (`--vision-slots 4 6 2`, 12,288); and default flags with no `--no-spmd`. It checks crashes and version
compatibility, **not** memory or speed. Run it on the target machine first:

```bash
python scripts/smoke_test_sft.py --traces data/traces/train_cot_traces.jsonl -- --canonical-resize --max-seq-len 8192
```

**12.8 Single 24 GB GPU (RTX 4090) configuration.** Native resolution is not viable: a native DENTEX base image is ~4.2k
tokens and with-tools traces carry up to five full-size images. Use canonical views with dynamic padding (no static slots, no
SPMD), and a manifest measured in the same mode:

```bash
python scripts/compute_exact_trace_lengths.py --canonical-resize --recompute
python scripts/train_sft.py --canonical-resize --no-pad-vision-to-slots --max-seq-len 10240 ...   # --spmd/--fsdp are inert off-TPU
```

Length filter at that setting, from the uploaded manifest: DENTEX with-tools 540 of 678 (79.6 %) fit 10,240 and 218 (32.2 %)
fit 8,192; the 880-trace set 736 (83.6 %) and 379 (43.1 %). The excluded traces are the longest (multi-finding) ones. Memory is
unmeasured: your earlier note of ~21.6 GB at 10,240 predates canonical views, so treat 10,240 as the starting point and
`--max-seq-len 8192` (or `--precision qlora`) as the fallbacks if it does not fit.

## 13. Deferred items (to attend after the hardware path is validated)

1. **Evaluation policy: evals run on original-size images**, whichever hardware trained the model. Today only
   `scripts/evaluate_models.py` has the switch (`--canonical-resize`, default off = original size). Still native-only and
   untouched: `dental_agent/evaluation/ablations.py` (5 call sites), `sweep.py`, `batch_runner.py` (2),
   `dental_agent/rewards/judge.py`. **Open question:** a checkpoint trained on canonical views and evaluated on original-size
   views sees a different input distribution than it was trained on (a native DENTEX image is 3.7x the tokens). Decide whether
   that is intended (report both settings) before quoting numbers.
2. **68 of 880 traces drop at a single 16,384 graph**; 18,432 drops 16, 20,480 drops 3, tiers drop ~3 at ~84 % of the compute.
   Kept at 16,384 until the first HBM measurement.
3. **GRPO decode shapes** on XLA (static cache / bucketed decoding).
4. **LoRA coverage of linear-attention layers** (12.5).
5. **Letterbox vs stretch ablation**, if the paper needs the accuracy claim.

