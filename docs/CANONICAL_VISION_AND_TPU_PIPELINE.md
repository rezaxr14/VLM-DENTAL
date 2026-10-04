# Canonical Vision Inputs, Static Vision-Slot Padding & the TPU Training Path

**Status:** implemented in the working tree, unit-tested on CPU, **not yet run on TPU hardware** (see §9).
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
| 8 | **"Zero-compromise GPU path" via `is_tpu` toggles** conflicts with the standing decision that a GPU path would be a *separate* clean build with no shared backend-switching. | Medium | We added **no GPU code**: every new flag defaults off outside TPU, and GPU/CPU run exactly the pre-existing native + dynamic-padding code. The pre-existing `is_tpu` collator branch was already in the repo. Revisit if you want a hard split. |
| 9 | **Hardware naming inconsistency.** Code/notebooks say Cloud TPU **v5e-8**; project notes say Kaggle **TPU v3-8**. Both are 8 × 16 GB, so the memory math is unaffected, but the paper must name the hardware that was actually used. | Medium | **[open]** — confirm and fix naming. |
| 10 | **Unverifiable evidence.** The 67.6 % / 54.5 % / 224-token reconciliation is arithmetically correct (see §8) but its scripts (`scratch/*.py`) are not in the repo, and the 91-shape census was not reproducible. | Low | Added `scripts/census_vision_slots.py` so the numbers regenerate from the committed traces. |

## 4. Decisions

Decided in this session on the "do what is best, authentic, paper-ready" instruction. **D3, D5 and D8 are the ones to
confirm or overrule.**

- **D1 — Views, not data.** The model sees canonical views; tools run on native images; ground-truth coordinates stay
  in native pixel space. The model never emits pixel coordinates itself (every bbox comes from a tool), so the
  view/coordinate scale difference is harmless and consistent between SFT, GRPO and eval.
- **D2 — One module for geometry** (`canonical.py`); no constants duplicated elsewhere.
- **D3 — Sequence length 16,384 in padded (TPU) mode; unchanged elsewhere.** `--max-seq-len` keeps its 10,240 CLI
  default (pinned by an existing test and AGENTS.md Rules 14/19); the TPU path raises it to 16,384 only when
  `--pad-vision-to-slots` is active and the value is still the default. Never lowered. *Trade-off:* every step runs a
  16,384-token sequence (~1.6× the LLM tokens of a 10,240 graph) regardless of trace size. A tiered budget (a few
  slot tiers → a few graphs) is the obvious optimization if throughput, not compile count, becomes the bottleneck; it
  needs the census output to size the tiers, so it is deferred rather than guessed.
- **D4 — Fail loudly instead of degrading.** Padded mode raises on: non-canonical grids, over-budget slots,
  overlength sequences. `safe_process_vision_info` warns (instead of silently returning `(None, None)`) when images
  are present but cannot be loaded.
- **D5 — Plain LANCZOS stretch, no letterbox (kept as specified).** This is the project's settled spec, but it is a
  methodological limitation for CROP/COMPARE (§10): `zoom_crop` boxes are near-square while CROP is 2:3. A
  letterbox variant is the recommended ablation; it was *not* implemented to avoid a mid-stream spec change.
- **D6 — XLA shim patches `torch.linalg.solve_triangular`, not the model function.** Copying upstream's function body
  would drift with `transformers`. The shim intercepts only `left=True, upper=False, unitriangular=True` on `xla`
  tensors and uses a log-depth matmul product expansion (6 factors for chunk size 64) instead of upstream's 63-step
  in-place loop. CPU/CUDA behaviour is untouched.
- **D7 — One resolver for "canonical or not".** `resolve_canonical_resize(value)`: explicit argument wins, else
  `DENTAL_CANONICAL_RESIZE`, else off. Training/GRPO/eval entrypoints expose `--canonical-resize`; library callers
  that don't pass it (ablations, sweep, judge, batch runner) inherit the env var, so a whole eval run is consistent.
- **D8 — Defaults.** TPU: `canonical_resize=True`, `pad_vision_to_slots=True`. GPU/CPU: both off. GRPO/eval default
  off unless the env var/flag says otherwise (they must be set to match the SFT checkpoint — a notebook cell does it).
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

## 6. Runbook

```bash
# 0. (once per resolution mode) lengths + census — CPU only, needs traces + images
python scripts/compute_exact_trace_lengths.py --canonical-resize --recompute
python scripts/census_vision_slots.py data/traces/train_cot_traces_dentex.jsonl \
    --manifest data/traces/trace_token_lengths.json --seq-lens 10240 16384

# 1. AOT warmup (TPU) — must equal the training graph
python scripts/warmup_xla_cache.py --spmd --fsdp --max-seq-len 16384 --pad-vision-to-slots

# 2. SFT (TPU: canonical + padded are the defaults)
python scripts/train_sft.py --spmd --fsdp --max-seq-len 16384 --canonical-resize --pad-vision-to-slots ...

# 3. GRPO — rollouts must see the same views as SFT
python scripts/run_grpo.py --canonical-resize ...

# 4. Evaluation — same views as the evaluated checkpoint
python scripts/evaluate_models.py --canonical-resize ...
```

GPU/CPU: run the same scripts without those flags — native resolution, `dynamic_padding=True`, no XLA code imported.
Notebook equivalents are the `CANONICAL_RESIZE` / `PAD_VISION_TO_SLOTS` cells.

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

## 9. Open items (honest list)

1. **No TPU run yet [open].** Not verified: that GSPMD shards the shim cleanly, that HBM fits at 16,384, that XLA
   compiles exactly once, that `save_pretrained` gathers sharded parameters under SPMD (already on the horizon list).
   First hardware check: `warmup_xla_cache.py --spmd --max-seq-len 16384` must not raise `RESOURCE_EXHAUSTED`; if it does,
   the fallback is tiered slot budgets (D3), **not** fewer cores or a lower token ceiling (standing rules).
2. **GRPO on TPU is not solved by this change.** Rollouts use HF `generate` with growing, dynamically-shaped inputs and
   still launch through legacy `xmp.spawn`; static slot padding addresses the *SFT training graph* only. The canonical
   plumbing makes GRPO *consistent* with SFT; it does not make it TPU-efficient.
3. **`--spmd` default.** `train_sft.py` defaults `--spmd` to **False** at this commit although project notes record it
   defaulting to True after patch 0004; the notebook passes it explicitly. Confirm patch 0004 fully landed.
4. **Census numbers (91 shapes, 100 % coverage, length fit) not reproduced here** — run `census_vision_slots.py`.
5. **Hardware naming** (v5e-8 vs v3-8) — fix in code comments, docs and the paper.
6. **Runs predating the manifest-key fix** may have trained on truncated targets (§5.2).
7. **`docs/SFT_RESULTS.md` memory figures** (e.g. "~6.86 GB … under native resolution") are historical measurements taken before canonical resizing and slot padding; re-measure on the new graph before quoting them.
8. Still outstanding from before: regenerate the 18 Tufts no-tools traces; zero-shot evaluation notebook; Tunisia loader.

## 10. Paper-ready statements and limitations

**Methods (suggested text).** *Radiographs are presented to the model as fixed-size views: the base image and
whole-image operators at 1536×768, `zoom_crop` outputs at 256×384 and `contralateral_compare` composites at 512×384
(Lanczos resampling; 1,152 / 96 / 192 visual tokens). Tools execute on the original-resolution image and all
coordinates are in original pixel space. The identical views are used for SFT, GRPO rollouts and evaluation.*

**Limitations to state.**
* **Anisotropic CROP resampling.** `zoom_crop` boxes (pad = max(25 %·side, 50 px)) are close to square; CROP is 2:3.
  Measured stretch (vertical ÷ horizontal): 60×120 box → 1.09; 110×150 → 1.26; 152×148 → 1.52; 200×160 → 1.73. FULL
  distortion is small (0.955 for the 2872×1504 scan). Morphology cues in crops are therefore scaled non-uniformly,
  identically at train and test time. **Recommended ablation:** letterbox/pad-to-canvas (same token counts) vs stretch.
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
