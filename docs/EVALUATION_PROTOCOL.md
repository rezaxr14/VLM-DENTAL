# Evaluation protocol

What is held out, what is compared, and how it is measured. This file records decisions; each one is marked
**implemented** (in the repo now) or **planned** (a later patch of the evaluation work).

## 1. Splits

| Split | Source | Used for |
|-------|--------|----------|
| Train | DENTEX train images, minus the dev split | SFT and GRPO training |
| Dev | About 70 DENTEX train images, fixed seed, `data/splits/dentex_dev.json` | GRPO group size K sweep, ablations, any hyper-parameter or checkpoint choice |
| Test | The official DENTEX validation set (`validation_triple.json`, 50 images) | Final numbers in the paper, and nothing else |

**Implemented.** `scripts/make_dev_split.py` draws the dev split once. `train_sft.py --dev-split` drops the traces of
dev images; `run_grpo.py` and `run_grpo_sweep.py` load the train images only (they previously loaded the default
split, which is the test set) and also take `--dev-split`.

Image ids collide between train (1 to 705) and validation (1 to 50), so an image is always identified by
`(dataset, image_id)`, written `dataset::image_id`, never by the id alone. `assert_not_test_split` refuses to
train on a dataframe whose annotation file is the validation file.

Healthy images (no annotations) stay in the GRPO pool with ground truth `[]`; the accuracy reward is 1.0 only for an
empty answer on them. Before this change they were skipped, so the policy never practised answering "no findings".
`--no-include-healthy` removes them if a run needs pathological images only.

## 2. Conditions

Base, SFT and GRPO checkpoints, each with and without tools, on the test set. Ablations (tool contribution, the
`locate_tooth` then `nudge_crop` self-correction loop, GRPO group size K) run on the **dev** split, and the single
chosen configuration is then evaluated once on the test set.

**Image size.** Every condition is evaluated at both canonical size and original size. Canonical size is the primary
result because it is what the model was trained on; original size is reported alongside it. `evaluate_models.py
--canonical-resize` selects the view; canonical results are written to `..._canonical.jsonl`, so the two never share a
file. Running both for every condition in one go is planned.

**One loop for training and evaluation.** Evaluation runs the same `run_agent` loop GRPO rolls out with, so prompts,
the `Result of <tool>:` observation format, canonical views and KV-cache reuse are identical. Evaluation used to have
its own loop with a different tool-result format and no cache reuse. Budgets are the training ones, set explicitly:
50 tool calls, turns bounded at tool calls plus 15, 2048 new tokens per turn (`--max-tool-calls`, `--max-turns`,
`--max-new-tokens`; GRPO takes `--max-new-tokens` too). The earlier defaults (15 turns, 25 tool calls) would have cut
off about a quarter and 43% of the training traces, and the implicit 512-token cap about 2% of their turns.
`--malformed-retries` (default 0, as in GRPO) answers an unparseable reply with an error message instead of ending
the episode.

**Prompts.** Both tracks train, roll out and evaluate under the current prompt of the track. At SFT load it replaces
the system prompt stored in each trace, the same way the teacher directive is stripped from the first user turn, so old
trace files stay valid. Without it, most stored with-tools traces carried an older prompt that lacks the healthy-scan
instruction, and the no-tools traces carried the teacher prompt. The no-tools prompt has a `thought` and a
`final_answer`. Stored Tufts traces keep their own prompt, since a record does not say which class list it used.
Recompute the token-length manifest after this change (`scripts/compute_exact_trace_lengths.py`).

**Healthy scans.** `final_answer: []` is a valid, correct answer. It used to count as a format failure, and a failed
episode (unparseable, or cut off by a cap) scored full accuracy on a healthy image. Both are fixed in the reward.

**Base no-tools.** `base_no_tools` uses the unified prompt like the other no-tools conditions. The earlier zero-shot
runs (a different prompt, used for the cross-model comparison) stay valid for that table; `--reuse-cached` loads them
and tags the records `zero_shot_cached`, which are not comparable with the unified-prompt conditions.

## 3. Metrics

**Implemented** in `dental_agent/evaluation/matrix.py`.

- Finding-level precision, recall and F1, pooled over all images, using `match_multi_findings` (never the first
  annotation row, Rule 13), at two levels: exact (right tooth and diagnosis) and FDI (right tooth). Counts are recomputed
  from the stored ground truth and predictions. Image-level "all findings right" is kept as a secondary number.
  Worked example: three images with (2 GT, 2 predicted, 1 exact), (1 GT, nothing predicted) and (healthy, 1 invented
  finding) pool to TP 1, predicted 3, GT 3, so precision 1/3, recall 1/3, F1 = 2 TP / (predicted + GT) = 2/6. The pooled
  numbers are repeated for pathological images only.
- Confidence intervals resample whole images, since the findings of one image are not independent.
- Paired bootstrap differences between conditions on the images both cover, using the same resample for both: tools vs
  no tools, SFT vs base, GRPO vs SFT, canonical vs original. A two-sided bootstrap p-value accompanies each.
- Per-class precision/recall/F1 (a right tooth with the wrong diagnosis is a miss for the true class and a false
  positive for the predicted one), and the healthy images on their own: how many get invented findings.
- A format failure counts as an empty answer in the pooled numbers and is reported separately (format ok rate).

## 4. Provenance to state in the paper

- The trace generator and verifier models are described generically, as role plus the cohort each handled.
- `locate_tooth` uses one fixed YOLO grounding checkpoint in every run. It is a supporting tool, not a contribution;
  state the checkpoint and that it is held fixed across all conditions. (The checkpoint was ranked on the validation
  images, so describe it as the grounding tool in use, not as a tuned component.)
- The base-model zero-shot file is missing one test image; rerun it so every condition covers the same 50 images.

## 5. GRPO start and resume

The trainable `grpo_policy` adapter is initialised as a copy of the frozen SFT `reference` adapter, so the first
rollout comes from the SFT model and the KL term is exactly zero at step 1. (Before this it started from the bare
base model while the reference was the SFT model.)

A GRPO run is resumable from its step checkpoints: policy adapter, step counter and reward baseline are restored and
the images already trained on are skipped. It refuses to resume if the image pool or group size differ. Optimizer
moments are not stored, so Adam restarts cold at the resumed step; report any resumed run as such.

## 6. Before a long GRPO run: preflight and time probe

**Implemented.** Two checks, both cheap, both read-only with respect to settings (nothing is changed or switched for you).

- `run_grpo.py ... --preflight` takes the exact launch command and checks it without loading the model: the SFT adapter
  is where GRPO will look, the SFT run's recorded settings (`sft_run_meta.json`, written next to the adapter by
  `train_sft.py`) agree with the GRPO command on track, canonical resize and the held-out dev split, the dev split is
  disjoint from the training pool, every pool image exists on disk, a `--resume` still matches the pool and group size,
  a hub token is present when uploads are on, and on a TPU the core count matches `--num-cores`. It never downloads data.
  Adapters trained before the metadata file existed give a warning instead of a comparison.
- Every GRPO step now logs its rollout seconds, update seconds, mean turns, mean tool calls and format-ok rate.
  `scripts/summarize_grpo_log.py` turns a short probe (`--max-images 10`, own `--log-path`, no hub uploads) into the
  projected hours of the full pool and of other group sizes. Time per step scales linearly with K, so a probe at K=4
  averaging 600 s per step over a pool of 635 images projects to 635 x 600 s = 106 h, and K=8 to 211 h.

The projection is an estimate from the probe images; the logged turns and tool calls show whether they were typical.

## 7. Running the matrix

**Implemented.** `scripts/run_eval_matrix.py` runs the grid (checkpoints x tools x image size) on one split. `--split
test` is the official validation set and is for final numbers only; `--split dev` evaluates the dev images held out of
training (`--dev-split`), which is where the ablations and the group-size sweep are measured. Each cell writes
`eval_<dataset>_<split>_<condition>[_canonical].jsonl`, one record per image. A record keeps the settings it was run with,
the seconds taken, and a compact transcript (every message except the constant system prompt, images replaced by
`<image>`, plus each turn's status and tool calls), so failures can be read and qualitative figures drawn without
re-running anything.

`--existing-results` decides what happens to a file that already has records: `error` (the default, instead of the old
silent appending of duplicates), `resume` (finished images are kept, only the rest run; refused when the settings differ,
so two settings can never be mixed in one file) or `overwrite`. A finished cell is skipped without loading the model.
The tables (`matrix_summary.md` and CSVs, including per-class and paired comparisons) are written to
`<output-dir>/matrix_<dataset>_<split>/` and can be recomputed without a model with `scripts/aggregate_eval.py`.
The cached zero-shot file used another prompt, so the matrix always runs live under the unified prompt.

## 8. Tool ablations and localization

**Implemented (evaluation switches only; nothing is removed from the repo and nothing is retrained).** The trained agent
is run with a tool withheld or with a deliberately wrong `locate_tooth` box, to measure how much it depends on its tools
and whether it can correct itself. Run them on `--split dev`, with `--conditions` of your choice (only `*_with_tools`
cells run), and read the tagged files next to the normal ones; the unablated results are never touched.

| flag | what it does |
|---|---|
| `--remove-tools nudge_crop` | the tool is absent from the tool list and a call to it gets the standard "not recognized" error |
| `--locate-shift-frac 0.3` | every `locate_tooth` box moves by 30% of its own size in a fixed pseudo-random direction |
| `--locate-scale 1.5` | every `locate_tooth` box is rescaled about its centre |
| `--ablation-seed` | seed of that direction; one (seed, image, tooth) always gets the same wrong box, so runs are paired |
| `--iou-threshold 0.5` | IoU at which a box counts as hitting the annotated tooth |

A result file is named `eval_<dataset>_<split>_<condition>[_canonical]__<tag>.jsonl`, for example
`..._sft_with_tools_canonical__no-nudge_crop.jsonl`.

**Reading the results.** `matrix_summary.md` gets an "Ablation" section per tag: the ablated exact F1, the unablated one
on the same images, and their difference with a paired bootstrap interval. Removing a tool measures a trained agent's
*dependence* on it. The prompt the checkpoint was trained with still mentions the tool, so how often the model asked for
it anyway is reported as attempts per image. It is not a model trained without the tool, and the paper should say so.

**Localization** (every `with_tools` cell, ablated or not). For each annotated finding whose tooth the agent located,
the IoU of its working box with the annotated tooth box is taken right after `locate_tooth` (initial) and after its
`nudge_crop` calls (final); a box counts as hitting at IoU >= the threshold. Worked example: annotated box
[100, 100, 100, 100], located box [150, 100, 100, 100] has IoU 5000 / 15000 = 0.33 (a miss); after a nudge by
dx_frac -0.5 the box is the annotated one, IoU 1.0 (a hit, recovered). Three findings with (initial, final) IoU of
(0.33, 1.0), (0.8, 0.8), (0.2, 0.3) give a hit rate of 1/3 initially and 2/3 finally, and 1 of the 2 initial misses
recovered. Also reported: how much of the annotated tooth the last `zoom_crop` of that box shows, nudges per finding, and
the share of findings whose tooth was located at all. Tables: `matrix_localization.csv`, `matrix_ablations.csv`.

Only findings with an annotated tooth box count; DENTEX annotations carry one. The detector (YOLOv8m) is the fixed
supporting tool, so the perturbation is a stress test of the agent's self-correction, not an evaluation of the detector.
