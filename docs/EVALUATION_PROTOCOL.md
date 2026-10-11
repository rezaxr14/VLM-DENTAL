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
result because it is what the model was trained on; original size is reported alongside it. (planned)

**No-tools prompt.** The no-tools track uses one prompt, with `thought` and `final_answer`. At SFT load it replaces the
system prompt stored in each trace, the same way the teacher directive is stripped from the first user turn, so old
trace files stay valid. Evaluation and GRPO rollouts use the same prompt. (planned)

## 3. Metrics

- Finding-level precision, recall and F1, pooled over all test images, using `match_multi_findings` (never the first
  annotation row, Rule 13). Image-level "exact match" is kept as a secondary number.
- Paired bootstrap confidence intervals for differences between conditions on the same images.
- Per-class results, and the healthy images reported separately (false-positive behaviour).

## 4. Provenance to state in the paper

- The trace generator and verifier models are described generically, as role plus the cohort each handled.
- The YOLO grounding checkpoint behind `locate_tooth` must be chosen on data other than the test set. The fold used
  today was ranked on the official validation images; confirm which fold the runs use before reporting.
- The base-model zero-shot file is missing one test image; rerun it so every condition covers the same 50 images.
