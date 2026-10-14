# Evaluation: what was done and why

Paper-ready account of the evaluation design. It states each decision with its reason and one worked example, and leaves
the results out: numbers belong in the tables and figures the aggregator writes, so this text never goes stale. The
mechanics (flags, file names, resume behaviour) are in `docs/EVALUATION_PROTOCOL.md`; the commands are in
`docs/RUNBOOK_SFT_GRPO.md`. Items marked **to run** have code but no results yet.

## 1. What is claimed, and what the evaluation has to show

The system is an agent: a vision-language model that calls image and grounding tools over several turns, trained first on
verified traces (SFT) and then by reinforcement learning (GRPO). The evaluation therefore answers four questions, each
with its own comparison and nothing else changed between the two arms:

1. Does training help? Base against SFT against GRPO.
2. Do the tools help? The same checkpoint with and without tool access.
3. How much does the trained agent depend on a particular tool, and can it repair a wrong grounding box? Evaluation-time
   ablations of the trained agent, section 5.
4. Which GRPO group size K to use? A sweep on held-out development images, section 7, **to run**.

## 2. Held-out data

Three disjoint sets are used and each has one job.

| Set | Images | Job |
|---|---|---|
| Train | the DENTEX training images minus the dev split | SFT and GRPO |
| Dev | about 70 training images, drawn once with a fixed seed | the K sweep, the ablations, every checkpoint or setting choice |
| Test | the 50 official validation images | final numbers, once, for the chosen configuration |

The reason for a dev split is that every choice made by looking at test scores is a leak. All choices are made on dev, and
the test set is run once at the end. An image is identified by its dataset and id together, because the id alone collides
between train and validation; training code refuses an annotation file that is the validation file.

Healthy images (no annotated finding) stay in training and in evaluation with ground truth `[]`. A correct answer for them
is an empty one. Leaving them out would let a model that invents findings look better than it is, and would never teach a
model to say that nothing is there.

## 3. Conditions and image size

Every condition is a checkpoint (base, SFT, GRPO) with or without tools, and every condition is evaluated at two image
sizes: the canonical size the model was trained on (the primary result) and the original size (reported alongside it, to
show how much the result depends on the training view).

Evaluation runs the same agent loop that GRPO rolls out with, with the budgets of training set explicitly. Prompts, the
format of tool observations and the canonical views are therefore identical in training and in evaluation, and a
difference between a training reward and an evaluation score cannot come from a different loop. Both tracks are evaluated
under the current prompt of the track, so no condition is advantaged by an older prompt.

## 4. How a result is measured

**Findings are matched as a set.** Each image has one or more annotated findings (a tooth and a diagnosis). Predictions
and annotations are paired one to one by optimal assignment, so a prediction counts once and order does not matter. A
finding is right at two levels: exact (right tooth and right diagnosis) and tooth-only (right tooth).

**Counts are pooled over all findings before the score is computed.** Worked example, three images:

| image | annotated | predicted | matched exactly |
|---|---|---|---|
| A | 2 | 2 | 1 |
| B | 1 | 0 | 0 |
| C | 0 (healthy) | 1 | 0 |

Pooled: 1 match, 3 predicted, 3 annotated, so precision 1/3, recall 1/3 and F1 = 2 x 1 / (3 + 3) = 1/3. Averaging the
per-image F1 would give (0.5 + 0 + 0) / 3 = 1/6 and would let image C, which has nothing to find, count for as much as an
image with seven findings. Pooling also charges the invented finding on the healthy image to precision, which is where it
belongs.

**Why every finding must be in the ground truth.** An earlier evaluator took the first annotation row of an image as its
whole ground truth. With three annotated findings that error is symmetrical and silent: a model that reports all three is
charged two false positives (F1 = 2 x 1 / (3 + 1) = 0.5, true value 1.0), and a model that reports only the first looks
perfect (F1 = 1.0, true value 0.5). The matrix runner, the ablation and batch evaluators and the sweep now share one
definition of a finding (`dental_agent/evaluation/ground_truth.py`): every annotation row, `[]` for a healthy image.

**Uncertainty resamples images, not findings.** The findings of one image share its quality, noise and difficulty, so they
are not independent. A confidence interval is obtained by resampling whole images with replacement and recomputing the
pooled score each time.

**Comparisons are paired.** To compare two conditions (tools against no tools, SFT against base, GRPO against SFT,
canonical against original), both are scored on the images they share and the same resample is applied to both, so image
difficulty cancels. Worked example, three images, condition A with matches 2, 1, 0 out of predictions 2, 1, 1 and condition
B with matches 1, 0, 0 out of predictions 1, 1, 0, with 2, 1, 1 findings annotated: A pools to 3 matches, 4 predicted,
4 annotated (F1 0.75); B to 1, 2, 4 (F1 0.33); the paired difference is +0.42. Three images cannot support a claim, and the
interval says so (0 to 1 at the default seed). With the 50 test images the intervals narrow, but they stay wide enough
that a difference whose interval touches zero should be reported as not established, whatever its point value.

Two further reports go with every cell: the share of healthy images on which findings were invented, and the share of
episodes that ended without a parseable answer. A failed episode counts as an empty answer in the pooled score and is
reported separately, so format failures cannot hide inside the accuracy.

## 5. Tool dependence and self-correction

The self-correcting grounding loop is the system's distinguishing mechanism: the agent calls `locate_tooth` for a box,
inspects the crop, and calls `nudge_crop` to move the box when it is off the tooth. Two evaluation-time switches test it,
both run on the trained agent with nothing retrained:

- **Remove a tool.** The tool disappears from the tool list and a call to it gets the standard "not recognized" error. The
  drop in score is the agent's measured dependence on that tool. It is not the score of a model trained without the tool,
  because the training prompt still mentions it, and the paper must say so.
- **Perturb the localizer.** Every `locate_tooth` box is moved by a fraction of its own size, or rescaled, in a fixed
  pseudo-random direction (the same wrong box for the same image and tooth in every run, so conditions stay paired).
  This is a stress test of the agent's self-correction; it is not an evaluation of the detector. The detector is one fixed
  grounding checkpoint, used identically in every condition and described as the grounding tool in use, not as a tuned
  component.

**Localization is measured against the annotated tooth box.** For each annotated finding whose tooth the agent located, the
overlap (intersection over union, IoU) between its working box and the annotated box is taken right after `locate_tooth`
(before) and after its `nudge_crop` calls (after); the box hits when IoU reaches 0.5. Worked example: annotated box
`[100, 100, 100, 100]` and located box `[150, 100, 100, 100]` overlap 5,000 of 15,000 pixels, IoU 0.33, a miss. After a
nudge by minus half a box width the box is the annotated one, IoU 1.0, a hit that was recovered. Three findings with
(before, after) IoU of (0.33, 1.0), (0.8, 0.8), (0.2, 0.3) give a hit rate of 1/3 before and 2/3 after, and one of the two
initial misses recovered. The same measurement under a perturbed localizer shows whether nudging repairs a bad box; with
`nudge_crop` removed the before and after rates coincide by construction.

## 6. Figures

All figures are drawn by `scripts/plot_eval_figures.py` from the tables of `scripts/aggregate_eval.py`, so each figure
has a table twin and the two cannot disagree. Blue is with tools and orange is no tools, a pair chosen to stay
separable under colour-blind simulation; text never wears a series colour, and hollow markers mean the interval includes
zero.

| Figure | Shows | How to read it |
|---|---|---|
| `matrix_forest_<level>` | pooled F1 with its 95% interval for every condition, one panel per image size | Non-overlapping intervals are a hint, the paired figure below is the test. Canonical is the left panel. |
| `paired_comparisons_<level>` | paired F1 differences with paired intervals, in blocks (canonical, original, size) | Filled marker: the interval excludes zero. No correction is applied for the number of rows. |
| `ablation_effects_<level>` | ablated minus unablated F1 per ablation run and checkpoint | Negative means the agent leans on what was removed or damaged. |
| `localization_before_after` | box hit rate after `locate_tooth` (hollow) and after `nudge_crop` (filled) | A long dumbbell is repair; a dot is no change. The text column gives located findings and recovered / recoverable. |

`<level>` is `exact` (primary) or `fdi` (tooth only). The footnote of every figure states the resample count and the number
of images per cell, flags any cell that came from the cached zero-shot run (different prompt, not comparable) and any cell
that covers fewer images than the others.

## 7. The GRPO group-size sweep

K is the number of rollouts GRPO samples for each training image. The cost grows with K (twice the rollouts per image for
twice the K), and a group whose K rewards are all equal teaches nothing, because every advantage is zero. The sweep asks how
much K the policy needs.

**Design.** One GRPO run per K, each from the same SFT adapter, on the same seeded training images in the same order
(`--dev-split`, `--seed`, `--max-images`), with every other setting identical. Each final checkpoint is evaluated on the
dev images with the agent loop and budgets of training, canonical size, next to the SFT reference on the same images. The
split is dev, never test: choosing K by test score would leak the test set.

**What is compared, all paired over the same images.** Each K against SFT (did GRPO at this K beat its starting point), each
K against the best K (is it distinguishable from the best), and each K against the next smaller one (does doubling still
pay). Illustrative numbers: K = 4 and K = 8 pool to exact F1 0.58 and 0.60. The 0.02 gap is inside the evaluation noise if the
paired interval of K = 4 minus K = 8 is [-0.09, +0.05], which contains 0, so K = 4 is not distinguishable from K = 8 and costs
half the rollouts. A paired interval of [-0.19, -0.02] for K = 2 minus K = 8 excludes 0, so K = 2 is worse at this sample size.

**What it cost.** From the step log of each run: hours, and the tie rate, the share of groups whose K rewards were all
equal. Worked example: 40 steps at K = 2 with 14 tied groups is a tie rate of 0.35, so about a third of the K = 2 rollouts
carried no learning signal. K = 1 has no group to tie; it uses a running-mean baseline instead of group normalization, so
K = 1 against K >= 2 differs in the estimator as well as in the group size. Hours are measured, not projected: on the TPU
path a policy forward carries one rollout per core, so a K below the core count leaves rows idle and the cost is not linear in K.

**Decision rule.** The aggregator reports, and does not apply, which K are not distinguishable from the best and which of them
is cheapest. The rule used in the paper (for example: the smallest K whose paired interval to the best K contains 0) is
fixed before the table is read and stated with it. The chosen K is then run once on the test split, at canonical and at
original size.

**Limits.** One training run per K: the intervals describe the sampling of the evaluation images, not the variation between
training runs with another seed, so a small gap between two K is not evidence about K. With about 70 dev images the
intervals are wide.

Figures: `k_sweep_<level>` (pooled F1 per K with the SFT reference, and the paired difference to SFT, hollow marker = the
interval includes 0) and `k_sweep_training` (hours, tie rate, first against last quarter reward). Each has its table twin in
the same folder.

## 8. Decisions, in one place

- Hyper-parameters, the K sweep and every ablation are chosen or run on the dev split; the test split is used once.
- Canonical size is primary; original size is reported with it for every condition.
- Findings are scored pooled over all findings, with intervals that resample images and paired comparisons for every
  claimed difference.
- The no-tools conditions use one unified prompt, so the cached zero-shot runs (another prompt) are shown only in their own
  table.
- Tool removal is reported as dependence of a trained agent, never as the value of a tool-free model.
- The trace generator and verifier models are described by role and by the cohort each handled, not by product name.
- K is chosen on the dev split from the sweep table and a rule fixed beforehand; the test split sees only the chosen K.
- The grounding detector is a fixed supporting tool. Its checkpoint was ranked on the validation images, so it is
  described as the tool in use and not as a tuned part of the method.

## 9. Limits to state in the paper

- The test set has 50 images (four of them healthy); intervals are wide and several comparisons will not separate.
- Many comparisons are shown without a multiplicity correction. The figures flag screening results, not confirmed effects.
- The ablations are evaluation-time interventions on one trained agent, not retrained models.
- The K sweep has one training run per K; its intervals do not cover training-run variation.
- A resumed GRPO run restarts the optimizer moments cold; a resumed run is reported as such.
- The base zero-shot file lacks one test image (id 38); it must be rerun so that every condition covers the same 50 images
  before any paired claim involving the zero-shot condition.
- Evaluation keeps every tool image in the context, whereas SFT traces replaced older tool images once the context got
  long, so very long episodes are out of distribution (open issue in `docs/CANONICAL_VISION_AND_TPU_PIPELINE.md`).
- The TPU training path replaces the triangular solve of the Gated-DeltaNet layers with a matmul expansion, because the
  native solver ran out of memory. That replacement has not yet been run on a TPU, so TPU-trained checkpoints must not be
  described as validated until the memory test in the runbook (B4) has passed.

## 10. Status

| Item | State |
|---|---|
| Dev split, matrix runner, pooled and paired tables | implemented |
| Tool removal and localizer perturbation, localization measure | implemented |
| One ground-truth definition shared by all evaluators; healthy images evaluated | implemented |
| Figures from the aggregate tables | implemented |
| K-sweep launcher (same flags as `run_grpo.py`, resume, preflight), per-K dev evaluation, tables and figures | implemented |
| Unablated and ablated dev runs, GRPO group-size K sweep (training and evaluation) | to run |
| Test-set run of the chosen configuration at both image sizes | to run |
| Rerun of the base zero-shot condition so it covers the same 50 images | to run |

## 11. Reproducing it

Every setting is explicit on the command line; nothing is switched for you.

```
python scripts/make_dev_split.py --data-dir data --n 70 --seed 42 --output data/splits/dentex_dev.json
python scripts/run_eval_matrix.py --split dev --dev-split data/splits/dentex_dev.json --sft-adapter <sft> --grpo-adapter <grpo> --precision qlora --max-tool-calls 50 --max-new-tokens 2048 --existing-results error
python scripts/run_eval_matrix.py --split dev --dev-split data/splits/dentex_dev.json --conditions sft_with_tools grpo_with_tools --sizes canonical --sft-adapter <sft> --grpo-adapter <grpo> --precision qlora --remove-tools nudge_crop
python scripts/run_eval_matrix.py --split dev --dev-split data/splits/dentex_dev.json --conditions sft_with_tools grpo_with_tools --sizes canonical --sft-adapter <sft> --grpo-adapter <grpo> --precision qlora --locate-shift-frac 0.3
python scripts/aggregate_eval.py --output-dir data/evaluations --dataset dentex --split dev
python scripts/plot_eval_figures.py --output-dir data/evaluations --dataset dentex --split dev --formats png pdf
```

The K sweep, after the SFT adapter and the dev split exist (the sweep and its evaluation share every setting with the single runs):

```
python scripts/run_grpo_sweep.py --track with_tools --sft-stage dentex_alone --sft-model-dir <sft> --k-values 2 4 8 --epochs 2 --lr 5e-6 --kl-beta 0.04 --clip-eps 0.2 --push-every-steps 25 --checkpoint-dir data/models --canonical-resize --precision qlora --temperature 0.7 --max-new-tokens 2048 --dev-split data/splits/dentex_dev.json --seed 42 --hf-repo <repo>
python scripts/run_eval_matrix.py --split dev --dev-split data/splits/dentex_dev.json --conditions sft_with_tools --sizes canonical --sft-adapter <sft> --precision qlora --output-dir data/evaluations --existing-results resume
python scripts/run_eval_matrix.py --split dev --dev-split data/splits/dentex_dev.json --conditions grpo_with_tools --sizes canonical --grpo-adapter data/models/qwen3_5_9b_grpo_with_tools_k4_dentex_alone/grpo-with_tools-final --precision qlora --output-dir data/evaluations/k_sweep/k4 --existing-results resume
python scripts/aggregate_k_sweep.py --sweep-dir data/evaluations/k_sweep --reference-dir data/evaluations --log-template "data/models/qwen3_5_9b_grpo_with_tools_k{k}_dentex_alone/grpo_training_log.jsonl"
python scripts/plot_k_sweep.py --sweep-dir data/evaluations/k_sweep --formats png pdf
```

(The K values above are placeholders; the sweep takes whatever `--k-values` the notebook sets, and the evaluation command is repeated for every K.)

The test run is the same matrix command with `--split test` and no ablation flag, followed by the same two scripts with
`--split test`.
