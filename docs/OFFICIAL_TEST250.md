# DENTEX test cohort (250 images)

`EVAL_SPLIT = "official_test250"` evaluates the 250-image DENTEX test split (`DENTEX/test_data.zip`) next to the existing 50-image `"test"` split. The 50-image branch,
the dev split, the training data and the SFT/GRPO code and notebooks are unchanged; nothing is tuned on the 250 images. Results of the two splits are separate cohorts (the split
is part of every result path and identity). Zero-shot (local and frontier providers), baseline, SFT and GRPO are all evaluated and scored on this cohort.

## Labels

The zip holds `disease/input/test_N.png` and `disease/label/test_N.json` (250 + 250), LabelMe polygons, 1600 in total. A label is `<code>-<Turkish word>-<FDI>`; the FDI tooth number is
the LAST token. They are converted once into one COCO file with the structure of `validation_triple.json` (`images`, `annotations`, `categories_1/2/3`; image id = N + 1;
`category_id_1/2` are 0-indexed FDI quadrant and tooth). Every polygon is kept.

## Classes: the English name of each Turkish word

| code | word | polygons | class | id |
|---|---|---|---|---|
| 1 | çürük | 747 | Caries | 1 (DENTEX) |
| 6 | gömülü | 221 | Impacted | 0 (DENTEX) |
| 7 | lezyon | 75 | Periapical Lesion (any lesion name a model writes, Tufts lesion classes included, is normalised to it by `evaluation/metrics.py`) | 2 (DENTEX) |
| 3 | kanal | 161 | Root Canal | 4 (appended) |
| 2 | küretaj | 265 | Curettage | 5 (appended) |
| 5 | çekim | 29 | Extraction | 6 (appended) |
| 8 | kırık | 11 | Fracture | 7 (appended) |
| 0 | saglam | 91 | none: the healthy-tooth polygon is a negative (a disease predicted on it is a false positive) | - |

`categories_3` keeps the four DENTEX classes first and identical (Impacted, Caries, Periapical Lesion, Deep Caries) and appends the four others. Nothing is ignored or dropped. Deep Caries stays in the
class list but no code of these labels is Deep Caries by name; see the proximity credit below.
The meanings come from Turkish dental sources (kanal tedavisi = root canal treatment, diş çekimi = tooth extraction, diş kürretajı = dental curettage, gömülü diş = impacted tooth, diş çürüğü = caries,
kron/kök kırığı = crown/root fracture); they are listed with their evidence in `dental_agent/data/label_maps/official_test250_v1.json`, which is versioned: edit it only through a new version (the COCO
records its SHA256 and the loader refuses a mismatch).

## Prompts (this cohort only)

The prompt for this cohort names the eight classes (the four DENTEX ones, spelled as the repository prompts spell them, plus Root Canal, Curettage, Extraction, Fracture), so a model that has never seen
the last four, or a frontier model, is told they exist and is scored when it finds them. The agent prompt (`dataset="official_test250"`, `args.prompt_dataset`), the zero-shot prompt and the no-tools prompt are the existing prompts
with only the class list replaced (derived in code, and a test proves the rest is identical; if a prompt is edited so the list changes shape the derivation raises instead of silently diverging).
`args.dataset`, and so every result path, is unchanged. Dev, training, SFT/GRPO and every other split keep their prompts byte for byte.

## Scoring (`dental_agent/evaluation/test250_scoring.py`, profile `test250_scoring_v2`)

The standard `match_multi_findings` over all classes, unchanged, so the existing proximity credit stays (Caries <-> Deep Caries 0.75, FDI-localisation F1, the closeness score). The only additions: prediction names a model
writes freely ("root canal treatment", "tooth extraction", "fractured crown") are put into the four cohort names (anything it cannot place is simply a non-match), and `family_exact_*` (Caries and Deep Caries as one class).
Extra record fields: `scoring_profile`, `n_renamed_predictions`, `family_exact_precision/recall/f1`; the COCO and mapping versions and SHA256 are in every result's settings and identity.
The primary metric is per-tooth F1 (FDI + class). The challenge-style box AP is not computed: the agent outputs findings, not boxes, and the repository has no AP code for them.

## Reading order: local disk, Kaggle, Hugging Face

The COCO and the 250 images are read from local disk first, then from an attached Kaggle dataset (`/kaggle/input/datasets/rezanadimikj/<slug>` or `/kaggle/input/<slug>`, found by the same input scan the other
notebooks use), then from the Hugging Face cache, then downloaded from `DENTEX_TEST_REPO` (default `Reza-Nadimi/dentex-official-test`, files `coco/official_test250_<v>.json`, `disease/input/test_N.png`). When copies differ the first
source wins and the disagreement is printed. The COCO is never re-converted at evaluation time. Exactly 250 images are required: otherwise `FileNotFoundError` lists the missing ids and nothing is launched.

## One-time data preparation (three commented cells at the end of both Final notebooks)

Uncomment one at a time, run, comment out again: (1) convert (`scripts/convert_official_test_to_coco.py`: stops if the zip hash, the validation file's area convention, an image size, an FDI number or an id check fails);
(2) upload to Kaggle (`scripts/stage_official_test250_dataset.py` stages `coco/`, `label_maps/`, `disease/input/`, `dataset-metadata.json` from `data/official_test250` AS IT IS, so a hand-corrected COCO goes up unchanged; then the same
`kaggle datasets version ... || create` pattern as the SFT notebook; credentials `KAGGLE_USERNAME`/`KAGGLE_KEY` or `~/.kaggle/kaggle.json`); (3) overwrite the Hugging Face copy
(`scripts/publish_official_test_coco.py --overwrite`: the two files only, one commit, never `disease/`, never creates the repository, refuses a repository that is not private). Without `--overwrite` it refuses to replace an existing version.

## Changes outside the cohort modules (all additive; defaults keep the old behaviour)

`agent/prompts.py` one new `dataset` key; `agent/loop.py` optional `system_prompt=None` on `run_agent_no_tools`; `scripts/evaluate_models.py` the cohort split, scoring and prompt hooks; `scripts/run_zero_shot.py` the cohort split, prompt and scoring;
`harness/notebook_eval.py` the split in `prepare_inputs`, the identity block, `--split official_test250` for the zero-shot command; the split name in the `--split` choices of `run_eval_matrix.py`, `aggregate_eval.py`, `plot_eval_figures.py`,
`flatten_results.py`; `EVAL_SPLIT`'s comment and the commented cells in the two Final notebooks.

## Verified, and not

Verified offline (no GPU, no torch, no Hugging Face or Kaggle access): FDI conversion, polygon geometry and the validation-file area check, the mapping validator, the conversion (real polygon counts), the 250-image gate, the cohort resolver
(local, Kaggle and Hugging Face-cache order, differing copies, Hub fallback, incomplete cohort, changed mapping), the loader, the scoring rules, that each cohort prompt equals its base prompt except for the class list, the staging script and its read-back
by the resolver, the publisher (including overwrite), the commands of the three notebook cells against the real argument parsers, and the session planning (identity, `--split official_test250` for zero-shot and baseline). Not verified: the real conversion (it
checks the validation file's area convention and stops if it does not match), the real Kaggle and Hugging Face uploads, `scripts/run_zero_shot.py` end to end (its imports need the full torch stack; only its additions were reviewed in place), and any model run.
