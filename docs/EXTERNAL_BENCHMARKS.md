# External benchmarks: MMOral-OPG-Bench

Purpose: measure the final checkpoint (GRPO trained with the rules + skills harness) on a benchmark we did not build, next to the raw
Qwen3.5-9B and two public dental models, all scored by one judge, and report how much the agent adds. Nothing here trains or tunes.
Control centre: `notebooks/VLM_Dental_Colab_External_Benchmarks.ipynb`. CLI: `scripts/run_external_benchmark.py`. Code: `dental_agent/benchmarks/`.

## What was checked, and what was found (read November 2026; sources are the public pages and repositories named)

| Item | Finding | Decision |
|---|---|---|
| MMOral-OPG-Bench (`OralGPT/MMOral-OPG-Bench`, MIT, not gated, 382 MB) | Two TSVs, images as base64. Closed: `index, image_id, file_name, question, category, image, type, option1-4, answer` (a letter). Open: `index, image_name, image, question, answer, category`. 100 radiographs, 491 closed, 578 open questions; categories Teeth, Patho, HisT, Jaw, SumRec (+ Report for open). | Run. |
| MMOral-Uni-Bench | Gated, no dataset card. Per OralAgent's description its modalities are intraoral photos, periapical and cephalometric radiographs, pathology images, video; no panoramic. | Not run: out of domain for an OPG-only model. |
| MMOral-OPG-Training | Gated training data (18.7 GB). | Not used: using it would change the training recipe. |
| OralDetect-Bench | Gated detection benchmark. | Not used: not VQA. |
| MMOral-X, OPG-Bench | No public release found (MMOral-X: "will be released"; OPGAgent's repository has code only). OPG-Bench scores (location, field, value) triples, the best match for our output format. | Not run. Ask the authors for access. |
| OralGPT-Omni-7B-Instruct, OralGPT-OPG-7B-Instruct | Gated, Qwen2.5-VL-based, loadable with transformers. | Run as public baselines (accept the terms with the `HF_TOKEN` account first). |
| OralGPT-Plus-7B | Gated, licence "other". Its zoom/mirror tool loop is in a separate code folder; plain generation would test it without its tools. | Optional, not wired. |
| Judges | The benchmark authors' EvalKit config uses `gpt-5-mini`; OralGPT-Plus's own scorer defaults to `gpt-5-nano`; the original paper used GPT-4-turbo. OralGPT-Omni scores 45.31 on MMOral-OPG in its own paper and 59.06 in OPGAgent's table. | The judge is a setting, stored in every judged row. Copied numbers are never mixed with measured ones. |

## The systems (paper table rows)

| Row | Image | Prompt | Answering model |
|---|---|---|---|
| Qwen3.5-9B raw | native | official prompt | base model, no adapter |
| GRPO + harness, agent-assisted (**Option A**) | native | `findings_ctx_v1` block + the official prompt, unchanged | **the same base model** |
| OralGPT-Omni-7B, OralGPT-OPG-7B | native | official prompt | their own weights |

The agent runs once per image (phase `policy`): native system prompt, 8 tools, the harness mode of the checkpoint, canonical views per the
hardware profile, greedy, 50 tool calls, 16384 new tokens (the frozen protocol; `scripts/evaluate_models.py` functions, not a copy). Its findings
are cached in `policy/<name>/findings.jsonl`. The agent never sees a question: it was trained on one answer format (findings in FDI notation for four
classes), so it is used as a perception front end. Because Option A and the raw row share the answering model and prompt, their paired
difference is what the findings add. The `findings_ctx_v1` block states that the agent reports four conditions only and that the absence of a
finding rules nothing out; an episode without a valid final answer is reported to the answering model as such, not as "healthy".

Precision. The policy, the raw model and the Option A answerer run under the hardware profile's precision and runtime flags (`quadro_rtx8000`:
fp16, SDPA; `workstation48_lora`: bf16). The public baselines use `EXTERNAL_PRECISION` (the profile precision by default and only that flag of the profile). The OralGPT releases
are bf16 checkpoints; on a card without native bf16 the profile precision is fp16, which can overflow in some vision-language models. If answers come back
empty or garbled (the diagnostics table shows it), the explicit alternative is `EXTERNAL_PRECISION = "fp32"` (about 32 GB of weights for an 8B model). The policy
checkpoint must have been trained under the same profile: its name carries the precision and the profile tag, and no other is substituted.

Option B (the agent answers in free text under a new prompt) is not built: the adapter was never trained on that format. It is the fallback if
Option A is low; it would be a different system with its own frozen prompt.

## Scoring: what is official and what is not

Ported verbatim from the benchmark authors' EvalKit (Apache-2.0, `MMOral-Bench-EvalKit/vlmeval/dataset/mmoral.py`, `utils/mmoral.py`): the closed and
open answer prompts, the closed-ended option extraction (`get_single_choice_prediction`), the open-ended judge prompt (8 worked examples, text only),
the 5-try rule (temperature 0, 0.5, 1.0, ...; no valid score after 5 tries = 0.0) and the category aggregation. The judge prompt is pinned by a
SHA-256 in `tests/test_external_benchmarks.py` (verified byte for byte against the official function).

Deliberate differences, all visible in the results:

1. **Unreadable closed answers.** The official extractor returns a random option. Here that choice is seeded per question (reproducible), and a
   `[strict]` score (unreadable = wrong) and the unreadable count are reported next to it. A quirk of the official extractor, reproduced: a letter
   followed by a period or comma ("C. is right", "D, because") is a candidate whose position is never found, so it is treated as unreadable.
2. **Category columns count each question once.** The official scripts add a question whose category is a single label (for example `Jaw`) to the
   main-category counter three times (closed) or twice (open), because the main category, the exact category and its comma-free copy share one key.
   Overall and the exact-category tables are not affected. Official-weighted values are kept in `mmoral_opg_scores.json` (`main_official_weighting`).
3. **`<think>` blocks** of a model are removed before extraction and judging, and counted (`thinking_stripped`, `unclosed_thinking`).
4. **Judge errors.** An API or transport error stops the run (it resumes); it is never scored 0.0.
5. **Intervals.** Every mean has a 95% percentile interval from resampling radiographs, because questions of one image are correlated. Paired
   comparisons use the same resampling.

## Commands

```bash
python scripts/run_external_benchmark.py prepare  --download
python scripts/run_external_benchmark.py leakage  --data-dir data
python scripts/run_external_benchmark.py policy   --policy-name P --adapter-path <grpo_policy> --harness-mode rules_skills_index \
       --max-new-tokens 16384 --no-canonical-letterbox <runtime flags of the profile>
python scripts/run_external_benchmark.py answer   --system-name qwen3_5_9b_raw --max-new-tokens 16384 <runtime flags>
python scripts/run_external_benchmark.py answer   --system-name optionA_P --findings-from P --max-new-tokens 16384 <runtime flags>
python scripts/run_external_benchmark.py answer   --system-name oralgpt_omni_7b --loader external --model-id OralGPT/OralGPT-Omni-7B-Instruct --max-new-tokens 16384
python scripts/run_external_benchmark.py judge    --system-name <s> --judge-model <name>
python scripts/run_external_benchmark.py score    --systems <s> ... --judge-model <name> --published published_numbers.json
python scripts/run_external_benchmark.py compare  --a optionA_P --b qwen3_5_9b_raw --judge-model <name>
python scripts/run_external_benchmark.py sync     --direction up --tag <profile tag>
```
The notebook builds exactly these, from visible settings. `--max-new-tokens` has no default; `policy` refuses to start unless the canonical view
(`--canonical-letterbox` / `--no-canonical-letterbox`) is stated, and it must be the one of the hardware profile.

## Files

```
data/benchmarks/mmoral_opg/raw/*.tsv                 the two Hub files (MD5 checked against the authors' values)
data/benchmarks/mmoral_opg/prepared/                 images/<sha16>.jpg, images.csv, closed.jsonl, open.jsonl, manifest.json
data/benchmarks/mmoral_opg/leakage_dentex.json       leakage screen against DENTEX combined train folders + validation
data/evaluations/external/mmoral_opg/<profile tag>/
    policy/<name>/findings.jsonl                     one record per image: findings, answered, trajectory, settings
    answers/<system>/{closed,open}.jsonl             one record per question: raw, prediction, flags, settings
    answers/<system>/model_info.json                 model id, requested revision, commit hash actually loaded
    judge/<system>__<judge>.jsonl                    score, log, judge identity per open question
    tables/mmoral_opg.md, mmoral_opg_scores.json, compare__*.json
    PROVENANCE.json                                  git commit, package versions, MD5s, checkpoint, settings, judges
```
Hub copy: `external/mmoral_opg/<profile tag>/` in the existing results repository (`sync`; never creates the repository, never overwrites a local file
on restore, no images).

## Resume and identity

Every record carries the settings that decide what it means (model, adapter and its file hashes, harness mode, token budget, prompt version,
findings file hash, benchmark MD5s; for judging also the judge model, endpoint, judge-prompt hash and temperature handling). A resume with different
settings stops; `overwrite` is explicit. Settings are verified before any 9B model is loaded.

## Reading the result

- *Option A minus raw* is the same-model, same-prompt, same-judge statement of what the agent adds. Report it with its interval.
- The public models were trained on MMOral-style data; the agent-assisted system was not. Say so next to their rows.
- Published numbers of other papers go into `published_numbers.json` with their `source` and `judge` (a row without a source is refused) and print as separate rows.
- Hypothesis, not a result: gains, if any, should concentrate in Teeth and Patho; past treatments, jaw and recommendation are outside the four classes the agent reports.
- The leakage screen compares against the combined DENTEX train folders (about 1,294 images: everything the grounder saw, a superset of the VLM's training images) and the 50 validation images. It finds copies, rescaled copies and re-encodings; it does not find crops, rotations or flips.

## Open items to settle with a smoke run (`SMOKE_LIMIT = 3`) before the full run

- Whether Qwen3.5-9B's chat template makes the raw model think by default. The repo never passes `enable_thinking`, and this adapter does not change shared
  generation code. If it thinks, answers can be long (up to the 16384-token budget, which is not lowered) and the full run takes correspondingly longer; the
  diagnostics table (mean new tokens, answers that hit the cap, unclosed `<think>` blocks) shows it after three questions. Options if that is too slow: accept it, or a small
  separate patch that passes a chat-template argument through `generate_agent_reply`.
- The real TSV schema (taken from the dataset card preview) and the image-reference convention; `prepare` stops with the column list if either differs.

## Verification status of this adapter

Verified in the development sandbox (no GPU, no torch, no Hub access to the benchmark): the official prompts and judge head against the official code,
extraction and scoring logic, preparation on synthetic TSVs of the documented schema, judge retry and resume rules, resume refusals, the leakage screen
on synthetic images, tables and bootstrap, the CLI end to end with the model, tools and judge stubbed, and every command the notebook cells build against
the real argument parser. Not verified until the first run on the server: loading the real TSVs (the schema is taken from the dataset card preview;
`prepare` stops with the column list if it differs), real model loading of the OralGPT releases, generation speed and memory (nothing here is a measured time or
memory figure), and the judge endpoint.
