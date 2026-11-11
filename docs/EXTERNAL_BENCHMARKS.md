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

| Row | Image | Prompt | Answering model | Hardware |
|---|---|---|---|---|
| Qwen3.5-9B raw | canonical FULL view (`--image-view canonical`) | official prompt | base model, no adapter | GPU or TPU |
| GRPO + harness, agent-assisted (**Option A**) | canonical FULL view | `findings_ctx_v1` block + the official prompt, unchanged | **the same base model** | GPU or TPU |
| OralGPT-Omni-7B, OralGPT-OPG-7B | native (`--image-view native`: their own preprocessing) | official prompt | their own weights | **CUDA only** |

The canonical FULL view is `to_canonical(image, "FULL")`, the call the agent itself makes for its first view (`run_agent`, `run_agent_no_tools`), so the
raw model, the Option A answerer and the agent see the same picture, and it is the evaluation protocol's primary view. On a GPU the profile's letterbox
setting applies (no bars on `quadro_rtx8000`/`workstation48_lora`, with the check `check_gpu_image_policy` that the other scripts run); on a TPU it is the
padded 1536x768 canvas.

The agent runs once per image (phase `policy`): native system prompt, 8 tools, the harness mode of the checkpoint, canonical views per the
hardware profile, greedy, 50 tool calls, the profile's token budget (16384 on every GPU profile, 24576 on the TPU: `generation_budget` in `notebook_eval.py`; the frozen protocol; `scripts/evaluate_models.py` functions, not a copy). Its findings
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
python scripts/run_external_benchmark.py answer   --system-name qwen3_5_9b_raw --image-view canonical --no-canonical-letterbox --max-new-tokens 16384 <runtime flags>
python scripts/run_external_benchmark.py answer   --system-name optionA_P --findings-from P --image-view canonical --no-canonical-letterbox --max-new-tokens 16384 <runtime flags>
python scripts/run_external_benchmark.py answer   --system-name oralgpt_omni_7b --loader external --image-view native --model-id OralGPT/OralGPT-Omni-7B-Instruct --max-new-tokens 16384
python scripts/run_external_benchmark.py judge    --system-name <s> --judge-model <name>
python scripts/run_external_benchmark.py score    --systems <s> ... --judge-model <name> --published published_numbers.json
python scripts/run_external_benchmark.py compare  --a optionA_P --b qwen3_5_9b_raw --judge-model <name>
python scripts/run_external_benchmark.py sync     --direction up --tag <profile tag>
```
The notebook builds exactly these, from visible settings (`<runtime flags>` come from `runtime_flags(profile)`; on the TPU they include the TPU generation flags and the budget is 24576). `--max-new-tokens` and `--image-view` have no default; `policy` refuses to start unless the canonical view
(`--canonical-letterbox` / `--no-canonical-letterbox`) is stated, and it must be the one of the hardware profile.

## TPU (Cloud TPU v5e-8)

Every phase except the public OralGPT baselines runs on the TPU, through the repository's own evaluator functions in the evaluator's order
(`docs/TPU_EVALUATION.md`); the notebook selects the `tpu` profile when a TPU is detected (without starting XLA, as the Final_Evaluation notebook does).

| Requirement of `docs/TPU_EVALUATION.md` | How the adapter meets it |
|---|---|
| Canonical letterbox views only (static vision slots: FULL 1536x768, CROP 256x384, COMPARE 512x384) | `policy` and `answer --image-view canonical` call `require_static_views` before any weight is read; the answering model gets `to_canonical(image, "FULL")`. `--image-view native` is refused on a TPU |
| SPMD enabled before the XLA device exists; weights sharded while moving | `setup_eval_hardware(..., spmd=True)` and `load_eval_model(...)` are the evaluator's own, called with the evaluator's TPU keywords |
| `max_new_tokens` must fit a cache tier and is never lowered | passed to `load_eval_model`, which checks and warms every reachable tier. The budget is `generation_budget(profile)`: 24576 on the TPU (tiers 32768/49152 hold it) |
| Flags `--tpu-generation-chunk-tokens 512 --tpu-generation-tiers 32768 49152 --tpu-vision-slots 8 16 6` | taken from `runtime_flags('tpu')`; stored in every result's settings (`tpu_generation`) |
| One process per sharded model | each phase and each system is its own child process |
| Refusals are recorded, never silent | a prompt beyond the tiers or the slots (`GenerationTooLong`, `SlotOverflow`) is stored with `generation_refused`, counted, and listed in `tpu_generation_report.json` next to the results |

Refused items are scored **wrong**: a refused closed-ended question is not given the official extractor's random option, and a refused open-ended
question gets 0.0 without a judge call (its log says so). A refused agent episode is stored as no answer, and Option A tells the answering model "no valid
result", not "healthy". Resume comparison is symmetric in the settings, so a TPU result file is never resumed by a GPU run or the other way round; TPU and
GPU results are different cohorts (`image_view`, `canonical_letterbox`, profile tag `tpu_bf16`).

Not on the TPU: OralGPT-Omni-7B / OralGPT-OPG-7B. The static generator exists for the hybrid Qwen3.5 stack (linear-attention state, Qwen3.5 M-RoPE positions,
Qwen3.5 vision patch layout); the OralGPT releases are Qwen2.5-VL-based. `answer --loader external` refuses on a TPU, before any weight loads. To include them in
a TPU table, run the notebook on a CUDA machine with `RUN_POLICY = RUN_RAW_BASELINE = RUN_OPTION_A = False`, then copy `answers/<name>/` and
`judge/<name>__<judge>.jsonl` into the TPU results folder (or `sync` through the Hub). The records keep their own settings (hardware precision, model commit),
so the table shows exactly under which conditions each row was measured; they are measured with the same judge.

## Files

```
data/benchmarks/mmoral_opg/raw/*.tsv                 the two Hub files (MD5 checked against the authors' values)
data/benchmarks/mmoral_opg/prepared/                 images/<sha16>.jpg, images.csv, closed.jsonl, open.jsonl, manifest.json
data/benchmarks/mmoral_opg/leakage_dentex.json       leakage screen against DENTEX combined train folders + validation
data/evaluations/external/mmoral_opg/<profile tag>/
    policy/<name>/findings.jsonl                     one record per image: findings, answered, trajectory, settings
    answers/<system>/{closed,open}.jsonl             one record per question: raw, prediction, flags, settings
    answers/<system>/model_info.json                 model id, requested revision, commit hash actually loaded, hardware
    policy/<name>/tpu_generation_report.json, answers/<system>/tpu_generation_report.json   (TPU) calls, tiers used, refused items
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

## Running diagnostics (no separate trial run)

Every answer phase prints, every 25 answers: mean new tokens, answers that hit the token cap, unclosed `<think>` blocks, empty answers and refusals. If the raw
Qwen3.5 thinks by default (the repository never passes `enable_thinking`, and this adapter does not change shared generation code), the numbers show it in the
first minutes of the full run: long answers up to the budget, which is not lowered. The `<think>` block is removed before scoring and counted. The notebook's
Phase 6 prints the same per system, plus the TPU generation reports. The real TSV schema (taken from the dataset card preview) is checked by `prepare` itself,
which stops with the column list if it differs.

## Verification status of this adapter

Verified in the development sandbox (no GPU, no TPU, no torch, no Hub access to the benchmark): the official prompts and judge head against the official code,
extraction and scoring logic, preparation on synthetic TSVs of the documented schema, judge retry and resume rules, resume refusals (including TPU against GPU
files), the leakage screen on synthetic images, tables and bootstrap, the CLI end to end with the model, tools, TPU generator and judge stubbed (call order of the
TPU contract, SPMD flag, `max_new_tokens` at load, refusal records scored wrong, generation report path, external models refused, canonical FULL canvas size),
and every command the notebook cells build, for the `quadro_rtx8000`, `workstation48_lora` and `tpu` profiles, against the real argument parser.

Not verifiable here, and so not verified: anything that needs the real XLA runtime (sharding the real 9B from host RAM, the warm-up compile of the cache tiers,
memory on a 16 GiB chip, TPU-only lowering, decode throughput; `docs/TPU_EVALUATION.md` lists the same items for the evaluators), loading the real TSVs, real model
loading of the OralGPT releases, generation speed and memory (nothing here is a measured time or memory figure), and the judge endpoint. The TPU path reuses the
evaluator's own functions and order instead of re-implementing them, so what holds for `evaluate_models.py` on your TPU holds for the load and generation here; the
adapter's own additions are the per-question answer loop and the refusal scoring above.
