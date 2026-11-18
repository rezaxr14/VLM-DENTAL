# Kaggle Benchmarks: a field guide (what we measured, how to use it, how to keep the results)

Written 2026-11-18 from nine probe notebooks (`kaggle_probe.ipynb`, `v5` to `v9`, 2026-11-12 to 2026-11-14) and the working
implementation in `dental_agent/external/`. **Everything marked "measured" was observed on our account on the date given; Kaggle
changes this surface often, so re-check with the cheap commands in section 12 before relying on a number.** Anything we did not
observe is listed as open in section 11, never assumed.

Companion document: `docs/KAGGLE_EXTERNAL_MODELS.md` (the VLM-DENTAL specifics: conditions, identity, commands).
This guide is written so that it can be reused for another project: only section 9 is VLM-DENTAL specific.

---

## 1. What Kaggle Benchmarks gives you

A way to call many frontier models (Claude, Gemini, GPT, Grok, Gemma, ...) **billed to a Kaggle quota instead of per-vendor keys**,
either from your own notebook or from server-side tasks:

| Route | Where your code runs | Which models it reaches | Typical use |
|---|---|---|---|
| **Interactive proxy** | your Kaggle/Colab/local session, calling the Model Proxy over HTTPS | a **curated subset** (our key: 6 vision models + 2 text-only) | quick experiments, probes, one-off calls |
| **Hosted task** (`kaggle benchmarks tasks push` + `tasks run`) | a Kaggle worker, server-side; you can close the page | the **full catalogue** (37 listed, 34 read images) | real evaluations over many models |

The two routes use the same SDK (`kaggle-benchmarks`) and CLI (`kaggle`, `kaggle benchmarks ...`) but **different credentials and
different model access**. The interactive key's roster is not the hosted catalogue.

Mental model of a hosted run:

```
your machine/notebook                       Kaggle
  kaggle benchmarks tasks push  -----------> task source (a .py with @kbench.task) + attached datasets
  kaggle benchmarks tasks run -m <model> --> one run = (task version, model)  -> a worker starts
                                               worker: your code gets `llm` (a proxy client for THAT model)
                                               worker: writes files to /kaggle/working
  kaggle benchmarks tasks status ----------> queued / running / completed / failed per model
  kaggle benchmarks tasks download --------> <out>/<task>/<version>/<model>/<run_id>/<files>
```

## 2. Setup and credentials

* Install: `pip install "kaggle>=2.2.5" "kaggle-benchmarks>=0.5.0"` (we ran CLI 2.2.5 and kaggle-benchmarks 0.6.3). Pin after your own spike.
* Interactive session: `kaggle benchmarks init -y --env-file F` (or `auth -y`) writes `MODEL_PROXY_URL`, `MODEL_PROXY_API_KEY`
  (about **1 h** expiry, renew it), `LLM_DEFAULT`, `LLMS_AVAILABLE`. A plain session has `kbench.llms == {}` unless `init` ran.
* Hosted run: the platform injects the proxy variables itself; the key lasts about **12 h**; runs of **at least 150 minutes** are allowed.
* **Kaggle Secrets do not work inside hosted runs** (measured, probe v9). Put your own tokens (`HF_TOKEN`, repo names) in a **private Kaggle
  dataset** holding a file named `env` or `.env` and attach it to the task; read it from `/kaggle/input/...` at start. Dataset uploads can
  drop dotfiles, so a plain name works. Never print the values, never write them into the task source.
* Never let that file override the runner's own wiring (`MODEL_PROXY*`, `KAGGLE_*`, `LLM*`); our loader skips those prefixes.

## 3. Quota and money

* `kaggle benchmarks quota --format json` gives Daily and Monthly allowances (ours: Daily $10, Monthly $100) with `remaining`, `used`,
  `total`, `refillAt`. Quota moves in step with the summed response costs (measured).
* Cost per request is in the response: `usage.cost.{input,output}_tokens_cost_nanodollars` plus `total_backend_latency_ms`.
  `completion_tokens` **includes hidden reasoning tokens**, so `usage.cost` is the true spend. A bare number or an unknown unit is stored as
  unknown (`None`), never summed as 0.
* The history is re-sent every turn, images included, so a multi-turn tool conversation grows roughly quadratically in input tokens.
  Worst case for 250 images x 65 turns is large; estimate before spending (`scripts/run_kaggle_external.py estimate`).
* Hitting the daily cap or an expired key is **not an error to record**: pause and resume later (section 8).

## 4. Talking to a model

### 4.1 Through the proxy (what the adapter uses)

`POST {MODEL_PROXY_URL}/openapi/chat/completions` (OpenAI-compatible). In a hosted run `MODEL_PROXY_URL` already ends in `/openapi`.
Measured to work: system message; base64 `image_url` parts; multi-turn history with a later image; `max_tokens` (and for some models
`max_completion_tokens`) enforced with `finish_reason: length`; 16384 output tokens; 5 MB images; **16 parallel requests** with no
rate-limit headers. `/genai/v1beta/models/{id}:generateContent` also works for Gemini.

Per-model quirks (measured; our adapter learns the token parameter and temperature policy from the provider's HTTP 400 and records it per request):

| Quirk | Models seen |
|---|---|
| rejects `temperature` (decoding is the provider default, not greedy) | `claude-sonnet-5` |
| not repeatable at temperature 0 (hidden thinking) | `gemini-3.6-flash` (the two flash-lite models were repeatable) |
| bills hidden reasoning by default; `thinking_budget: 0` or `reasoning_effort: none` switches it off on 3.6 Flash | `gemini-3-flash-preview`, `gemini-3.6-flash` |
| accepts `temperature` only while `reasoning_effort` is unset | `gpt-5.4-nano` |
| image tokens: Gemini ~1.1k at any size; GPT/Claude scale with pixels (2048x1024: 2466 / 2753) | `detail` and `media_resolution` were ignored |
| prompt caching: no discount observed | all |

### 4.2 Through the SDK

`import kaggle_benchmarks as kbench`; `@kbench.task(name=...)` defines a task (a return annotation is required if it returns a value);
`llm.prompt(text, image=..., temperature=...)`; images via `kaggle_benchmarks.content_types.images` (`from_path`, `from_url`,
`from_base64`); `chats.new(name, system_instructions=...)`; build a model by hand with `ModelProxy(model=, api_key=, base_url=)`.
The SDK itself disables `temperature` on the OpenAI route. We bypass the SDK's chat object and call the proxy directly (section 9), because
the project's agent loop needs full control of the transcript.

## 5. The model roster

Two different lists; do not confuse them.

* `kaggle benchmarks tasks models` lists **37** slugs (bare ids such as `claude-haiku-5-5`, `gpt-6-sol`).
* **Interactive key (probe v7):** only `anthropic/claude-sonnet-5@default`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`,
  `gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gpt-5.4-nano-2026-03-17` read images; `gpt-oss-120b` and `ibm/granite-4.0-h-small`
  are text-only; everything else answered "model not found" under every slug form.
* **Hosted runs (probe v9):** each model scheduled on its own and asked to read a two-colour image: **34 read images**,
  `gpt-6-luna` answered wrongly (unreliable), `gpt-oss-120b` is text-only, `grok-4.5-0708` is listed but unserved (404).

Inside a hosted run the slug the proxy accepts is provider-prefixed (`llm.model`, e.g. `anthropic/claude-haiku-5-5`); `tasks run -m`
takes the bare CLI slug. Both are in `HOSTED_CATALOG` (`dental_agent/external/model_profiles.py`) with indicative prices
(USD per million tokens, from one tiny request's `usage.cost`, so treat them as order-of-magnitude):

| Provider | Models (CLI slugs, abridged) | Price range in/out |
|---|---|---|
| Anthropic (11) | `claude-haiku-4-5-20251001`, `claude-haiku-5-5`, `claude-sonnet-4-6/5/5-5-default`, `claude-opus-4-5-20251101` to `claude-opus-5-5-default` | haiku-5-5 0.1/0.5 up to opus 5.0/25 |
| Google (11) | `gemini-2.5-flash/pro`, `gemini-3-flash-preview`, `gemini-3.1-flash-lite/pro-preview`, `gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gemini-3.8-flash`, `gemini-4-argon`, `gemma-4-26b-a4b-it`, `gemma-4-31b-it` | 0.15/0.6 to 2.0/12 |
| OpenAI (12) | `gpt-5.4`, `-mini`, `-nano`, `gpt-5.5`, `gpt-5.6-luna/sol/terra`, `gpt-6-astra/luna/sol`, `gpt-6.1-sol`, `gpt-oss-120b` | 0.1/0.5 to 10/50 |
| xAI (3) | `grok-4.5-0708` (unserved), `grok-4.6`, `grok-4.7` | 1.3/6 to 2/6 |

The catalogue changes. Refresh it with `kaggle benchmarks tasks models` and re-run a tiny vision probe (section 12) before trusting a new slug.

## 6. The hosted worker environment (measured, probe v9)

* Python **3.12.10** (interactive notebooks may be 3.13: anything hashing `ast.dump` text differs between them; we made that check version independent).
* **4 CPU, no GPU, no torch.** Internet works for pip, git and Hugging Face (other sites are usually blocked).
* Attached datasets mount read-only at `/kaggle/input/<name>` (interactive notebooks: `/kaggle/input/datasets/<owner>/<name>`);
  read the mount path back instead of typing dataset ids.
* Files written to `/kaggle/working` are what `tasks download` returns. Keep them small and write a **manifest** (status, stage, error) so
  a failure is diagnosable without reading logs.
* A `tasks push` also **starts one run on the account's default model** (`gemini-3.8-flash`). Adopt that run instead of scheduling it again.
* `tasks run` takes **one model per call**; several `-m` flags return HTTP 400.
* The worker can `git clone` your repo (we clone GitHub's default branch, unpinned: **push first**) and `pip install`. A bare worker needs
  everything the import chain touches (our project's package `__init__` imports torch, langgraph, scikit-learn, tqdm, huggingface_hub...), so
  list requirements explicitly and test that list against the real import chain.
* Tool conditions that need a detector (here YOLO via ultralytics) must install it and fetch weights on the worker; that part was verified
  only by the in-notebook smoke test, not by a probe.

## 7. Anatomy of a hosted task

```python
import kaggle_benchmarks as kbench

SPEC = {...}                     # everything the shard needs, embedded as JSON in the source

@kbench.task(name="my-task-name", description="...")
def shard(llm) -> None:          # llm = the proxy client for the model this run was scheduled with
    ...                          # clone/pip, run the loop with llm.model as the slug, write /kaggle/working/*
shard.run(kbench.llm)
```

* `tasks push NAME -f source.py -d owner/dataset ... --wait 60` creates or versions the task and attaches datasets.
* `tasks run NAME -m <cli-slug>` schedules one run. `tasks status NAME` shows one row per model: queued/running/completed/failed.
* `tasks download NAME -o OUT -m <cli-slug> -s -f` gives `OUT/NAME/<version>/<model>/<run_id>/<files>`.
* One task = one unit of parallelism. We split work into **shards of 25 images** so one failure or one daily cap costs 25 images, not all of them.
* Task names are limited to a safe character set; ours are `vlmdental-<cohort6>-<cond>-sNN`. A name is the identity of a task, so changing
  the cohort or settings should change the name.

## 8. Pause, resume and "never pay twice"

Design rules that held up:

1. **Classify failures.** `auth` (expired key) and `quota` (cap reached) **pause** the run and record **nothing** for the image in flight;
   `rate_limit` is retried after a back-off (30/60/120 s) and then pauses; anything else is stored as evidence and stops a cell after three in a row.
2. **Journal every answered turn** the moment it arrives. A resumed image replays its answered turns for free and only the failed turn is sent
   again; if the request would differ from what was journalled, fail closed (`HistoryMismatch`) instead of mixing.
3. **Identity on every record.** Provider, model, condition, prompts (hashes), decoding, tool pack, harness, cohort and (for the 250-image
   cohort) the COCO and mapping hashes. Resume refuses a record with a different identity and names the keys that differ.
4. **Persist the driver state** (`campaign.json`: which task/model runs exist and their status) to the Hub as well as locally, so a new
   session adopts it instead of pushing or scheduling anything twice. Several Kaggle accounts can contribute (`ACCOUNT_LABEL`).
5. **Remainders.** Images a model still lacks become small remainder tasks for that model only.
6. **Smoke test first.** Run one cheap model on two images through every condition as real hosted tasks and refuse to start the campaign until it passes.

## 9. Saving results and capturing findings

Two record layers, so ground truth never travels to Kaggle:

* **Transport record** (written by the worker): answers, raw turns, tool calls, usage per request, identity. No annotations, no metrics.
* **Evaluated record** (built locally): the exact schema `scripts/evaluate_models.evaluate_condition` writes, so the project's
  `matrix.summarize_cell`, `paired_bootstrap_diff` and `build_summary` read it unchanged. An extra `external` block carries provider, usage and error.
* On the 250-image cohort the evaluated record also carries `scoring_profile`, `n_renamed_predictions` and `family_exact_*`, and the cohort
  prompts, scoring and COCO/mapping identity are the ones the trained checkpoint is evaluated with.

Where things go (existing private Hugging Face repos only; nothing is ever created):

| What | Where |
|---|---|
| results bundles (atomic, identity-checked, per shard) | `HF_EVAL_REPO` (default `Reza-Nadimi/vlm-dental-evaluations`) under `external/...` |
| campaign state | the same repo under `external/campaign/...` |
| hindsight rows (one per episode: trajectory, tools, harness events, answer, ground truth, usage, `outcome` success/partial/failure/format_failure/error) | `HF_TRACES_REPO` (`Reza-Nadimi/vlm-dental-traces`), only when asked |
| raw downloads, report, comparison tables | local `results/kaggle/hosted/` or `/kaggle/working/kaggle_hosted/` (`report.md`, `comparison.md`) |

Capturing findings: keep **the capture format identical for probes and real runs** (probe v6 already wrote every request/response, images
redacted, to a JSONL file that became the hindsight format). Store what the provider returned verbatim (finish reason, usage, latency, errors),
store unknowns as `None`, and never delete a failed attempt: it is evidence.

## 10. How to run our stuff on it (recipes)

**A. One model, interactive (cheap check).**
`python scripts/run_kaggle_external.py run --model gemini-3.5-flash-lite --condition kaggle_zero_shot --split official_test250 --confirm-spend --max-requests 300`

**B. The whole catalogue, hosted (the main route).** Open `notebooks/VLM_Dental_Kaggle_Frontier_Benchmarks.ipynb`, attach the private `.env`
dataset and the cohort dataset, set the settings cell (`SPLIT`, `CONDITIONS`, `MODELS`), and Run All. Re-run the campaign cell after a pause.
Command-line equivalent: `scripts/run_kaggle_external.py hosted push|status|collect|evaluate --split official_test250 ...`.

**C. A new task (not dental).** Copy `sharding.TASK_TEMPLATE`: keep the `SPEC` embedding, the bootstrap, the manifest-on-failure, and a
`run_shard` that writes only transport records. Reuse `kaggle_cli.KaggleCLI`, `pause.QuotaGuard`, `journal.TurnJournal`, `persist.ExternalHub`.

**D. A new condition / tool set / scorer / plugin.** `conditions.register_condition`, `tool_packs.register_pack`, `evaluators.register_evaluator`,
and list the module in `PLUGINS` so the driver and every worker import it. Examples we planned: segmentation masks, step-by-step image editing
(zoom, contrast, filters, kernels, re-look), hindsight retrieval-augmented prompting.

**E. Compare with the trained checkpoint.** Put the checkpoint's evaluated records in `EXTRA_EVALUATED`; the tables use the paired bootstrap, so a
harness effect (`tools vs zero-shot`, `rules+skills vs tools`) comes with a confidence interval.

## 11. Open questions (not verified; do not assume)

* The tool conditions on a hosted worker end to end (torch + ultralytics install and the YOLO weights): covered only by the notebook smoke test.
* Claude determinism, cross-run variance of thinking models, prompt caching discounts.
* Whether the hosted catalogue or prices have changed since 2026-11-14.
* Free-tier limits for scheduled runs and how the daily cap behaves across several accounts.
* Exact SDK details we avoided relying on (`chats.new` internals, assistant-message injection, any max-token option on the SDK path).

## 12. Cheap checks before you spend

```
kaggle --version && kaggle benchmarks --help           # CLI present
kaggle benchmarks quota --format json                  # allowance left, refill time
kaggle benchmarks tasks models                         # the listed slugs (changes over time)
python scripts/run_kaggle_external.py roster --probe   # one tiny request per model: who actually answers, who reads images
python scripts/run_kaggle_external.py estimate --n-images 250 --model <slug>
```

## 13. Pitfalls we hit (so you do not)

* Thinking that the interactive roster equals the hosted catalogue.
* Batch `tasks run -m a -m b ...` (HTTP 400): one model per call.
* Expecting a `tasks push` not to start a run: it starts one on the default model.
* Relying on Kaggle Secrets in hosted runs: use the private `.env` dataset.
* An unpinned clone: the worker runs GitHub's default branch, not your local checkout. Push first.
* A bare worker failing to import the project: list real requirements, write an import-failure manifest.
* Python-version drift (3.12 hosted vs 3.13 interactive) breaking any hash of `ast.dump` text.
* A shared scratch folder for several models' bundles (bundles conflicted): key it by the settings fingerprint.
* Treating quota or an expired key as a model failure: pause, do not record.
* Running a smoke test again in every new session: build or restore the campaign first and skip the smoke test if it already ran.

## 14. Probe history (what each one settled)

| Probe | Date | Settled |
|---|---|---|
| `kaggle_probe` | 2026-11-12 | SDK basics, raw proxy reachability (cap: 40 requests) |
| v5 | 2026-11-14 | credentials flow, `/openapi/chat/completions`, system + image + history, usage/cost fields, quota (cap: 30 requests) |
| v6 | not recorded | capability matrix: token parameter and temperature policy per model, vision, multi-turn with a later image, big images, concurrency, real prices; captured every request in the hindsight format |
| v7 | 2026-11-14 | full roster (37 listed vs served), image-token fidelity, thinking/reasoning knobs |
| v8 | not recorded | hosted (non-live) runs, full catalogue; batch `-m` of 37 models fails with 400 |
| v9 | 2026-11-14 | one model at a time works; worker environment (Python 3.12, 4 CPU, no GPU/torch, pip/git/HF reachable, dataset mount, Secrets do not work, >=150 min runs, 12 h key); 34 vision-capable models |
