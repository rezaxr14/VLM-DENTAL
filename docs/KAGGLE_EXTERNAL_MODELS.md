# Kaggle Benchmarks as an external model provider

Hosted models reached through Kaggle are evaluated with the project's OWN agent loop, tools, harness and clinical
evaluator, so a comparison with the VLM-DENTAL agent changes only the model.

## Conditions (`dental_agent/external/conditions.py`)

| condition | tools | harness | prompt |
|---|---|---|---|
| `kaggle_zero_shot` | none | off | `NO_TOOLS_SYSTEM_PROMPT` + `FIRST_USER_PROMPT` (one turn) |
| `kaggle_tools` | all 8 | off | unified agent prompt |
| `kaggle_tools_rules_skills` | all 8 | `rules_skills_index` | unified agent prompt + rules + skill index |
| `kaggle_zero_shot_legacy_prompt` (optional) | none | off | `ZERO_SHOT_PROMPT` (Baseline #1 prompt) |

Always: canonical views without letterbox, greedy decoding, 50 tool calls (65 turns), 16384 tokens per turn,
`malformed_retries=0`, fail-fast with no automatic retry.

## How it fits

* `run_agent_no_tools(..., reply_fn=None)` (additive; the default path is unchanged; it coexists with the cohort's `system_prompt`) and the
  version-independent rule-hash check in `harness/rules.py` are the only edits to existing project code besides `config.py` (env-dataset
  loading). `agent/` and `harness/` are hashed by `NotebookEvaluations._source_identity`, so **land them in the same batch as any other
  edit in the hashed folders** to change bundle identities once.
* `KaggleReply` is a `reply_fn`: returns `(text, prompt_tokens, [], None, None)`; token ids are never invented.
* `TranscriptSync` sends only the new user messages each turn and fails closed (`HistoryMismatch`) if the loop's transcript
  no longer extends what the remote chat holds. One chat per image.
* Two record layers: the *transport* record (answers + usage, no annotations) and the *evaluated* record built locally with
  the exact `evaluate_condition` schema, so `matrix.summarize_cell`, `paired_bootstrap_diff` and `build_summary` work unchanged.
* Missing usage / cost stays `None` and is never summed as 0.

## Commands (`scripts/run_kaggle_external.py`)

```
estimate   --n-images 250 [--expected-turns 12 --price-in P --price-out P]   # worst case 250 x 65 requests per tools condition
preflight  --split official_test250 [--ids-file subset.txt] [--yolo-train-ids ids.txt]
roster     [--listing file] [--probe]                                         # --probe spends one tiny request per model
run        --model SLUG --condition kaggle_tools --confirm-spend --max-requests N
import     --model SLUG --condition kaggle_tools
table      --cell "SLUG/kaggle_zero_shot=...jsonl" ... --model SLUG
```

`run` refuses without `--confirm-spend` and a request budget.

## Gates before any real run

* **G1 SDK/proxy spike (done, probe v5, 2026-11-14, kaggle-benchmarks 0.6.3, CLI 2.2.5).** Credentials: `kaggle benchmarks init -y
  --env-file F` (or `auth -y`) writes `MODEL_PROXY_URL`, `MODEL_PROXY_API_KEY` (about 1 h expiry), `LLM_DEFAULT`, `LLMS_AVAILABLE`.
  Models: `kaggle benchmarks tasks models` (37 slugs, bare ids). Quota: `kaggle benchmarks quota --format json` (Daily $10,
  Monthly $100). Verified on `{MODEL_PROXY_URL}/openapi/chat/completions` for Gemini and GPT: system message, base64
  `image_url` parts, history re-sent every turn (image re-billed each turn), `max_tokens` enforced (`finish_reason: length`),
  `temperature` accepted, usage `prompt_tokens` / `completion_tokens` and `usage.cost.{input,output}_tokens_cost_nanodollars`
  plus `total_backend_latency_ms`. `/genai/v1beta/models/{id}:generateContent` also works for Gemini. SDK: a plain session has
  `kbench.llms == {}` unless `init` ran; build models with `ModelProxy(model=, api_key=, base_url=)`; `chats.new(name,
  system_instructions=)`; images via `kaggle_benchmarks.content_types.images`. The SDK itself disables `temperature` on the
  OpenAI route, so `send_temperature` is configurable and recorded per turn.
  **Roster (probe v7).** `tasks models` lists 37 models but this key's proxy serves only: `anthropic/claude-sonnet-5@default`,
  `gemini-3-flash-preview`, `gemini-3.1-flash-lite-preview`, `gemini-3.5-flash-lite`, `gemini-3.6-flash`, `gpt-5.4-nano-2026-03-17`
  (all vision), plus text-only `gpt-oss-120b` and `ibm/granite-4.0-h-small`. Everything else (other Claude, GPT-5.x/6.x, Grok, Gemma,
  Gemini 2.5/3.1-pro, DeepSeek, Qwen) answered "model not found" under every slug form, so the roster comes from
  `dental_agent/external/model_profiles.py` (measured settings and prices), not from the listing. Whether hosted task runs
  (`kaggle b t run`) reach the other models is NOT verified.
  **Measured behaviour.** `claude-sonnet-5` rejects `temperature` (not greedy; stored in the run identity as `decoding`).
  Gemini 3.6 Flash is not repeatable at temperature 0 (two flash-lite models were); Gemini 3 Flash / 3.6 Flash bill hidden reasoning
  (`completion_tokens` includes it; `thinking_budget: 0` or `reasoning_effort: none` turns it off on 3.6 Flash). Image handling
  differs: Gemini ~1.1k tokens at any size, GPT/Claude scale with pixels (2048x1024: 2466 / 2753); `detail` and `media_resolution`
  were ignored. gpt-5.4-nano accepts `temperature` only while `reasoning_effort` is unset. 16384 output tokens, 5 MB images,
  a later image in a multi-turn chat, and 16 parallel requests all worked with no rate-limit headers. Quota moves in step with the
  summed response costs. The proxy key lives about one hour; `kaggle_cli_refresher` renews it before expiry.
  STILL OPEN: hosted-task roster, Claude determinism, cross-run variance of thinking models, prompt caching (no discount seen).
* **G2 hosted-environment probe (done, probe v9, 2026-11-14).** Hosted runs: Python 3.12.10, 4 CPU, no GPU, **no torch**; pip, git
  and Hugging Face are reachable; an attached dataset mounts at `/kaggle/input/<name>` (interactive notebooks:
  `/kaggle/input/datasets/<owner>/<name>`); **Kaggle Secrets do not work** in hosted runs (the `.env` dataset replaces them);
  `MODEL_PROXY_URL` ends in `/openapi`; the hosted key lasts about 12 h; runs of at least 150 minutes are allowed. A `tasks push`
  also starts one run on the account's default model (gemini-3.8-flash); the driver adopts it instead of scheduling it twice.
  `tasks run` with several `-m` flags returns 400, one model at a time works for every listed model. Of the 37 listed models 34
  read images (`gpt-6-luna` unreliable, `gpt-oss-120b` text only, `grok-4.5-0708` unserved); see `HOSTED_CATALOG`.
* **G2b gating smoke test (in the notebook, section 10).** The one thing probe v9 did not verify is the tool conditions on a
  worker (torch + ultralytics install and the YOLO weights from the private models repo). Instead of another throwaway probe, the
  notebook runs one cheap model on two images through every condition as real hosted tasks and refuses to start the campaign until
  that passes.
* **G3** dev pilot to measure mean turns and tokens (replaces the `estimate` defaults).
* **G4** the 250-image test run.

## Known limits

* `locate_tooth` degrades silently without ultralytics or its checkpoint; `preflight` loads it and pins the SHA-256.
* The grounder must not have trained on the test images: pass `--yolo-train-ids` (the earlier exclusion covered only the
  50 validation images).
* The harness rule bindings used to pin `ast.dump` hashes, which differ between Python versions, so `render_prompt()` failed on
  3.13 (Kaggle's interactive notebooks) while the hosted runtime is 3.12. `harness/rules.py` now hashes a version-independent dump
  (`_stable_dump`, byte-identical to the 3.12 text), so the 16 hashes in `rules.md` are unchanged and saved checkpoint/trace
  identities (which hash `rules.md`) stay valid. `rules.md` must not be edited. This touches a hashed folder (`harness/`), so it
  changes `NotebookEvaluations._source_identity` once, like any other edit there: land it with the rest of the batch.
* No test-split MTP run of the final target exists yet; see the plan for the reporting options.
* Data is loaded through the project's own loaders (`external/cohort.load_split`: `notebook_eval.prepare_inputs` for the cohort, then
  `scripts.evaluate_models.load_eval_split`); `notebook_eval.py` is not touched.

## The 250-image cohort (`--split official_test250`, `SPLIT = "official_test250"`)

Built on `docs/OFFICIAL_TEST250.md`; `dental_agent/external/cohort.py` only chooses which project piece to call, nothing is re-implemented:

* **Data.** COCO and images read local disk, then attached Kaggle dataset, then Hugging Face cache, then `DENTEX_TEST_REPO` (`prepare_cohort`).
  Attach the dataset made by `scripts/stage_official_test250_dataset.py` so hosted workers can mount the images; `locate_image_datasets`
  reads the `owner/name` back from the mount path.
* **Prompts.** The unified agent prompt uses the `official_test250` dataset key, the no-tools condition passes `official_test250.no_tools_prompt()`
  as `system_prompt`, and the legacy supplement uses `zero_shot_prompt()`. Prompt hashes in the identity follow the split.
* **Scoring.** `records.evaluated_record` runs `test250_scoring.score_cohort` (canonical class names, `family_exact_*`) when the run's settings carry
  the cohort block, exactly as `evaluate_condition` does; `split` is `official_test250`, `dataset` stays `dentex`.
* **Identity.** The cohort settings (COCO/mapping versions and SHA256, scoring profile, prompt classes) sit under the `official_test250` key of
  `run_settings`, travel to the workers in `ShardSpec.cohort_settings`, and change `cohort_sha` (so a new COCO is a new set of task names).
  `run_settings`/`build_campaign` refuse a cohort run without them and any other split with them. Other splits keep their old fingerprints.
* **Qwen3.5-4B.** Untouched here: external models never load a local checkpoint; the 4B work only matters when the trained checkpoint's records
  are added through `EXTRA_EVALUATED`.

## Segmentation foundation

`dental_agent/segmentation/` (contract, masks, metrics; numpy/PIL/scipy only, not wired into the loop).

## Hosted (non-live) runs: the main route

`notebooks/VLM_Dental_Kaggle_Frontier_Benchmarks.ipynb` (and `scripts/run_kaggle_external.py hosted ...`) drive hosted runs over the
whole catalogue:

* **Shards.** Tasks are `(condition, 25-image slice)` named `vlmdental-<cohort6>-<cond>-sNN`; one hosted run per (task, model).
  A worker clones this repository (GitHub's default branch, **not pinned**; push your code first), installs what it needs,
  runs `run_cell` on its slice and returns transport records. Scoring is local; annotations never leave the machine.
* **Pause / resume.** `auth` and `quota` provider failures and `QuotaGuard` readings pause a run without recording an error;
  `rate_limit` is retried with back-off, then pauses; other errors are stored as evidence and stop a cell after three in a row.
  Every answered turn is journalled (`journal.py`), so a re-run replays them for free and a changed request fails closed.
  Images a model still lacks become remainder tasks for that model only. The driver state (`campaign.json`) is synced to the
  existing results repo, so a new session continues without pushing or scheduling anything twice.
* **Credentials.** Put your `.env` in a **private Kaggle dataset** (a file named `env` or `.env`; Kaggle's upload may drop
  dotfiles) and attach it to every notebook. `dental_agent.config.load_env` fills unset variables from any attached dataset that
  holds one (never `MODEL_PROXY*`, `KAGGLE_*`, `LLM*`); `build_campaign` attaches the same dataset to the hosted runs, which read it
  too. The values are never printed or written into task sources.
* **Images.** The image datasets are read back from where the loader found the images (`sharding.locate_image_datasets`);
  `image_dataset.stage_image_dataset` + `KaggleCLI.dataset_publish` stage an images-only private dataset when the cohort is not
  on Kaggle yet (the 50-image test set). The 250-image cohort has its own staging script (`scripts/stage_official_test250_dataset.py`).
* **Worker environment.** A bare worker must install torch, torchvision, langgraph, scikit-learn, tqdm and friends just to import
  the project (the scoring helpers live in `dental_agent.evaluation`, whose package `__init__` imports the whole project);
  `sharding.BASE_REQUIREMENTS` lists them and `test_worker_requirements_cover_everything_the_import_chain_needs` walks the real
  import chain, so it fails when the list falls behind. The tool conditions also install ultralytics and fetch `best.pt` through
  `utils.inputs.resolve_yolo_checkpoint` (needs `HF_TOKEN` from the env dataset, or a weights dataset in `extra_datasets`).
  A worker that cannot import writes a `shard_manifest.json` with the stage and error; nobody reads logs.
* **Hub.** Results go to the EXISTING private repo `HF_EVAL_REPO` through `harness.notebook_eval.HubResults` (atomic, identity-checked,
  per-shard bundles so several Kaggle accounts can contribute); hindsight rows go to `HF_TRACES_REPO`. No repository is created.
  Set `ACCOUNT_LABEL` per Kaggle account so their campaign states do not share a bundle.

## Extending it

* **New conditions**: `conditions.register_condition(Condition(...))` (`pack`, `evaluator` fields).
* **New tools** (segmentation, step-by-step image editing such as contrast / filters / zoom / kernels / re-look):
  `tool_packs.register_pack(name, factory)` returning a `ToolRegistry`; name it in `Condition(pack=...)`. The tool descriptions are in
  the prompt hash and the pack name is in the run identity, so results of different tool sets are never mixed.
* **New scoring** (segmentation masks, retrieval-augmented conditions): `evaluators.register_evaluator`, named in `Condition(evaluator=...)`.
* **Plugins**: modules listed in `PLUGINS` (and `ShardSpec.plugins`) are imported by the driver and by every hosted worker before
  they run; a plugin is ordinary repository code that registers the above. Existing conditions keep their fingerprints (the pack
  and evaluator appear in the identity only when they are not the defaults).
* **Hindsight RAG**: `hindsight.export_hindsight` writes one row per episode (trajectory, tools, harness events, answer, ground
  truth, usage and an `outcome` of success / partial / failure / format_failure / error) to the traces repo; it runs only when asked.
