# Input resolution: how every SFT / GRPO input is found

Code: `dental_agent/utils/inputs.py` (one resolver per input), used by `scripts/train_sft.py`, `scripts/run_grpo.py`, `scripts/run_grpo_sweep.py`, `scripts/compute_exact_trace_lengths.py` and the `--preflight` doctor (`dental_agent/training/preflight.py`). Tests: `tests/test_input_resolution.py` (no network: the three Hub functions are replaced).

## Rules

1. **Fixed order** for every input: explicit flag, local folders, attached Kaggle inputs, Hugging Face Hub. The first hit wins and its source is printed (`flag`, `local`, `kaggle`, `hub-cache`, `hub`).
2. **Paths only.** A resolver never changes a setting. Precision, `--canonical-resize`, `--max-seq-len`, slots and the shim come from the command; the notebook is where they are set.
3. **Working-directory independent.** A relative path is tried against the working directory and against the repository root. A trace image recorded as `C:\Users\...\13.png` or `/content/VLM-DENTAL/data/dentex/images/13.png` is rebased onto the local data roots.
4. **Fail before the model loads**, with what was searched and one fixing command (`[INPUT NOT FOUND] ...`). Nothing falls back to a different input silently: a wrong `--sft-model-dir` fails, a missing image fails, a missing stage trace file fails.
5. **Surgical download (Rule 16).** Only the missing delta is fetched: one trace file, one image (or one `list_repo_files` call per dataset, then the missing ids), the adapter files `adapter_config.json`, `adapter_model.*`, `sft_run_meta.json` (never optimizer state).
6. **Leakage guard.** DENTEX train and the 50 official validation images share ids 1-50. A DENTEX *train* trace is never resolved to a file under a `val` / `validation` / `test` named path.

## Resolvers

| Input | Function | Order |
|-------|----------|-------|
| Trace image | `resolve_image`, `ensure_trace_images`, `ensure_dataset_images` | recorded path, rebased on data roots, `<id>.png` in image folders, Kaggle inputs, recursive search, HF cache, one-image Hub download |
| SFT adapter | `resolve_sft_adapter` | `--sft-model-dir` (must be valid, else error), `data/models/qwen3_5_9b_sft_<track>_<stage>_<precision>`, unsuffixed name, `checkpoints/sft_<track>_<stage>`, `data/models/qwen3_5_9b_sft_<track>`, Kaggle (by folder name or by `sft_run_meta.json`), Hub `sft/<name>_<precision>` then `sft/<name>` |
| Stage traces | `resolve_stage_traces`, table `stage_trace_names` | `<data-dir>/traces`, `data/traces`, repo `data/traces`; missing files from the Hub traces repo. **All** files of the stage are required (disease and healthy); the legacy single file is used only when no stage file exists anywhere |
| Token manifest | `check_token_manifest` | `<data-dir>/traces/trace_token_lengths.json`, repo `data/traces`, `/kaggle/working/data/traces`; then judged by `manifest_staleness` |
| Dev split | `resolve_dev_split` | `--dev-split`, repo-root copy, `<data-dir>/splits/` |
| YOLO checkpoint | `resolve_yolo_checkpoint` | `GROUNDING_MODEL_PATH`, default folders, Kaggle, Hub `yolo_cv/yolo_cv_best/weights/best.pt` (copied to the default path) |
| Base model | `resolve_base_model` | existing local folder (needs `config.json` and weights), or a Hub id (cache, then Hub check in `--preflight`); a nonexistent path fails |

`--preflight` calls the same functions with `hub="probe"` / `"off"`: it lists, never downloads, never writes.

## When the token-length manifest counts as stale

`compute_exact_trace_lengths.py` writes `_meta.fingerprint` = sha1 of `sft.py`, `canonical.py`, `prompts.py` (line endings normalised), sha1 and record count of every trace file, and the canonical flag; plus `_meta.complete` and `_meta.estimated_traces`. A manifest is stale when any of these holds: no fingerprint (written before patch 0034), the measurement did not finish, any entry is an estimate, canonical flag differs, the three source files changed, a trace file changed (for example after the patch-0028 regeneration).

* Under `--pad-vision-to-slots` stale or missing is an error: the length filter would use wrong numbers and Rule 21 (no silent truncation) depends on it.
* Without it, a warning with the recompute command.
* An interrupted measurement resumes only from a partial manifest of the same code and traces.

## Worked example

Illustrative numbers. TPU run, `dentex_alone`, `with_tools`, notebook folder `/kaggle/working`, repo checked out at `/kaggle/working/VLM-DENTAL`:

1. Traces: `data/traces/train_cot_traces_dentex.jsonl` is local, `train_cot_traces_healthy_dentex.jsonl` is not. The Hub traces repo is asked for that one file only.
2. 705 train traces minus 70 dev images = 635 images to resolve. 600 are in the attached Kaggle image dataset, 35 are not: one `list_repo_files` call, then 35 single-image downloads. If 2 are on no source, the run stops with `Trace images (2 of 635 images missing ...)` and `python scripts/fetch_hf_assets.py --images`.
3. Manifest was measured before patch 0028, so it has no fingerprint: the run stops with the exact `compute_exact_trace_lengths.py --recompute --vision-slots 5 10 4 --canonical-resize` command instead of filtering with old lengths.
4. Only then is the 9B model loaded.

## What was not verified against the real Hub

The sandbox that built this has no route to Hugging Face. Hub listing, snapshot and file downloads are tested with in-memory fakes. The first real test is the doctor on your machine: `python scripts/train_sft.py ... --preflight` with `HF_TOKEN` set.
