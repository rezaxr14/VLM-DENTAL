# Checkpoint uploads and notebook settings (0043)

Author: Reza Nadimi <rezaxr14@gmail.com>

## Choose the patch that matches your checkout

- **Fresh upstream at `e1e33e29a99a96c7ce7bbc821d3c33d93dc202c9`:** apply
  `0043-tpu-hub-upload-notebook.patch`. It includes the complete TPU loss patch
  0042 as well as this upload/notebook fix. Do not apply 0042 separately.
- **That upstream plus patch 0042 already applied:** apply
  `0043-on-top-of-0042.patch` instead.

For either file:

```bash
git apply --check PATCH_FILENAME.patch
git apply PATCH_FILENAME.patch
```

The mail header names Reza Nadimi <rezaxr14@gmail.com>. `git apply` changes files;
it does not create a commit. No agent commit or push was performed.

Before a new Kaggle Commit & Run, make the patched source available from the
repository that the notebook clones (commit/push it yourself), or include a patch-application cell after the clone using an attached patch
file. An interactive checkout edit alone is not carried into a new Commit & Run
container. The updated notebook
contains no embedded base64 repair. It checks for required flags and stops
before model loading if the checkout is stale. Editing the notebook/repository
does not modify a Commit & Run job that is already executing.

## Why the upload failed

PEFT generated a model card containing this local loading path:

```yaml
base_model: /kaggle/input/datasets/rezanadimikj/qwen3-5-9b
```

The Hub requires a model identifier in that optional metadata field:

```yaml
base_model: Qwen/Qwen3.5-9B
```

The shared uploader repairs the checkpoint root and nested adapter cards,
including `best_adapter`, `reference` and `grpo_policy`. With
`--hub-base-model-id`, it also makes the on-disk `adapter_config.json`
`base_model_name_or_path` portable. It preserves adapter weight bytes and does
not change the in-memory model configuration, local model loading, precision,
quantization, attention, chunking, labels or optimizer math.

Without an explicit override, valid existing metadata is preserved. A valid
Hub identifier from an adapter config can repair a missing/invalid card field.
If no trustworthy Hub identifier is available, invalid optional `base_model`
metadata is omitted instead of inventing a model origin. Local loading paths
in adapter configs are retained unless a Hub ID is explicitly supplied.

The fix also repairs GRPO cards before its existing uploads. GRPO training and
its existing checkpoint destination convention are otherwise unchanged.

## SFT destination identity

The SFT Hub folder always uses **track + curriculum stage + precision**, even
when `OUTPUT_DIR` has a custom unrelated name:

```text
sft/qwen3_5_9b_sft_with_tools_dentex_alone_qlora/
sft/qwen3_5_9b_sft_with_tools_dentex_alone_bf16/
sft/qwen3_5_9b_sft_with_tools_dentex_tufts_overlap_qlora/
sft/qwen3_5_9b_sft_with_tools_multicohort_all_bf16/
```

Uploads, resume downloads and read-only preflight use the same destination
function. Default canonical destinations are unchanged. Previously custom
output-derived Hub names are no longer the default; recover/re-upload those
checkpoints into a canonical destination with the utility below.

QLoRA FP16 compute on T4 and QLoRA BF16 compute on RTX 4090 both have storage
precision `qlora`. To keep those experiments separate, set an explicit tag:

```python
HF_RUN_TAG = "dual_t4_fp16"  # Another run could use "rtx4090_bf16".
```

The folder then ends in `..._qlora__dual_t4_fp16/`. Use the same tag to resume.
The tag is Hub-only; choose a separate `OUTPUT_DIR` too if retaining distinct
local runs on persistent storage. A new experiment should use fresh output
locations rather than overwrite an existing experiment. GRPO's automatic SFT
reference search keeps its existing untagged convention; for tagged runs,
download the chosen adapter and pass its local `--sft-model-dir` explicitly.

## Updated SFT notebook

GPU profile precision, compute dtype, chunk sizes, attention, loader settings,
canonical views, token limits and optimization hyperparameters are unchanged.
The notebook now exposes:

```python
HF_BASE_MODEL_ID = "Qwen/Qwen3.5-9B"  # Metadata only; MODEL_ID remains local.
HF_RUN_TAG = ""                     # Optional experiment suffix.
HF_UPLOAD_CHECK = True
SAVE_STRATEGY = "both"
PUSH_EVERY_STEPS = 5
```

Every five optimizer steps, and at epoch boundaries, the existing checkpoint
routine saves and uploads the adapter, resume state and available best adapter.
This adds I/O/upload overhead; adjust the cadence explicitly if desired.
`DEBUG_STEPS=2` still means diagnostics without checkpoint uploads. Set it to
`0` for actual training as before. Epochs, LR, batch/accumulation and LoRA
settings have not been shortened or retuned.

With uploads enabled and debug disabled, an actual launch first writes a small
`upload_check.json` to the selected Hub folder **before loading model weights**.
This checks credentials, repository access and that destination's upload path.
It is not a model checkpoint. Failure stops early instead of waiting until the
first epoch. It does not prove that a later large transfer cannot fail.
The repository must already exist. `--no-hf-upload-check` explicitly skips this
check; `--hf-repo ''` disables checkpoint uploads. `--preflight` remains read-only
and performs no write probe. Smoke-test scripts disable uploads for synthetic
artifacts.

For the TPU profile:

```python
TPU_LOSS_CHUNK_TOKENS = 256
RUN_TPU_LOSS_PROBE = True
```

The GPU flag remains `LOSS_CHUNK_TOKENS`; TPU keeps that value at 0 and uses
`--tpu-loss-chunk-tokens` separately. The notebook passes the TPU value to both
training and AOT, runs the real-TPU output-head probe once per core/chunk setting,
and passes matching LoRA rank/alpha/dropout/LR to AOT. Unsupported training flags
are never appended to the CUDA attention probe. Printed local/Hub checkpoint
paths and the plot's local precision suffix are corrected.

The TPU implementation is the opt-in 0042 path. Read `TPU_CHUNKED_LOSS.md` for
its experimental XLA API, hardware probe and HBM limitations. Physical TPU fit
is not established by CPU/software tests.

## Upload an already recovered checkpoint without retraining

Run this from the patched repository, on CPU or your local computer. Authenticate
with the Hugging Face account that can write to the destination (for example,
`hf auth login`). The utility also reads the repository `.env` without overriding
existing environment variables. Never paste tokens into shared notebooks/chat.

```bash
python scripts/upload_saved_checkpoint.py \
  --checkpoint-dir FULL_PATH_TO_SAVED_CHECKPOINT_FOLDER \
  --hf-repo Reza-Nadimi/vlm-dental-models \
  --hub-base-model-id Qwen/Qwen3.5-9B
```

If `sft_run_meta.json` is present, stage/track/precision are inferred from that
saved metadata. Otherwise add:

```bash
--track with_tools --stage dentex_alone --precision qlora
```

Use `--hf-run-tag NAME` for a distinct experiment or `--path-in-repo PATH` for
an explicit destination (including a GRPO checkpoint). Explicit stage/precision
values that conflict with saved SFT metadata are rejected. `--prepare-only`
repairs local metadata and prints the destination without initializing a Hub
client. The utility checks for adapter config and nonempty saved weight files;
it never loads the model, allocates CUDA tensors, or downloads base weights.

Use a writable copy of recovered files: `/kaggle/input` attachments are read-only.
Select the actual checkpoint folder, not the entire notebook-output parent.
Keep `best_adapter`, optimizer, scheduler, scaler and `training_state.json` when
available so the existing resume flow has its expected files. No new utility can
retrieve unpublished files from a different live Commit & Run container.

## Verification scope

Local tests reproduce a real PEFT save with the invalid Kaggle metadata, repair
root/nested cards and configs, verify unchanged weight bytes/live model paths,
exercise 18 stage/track/precision destinations, test permission failures and
recovery upload routing with fake Hub clients, and execute all four notebook
profiles against the real CLI parsers. Notebook platform flags/subprocesses are
simulated solely for command-building tests; that is not hardware validation.
No actual model upload or token permission check was performed by the agent.
Existing CPU SFT smoke/numerical tests and XLA software compiler checks provide
additional regression coverage; they do not measure real GPU speed or TPU HBM.

Primary references:

- https://huggingface.co/docs/hub/model-cards
- https://huggingface.co/docs/huggingface_hub/en/guides/upload
