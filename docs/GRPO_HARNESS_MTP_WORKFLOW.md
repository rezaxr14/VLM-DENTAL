# GRPO with and without rules/skills, followed by MTP draft fine-tuning

This workflow keeps the existing GRPO architecture: the trainable policy adapter is
initialized from SFT, while the original SFT adapter remains the frozen KL reference.
It does not retrain SFT, stack SFT twice at inference, or change base/adapter loading,
Kaggle dataset resolution, vLLM server-command construction, or cleanup policy.

## Apply

Apply `0058-grpo-harness-mtp-workflow.patch` after corrected 0055 and 0057. Patch 0056
is independent and can already be installed. This is an incremental patch; do not
reapply 0055/0057 over an installation that already includes them.

## 1. Train the two GRPO variants

In `VLM_Dental_Colab_GRPO.ipynb`, the execution-manifest cell exposes:

```python
TRACK = "with_tools"  # Keep the actual configuration cell's track selection.
GRPO_HARNESS_MODES = ["off", "rules_skills_index"]
```

The existing checked launch cell runs each mode as a separate process, using the
same SFT reference, image pool, seed and hyperparameters. A failure stops the cell.
Both variants have dental tools. `no_tools` is a separate control and cannot use
the rules/skills harness. `rules_skills_index` exposes the skill index and on-demand
`read_skill`; `rules_skills_full` injects all skill bodies. To choose full injection,
change the selected mode consistently in training and evaluation cells.

The original off checkpoint name is preserved. The harness run appends a readable
suffix, for example:

```text
grpo/qwen3_5_9b_grpo_with_tools_k4_dentex_alone_qlora__dual_t4_fp16/
grpo/qwen3_5_9b_grpo_with_tools_k4_dentex_alone_qlora__dual_t4_fp16--harness_rules_skills_index/
```

Each root contains its own `grpo-with_tools-final/grpo_policy`. Step/final metadata
records the training harness and rule/skill file fingerprints; resume rejects a
different mode or changed harness contents. Legacy off checkpoints remain valid.
The existing resume limitation remains: GRPO optimizer moments restart cold.

## 2. Evaluate both trained variants

All six notebook evaluation setups expose `EVAL_GRPO_HARNESS_MODES`,
`SPEC_HARNESS_MODE`, `SPEC_TARGET_KIND`, and `SPEC_DRAFT_VARIANT` beside the literal
editable profile dictionaries. Primary GRPO evaluations use the corresponding
training harness. Zero-shot, base controls and ordinary SFT controls retain their
original selection and loading behavior. Rules/skills ablations remain explicit,
separate conditions on their selected target.

Results go to the existing `Reza-Nadimi/vlm-dental-evaluations` repository. Named
paths and identities distinguish harness cohorts. Paired summaries compare matching
GRPO harness-on versus harness-off runs at the same profile, stage, K, image cohort,
image size and ablation settings. Positive differences mean harness-on minus off.

## 3. Select and merge the final harness-GRPO target

In `VLM_Dental_Colab_Speculative_Decoding.ipynb`, select the real hardware profile,
stage and K. Defaults select `SPEC_TARGET_KIND = "grpo"` and
`SPEC_HARNESS_MODE = "rules_skills_index"`. Set `RUN_MERGE = True` when preparing a
new target. The existing merge implementation is unchanged; a metadata sidecar
`dental_target.json` binds the exact source adapter and training harness.

`MTP_ADAPTER_PREFIX` is selected automatically only when one matching final policy
exists for `MTP_PROFILE`; otherwise set the exact prefix printed by the inventory.
No reference, off-harness policy, SFT-only adapter or other profile is substituted.

## 4. Generate draft data and fine-tune MTP

Set `RUN_MTP_FINETUNE = True` only in the intended training/serving environment.
The cell will:

1. Serve that exact merged final GRPO target using the existing hidden-state wrapper.
2. Reuse image IDs from the SFT TRAIN trace files, excluding held-out dev images.
3. Generate new target conversations with the selected rules/skills enabled.
4. Prepare data, extract hidden states, then release only that owned server.
5. Train the native MTP head and stitch it into a new checkpoint.
6. Verify every non-MTP tensor is unchanged and at least one MTP tensor changed.
7. Save an explicit fine-tuned-draft manifest and, with `PUSH_MTP_DRAFT = True`,
   upload only the trained head/provenance to the existing model repository.

`TRAIN_EPISODES = None` selects the full eligible training pool; set a positive
number for a smoke run. Generation and MTP sequence budgets remain 16384. Existing
conversation files are reused only after exact target/harness/provenance validation.
Old SFT assistant answers or intermediate-policy GRPO rollouts are not silently
substituted for final-target conversations. No new manual annotation is required.

The workflow reports the native MTP parameter count from safetensors shapes,
excluding shared embeddings and the output head. Weights, optimizer tensors and
activation memory are different quantities; the parameter count is not a VRAM fit
guarantee. Native MTP, multimodal rendering and hidden-state extraction still require
the pinned compatible Speculators/vLLM environment already configured in the notebook.
No generation budget or dependency version is lowered by this patch.

Named artifacts are saved under:

```text
data/mtp_train/<GRPO experiment name>/
HF_ARTIFACT_REPO/mtp/<GRPO experiment name>/trained_head/
```

Failed or partial data generation remains visible and is not certified as complete.
The original strict output/provenance checks remain; select a new output directory
when intentionally replacing a target or rerunning a different experiment.

## 5. Compare original and fine-tuned drafts

Ordinary decoding and MTP decoding use the same target, harness, tools, images,
seed, geometry and generation settings. `baseline` inside this benchmark means
ordinary decoding of this same trained target, not the untrained base model.

The after-training cell explicitly selects the fine-tuned draft and the one selected
source adapter. It checks matching non-MTP target weights, harness identity, decoded
outputs/tool actions and F1 before comparing acceptance. Only requested MTP conditions
are compared. Timing and equivalence failures remain visible and suppress valid
speedup claims; no acceleration or diagnostic result is assumed in advance.

## Reuse a saved head in another runtime

Download only its named `mtp/.../trained_head` folder using your existing HF tools.
Prepare the same merged final harness-GRPO target locally, then run:

```bash
python scripts/bench/restore_mtp.py --checkpoint LOCAL_MERGED_TARGET --draft-dir LOCAL_DOWNLOADED_HEAD --output NEW_STITCHED_TARGET --harness-mode rules_skills_index --speculators-executable PATH_TO_SPECULATORS --execute
```

The helper checks the source adapter, harness fingerprint and target tensor identity
before stitching, and audits the resulting non-MTP weights. It never downloads a base
model. In the final evaluation notebook, set `SPEC_DRAFT_VARIANT = "finetuned"` and
map that exact source adapter to `NEW_STITCHED_TARGET` in `SPEC_CHECKPOINTS`.

## Validation scope

CPU tests exercise harness propagation, distinct checkpoint names/cohorts, legacy-off
compatibility, resume/provenance rejection, TRAIN/dev exclusion, head-only publishing,
requested-condition comparisons, and subprocess/upload reliability. Notebook schemas,
transformed code syntax and protected-code comparisons are checked separately.
GPU/TPU training, vLLM startup, multimodal hidden-state extraction, live HF uploads,
VRAM consumption, final diagnostic metrics and speedups require real runtime execution.
