# Shared notebook evaluations

All evaluation cells in SFT, GRPO, ZeroShot (native local branch only), Rules_Skills,
Speculative_Decoding and Final_Evaluation use `dental_agent/harness/notebook_eval.py`.
The evaluators still own inference, prompts, dispatch, matching and scoring. Training
cells, reward functions, harness defaults and the frozen paper protocol are unchanged.

## Controls visible in the evaluation cells

- `EVAL_PROFILES`: a literal editable dictionary in every setup cell, passed to the runner. Edits control discovery, runtime flags, compatibility checks and result identity.
- `EVAL_TARGET_PROFILES`: a profile name, list of names, or `"all"`.
- `EVAL_STAGES`: an exact training stage or `"all"` (all uploaded stages).
- `EVAL_TRACK`: `TRACK` in SFT/GRPO; explicit `with_tools`, `no_tools`, or `all` elsewhere. Selects matching baselines and trained adapters. Missing adapters are reported per track.
- `EVAL_SELECTED_K`: dev-selected GRPO K; `"all"` is permitted only on dev.
- `RUN_EVALUATIONS` / `RUN_MISSING_EVALUATIONS`: whether to launch missing inference.
- `EVAL_SIZES`: canonical and/or original primary evaluations. These never share a result file.

Canonical names are `dual_t4` (FP16 compute QLoRA), `rtx4090_qlora` (BF16 QLoRA),
`workstation48_lora` (BF16 unquantized LoRA), `tpu`, and the existing legacy
`dual_t4_fp16` unquantized FP16 zero-shot profile. `rtx4090` and `rtxpro6000_48gb`
are aliases. `"all"` selects checkpoint cohorts; it does not change the physical GPU.
A 48-GB CUDA runtime can attempt multiple numerical profiles; a T4 cannot execute
native BF16 profiles. No precision is silently substituted. Unsupported jobs are
listed as blocked. Eight TPU cores are not pooled memory: sharded 9B autoregressive
generation is not implemented in the current evaluators, so TPU stays pending.
No CPU/disk offload is silently enabled to pretend an unsupported runtime fits.

Generation remains **16384** for every managed local cohort, with 50 tool calls,
zero malformed retries and no sample limit. Speculative serving exposes a separate
**32768** total context capacity, because a 16384-token output request plus nonempty
text/image input cannot fit a 16384 total context. This changes no training length or
generation headroom. Long histories can still exceed the selected serving capacity;
no prompt truncation or automatic output-budget reduction is introduced. VRAM fit and
kernel support must be verified on the actual serving hardware/environment.

## Cohorts and reuse

The final notebook lists source checkpoints, saved bundles and missing conditions,
then runs zero-shot, baseline without/with tools, separately trained SFT controls,
separately trained GRPO policies, rules/skills and speculative comparisons. A missing
no-tools adapter is reported; a with-tools checkpoint is never substituted. Only final
GRPO policy adapters are selected, never the reference adapter. Both SFT and GRPO
with-tools checkpoints may have separate dev-only harness/serving experiments.

Zero-shot keeps `ZERO_SHOT_PROMPT`; it is separate from the unified no-tools baseline.
Rules/skills and speculation always use the committed dev split. The selected K is not
swept on test. Optional K sweeps and tool/grounding probes remain dev-only.
The official test cohort must resolve all 50 images; the dev cohort must match the
committed membership exactly. Healthy images are retained. Local originals are reused;
only missing per-image files are fetched from `DENTEX_IMAGES_REPO`, with train and
validation namespaces kept separate. No new split is generated.

Managed notebooks share content identities built from source/prompt code, software,
profile runtime, exact adapter content IDs, cohort image/annotation hashes, split,
stage, K, image geometry and decoding settings. Adapter-content fingerprints exclude
evaluation uploads, so uploading results does not invalidate checkpoint identity.
A prior matching manifest can be selected before downloading images/model weights;
conflicting prior cohort identities stop rather than choosing silently.

Validated complete bundles skip model loading. Partial episodes restore and resume;
local and remote progress merge only when overlapping records agree. Changed settings,
weights, duplicate IDs, corrupt checksums and conflicting evidence fail closed. Good
results are never moved to an invalid folder. The canceled 2K evaluation is outside
this managed namespace and is never reused. Legacy files without a compatible identity
remain visible as unverified; they are not silently certified or overwritten. These
cross-notebook identities apply to evaluations run with this patch; arbitrary historical
ZIPs/JSONLs cannot prove matching checkpoints/settings by filename alone.

## Existing Hugging Face repository only

Results go to the existing `Reza-Nadimi/vlm-dental-evaluations` repository, configured by `EVAL_RESULTS_REPO` in every setup cell. `EVAL_RESULTS_REPO_TYPE = None` detects its existing dataset/model type; set the type explicitly if needed. Checkpoints remain in `HF_ARTIFACT_REPO`, defaulting to `Reza-Nadimi/vlm-dental-models`. No repository is created.

```
evaluations/dentex/<split>/<profile_tag>/<stage>/<kind>/<condition>/[k<K>/]<size>/run_<identity16>/
    bundle.json        # identity, per-file hashes, progress and completion
    results.jsonl      # primary / local zero-shot records
    results.json       # harness / speculative records instead
    runtime.json       # actual hardware/software/command, no environment secrets
    summary.json       # measured metrics once complete

evaluations/summaries/dentex/<split>/<profile>/<stage>/<controls_or_kK>/run_<content16>/
    summary.json
    summary.md
```

Progress syncs every 60 seconds and after completion/handled interruption. A hard runtime
shutdown can lose the unsaved interval. Uploads contain results and metadata, not images,
base weights, optimizer states or credentials. Optimistic parent-commit checks prevent
silently overwriting concurrent Hub writes; a conflict leaves local results intact.
Authentication and write access to the existing results repo are required. Failed greedy
speculative equivalence remains a terminal failed report, never an accepted speedup.

`SPEC_CHECKPOINTS[profile][exact_adapter_prefix]` explicitly associates an audited
merged target with its source adapter. The speculative notebook retains an opt-in CPU
merge/audit and MTP-only training workflow. Missing compatible vLLM executables or
merged targets remain pending rather than starting unrelated training or choosing
another target. A merged 16-bit serving target has its own baseline; equivalence to
4-bit QLoRA inference is not assumed. MTP training remains opt-in and separate from
SFT/GRPO. Non-MTP byte checks remain in the existing stitch/audit scripts.

The final notebook produces inventory, raw records and JSON/Markdown summaries. It
never generates paper figures, LaTeX assets or a PDF. Rebuild the paper independently
with `python scripts/paper/build_paper.py` when desired.

## Verification

```bash
python -m pytest tests/test_notebook_evaluations.py tests/paper/test_final_evaluation.py tests/test_harness_rules.py tests/test_harness_skills.py tests/test_harness_specdec.py -q
python scripts/paper/verify_paper.py
```

CPU tests cover coordination and notebook/CLI contracts; they do not establish hardware
fit, diagnostic performance, live Hub permissions, or vLLM kernel compatibility. Run
actual inference in the selected evaluation runtime, separately from active training.
