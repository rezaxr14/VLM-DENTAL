# Runbook: SFT and GRPO, GPU path and TPU path

Run the commands of one section top to bottom. The notebooks (`notebooks/VLM_Dental_Colab_SFT.ipynb`, `VLM_Dental_Colab_GRPO.ipynb`) generate exactly these commands. Nothing is chosen for you: precision, view mode and sequence length are always explicit flags.

Replace `<repo>` with your Hugging Face model repo. Commands are one line each (PowerShell and bash).

---

## Step 0. The doctor (both paths, both stages)

Every input is found by itself, in this fixed order: **explicit flag, then local folders (the working directory and the repository root), then attached Kaggle inputs, then the Hugging Face Hub.** Discovery resolves paths only; it never changes a setting (precision, `--canonical-resize`, `--max-seq-len` are only ever what you pass). Details: `docs/INPUT_RESOLUTION.md`.

Add `--preflight` to the exact command you are about to run. It loads no model, changes nothing on disk, and prints `[PASS]`, `[INFO]`, `[WARN]` or `[FAIL]` per input, then exits non-zero on any FAIL:
```
python scripts/train_sft.py --track with_tools --stage dentex_alone --precision qlora --canonical-resize --no-canonical-letterbox --dev-split data/splits/dentex_dev.json --preflight
python scripts/run_grpo.py --track with_tools --sft-stage dentex_alone --precision qlora --canonical-resize --no-canonical-letterbox --dev-split data/splits/dentex_dev.json --group-size 4 --preflight
```
The notebooks do this for you (`PREFLIGHT_FIRST = True`): the launch cell re-runs its own command with `--preflight` and stops before anything loads if there is a FAIL. Without `--preflight` the real run performs the same resolution first and stops with `[INPUT NOT FOUND]`, naming everything it searched and the one command that fixes it, before the 9B model is touched.

| Input | Searched (in order) | Fix when missing |
|-------|--------------------|------------------|
| SFT adapter (GRPO) | `--sft-model-dir`, `data/models/qwen3_5_9b_sft_<track>_<stage>_<precision>`, the same without suffix, `checkpoints/sft_<track>_<stage>`, Kaggle inputs, Hub `sft/<name>` | `train_sft.py` for that track, stage and precision |
| Traces (SFT) | `<data-dir>/traces`, `data/traces`, repo `data/traces`, Hub traces repo (only the missing files) | `python scripts/sync_traces_hf.py --download` |
| Trace and pool images | recorded path, rebased on the data roots, `<id>.png` names, Kaggle inputs, HF cache, Hub (one image per request) | `python scripts/fetch_hf_assets.py --images` |
| Token-length manifest | `<data-dir>/traces/trace_token_lengths.json`, repo `data/traces`, `/kaggle/working/data/traces` | `compute_exact_trace_lengths.py ... --recompute` (the doctor prints the full command) |
| Dev split | `--dev-split`, repo-root copy, `<data-dir>/splits` | `python scripts/make_dev_split.py --data-dir data --n 70 --seed 42` |
| YOLO checkpoint (`with_tools`) | `GROUNDING_MODEL_PATH`, `data/models/yolo_cv_best/weights/best.pt` and sibling folders, Kaggle, Hub `yolo_cv/yolo_cv_best/weights/best.pt` | `python scripts/fetch_hf_assets.py --models` |
| Base model (notebooks) | `resolve_qwen_model()`: `QWEN35_9B_LOCAL_DIR`, attached Kaggle model / Drive copy, `<repo>/data/models/Qwen3.5-9B`, an existing HF cache snapshot, else a resumable download into `data/models/Qwen3.5-9B` | put the 19 GB folder in `data/models/Qwen3.5-9B` or set `QWEN35_9B_LOCAL_DIR` (see "Base model location" below) |
| Base model (scripts) | the `--model-id` you pass: an existing local folder with `config.json` and weights, or a Hub id | pass the folder the notebook printed as `Qwen3.5-9B source: <source> -> <path>` |

Behaviour changes in patch 0034 you should know about:
* A wrong explicit `--sft-model-dir` now **fails**. It used to fall back to a different folder, and GRPO could start from a fresh LoRA without saying so.
* A trace whose image cannot be found now **stops the run** (it used to train on a gray 512x512 square). A DENTEX *train* trace is never resolved to a validation (test) image, even though the ids 1-50 exist in both sets.
* The token-length manifest carries a fingerprint (code, trace files, canonical flag, whether every trace was measured). A manifest without one, from an interrupted run, with estimated entries, or from older traces or code is stale: it is a **FAIL under `--pad-vision-to-slots`** (the filter would use wrong lengths) and a WARN otherwise.
* The doctor warns when a `no_tools` run uses `--pad-vision-to-slots` with the default slots: every sample then carries 7,488 vision tokens for 1,152 used.

### Base model location (patch 0035)

One folder, one rule, in every notebook that loads Qwen3.5-9B (SFT, GRPO, ZeroShot, TraceGen's vLLM start): `dental_agent.utils.inputs.resolve_qwen_model()` returns `(path, source)` and prints `Qwen3.5-9B source: <source> -> <path>`. The first **valid** candidate wins; nothing is copied or moved:

| # | Candidate | `source` |
|---|-----------|----------|
| 1 | environment variable `QWEN35_9B_LOCAL_DIR` (set but unusable = error, never skipped) | `env` |
| 2 | attached Kaggle model / dataset, read-only and in place (`KAGGLE_BASE_MODEL_DIRS` plus any Kaggle input folder named like Qwen3.5-9B); on Colab `<persist dir>/models/Qwen3.5-9B` | `kaggle-input`, `colab-drive` |
| 3 | `<repo root>/data/models/Qwen3.5-9B` (exact casing; found from the repo, not from the working directory; git-ignored) | `repo-data` |
| 4 | an existing Hugging Face cache snapshot of `Qwen/Qwen3.5-9B` (`HF_HUB_CACHE`, `$HF_HOME/hub`, `~/.cache/huggingface/hub`, `data/models/vllm_cache/hub`) | `hf-cache` |
| 5 | none found: `snapshot_download("Qwen/Qwen3.5-9B", local_dir=<canonical folder>)`, which resumes a partial copy (Kaggle with a read-only repo: `/kaggle/working/models/Qwen3.5-9B`, with a warning about the 19 GB working-disk limit). With `HF_HUB_OFFLINE=1` or no network: an error that says where to put the model | `downloaded` |

A folder is valid only if `config.json`, a tokenizer file, a processor file and every shard listed in `model.safetensors.index.json` (or `model.safetensors`) exist and are non-empty; a partial folder is skipped (or resumed in step 5). The notebooks then export the chosen folder as `QWEN35_9B_LOCAL_DIR`, so evaluation and the zero-shot provider use the same one. `FORCE_HF_MODEL_DOWNLOAD = True` skips 1-4. The Hub id is never handed to `from_pretrained`, and `HF_HUB_OFFLINE` is never set by a notebook (datasets still come from the Hub).

Name-based logic (checked in ZeroShot / TraceGen): nothing keys off "qwen3.5" in code. Only result file names (`zero_shot_<dataset>_<split>_transformers_Qwen--Qwen3.5-9B.jsonl`, `--reuse-cached` lookup) and the `model` field of records use the name, so the name stays `Qwen/Qwen3.5-9B` and the folder is passed separately: the `transformers` provider loads the folder (`local_model_alias`), the cached-file lookup maps the folder back to the same tag (`model_file_tag`), and TraceGen starts vLLM with `--model <folder> --served-model-name Qwen/Qwen3.5-9B` so generator requests still match.

### Image view on a GPU: nothing added, nothing stretched (patch 0036)

Three views exist; the notebook states which one in `CANONICAL_RESIZE` / `CANONICAL_LETTERBOX`, and SFT, GRPO, the token-length manifest and evaluation must all use the same one.

| Notebook setting | Flags | What the model sees | Use |
|---|---|---|---|
| `CANONICAL_RESIZE = True`, `CANONICAL_LETTERBOX = True` | `--canonical-resize --canonical-letterbox` | uniform scale, then centred on a fixed black canvas (FULL 1536x768, CROP 256x384, COMPARE 512x384) | **TPU only** (static vision slots need one fixed shape) |
| `CANONICAL_RESIZE = True`, `CANONICAL_LETTERBOX = False` | `--canonical-resize --no-canonical-letterbox` | the same uniform scale, **nothing added**; each side rounded to a multiple of 32 | RTX 4090 / A100 / Pro6000, QLoRA or LoRA |
| `CANONICAL_RESIZE = False` | `--no-canonical-resize` | the full image at its native size | GPU, no canonical size at all |

`CANONICAL_LETTERBOX = IS_TPU` in the notebooks (TPU pads, GPU never does); the script refuses the other combination instead of changing it: `--canonical-resize --canonical-letterbox` on a CUDA GPU stops with the fix, `--pad-vision-to-slots` without letterbox stops, and the GRPO doctor FAILs `canonical-letterbox vs SFT` when the two stages differ (an SFT run from before this patch counts as padded).

Worked example, a 2872 x 1504 panorama, FULL view: scale = min(1536/2872, 768/1504) = 0.5106, giving 1466.5 x 768, rounded to **1472 x 768** (46 x 24 patches = 1104 tokens; the padded canvas is 1536 x 768 = 1152 tokens with 32-pixel-wide black bars left and right). Aspect 1.9096 becomes 1.9167. Token counts now follow the image, so GPU batches are dynamically padded in the text only.

What this cannot remove, stated plainly: the Qwen processor snaps every image to a multiple of 32 pixels and enforces `min_pixels` = 65,536 (256 x 256) and `max_pixels` = 16,777,216. In the no-bars canonical view the rounding is done once by us (at most 16 pixels per side, about 1 % on a FULL view). In native mode the processor does it (same bound). A very small tool crop (below 65,536 pixels, for instance a CROP view of 256 x 224) is scaled up by the processor to reach the minimum, in both modes. The notebooks and scripts do not change `min_pixels`; if you want that limit lowered for crops, it is a separate, explicit setting.

After changing either setting recompute the token-length manifest (the check says so: `measured with canonical_letterbox=...`); a result file written with the other setting is not resumed.

### Behaviour changes in patch 0035

* **Resume restores everything.** `--resume` now reloads the adapter weights, the optimizer and the LR schedule from the checkpoint (before, only the step counter came back and training continued from a fresh LoRA). A checkpoint saved mid-epoch (SIGTERM or `--push-every-steps`) rewinds to the first step of that epoch, so no step is counted twice: epoch 2 began at step 76, the checkpoint says step 120, resume continues at step 76 with the schedule at step 76. A missing or foreign adapter file is `[RESUME FAILED]`, never a silent restart. First check on resume: the first logged loss is close to the last logged loss before the interruption.
* **SPMD batches are whole.** Both loaders use `drop_last`: a batch dimension not divisible by the core count would be replicated and compiled as a second graph. Worked example, 8 cores, 604 training samples: 75 steps x 8 = 600 used, 4 dropped (a different 4 each epoch); 31 validation samples: 3 x 8 = 24 used. The run prints `[LOADER] ... dropped`.
* **No silent unsharded model.** `--spmd --fsdp` without FSDPv2 in `torch_xla` now stops with an error (before: the full 18 GB model on every core, then out of memory).
* **Tool-image fallbacks are counted and warned** (`[TOOL IMAGE]`); a default crop replaced a missing tool image before without a word.
* **QLoRA quantizes the language model only.** The vision tower (`model.visual`: patch embed, 27 blocks, `merger`) and `lm_head` stay 16-bit; every 4-bit site (SFT, GRPO and its K sweep, evaluation, API pool) builds its config with `dental_agent.model.backbone.qlora_bnb_config`, whose skip list is the single constant `QLORA_SKIP_MODULES` in `dental_agent/config.py`. The projector LoRA targets `model.visual.merger.linear_fc1/2`, now plain Linear layers. `prepare_qlora_model` stops peft from upcasting the vision tower to fp32. NF4, double quantization and the compute dtype are unchanged. **A QLoRA adapter trained before this patch has a projector LoRA fitted on an NF4 merger**; the GRPO doctor warns (`QLoRA vision tower vs SFT`) and the clean fix is to retrain that SFT. bf16 runs and TPU runs are unaffected.
* `sft_run_meta.json` records `qlora_skip_modules`; the GRPO doctor compares precision and skip list with the SFT run.

---

## Section A. GPU path (RTX 4090: QLoRA; A100: bf16 LoRA)

The GPU path pads each batch to its longest sample. There are no vision slots, no solver shim and no SPMD here, because they exist only to stop XLA recompiling on TPU. `--max-seq-len` is a ceiling: traces longer than it are dropped, and the longest remaining trace sets the peak memory.

**A1. Install**
```
pip install -r requirements.txt
pip install -r requirements-gpu.txt
huggingface-cli login
```
`requirements-gpu.txt` installs `flash-linear-attention` (fast kernels for the Gated-DeltaNet layers) and `bitsandbytes` (QLoRA). Without the first, training works but is much slower. If `flash-linear-attention` will not install on Windows (it needs Triton), use WSL or Linux.

**A2. Measure trace lengths with canonical views** (once; reuse afterwards)
```
python scripts/compute_exact_trace_lengths.py --canonical-resize --no-canonical-letterbox --recompute
```

**A3. See how many traces fit each ceiling, and write the 4 longest traces as a probe set**
```
python scripts/select_longest_traces.py --traces data/traces/train_cot_traces_dentex.jsonl --manifest data/traces/trace_token_lengths.json --top 4 --output data/traces/probe_longest.jsonl
```

**A4. Smoke test** (tiny random model; catches crashes and library-version problems, not memory)
```
python scripts/smoke_test_sft.py --traces data/traces/train_cot_traces_dentex.jsonl -- --canonical-resize --no-canonical-letterbox --precision qlora
```
Expect `[SMOKE] PASSED`. Stop and report the traceback if it fails. The same check exists for GRPO and evaluation (4-bit, as on a 24 GB card):
```
python scripts/smoke_test_grpo.py --traces data/traces/train_cot_traces_dentex.jsonl --track with_tools --eval-precision qlora -- --canonical-resize --no-canonical-letterbox
```
Expect `[SMOKE-GRPO] PASSED`.

**A4b. QLoRA load check** (RTX 4090, loads the 9B model in 4-bit, no training, about 1 minute; `--names-only` runs anywhere and loads no weights)
```
python scripts/check_qlora_load.py --model-id <folder the notebook printed>
```
Expect `[PASS]` for: no 4-bit layer under `model.visual`, every language-model layer 4-bit, `lm_head` 16-bit, all LoRA targets found, projector LoRA on `model.visual.merger.linear_fc1/2`, then `[RESULT] QLoRA quantizes the language model only` and the CUDA memory. Exit code 1 = a check failed, 2 = cannot run here (no CUDA or bitsandbytes).

**A5. Memory probe at 16384** (a few steps on the longest traces; 4090: `--precision qlora`, A100: `--precision bf16`)
```
python scripts/train_sft.py --track with_tools --stage dentex_alone --dataset-path data/traces/probe_longest.jsonl --output-dir data/models/probe --epochs 1 --batch-size 1 --gradient-accumulation-steps 1 --canonical-resize --precision qlora --max-seq-len 16384
```
Watch the progress bar: `peak_gib` is the highest memory PyTorch has allocated so far (also written to `training_loss.jsonl`). `nvidia-smi` shows the larger real total (allocator cache and CUDA context add roughly 1-2 GiB on top). It fits when the run finishes without an out-of-memory error and `peak_gib` leaves that headroom below the card's memory.

**A6. Choose the sequence length**

| Result of A5 | What to do |
|--------------|------------|
| Fits | Keep `--max-seq-len 16384`. Raising it changes nothing for the current traces: the longest canonical trace is below 16384 (printed in A3). It only matters if you regenerate longer traces or train without `--canonical-resize`. |
| Out of memory | Lower the ceiling, and probe the new ceiling: `python scripts/select_longest_traces.py --traces data/traces/train_cot_traces_dentex.jsonl --manifest data/traces/trace_token_lengths.json --top 4 --at-most 12288 --output data/traces/probe_12288.jsonl`, then repeat A5 with `--dataset-path data/traces/probe_12288.jsonl --max-seq-len 12288`. A3 printed how many traces each ceiling keeps; lower ceilings drop the longest, multi-finding traces. |
| Still too large | Other levers that cost no data: lower `--lora-r`; on a 24 GB card make sure it is `--precision qlora`. |

**A6a. Create the dev split (once; SFT, GRPO, the K sweep and the ablations all use the same file)**
```
python scripts/make_dev_split.py --data-dir data --n 70 --seed 42 --output data/splits/dentex_dev.json
```
It draws about 70 of the DENTEX *train* images, keeping the healthy share, and writes `dataset::image_id` keys. Commit the file so every machine holds out the same images. The 50 official validation images stay the untouched test set: GRPO now loads the train images only and refuses any annotation file that looks like the validation split.

**A7. Full SFT** (use the same `--precision` and `--max-seq-len` as A5; other hyperparameters as set in the notebook)
```
python scripts/train_sft.py --track with_tools --stage dentex_alone --canonical-resize --no-canonical-letterbox --precision qlora --lora-linear-attn --mask-malformed-turns --dev-split data/splits/dentex_dev.json --max-seq-len 16384 --epochs 3 --batch-size 1 --gradient-accumulation-steps 8 --eval-strategy epoch --save-strategy epoch --hf-repo <repo> --push-every-steps 50
```
The adapter is saved to `data/models/qwen3_5_9b_sft_with_tools_dentex_alone_<precision>` (for example `..._qlora`).

**A8. GRPO** (same `--canonical-resize` and `--precision` as SFT). Without `--sft-model-dir` GRPO searches `data/models/qwen3_5_9b_sft_<track>_<stage>_<precision>` for its own `--precision` first, then the unsuffixed folders; the command below names the folder, which is the explicit form:
```
python scripts/run_grpo.py --track with_tools --sft-stage dentex_alone --sft-model-dir data/models/qwen3_5_9b_sft_with_tools_dentex_alone_qlora --dataset dentex --dev-split data/splits/dentex_dev.json --group-size 4 --epochs 2 --lr 5e-6 --kl-beta 0.04 --clip-eps 0.2 --temperature 0.7 --canonical-resize --no-canonical-letterbox --precision qlora --max-new-tokens 2048 --hf-repo <repo> --push-every-steps 25
```
GRPO starts its trainable adapter as a copy of the SFT adapter. To continue a run that was interrupted, repeat the same command with `--resume` (same `--dev-split`, `--seed`, `--max-images`, `--include-healthy` and `--group-size`); on a fresh Kaggle/Colab session add `--resume-hf <repo>` to download the newest step checkpoint first. `--resume-hf` alone is refused, because it only downloads.

**Before a long GRPO run (A8 or B7).** Add `--preflight` to the exact launch command: it checks the SFT adapter and its recorded settings, the dev split, the pool, the image files, any `--resume` state, the hub token and the hardware, and exits without loading the model. Fix every `[FAIL]` first. Then time a probe on a few images, with its own log and no hub uploads:
```
python scripts/run_grpo.py <same flags as the real run> --max-images 10 --hf-repo "" --log-path data/grpo_probe_k4.jsonl
python scripts/summarize_grpo_log.py --log data/grpo_probe_k4.jsonl --pool-size <training images> --project-k 2 4 8 16
```
The summary gives seconds per step (rollout and update), turns and tool calls per trajectory, and the projected hours for the whole pool at each K. Delete the probe's checkpoint folder afterwards, so the real run does not find its step checkpoints. In the notebook the same two steps are `PREFLIGHT = True`, and `MAX_IMAGES = 10` with `HF_REPO = ""`.

GRPO on a GPU has no sequence-length ceiling: rollout length is set by the agent loop, and memory is dominated by generation. Watch `nvidia-smi` through the first step. If it runs out of memory, report the step it reached.

**A9. Evaluate** (canonical size is the primary result; run it again without `--canonical-resize` for original size)
```
python scripts/evaluate_models.py --condition sft_with_tools --adapter-path data/models/qwen3_5_9b_sft_with_tools_dentex_alone_qlora --dataset dentex --split test --precision qlora --canonical-resize --no-canonical-letterbox --max-tool-calls 50 --max-new-tokens 2048
python scripts/evaluate_models.py --condition grpo_with_tools --adapter-path data/models/qwen3_5_9b_grpo_with_tools_k4_dentex_alone/grpo-with_tools-final --dataset dentex --split test --precision qlora --canonical-resize --no-canonical-letterbox --max-tool-calls 50 --max-new-tokens 2048
```
Evaluation runs the same agent loop as GRPO rollouts, so keep `--max-new-tokens` equal to the GRPO run's. Canonical results go to `..._canonical.jsonl`.
**Evaluation matrix (all conditions, both sizes, one command).** Loads each checkpoint once; results and the pooled tables go to `data/evaluations/`:
```
python scripts/run_eval_matrix.py --split test --sft-adapter data/models/qwen3_5_9b_sft_with_tools_dentex_alone_qlora --grpo-adapter data/models/qwen3_5_9b_grpo_with_tools_k4_dentex_alone/grpo-with_tools-final --precision qlora --max-tool-calls 50 --max-new-tokens 2048 --existing-results error
```
Use `--split dev --dev-split data/splits/dentex_dev.json` for ablations and the K sweep, never `test`. After an interruption repeat the command with `--existing-results resume`. Read `data/evaluations/matrix_dentex_test/matrix_summary.md`; recompute the tables from the result files with `python scripts/aggregate_eval.py --output-dir data/evaluations --dataset dentex --split test`. The two commands above this paragraph run a single cell; `evaluate_models.py` also takes `--existing-results`.

**Tool ablations (dev split, evaluation only).** Same command with a switch; only `*_with_tools` cells run and the files are tagged, so the normal results stay untouched:
```
python scripts/run_eval_matrix.py --split dev --dev-split data/splits/dentex_dev.json --conditions sft_with_tools grpo_with_tools --sizes canonical --sft-adapter <sft> --grpo-adapter <grpo> --precision qlora --remove-tools nudge_crop
python scripts/run_eval_matrix.py --split dev --dev-split data/splits/dentex_dev.json --conditions sft_with_tools grpo_with_tools --sizes canonical --sft-adapter <sft> --grpo-adapter <grpo> --precision qlora --locate-shift-frac 0.3
```
Run the unablated dev matrix for the same conditions first: the effect is ablated minus unablated on the same images. Protocol and how to read it: `docs/EVALUATION_PROTOCOL.md` section 8.

**Figures** (no model; run after `aggregate_eval.py` for the same split). Forest plots of the matrix with intervals, the paired differences, the tool-ablation effects and the box hit rate before and after `nudge_crop`:
```
python scripts/plot_eval_figures.py --output-dir data/evaluations --dataset dentex --split test --formats png pdf
python scripts/plot_eval_figures.py --output-dir data/evaluations --dataset dentex --split dev --formats png pdf
```
Files go to `data/evaluations/matrix_dentex_<split>/figures/`. A figure without rows in its table (no ablation run yet) is skipped and named in the output. What each figure shows: `docs/EVALUATION_WRITEUP.md` section 6.

**GRPO group-size (K) sweep (dev split).** Train every K from the same SFT adapter on the same seeded images, evaluate each on dev next to the SFT reference, then tabulate and plot. The sweep launcher takes the flags of `run_grpo.py` (note `--epochs`: its default is 1 here and 2 in `run_grpo.py`, so pass it explicitly to make a K of the sweep equal a single run of that K), `--checkpoint-dir` for what `run_grpo.py` calls `--output-dir`, and `--preflight`, which checks every K without loading the model.
```
python scripts/run_grpo_sweep.py --track with_tools --sft-stage dentex_alone --sft-model-dir <sft> --k-values <K ...> --epochs 2 --lr 5e-6 --kl-beta 0.04 --clip-eps 0.2 --push-every-steps 25 --checkpoint-dir data/models --canonical-resize --precision qlora --temperature 0.7 --max-new-tokens 2048 --dev-split data/splits/dentex_dev.json --seed 42 --hf-repo <repo>
python scripts/run_eval_matrix.py --split dev --dev-split data/splits/dentex_dev.json --conditions sft_with_tools --sizes canonical --sft-adapter <sft> --precision qlora --output-dir data/evaluations --existing-results resume
python scripts/run_eval_matrix.py --split dev --dev-split data/splits/dentex_dev.json --conditions grpo_with_tools --sizes canonical --grpo-adapter data/models/qwen3_5_9b_grpo_with_tools_k<K>_dentex_alone/grpo-with_tools-final --precision qlora --output-dir data/evaluations/k_sweep/k<K> --existing-results resume
python scripts/aggregate_k_sweep.py --sweep-dir data/evaluations/k_sweep --reference-dir data/evaluations --log-template "data/models/qwen3_5_9b_grpo_with_tools_k{k}_dentex_alone/grpo_training_log.jsonl"
python scripts/plot_k_sweep.py --sweep-dir data/evaluations/k_sweep --formats png pdf
```
Repeat the second-to-last evaluation command for every K (the notebook loops it). After an interrupted sweep repeat the sweep command with `--resume` (a K whose final checkpoint exists is skipped without loading the model; on a fresh session add `--resume-hf <repo>`). Tables and figures land in `data/evaluations/k_sweep/ksweep_dentex_dev_grpo_with_tools_canonical/`; read `k_sweep_summary.md`. The table reports which K are not distinguishable from the best and which of those is cheapest; it does not pick one. Add `--sizes canonical original` and run the aggregation with `--size original` for the second size. What it shows and its limits (one training run per K): `docs/EVALUATION_WRITEUP.md` section 7.

A GRPO checkpoint folder holds the trained `grpo_policy` and a frozen `reference` adapter; evaluation uses `grpo_policy` automatically when you pass the checkpoint folder. Conditions: `base_no_tools`, `base_with_tools`, `sft_no_tools`, `sft_with_tools`, `grpo_no_tools`, `grpo_with_tools`, or `all`.

---

## Section B. TPU path (Cloud TPU v5e-8, single-process SPMD)

TPU needs static shapes so XLA compiles one graph. Every sample is padded to exactly `--max-seq-len` tokens, and its images are padded to `--vision-slots` (default `5 10 4` = 19 images = 7,488 vision tokens). So **text capacity = `--max-seq-len` − 7,488**, and memory is the same at every step: a single warmup run at the chosen length is a faithful memory test.

Do not install `requirements-gpu.txt` on a TPU: it would send the linear-attention layers to GPU kernels that cannot run on XLA.

**B1. Install** (no GPU extras)
```
pip install -r requirements.txt
huggingface-cli login
```

**B2. Measure trace lengths with canonical views** (once)
```
python scripts/compute_exact_trace_lengths.py --canonical-resize --recompute --vision-slots 5 10 4
```
The summary reports the GPU cost (real tokens) and the TPU cost (text plus the static vision tokens of `--vision-slots`) separately; the TPU one is what `--max-seq-len` is checked against on TPU.

**B3. How many traces fit each static length, and does the slot budget cover them**
```
python scripts/census_vision_slots.py data/traces/train_cot_traces_dentex.jsonl --manifest data/traces/trace_token_lengths.json --seq-lens 16384 18432 20480 --vision-slots 5 10 4
```
It prints, per candidate length, how many traces fit (text + 7,488), and exits with an error listing any trace that needs more images than the slot budget.

**B4. Memory test at 16384, with and without the solver replacement** (this answers whether replacing `solve_triangular` fixes the out-of-memory)
```
python scripts/warmup_xla_cache.py --spmd --fsdp --num-cores 8 --max-seq-len 16384 --pad-vision-to-slots --vision-slots 5 10 4 --triangular-shim --lora-linear-attn
python scripts/warmup_xla_cache.py --spmd --fsdp --num-cores 8 --max-seq-len 16384 --pad-vision-to-slots --vision-slots 5 10 4 --no-triangular-shim --lora-linear-attn
```
Read the `[XLA MEMORY | ...]` lines (after forward and after backward) and look for `RESOURCE_EXHAUSTED`. The warmup also fills the XLA cache that training reuses.

The solver replacement (`--triangular-shim`) has not yet been run on a TPU: the native solver ran out of memory, and this test is the first time the replacement is exercised. Treat it as unverified until the `--triangular-shim` line completes without `RESOURCE_EXHAUSTED`.

**B5. Choose the sequence length**

| Result | What to do |
|--------|------------|
| 16384 compiles and runs | Keep it, or try a longer length to keep more traces: repeat B4 with `--max-seq-len 18432`, then `20480` (steps of 2048). Use the largest length that completes; B3 shows how many traces each keeps. |
| 16384 fails with `RESOURCE_EXHAUSTED` | Do not reduce cores. Try a smaller slot budget if B3 shows it still covers every trace (for example `--vision-slots 4 8 3`; this lowers the static vision tokens, so a shorter `--max-seq-len` keeps the same text capacity). Otherwise the next option is tiered static lengths (several graphs), which is not implemented yet. |

The same `--max-seq-len`, `--vision-slots`, `--triangular-shim` and `--lora-linear-attn` setting must be used in warmup and SFT (the LoRA targets change the compiled graph), and the same view/shape flags in GRPO.

**B6. Full SFT** (create the dev split first, as in A6a)
```
python scripts/train_sft.py --track with_tools --stage dentex_alone --spmd --fsdp --num-cores 8 --canonical-resize --pad-vision-to-slots --vision-slots 5 10 4 --triangular-shim --lora-linear-attn --mask-malformed-turns --dev-split data/splits/dentex_dev.json --max-seq-len 16384 --epochs 3 --batch-size 1 --eval-strategy epoch --save-strategy epoch --hf-repo <repo> --push-every-steps 50
```
`--gradient-accumulation-steps` defaults to 1 on multi-core TPU (effective batch 8). Output: `data/models/qwen3_5_9b_sft_with_tools_dentex_alone_bf16`.

**B7. GRPO** (same view and shape flags as B6)
```
python scripts/run_grpo.py --track with_tools --sft-stage dentex_alone --sft-model-dir data/models/qwen3_5_9b_sft_with_tools_dentex_alone_bf16 --dataset dentex --dev-split data/splits/dentex_dev.json --num-cores 8 --spmd --fsdp --canonical-resize --pad-vision-to-slots --max-seq-len 16384 --vision-slots 5 10 4 --triangular-shim --group-size 4 --epochs 2 --lr 5e-6 --kl-beta 0.04 --clip-eps 0.2 --temperature 0.7 --max-new-tokens 2048 --hf-repo <repo> --push-every-steps 25
```
The policy update runs at the same static shapes as SFT. Rollout generation does not: its shapes still vary and will recompile on XLA (see `docs/CANONICAL_VISION_AND_TPU_PIPELINE.md` section 7). Rollouts longer than `--max-seq-len` are excluded from the update and counted. `--max-images N` trains on a seeded random subset of the pool (one GRPO step = one image), `--include-healthy` keeps healthy images with ground truth `[]`.

**B8. Evaluate** on a GPU or CPU machine, as in A9.

---

## Section C. Troubleshooting

**Both paths**
- `[DATASET MASK] ... filtered out N overlength traces` is larger than expected, or a warning says the manifest was ignored: the manifest was measured with a different `--canonical-resize` setting. Re-run `compute_exact_trace_lengths.py` with the same setting as training. The summary header prints the image mode; DENTEX lengths several times larger than expected mean `--canonical-resize` was left off.
- `[DATASET] N malformed assistant turns ... masked from the loss`: expected. Those turns (empty, truncated or mis-nested JSON that the agent loop rejected) stay as context and are not training targets. `--no-mask-malformed-turns` trains on them.
- A `[TOOL IMAGE]` warning means an image had no matching tool call and a default crop was rendered. It should not appear on the current traces; send the warning text if it does.
- GRPO reward is flat and `kl` stays 0.0000 for many steps: check that `--temperature` is above 0 (rollouts of one image must differ) and that the SFT adapter was found (the log prints `[SFT-REF] ...`). With a fresh adapter an early `kl` of 0.0000 is normal.
- GRPO `--preflight` says `SFT adapter ... not found`: `train_sft.py` writes `..._<precision>` folders (for example `..._qlora`). GRPO and the K sweep search the suffix of their own `--precision` (a bf16 run does not pick up a `_qlora` adapter, and nothing switches it), so a mismatch means the adapter was trained with the other precision or lives elsewhere: pass the folder with `--sft-model-dir` (the sweep takes the same flag).
- `--preflight` says `canonical-resize vs SFT` or `dev split vs SFT` FAIL: GRPO must use the SFT run's `--canonical-resize` and `--dev-split`, otherwise the policy trains on a different image distribution or on images SFT held out.
- Any crash: run the matching smoke test (A4) on the same machine first; it separates library-version problems from data problems.

**GPU path**
- `check_qlora_load.py` fails with `(a) ... vision-tower layers are 4-bit`: `QLORA_SKIP_MODULES` no longer matches this transformers version's module names. In transformers 5.x an entry matches from the START of the dotted name, so `model.visual` works and a bare `visual` does not; run `--names-only` and add the printed prefix.
- Out of memory in the probe (A5): lower the ceiling (A6), confirm `--precision qlora` on a 24 GB card, then lower `--lora-r`. `--no-lora-linear-attn` frees only about 250 MiB; it also removes the Gated-DeltaNet adapters, so use it only to compare.
- Very slow steps and a log line that `chunk_gated_delta_rule` falls back to a PyTorch implementation: `flash-linear-attention` is not installed (or not importable). Install it from `requirements-gpu.txt`; on Windows use WSL or Linux.
- `causal-conv1d` fails to build: it is optional, skip it.
- bitsandbytes CUDA errors: run `python -m bitsandbytes`, which prints what it found, and match the installed CUDA/PyTorch build.
- Loss is `nan`: do not use `--precision fp16`; use `bf16` (A100) or `qlora` (4090, bf16 compute).
- GRPO runs out of memory during rollouts: lower `--group-size` (the no-tools track generates the whole group in one batch). The with-tools track rolls out sequentially, so its peak is the longest single rollout.
- Rollouts longer than the training contexts: the agent loop keeps every tool image in the context, whereas the SFT traces replaced older tool images with an "[Earlier tool result omitted]" marker once the context got long. Very long GRPO or evaluation episodes are therefore out of distribution and use more memory than any training example. This is an open issue (see `docs/CANONICAL_VISION_AND_TPU_PIPELINE.md` section 7).

**TPU path**
- `RESOURCE_EXHAUSTED` in warmup: do not reduce the core count. Compare `--triangular-shim` with `--no-triangular-shim`, then try a smaller `--vision-slots` if the census says every trace still fits, then a shorter `--max-seq-len` only as a last resort (it drops the longest traces).
- Host RAM fills up: `--spmd` must be on (it is the default). `--no-spmd` loads a full model copy per process.
- Every step prints a long compilation pause: shapes are varying. Check that warmup and training use identical `--max-seq-len`, `--vision-slots`, `--triangular-shim` and `--lora-linear-attn`, and that `--canonical-resize --pad-vision-to-slots` are both set. GRPO rollout generation is expected to recompile (open issue).
- `pad_vision_to_slots: trace uses N FULL images but the static budget is M`: a trace needs more images than `--vision-slots`; the census (B3) lists them. Raise the slot budget.
- `Slot-padded sequence ... exceeds target length`: text plus the static vision tokens does not fit `--max-seq-len`; B3 shows how many traces fit each length.

## Appendix: where the Gated-DeltaNet layers are

Qwen3.5 text layers alternate three Gated-DeltaNet (linear-attention) layers and one full-attention layer. In the model code a layer is full attention when `(layer_index + 1) % 4 == 0`, otherwise Gated-DeltaNet; with 32 layers that is full attention at 3, 7, 11, ..., 31 and Gated-DeltaNet at the other 24. Print the exact layers of your checkpoint:
```
python -c "from transformers import AutoConfig; c=AutoConfig.from_pretrained('Qwen/Qwen3.5-9B').text_config; print(c.num_hidden_layers, [i for i,t in enumerate(c.layer_types) if t=='full_attention'])"
```
Module names: Gated-DeltaNet layers hold `model.language_model.layers.<i>.linear_attn` (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `conv1d`, `norm`, `out_proj`); full-attention layers hold `...layers.<i>.self_attn` (`q_proj`, `k_proj`, `v_proj`, `o_proj`); every layer has `...mlp` (`gate_proj`, `up_proj`, `down_proj`). The vision tower is a separate ViT with ordinary attention. LoRA targets: `q/k/v/o_proj` (full-attention layers), `in_proj_qkv`, `in_proj_z`, `out_proj` (Gated-DeltaNet layers, `--lora-linear-attn`, default on), the MLP projections (all layers) and the vision projector. `in_proj_a` / `in_proj_b` (small decay and write gates) are not adapted. At startup the log prints `[LORA] adapted modules: {...}` with the count per kind; `--no-lora-linear-attn` turns the Gated-DeltaNet adapters off. The extra adapters add about 22M trainable parameters at r=32 (roughly 250 MiB with optimizer states).
