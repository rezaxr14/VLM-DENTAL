# TPU evaluation (patch 0063)

Every evaluator now runs on a Cloud TPU v5e-8 (8 x 16 GB HBM) through the static-shape generator of patch 0062
(`docs/TPU_GENERATION.md`). GPU/CPU paths, flags, result files and identities are unchanged.

## What runs

| Evaluator | TPU |
|---|---|
| `scripts/evaluate_models.py`, `scripts/run_eval_matrix.py` (zero-shot-unified baseline, SFT, GRPO; with and without tools) | yes |
| `scripts/harness/run_dev.py` (rules / skills, dev split) | yes |
| `scripts/run_zero_shot.py --provider transformers` (legacy zero-shot prompt) | yes, canonical FULL view |
| speculative decoding (vLLM) | no: patch 0064 |

The notebooks (`Final_Evaluation`, `Rules_Skills`, `ZeroShot`) detect a TPU **without starting XLA** (the evaluation child process needs it) and
select the `tpu` profile. `EVAL_SIZES` becomes `["canonical"]`.

## Contract

* **Canonical letterbox views only.** Static vision slots exist only for the fixed canvases (FULL 1536x768, CROP 256x384, COMPARE 512x384).
  A TPU run needs `--canonical-resize --canonical-letterbox` and stops before loading weights without them. Original-size cells are
  `unsupported` on TPU. Results carry `image_view: canonical`, `canonical_letterbox: true`, profile tag `tpu_bf16`; compare them with
  GPU canonical results as a different cohort, not as the same one.
* **Flags.** `--tpu-generation-chunk-tokens 512 --tpu-generation-tiers 32768 49152 --tpu-vision-slots 8 16 6` (the notebooks pass them; the
  values are part of the result identity). Evaluation generates up to 16384 new tokens, so only tiers of at least chunk + 16384 are
  reachable; a `max_new_tokens` that fits no tier is an error at start-up, never lowered (Rule 19).
* **Load.** SPMD is enabled before the XLA device exists. The adapter is merged on the host, then FSDPv2 shards the merged weights while
  moving them (never `model.to(device)`, which would put 9B on one chip). Every reachable tier is compiled and warmed at start-up, so an
  out-of-memory appears in the first minutes.
* **Repetition penalty.** The generator applies `repetition_penalty` exactly like `generate` (verified equal on CPU); the zero-shot path keeps its
  1.10 default. `streamer` is accepted and ignored (tokens are read back once per step).
* **Refusals are recorded, never silent.** A prompt beyond the largest tier, or with more images than the slots, raises
  `GenerationTooLong` / `SlotOverflow`. The episode is stored with no answer (`format_ok: false`, no findings), the field
  `generation_refused` holds the reason, the run prints a count, and `*_tpu_generation_report.json` next to the results lists the images and
  tier/compile usage. Metrics count such episodes as failures; `pooled` adds `generation_refused` only when it is nonzero. A refusal in the middle
  of a multi-turn episode discards that episode's partial transcript.

## Identities and resuming

* `source_sha256` hashes `dental_agent/{agent,model,tools,evaluation,harness}`; patches 0062 and 0063 change those files, so **previously saved
  evaluation bundles no longer match for reuse**. That is the protocol working as designed.
* TPU identities add `tpu_generation` (chunk, tiers, slots). `eval_settings` of a TPU result file adds the same block; a GPU file never has it.
* `validate_resume_settings` accepts a result file written before the TPU settings existed when TPU generation is off (and letterbox is off);
  a TPU run never resumes a GPU file and vice versa.

## Not verified without a TPU

Verified offline (CPU and 8 virtual XLA-CPU devices): equality with stock `generate`, bounded compiles, sharded loading of a merged model through
`finalize_tpu_inference_model`, refusal records, flags, gating and identities. **Not** verifiable here: 32k/48k prefill attention memory on a
16 GiB chip, decode throughput, TPU-only lowering of the Gated-DeltaNet reference loop, and sharding the real 9B from host RAM. The start-up
warm-up exists to answer these first; `docs/TPU_DIAGNOSTICS.md` phases record each step.

## Commands

```
python scripts/evaluate_models.py --condition sft_with_tools --adapter-path <sft> --canonical-resize --canonical-letterbox \
  --precision bf16 --max-new-tokens 16384 --tpu-generation-chunk-tokens 512 --tpu-generation-tiers 32768 49152
python scripts/harness/run_dev.py --model-id <base> --adapter-path <sft> --precision bf16 --canonical-letterbox \
  --max-new-tokens 16384 --tpu-generation-chunk-tokens 512 --tpu-generation-tiers 32768 49152
```

Run one condition per process: a process loads one sharded model, and the matrix runner reuses it across the cells of one checkpoint (the sidecar
report is cumulative for that loaded model).
