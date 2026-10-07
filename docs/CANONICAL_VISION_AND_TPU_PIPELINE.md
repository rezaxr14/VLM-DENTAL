# Canonical Vision Views, TPU Static Slots and GPU/TPU Runs

**Status:** implemented and unit/CPU-smoke tested; not yet run on a GPU or TPU. Target hardware: Cloud TPU v5e-8 and single CUDA GPUs (RTX 4090 with QLoRA, A100 with LoRA).

## 1. Why

- Canonical resizing existed only as a plan; earlier runs fed native-resolution images to the model.
- XLA compiles one graph per distinct `(pixel_values, image_grid_thw)`. Static vision-slot padding gives TPU training a single graph.
- Qwen3.5's Gated-DeltaNet layers call `torch.linalg.solve_triangular`, which GSPMD cannot partition (replication, HBM exhaustion). A matmul replacement is installed on XLA tensors only.

## 2. Geometry

The model sees aspect-preserving views (LANCZOS, centred on a constant black canvas); tools always run on the native image, so all coordinates stay in native pixels.

| Family  | Canvas (W×H) | Produced by | LLM tokens |
|---------|--------------|-------------|-----------:|
| FULL    | 1536×768     | base image, `denoise`, `window_level`, `enhance_contrast` | 1152 |
| CROP    | 256×384      | `zoom_crop` | 96 |
| COMPARE | 512×384      | `contralateral_compare` | 192 |

Static slot budget (`--vision-slots FULL CROP COMPARE`, default `5 10 4`) = 19 images = 7,488 vision tokens; with `--pad-vision-to-slots` (TPU only; rejected with an error on GPU/CPU, which pad dynamically) every sample is `text + 7,488` tokens. Sizes live only in `dental_agent/utils/canonical.py`.

## 3. Decisions

- **Explicit, deterministic configuration.** Every behaviour is a CLI flag; the notebooks hold literals and pass them verbatim. No environment-variable behaviour, no TPU-dependent defaults, no silent precision switching (`bf16` stays `bf16`, `qlora` is chosen by the user; a hint is printed when a small GPU runs bf16 LoRA).
- **Defaults:** `--max-seq-len 16384` everywhere; `--spmd` on (inert off-TPU); `--canonical-resize` and `--pad-vision-to-slots` off unless passed.
- **GPU (RTX 4090 / A100):** `--canonical-resize`, dynamic padding. No slots, no shim, no SPMD (slot padding exists only to avoid XLA recompilation). Native resolution is not viable for with-tools traces (several full-size images per trace). 24 GB cards use `--precision qlora`; A100 uses `bf16` (`train_sft.py`, `run_grpo.py` and `evaluate_models.py` all take `--precision`).
- **TPU:** `--canonical-resize --pad-vision-to-slots --vision-slots 5 10 4 --triangular-shim --spmd --fsdp`. Padding does not change the loss: the tail is causally after every real token, with `labels=-100` and `attention_mask=0` (verified on the real Hugging Face Qwen3.5 code, loss and gradients identical to numerical precision).
- **Fail loudly:** padded mode raises on a non-canonical grid, an over-budget trace or an over-length sequence (truncation would desynchronise image tokens from vision features).
- **Length manifest** (`compute_exact_trace_lengths.py`) records its resolution mode and per-trace vision-token counts; a manifest from the other mode is ignored with a warning.
- **GRPO** runs single-process SPMD with the same view/shape flags as SFT. With `--pad-vision-to-slots` each policy update runs at the SFT static shapes (one rollout per core); rollouts longer than `--max-seq-len` are excluded and counted.
- **Evaluation** runs on original-size images regardless of training hardware (`evaluate_models.py --canonical-resize` is off by default).

## 4. What changed

| Area | Files |
|------|-------|
| Geometry, letterbox, slot budget | `dental_agent/utils/canonical.py` |
| Dataset / collator | `dental_agent/training/sft.py` (canonical views, static slots, bounded view cache, manifest lookup) |
| TPU shim | `dental_agent/training/xla_patches.py` |
| Vision input extraction | `dental_agent/model/backbone.py` |
| Launchers | `scripts/train_sft.py`, `warmup_xla_cache.py`, `run_grpo.py`, `run_grpo_sweep.py`, `compute_exact_trace_lengths.py`, `evaluate_models.py` |
| GRPO | `dental_agent/training/grpo.py`, `dental_agent/agent/loop.py` |
| Tooling | `scripts/census_vision_slots.py`, `scripts/smoke_test_sft.py`, `requirements-gpu.txt` |
| Notebooks | `notebooks/VLM_Dental_Colab_SFT.ipynb`, `VLM_Dental_Colab_GRPO.ipynb` |

## 5. Runbook

```bash
# Token-length manifest (once per resolution mode)
python scripts/compute_exact_trace_lengths.py --canonical-resize --recompute

# GPU extras (CUDA only, never on TPU)
pip install -r requirements-gpu.txt

# Pre-flight on the target machine (tiny random model, minutes; checks crashes and library versions, not memory)
python scripts/smoke_test_sft.py --traces data/traces/train_cot_traces.jsonl -- --canonical-resize --precision qlora

# RTX 4090
python scripts/train_sft.py --canonical-resize --precision qlora ...
# A100
python scripts/train_sft.py --canonical-resize --precision bf16 ...
# TPU v5e-8 (warmup first; run it with and without --triangular-shim to compare HBM)
python scripts/warmup_xla_cache.py --spmd --fsdp --pad-vision-to-slots --vision-slots 5 10 4 --triangular-shim
python scripts/train_sft.py --spmd --fsdp --canonical-resize --pad-vision-to-slots --vision-slots 5 10 4 --triangular-shim ...

# GRPO / evaluation: same view flags as the SFT checkpoint
python scripts/run_grpo.py --canonical-resize [TPU: --spmd --fsdp --pad-vision-to-slots ...]
python scripts/evaluate_models.py
```

Each training script prints a `[CONFIG]` line with the effective settings.

## 6. Bugs found and fixed

- **Duplicate crops.** When one assistant turn called a tool several times (e.g. three `zoom_crop`s), the dataset rendered every image from the *first* call's arguments, so the model saw copies of one crop next to text about several teeth. The k-th image of a tool now uses the k-th call. Earlier SFT runs trained on this; re-run before quoting results.
- **Length filter never filtered.** The dataset looked up manifest keys that `compute_exact_trace_lengths.py` does not write, so over-length traces reached the collator and were truncated (cutting off the assistant answer). All key forms are now matched.
- **GRPO optimizer** was rebuilt for every image, resetting Adam's moments. It is now created once per run.
- **Default `--spmd` crashed every non-TPU run** (`torch_xla` import); it is now gated on the real backend. `--pad-vision-to-slots` was silently ignored off-TPU; it now applies everywhere.
- **Notebook sweep mode** passed `--model-id`, which the sweep launcher rejected.
- **GRPO precision was implicit:** 4-bit loading came from `configs/default.yaml` and the non-4-bit GPU path loaded fp16. GRPO now has an explicit `--precision {bf16,qlora}` (bf16 means bf16).
- **Silent precision switches removed** (bf16→fp16 on GPUs without native BF16 is now a hint; QLoRA requested on TPU is an error instead of a switch).
- **Unbounded crop cache** (native-resolution images held in host RAM) replaced by a bounded LRU of finished views.

## 7. Open and deferred

- No GPU/TPU run yet: HBM/VRAM fit, GSPMD sharding of the shim and single compilation are unverified.
- GRPO `generate()` still produces dynamically shaped graphs on XLA; static update shapes do not remove decode recompiles.
- A single 16,384 length excludes the longest (multi-finding) traces; measure with `scripts/census_vision_slots.py`. Alternatives: a larger length or tiered graphs.
- Qwen3.5 is hybrid: LoRA targets `q/k/v/o/gate/up/down` reach attention in only the full-attention layers; the Gated-DeltaNet layers (`in_proj_*`, `out_proj`) get MLP LoRA only. Candidate: an explicit `--lora-targets` flag.
- Evaluation at original size: only `evaluate_models.py` has the switch; `evaluation/ablations.py`, `sweep.py`, `batch_runner.py` and `rewards/judge.py` call `run_agent` natively. A canonical-trained checkpoint evaluated at original size sees a different input distribution than it trained on; decide whether to report both.
- Optional ablation: letterbox vs plain stretch.

## 8. For the paper

*Radiographs are presented to the model as fixed-size, aspect-preserving views (Lanczos resampling onto a constant black canvas): 1536×768 for the base image and whole-image operators, 256×384 for `zoom_crop`, 512×384 for `contralateral_compare`. Tools execute on the original-resolution image and all coordinates are in original pixel space.* Limitations: small crops are upscaled and letterboxed (effective resolution is bounded by the source pixels); static padding is an engineering device with no effect on the loss; report how many traces the length filter excluded at each stage.
