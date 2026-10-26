# SFT GPU/TPU runbook: consolidated repair and evidence

Scope: Stage-1 SFT in `scripts/train_sft.py` and `notebooks/VLM_Dental_Colab_SFT.ipynb`. Base model: Qwen3.5-9B. This is a cumulative repair against upstream commit `e12f88064858f9f26ab864988e20e80135aff3d4`, replacing patches 0037/0038/0039 and all temporary notebook troubleshooting edits from the session. It does not modify GRPO's training loop or claim GRPO has the same memory behavior.

## Install once on a clean clone

```bash
git clone https://github.com/rezaxr14/vlm-dental.git VLM-DENTAL
cd VLM-DENTAL
git checkout --detach e12f88064858f9f26ab864988e20e80135aff3d4
git apply --check /absolute/path/0040-sft-gpu-portability.patch
git apply /absolute/path/0040-sft-gpu-portability.patch
```

The pin gives a reproducible baseline even after upstream changes. For today's default branch the same SHA was freshly cloned and used to build the patch. A later upstream version needs a fresh apply check; clean application is not guaranteed forever. Do not layer this cumulative patch on top of earlier patches. Do not erase a running/modified checkout to install it: use a separate clone for the next run. No changes were pushed to GitHub.

The delivered notebook also embeds the **runtime portion** of the repair. After cloning, its bootstrap applies that payload if it matches, or recognizes that it is already installed. The standalone patch adds tests/documentation as well. A partially modified/incompatible checkout produces a clear error rather than being overwritten. Remove the old manual loading/GQA patch cells when using this consolidated notebook; they should not be stacked.

## Explicit profiles

Choose `HARDWARE_PROFILE` in the notebook configuration. There is no GPU-name guess that silently changes the requested precision. Profiles are conveniences that set visible variables; the command prints the effective values.

| Profile | Base weights / adapters | Compute | Head loss chunk C | Attention policy | Loader |
|---|---|---|---:|---|---|
| `dual_t4` | NF4 QLoRA, adapters trainable | FP16 | 256 candidate; 128 observed working | SDPA; explicit KV repetition; fused-only guard | staged CPU cast/copy, 1 CPU thread during loading |
| `rtx4090_qlora` | NF4 QLoRA, adapters trainable | BF16 | 1,024 candidate | auto: importable FA2 preferred; otherwise fused SDPA; native GQA retained | standard |
| `workstation48_lora` | **unquantized BF16 base + LoRA** | BF16 | 2,048 candidate | auto: FA2 or fused SDPA; native GQA retained | standard |
| `tpu` | unquantized BF16 base + LoRA | BF16 | **0: stock loss** | XLA/SDPA path; no CUDA attention guard | standard |

All profiles keep the 16,384-token limit. The current canonical image policy remains unchanged: GPU canonical resize without letterboxing, TPU static slots with the existing letterbox requirements. Token manifests must match their actual geometry and dataset/code fingerprint.

The label "RTX Pro 6000 48 GB" does not uniquely establish architecture. Use the probe's reported GPU name, memory and compute capability. Choose `workstation48_lora` for the requested 48-GB BF16-LoRA budget and verify its actual hardware/software combination. A newer GPU or a successful `import flash_attn` alone does not prove that its installed wheel contains working forward and backward kernels for that SM, head dimension, dtype and dropout.

Larger chunks are **unmeasured candidates**. They do not follow merely from a live dashboard reading of 9.2/10.1 GB. Longer traces, more image views, accumulation, loss buffers and optimizer allocation can change the peak. Keep the existing successful 128-chunk T4 run running; evaluate a larger chunk on the next run. See `SFT_CHUNKED_LOSS.md` for the objective and memory math.

## CLI controls

- `--precision qlora --qlora-compute-dtype bf16`: NF4 storage with BF16 arithmetic, for RTX4090 and other native-BF16 CUDA devices. This does not quantize the vision tower or output head.
- `--precision bf16`: ordinary LoRA on BF16 base weights; no bitsandbytes quantization config is supplied. QLoRA compute dtype is irrelevant to this mode.
- `--precision fp16`: explicitly FP16, including on modern GPUs.
- `--qlora-compute-dtype auto`: compatibility default; select BF16 only when all visible CUDA GPUs have native SM80+ support, else FP16. Notebook profiles specify BF16/FP16 explicitly for QLoRA.
- `--loss-chunk-tokens C`: tokens per output-head loss chunk. `0` uses the stock full-logits loss; there is no automatic OOM fallback or context reduction.
- `--attn-implementation auto|sdpa|flash_attention_2`: auto selects FA2 only if importable on SM80+ CUDA; otherwise it logs selection of SDPA. An explicit FA2 request fails if unavailable rather than silently changing to SDPA. The small probe is still required to check actual kernel support.
- `--sdpa-repeat-kv auto|on|off`: auto activates the observed T4 fix on pre-Ampere CUDA only; `on` forces HF's existing repeated-KV branch; `off` preserves the installed native GQA policy. Modern auto profiles do not inherit the T4 override. Policy is restored after training exits, including exceptions.
- `--require-efficient-sdpa`: disable the quadratic math fallback during CUDA forward and backward/checkpoint recomputation. This leaves fused Flash/efficient SDPA eligible. External FA2 is separate from this PyTorch SDPA selection.
- `--loader-mode standard|staged`: staged reads the tensor, converts CPU dtype first, then performs device transfer. The Transformers materializer is patched only inside a context during model loading and restored afterward.
- `--load-cpu-threads N`: `0` preserves the original count; positive `N` changes it only during loading and restores it even after errors. The observed successful T4 setting was 1.
- `--load-traceback-seconds 60`: periodic Python stack dumps **during loading only**. A printed `Timeout (0:01:00)!` is an inspection timer, not an exception or process kill. Default `0` disables it.
- `--debug-steps N`: stop after N completed optimizer windows with finite loss/gradient and adapter-update checks. Debug mode saves no checkpoints/uploads, including SIGTERM, and does not resume an adapter. Use a separate debug output directory. Notebook GPU profiles start with 2 windows and accumulation 1; `DEBUG_STEPS=0` restores your configured accumulation 8 and normal checkpointing.

NF4 preparation restores frozen vision/embedding/output-head weights to their loaded dtype after PEFT's upcast; normalization and other intended FP32 parameters remain FP32. Adapters and optimizer state may remain FP32. BF16 compute does not mean every tensor in the process is BF16. On FP16 CUDA, autocast and GradScaler are enabled. Token-summed losses are scaled by a fixed divisor before backward to reduce overflow risk, then the accumulation boundary restores the token-mean gradient. BF16 uses no FP16 GradScaler or that FP16-only divisor. Scaler state is saved/restored for normal FP16 CUDA checkpoints. If GradScaler skips an overflowed optimizer update, the scheduler does not advance.

## Probe actual attention before loading 9B

The standalone probe loads **only the local config.json**. It derives Q/KV head counts, head dimension and attention dropout and executes a small synthetic causal attention forward/backward on every visible CUDA GPU. It checks output/gradient finiteness and disallows math fallback. It does not download weights or establish full-length VLM memory use. The notebook runs it in a child process before the training launch; VRAM dropping to zero after this child exits is expected.

```bash
# Dual T4
python scripts/probe_sft_attention.py --model-id /path/to/Qwen3.5-9B --precision qlora --qlora-compute-dtype fp16 --attn-implementation sdpa --sdpa-repeat-kv on

# RTX4090 (select flash_attention_2 explicitly to verify that installed wheel)
python scripts/probe_sft_attention.py --model-id /path/to/Qwen3.5-9B --precision qlora --qlora-compute-dtype bf16 --attn-implementation flash_attention_2 --sdpa-repeat-kv off

# 48-GB workstation, ordinary BF16 LoRA
python scripts/probe_sft_attention.py --model-id /path/to/Qwen3.5-9B --precision bf16 --attn-implementation auto --sdpa-repeat-kv auto
```

For FA2 the probe checks the installed package's direct kernel, including backward. Real VLM wrapper/padding integration must still be checked by debug training. For SDPA it checks the actual Transformers SDPA adapter and its KV-repetition branch.

A matching wheel must fit Python ABI, PyTorch ABI/build, CUDA runtime and actual GPU architecture. This patch does not compile or automatically install FlashAttention, does not downgrade your dependencies, and does not assume every new FA3/FA4 package is a drop-in replacement for the Transformers FA2 integration. Test a newly supported backend explicitly before adding it to the choices. Follow upstream compatibility information for the exact release you install.

Qwen3.5 has both full-attention and Gated-DeltaNet layers. FlashAttention accelerates the former; `flash-linear-attention` and optionally `causal-conv1d` accelerate different work in the latter. Fixing one is not evidence that the other kernel family is installed/supported. A reference `causal_conv1d` warning explains a speed limitation, not the observed full-logits OOM. Do not install CUDA-only FLA kernels into the TPU environment. Keep existing dependency specifications intact.

## What the session actually established

| Stage | Evidence | Resolution / interpretation |
|---|---|---|
| Original CUDA OOM | Stock `Qwen3_5ForConditionalGeneration.forward -> lm_head`, 5.46-GiB allocation on GPU 1 with 4.50 GiB free; no loss-chunk argument in that command | That specific run used the full-logits path. Earlier reports of a chunk-enabled failure cannot be fully reconstructed without those traces. |
| Early chunk helper | Code inspection: body and output head could occupy different GPUs, labels/scalar accumulation lacked explicit alignment | Align hidden chunk and target with head; move scalar NLL back. |
| Apparent loading stall | Repeated dumps inside `_materialize_copy -> tensor.to(device,dtype)` at 0/760 | Staged CPU cast/device transfer plus 1 CPU thread was followed by successful loading. This supports the workaround, not a proved universal root cause. |
| Loading after workaround | First successful log: 3:14; later run: 0:27 | Loading speed can also vary with storage/page cache. Do not attribute the full speedup to code alone. |
| T4 attention failure | Q `[1,16,10964,256]`, K/V `[1,4,10964,256]`; efficient kernel rejected unequal heads, flash rejected SM75 | Use HF's explicit K/V repetition branch and retain efficient-only guard. |
| Real dual-T4 debug | Full model; traces 10,964/4,905 supervised and 8,310/1,227 supervised; head rows <=128 | Both forward/backward windows completed. All 404 gradient tensors finite; 202 nonzero in those first windows. Zero A-factor gradients at zero-initialized LoRA B are expected initially. |
| Optimizer update | First window delta 0 at warmup LR=0; second projector LoRA B max delta `5.5627532e-07` | A real adapter update occurred. This is not a downstream-quality metric. |
| Memory in debug | Lifetime peak allocated 9.03 GiB GPU0, 10.16 GiB GPU1; reservations higher | Each device has its own allocation ceiling; memory is not a pooled contiguous 30-GB space. Dashboard readings need not equal allocated/peak/reserved metrics. |
| Normal training | Latest supplied log: accumulation 8, debug disabled, epoch 1 at 8/604 batches, loss 0.7045 | The current run progressed beyond the two-step probe. No completed epoch, validation, checkpoint round-trip or clinical evaluation is supplied. |

The original output included misleading `spmd=True fsdp=True` on CUDA from TPU-oriented flags; the effective CUDA mode is single-process **layer placement** from `device_map='auto'`, not DDP replicas or CUDA FSDP. The new configuration log reports effective TPU-only booleans. Device placement is printed and CPU/disk weight offload is rejected for this training path. GPU1 carrying more decoder layers is visible in the map and explains why device0-only memory logs were insufficient.

The final consolidated implementation scopes workarounds and adds flags/metadata compared with the exact ad-hoc code used in Kaggle. CPU checks verify those refactors; the full consolidated version and larger chunks still need their own hardware debug run. Preserve the known-working current run as evidence.

## Debugging by the last successful phase

1. **Preflight fails:** resolve paths, trace images, token manifest or unsupported flags. Reuse attached local weights/images; do not re-download large assets to diagnose a path error.
2. **Weights 0/760:** turn on loading traceback timer. Inspect actual wait/cast/transfer frames. T4's recorded workaround is staged loading + load-cpu-threads 1. Do not claim every loading pause has the same cause.
3. **No available attention kernel:** read rejection warnings. On this T4/PyTorch combination, unequal Q/KV heads required explicit KV repetition. On new hardware verify dtype, head dimension, dropout, mask/stride, wheel/SM support and backward, not just imports. Do not blindly enable math fallback for a long sequence.
4. **OOM before `[HEAD]`:** transformer/vision activations or attention/linear-attention workspace; decreasing loss chunk size cannot solve a body allocation. Check actual sequence/image count and layer placement. Do not reduce the user's 16k limit as an implicit fix.
5. **OOM in head/CE:** use a smaller C and verify `[HEAD]` rows are bounded. Check head dtype/device and target alignment. Remember one FP32 `[C,V]` array is not total loss memory.
6. **OOM only in backward:** checkpoint recomputation, gradients, KV repetition and loss intermediates can peak here. Keep the efficient-attention policy active during recomputation and record both devices.
7. **Nonfinite gradients:** inspect compute precision, GradScaler state/scale, loss normalization and failing tensor name. Do not hide the problem by replacing NaNs or declaring success after forward only.
8. **OOM only at first optimizer step:** Adam state allocation is a separate phase. Inspect trainable parameter count and actual parameter/optimizer dtypes.
9. **VRAM goes to zero:** check child exit status and last line. `[DEBUG COMPLETE]` means deliberate successful exit; a traceback or killed process means failure. The notebook's tiny probe also releases memory on exit.
10. **Slow step:** distinguish transformer checkpointing, output-head recomputation, reference convolution/linear attention, image preprocessing and inter-GPU placement. Increasing C may improve only the head component; benchmark comparable batches after warmup.

For reproducibility save the complete launch command, `sft_run_meta.json`, software/GPU metadata from the probe, batch sequence/supervised counts, `[PLACEMENT]`, `[HEAD]`, `[GPU i]`, `[GRAD]`, `[UPDATE]` lines, kernel warnings and full traceback. Normal training logs now include `peak_gib_per_device`; the display shows per-device peaks. Record C and numerical precision as implementation details in experiments.

## TPU is a separate path

Use the `tpu` profile: BF16 unquantized LoRA, loss chunk 0, standard loading, no CPU loading override/timer, no forced SDPA KV repetition, no CUDA SDPA guard, no CUDA debug-step mode. Keep the existing XLA SPMD/FSDP, image-slot padding, letterbox, solve-triangular shim and canonical geometry consistent. Avoid initializing the TPU PJRT client in the notebook parent before launching the child process. Preserve the single-process multi-device setup already present.

The GPU helper selects a data-dependent number of supervised rows. Routing it directly into XLA would change shapes and can cause recompilation; this patch rejects it. Stock TPU loss can still be large. Static-shape chunked/fused loss or vocabulary sharding would need separate XLA design and real hardware verification. CUDA chunks are not a universal TPU OOM fix. Likewise, static slots add padded vision tokens; a trace manifest measured under different geometry is not valid evidence that the XLA batch fits.

Recompute stale manifests with the exact model, canonical/letterbox flags and dataset code used for the target run. Changing only GPU precision or loss C does not change tokenization, but switching GPU no-letterbox views to TPU letterboxed/static geometry does.

## Validation status of the final package

The final validation record (CPU host, 2026-10-26):

- `PYTHONPATH=final_repo OMP_NUM_THREADS=2 python -m pytest final_repo/tests/test_chunked_loss.py final_repo/tests/test_gpu_runtime.py final_repo/tests/test_gpu_debug.py final_repo/tests/test_qlora_skip_modules.py final_repo/tests/test_grpo_probe_preflight.py -q`: **50 passed, 1 skipped** (the skipped case requires two real CUDA GPUs).
- `PYTHONPATH=final_repo OMP_NUM_THREADS=2 python -m pytest final_repo/tests/test_smoke_sft_cpu.py -k 'with_chunked_loss or negative_loss_chunk' -q`: **2 passed**, exercising normal tiny training/evaluation/checkpointing plus staged-loader/forced-KV debug integration, metadata and rejection of negative chunk size.
- A separate targeted smoke selection for default flags, no-letterbox behavior and the above loss cases: **6 passed**; some cases overlap the two-case selection, so these counts must not be added as distinct tests.
- A separate pristine clone at the exact baseline passed `git apply --check`, actual patch application, and byte-for-byte equality for all 15 task files. Both standalone-notebook runtime bootstrap and its repeated execution were checked; it also recognizes an already-applied cumulative patch. All notebook code cells were parsed after IPython transformation.
- Defensive review: explicit task-file whitelist avoids unrelated assets; `Path`/`shlex.join` preserve paths; model/config/flag guards fail clearly; no new dataset cache or tokenization mutation; temporary global hooks/thread limits restore on errors; fixed ordering and seeded small-model checks support reproducibility without claiming deterministic CUDA training.

 CPU tests cover loss/gradient equivalence, real multimodal projector LoRA, chunk tails/sizes, GQA repetition equivalence, real safetensors conversion and restoration on error, CLI training/evaluation/checkpoint/debug integration, metadata and flag contracts. The two-real-GPU regression is included and skipped on this CPU host. No 9B model weights or large dataset were downloaded here.

Unverified on this host: actual RTX4090 BF16 QLoRA, 48-GB BF16 LoRA, their larger chunk candidates, installed CUDA kernel/wheel compatibility, full-length 16k worst-case samples, TPU compilation/HBM/sharding, completed epoch quality, full-size checkpoint resume and end-to-end dental benchmark scores. Passing small tests or the supplied two-window log must not be described as proof of all those properties.

## Primary references

- [Transformers SDPA integration](https://github.com/huggingface/transformers/blob/main/src/transformers/integrations/sdpa_attention.py)
- [Transformers model loading](https://github.com/huggingface/transformers/blob/main/src/transformers/core_model_loading.py)
- [PyTorch SDPA](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
- [FlashAttention upstream compatibility](https://github.com/Dao-AILab/flash-attention)
- [PyTorch checkpointing](https://docs.pytorch.org/docs/stable/checkpoint)
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
