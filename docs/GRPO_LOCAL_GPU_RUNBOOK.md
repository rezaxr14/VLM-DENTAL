# GRPO and native local inference runtime (patch 0041)

Author: Reza Nadimi <rezaxr14@gmail.com>

Apply this incremental patch to `rezaxr14/vlm-dental` at
`8f1e4f0f2b05e00c6f652b1d32cd119620ad4d5c` (the upstream commit that already
contains SFT patch 0040). It does not require a notebook base64 runtime payload.

```bash
git apply --check 0041-grpo-local-gpu-runtime.patch
git apply 0041-grpo-local-gpu-runtime.patch
```

`git apply` applies files; it does not create a commit or store a commit author.
The patch's mail header names the author above. To commit it yourself with that
identity, use `git commit --author='Reza Nadimi <rezaxr14@gmail.com>'` after staging.

## What changed

The shared local loader now explicitly sets the dtype of non-quantized weights
as well as the NF4 compute dtype. GRPO, native Transformers zero-shot, and local
evaluation share the SFT runtime's attention selection, scoped T4 grouped-query
attention workaround, staged loading and CPU loading-thread controls. T4 uses
FP16; modern CUDA can use native BF16. Model placement and runtime settings are
logged. CPU/disk offload requires an explicit inference opt-in and is rejected
for GRPO training. Layer placement across two GPUs is one model replica; their
VRAM is not a single pooled allocation.

GRPO now prepares k-bit training while preserving the loaded dtype of frozen
vision/embedding/head weights, uses CUDA non-reentrant gradient checkpointing,
and uses gradient scaling for FP16 updates. Frozen vision blocks use the SFT
vision guard; trainable projector adapters remain trainable. Old-policy and
reference scoring respect `no_grad`. Reference adapters remain frozen and the
policy adapter is restored before checkpointed backward recomputation. The
no-tools grouped rollout repeats whole image patch matrices in group order,
rather than repeating each patch row individually.

Native decoding explicitly requests KV caching and last-position logits when
the model supports `logits_to_keep`. Current Transformers may already select
last-position logits internally; that flag alone is not evidence of a speedup.
Scoring explicitly disables the decode cache and does not inherit the
last-position-only setting. Cross-turn agent cache reuse requires an exact token
prefix match, uses the cache's actual length (including the often-uncached final
sampled token), preserves full-prompt span offsets and falls back to full image
prefill when reuse is unsafe. New image turns always prefill fully.

The local zero-shot branch preserves PIL views through the existing
`safe_process_vision_info` path, avoiding legacy Qwen vision utility patch-grid
resizing and text-only fallback. Per-request/rollout allocator purges are removed.
Local output metadata records compute dtype, attention selection, generation
seconds and tokens/second. The pipeline cache includes runtime settings and the
resolved weight folder, preventing reuse of a model with different settings.

Frontier API provider implementations and notebook option cells are unchanged.
The HTTP `local` provider is an external vLLM server; these flags affect only
`transformers` / `local_hf`, not that server. No real API benchmark was rerun.
User token limits, sampling values, image resolution controls, group sizes,
canonical-view policy and dependency version requirements are preserved.

## Explicit hardware settings

These are starting points, not hardware measurements or fit guarantees:

| Profile | Precision | QLoRA compute | Attention | SDPA repeat KV | Loader / CPU threads | GRPO scoring chunk |
| --- | --- | --- | --- | --- | --- | --- |
| Dual T4 | qlora | fp16 | sdpa | on | staged / 1 | 128 |
| RTX 4090 | qlora | bf16 | auto | auto | standard / 0 | 1024 |
| Workstation with 48 GiB | bf16 (ordinary LoRA) | n/a | auto | auto | standard / 0 | 2048 |
| TPU GRPO | bf16 | n/a | sdpa | off | standard / 0 | 0 |

`auto` attention chooses FlashAttention-2 only when all CUDA devices are SM80+
and the package imports; otherwise it selects SDPA. Import success does not prove
that every forward/backward kernel supports the installed stack. The efficient
SDPA guard refuses quadratic math fallback. FlashAttention accelerates full
attention; Qwen's separate linear-attention/convolution kernels can still fall
back to slower implementations if their compatible packages are absent.

KV repetition implements the same grouped-query attention mathematically. It
does not change Qwen's head counts or turn GQA into MQA. Force it on the T4 SDPA
path; leave modern CUDA on `auto` and TPU on `off`. This is a CUDA kernel
compatibility policy, not an architecture change to apply everywhere.

The GRPO notebook exposes `HARDWARE_PROFILE`; both single and sweep launchers,
and the evaluation matrix, receive the selected controls. The zero-shot
notebook exposes `LOCAL_HARDWARE_PROFILE` in Option 5. It defaults to
`dual_t4_fp16` to preserve the original unquantized FP16 benchmark. Select
`dual_t4` or `rtx4090` explicitly to use 4-bit weights, or
`workstation48_lora` for unquantized BF16 inference. Quantization and dtype can
change predictions: use a separate `--output` file when comparing them; do not
resume a new precision run into an old results file. Zero-shot TPU generation
is not implemented by this native CPU/CUDA launcher.

## GRPO launch example

Add these controls to the existing command, retaining its data, reference
adapter, canonical-view, rollout and checkpoint settings:

```bash
# Dual T4: append to scripts/run_grpo.py OR scripts/run_grpo_sweep.py
--precision qlora --qlora-compute-dtype fp16 \
--attn-implementation sdpa --sdpa-repeat-kv on \
--loader-mode staged --load-cpu-threads 1 \
--logprob-chunk-tokens 128 --gradient-checkpointing --require-efficient-sdpa
```

For RTX 4090 change compute to `bf16`, repeat policy to `auto`, loader to
`standard`, threads to `0`, chunk to `1024`, and attention to `auto`. For a 48 GiB
workstation use `--precision bf16` and try chunk `2048`. Ordinary LoRA retains
16-bit base weights; it does not mean full-model fine-tuning.

For the zero-shot CLI, use `--provider transformers --local-precision ...`
with the same compute/attention/loading controls. Do **not** pass
`--logprob-chunk-tokens`: zero-shot does not compute training log probabilities.
Keep `--max-tokens 16384` when reproducing the current notebook benchmark.
The evaluation scripts use `--precision` rather than `--local-precision`.

## What chunking means scientifically

GRPO needs the probability of each sampled assistant token under the current,
old and reference policies. Each probability still requires a softmax over the
**entire vocabulary**. The stock implementation projects all sequence positions
into a `[batch, sequence, vocabulary]` tensor before selecting token probabilities.
That includes prompt/tool positions that do not contribute to the objective.
The large tensor is created for loss/scoring computation, not sent to an API.

`--logprob-chunk-tokens C` computes contextual hidden states for the full sequence
first, selects supervised assistant positions, then projects at most `C` of those
positions at a time through the same vocabulary head. FP32 cross-entropy per
target gives its negative log probability. Each chunk is checkpointed during
training so vocabulary-sized intermediates are recomputed rather than retained
for backward. Outputs are reassembled into the original position order with
masked positions zeroed. PPO ratios, advantages, clipping, KL, token weighting
and optimizer-step frequency are unchanged. This is not context windowing,
truncated backpropagation, vocabulary sampling or smaller optimizer batches.

The chunked and stock paths have the same objective and mathematical gradients;
floating-point rounding can differ. Scoring now uses FP32 log-softmax/CE even
when model weights use 16-bit compute. Different hardware, precision or kernels
can also produce numerical differences, independently of chunk size.

Smaller chunks reduce peak vocabulary-logit memory but incur more projection,
CE and checkpoint launches. Larger chunks often improve throughput until memory
pressure dominates; the fastest value must be measured. Chunk size is tokens
per projection, not the number of chunks: `ceil(supervised_tokens / C)` chunks
are evaluated. Head memory is approximately proportional to `C * vocabulary`,
with additional softmax/backward workspaces. Full transformer activations,
attention and generation caches are separate memory costs; chunking cannot
make an arbitrarily long rollout or large group fit. No learning-quality benefit
is expected from selecting 128 versus 1024 at the same precision/objective.

Chunk `0` retains stock scoring and is the CLI default for compatibility.
The GRPO notebook enables a nonzero value explicitly for CUDA profiles. TPU
requires `0`: dynamic boolean selection and a varying Python chunk loop are not
an XLA static-shape implementation. A notebook line does not add TPU chunking;
this patch rejects nonzero TPU scoring chunks. A future TPU implementation needs
fixed-shape padded chunks, correct global masked normalization and gradient
comparison under the actual SPMD mesh. Existing TPU static slots, FSDP/SPMD and
full-vocabulary scoring remain available.

## Before an expensive run

1. Start from a clean kernel after changing the checkout or dependencies. Run the
   existing preflight; verify the exact SFT adapter and canonical-view metadata.
   Preflight is an input check, not a fused-kernel or VRAM test.
2. On CUDA, run `scripts/probe_sft_attention.py --help` and probe the selected
   precision/attention/KV policy. The existing probe uses representative full
   attention forward/backward and downloads no model. GRPO still needs its own
   short real run because generation caches, reference scoring and grouping add
   memory beyond SFT.
3. Use a separate debug output folder, `--max-images 1`, `--group-size 2` and
   `--hf-repo ''` for a short GRPO probe. These are explicit debug overrides,
   not new training defaults. Preserve the intended token budget and view policy.
4. Check finite loss/KL, a nonzero policy update, frozen reference parameters,
   checkpoints and peak allocated/reserved memory **on each GPU**. Measure
   rollout and scoring time separately before increasing group size or chunk.
5. If loading stalls, use `--load-traceback-seconds 60`. On T4 keep staged
   loading/one CPU thread. CPU offload can explain painfully slow inference;
   the default now fails clearly instead of silently accepting it.

## Validation and limits

The added CPU tests use real tiny random Qwen3.5 models and PEFT to compare
selected log probabilities and gradients, checkpointed reference/policy switching,
multimodal projector gradients, bounded head projection rows and saved tensors,
and cached greedy decode against stock generation. Contract tests check cache
prefix/span offsets and unchanged remote provider branches. A two-CUDA-device
head/body backward test is included and skipped without that hardware.

The existing tiny GRPO launcher smoke test now exercises chunk size 16 through
the real CLI, rollout, score, update and checkpoint flow. Its synthetic images,
random weights, capped test-only generation and deterministic stand-in rewards
check plumbing, not dental quality or real training convergence. No 9B weights,
real data benchmark, T4/4090/workstation/TPU speed or memory result was obtained
while creating this patch. The earlier successful dual-T4 SFT run is not evidence
that GRPO groups or unquantized zero-shot inference fit.
