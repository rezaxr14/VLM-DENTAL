# TPU SFT bounded loss: patch 0042

Author: Reza Nadimi <rezaxr14@gmail.com>

Base: `rezaxr14/vlm-dental` commit
`e1e33e29a99a96c7ce7bbc821d3c33d93dc202c9` ("fixing the notebook args").
This includes patches 0040/0041 and the corrected SFT launch cell.

## Apply and enable

```bash
git apply --check 0042-tpu-static-chunked-loss.patch
git apply 0042-tpu-static-chunked-loss.patch
```

No existing notebook or GPU helper is edited. CUDA retains its existing
`--loss-chunk-tokens` implementation, attention/loading controls and defaults.
The new setting is **`--tpu-loss-chunk-tokens`**, default **0** (off).
This implementation covers **SFT training, validation and AOT warmup**.
GRPO's TPU token-log-probability path is unchanged.

For TPU, keep the existing `tpu` profile, BF16, single-process SPMD/FSDPv2,
static vision slots, canonical letterboxing, `TARGET_SEQ_LEN=16384`,
`LOSS_CHUNK_TOKENS=0`, `DEBUG_STEPS=0`, and `REQUIRE_EFFICIENT_SDPA=False`.
Do not reuse the GPU chunk flag on TPU.

Add these two lines **after the SFT `cmd` list is built, before `cmd_str` and
preflight**, in your existing launch cell:

```python
if IS_TPU:
    cmd += ["--tpu-loss-chunk-tokens", "256"]
```

If using the AOT warmup cell, add the matching setting **before
`warmup_cmd_str` is built**:

```python
if IS_TPU:
    warmup_cmd += ["--tpu-loss-chunk-tokens", "256"]
```

These additions are conditional on TPU. Leave the working GPU cells/settings
alone. `256` is a starting point, not a measured 9B HBM-fit guarantee. Use the
same value for AOT and training. A changed loss implementation/chunk size needs
new compiled graphs; an old stock-loss AOT cache does not precompile this loss.
The AOT cache records the selected configuration in `tpu_loss_config.json`.
Use a separate writable cache directory for the probe/new configuration.
Do not assume that matching these flags alone guarantees the entire existing
AOT and training graphs are identical.

The CLI equivalent is to add:

```bash
--loss-chunk-tokens 0 --tpu-loss-chunk-tokens 256 \
--spmd --fsdp --pad-vision-to-slots
```

Keep the remaining model, data, canonical-view, slot, sequence and optimizer
settings exactly as intended. Warmup accepts `--tpu-loss-chunk-tokens` but not
`--loss-chunk-tokens` (that GPU flag belongs to the SFT launcher).

## Probe before loading the 9B weights

First run this **in a fresh TPU session/process**, with the repository's
matching torch/torch_xla environment:

```bash
PJRT_DEVICE=TPU python scripts/probe_tpu_chunked_loss.py \
  --num-cores 8 --chunk-tokens 256 \
  --output-dir data/tpu_chunk_probe
```

It downloads no model or data. It runs a small frozen linear head on the actual
TPU, shards the batch across eight cores and the head weight like FSDPv2, compares
loss and hidden-state gradients with a full-vocabulary CPU oracle, and repeats
fixed shapes with different labels/mask counts. An entirely masked row tests
normalization across the mesh. It fails on CPU fallback, numerical mismatch,
missing XLA loop lowering, or repeated compilation of the final loss/backward
execution. Whole-gradient host-readback compilation is counted separately from
the loss/backward graph, because diagnostic gathers can compile their own small
graphs on some XLA versions.

Reports under the output directory:

- `loss_and_backward.hlo.txt`: lowered graph, including the while loops.
- `report.json`: versions, device count, timings, compilation counts and raw
  memory readings (or an explicit unavailable reason).
- `metrics.txt`: XLA metrics/counters.

A second, opt-in test exercises a real tiny random Qwen3.5 with vision inputs,
PEFT projector adapters, activation checkpointing, the real FSDPv2 wrapper,
loss/gradient comparison and an optimizer update:

```bash
PJRT_DEVICE=TPU RUN_REAL_TPU_TESTS=1 python -m pytest \
  tests/test_tpu_chunked_loss_hardware.py -q -s
```

Neither small test proves that the full 9B transformer/vision graph fits each
16 GiB chip. Run these before an expensive full-model AOT run. If the full run
still fails, inspect its largest HBM allocations/sharding and whether the delay
occurs before or after graph execution; head chunking does not bound transformer
activations, vision slots, attention workspaces or other compiler graphs.

## Implementation

The ordinary CUDA `causal_lm_loss` is untouched. TPU opts into a labeled-forward
adapter installed on the Qwen model **before FSDPv2 wrapping**. The trainer still
calls `model(**inputs)` through PEFT/FSDP; it does not bypass the wrapper to call
an unwrapped body. The loss-only result is a named tuple with `.loss` and
`.logits=None`, so FSDPv2's tuple/Tensor backward-hook protocol remains usable.
The usual unlabeled forward remains available.

The body produces full contextual hidden states `[B,T,H]`. Labels shift by one
position, and a final ignored slot preserves static `T`. If needed, both tensors
pad to a whole number of chunks, never truncate. Each iteration projects
`[B,C,H]` through the same frozen output head and computes FP32 full-vocabulary
cross-entropy with `-100` masking. Batch dimensions stay explicit and are marked
on the existing mesh's `fsdp` axis. The summed loss divides by the **global**
shifted supervised-token count, clamped to one for an all-masked batch. This
matches the existing token-weighted accumulation scheme.

A custom first-order autograd function retains hidden states, targets and the
frozen head by reference, not per-chunk vocabulary logits/softmax activations.
Backward recomputes each chunk's probabilities, forms `softmax - one_hot`, masks
and scales by the same global denominator and upstream gradient, then multiplies
by the head weight to obtain hidden-state gradients. These flow into the
original transformer and projector LoRA adapters. It does not detach the body,
sample the vocabulary, shorten context, change optimizer-step frequency, or
train on prompt/tool labels.

Both passes use the **non-autograd core of XLA scan**, which lowers the chunk
iteration into a compact XLA `while` body. Public autograd scan's per-iteration
saved activations are deliberately avoided. The core
`torch_xla.experimental.scan._scan_impl_pytree` is experimental/private:
capability checks run before weight loading and fail clearly if the installed
API differs. There is no silent Python-unrolled fallback on XLA. CPU execution
is an explicit mathematical reference only.

Supported head: frozen plain `nn.Linear` (Qwen's normal LoRA configuration),
optionally with frozen bias. Trainable/quantized/LoRA-wrapped heads and external
loss denominators are rejected. Higher-order derivatives are unsupported; SFT
uses ordinary first-order backward. This patch does not install dependencies,
change dependency versions, or migrate the user's XLA stack.

## Memory, speed and numerical meaning

At vocabulary size 248,320, one materialized FP32 `[1,256,V]` buffer is about
**0.237 GiB**, compared with **15.16 GiB** for `[1,16384,V]`. Additional softmax,
backward and sharding workspaces still matter. These are tensor-size estimates,
not physical HBM measurements; actual compiler buffers require the TPU report.
Eight 16 GiB chips are not a single 128 GiB allocation.

Chunks count fixed sequence positions **per row**, including ignored positions.
Unlike the CUDA implementation, TPU does not dynamically gather only assistant
positions. At length 16,384 and chunk 256, each pass executes 64 iterations of
one compiled body. This bounds head memory without growing a Python-unrolled
64-copy graph. Smaller chunks reduce head workspaces but increase loop overhead;
larger chunks may improve throughput until memory pressure dominates.

The mathematical loss and first-order gradients are unchanged. FP32 reduction
order and BF16 rounding can differ from the stock implementation; bitwise
identity or identical training trajectories is not promised. Context, vocabulary,
labels, supervised-token weighting and token budgets are preserved.

## Verification scope

CPU numerical tests compare loss and gradients across chunk sizes, bias,
BF16, uneven tails, masked rows and real tiny multimodal Qwen/PEFT projector
training. Existing GPU loss/runtime and SFT smoke tests remain available.

An isolated matching PyTorch/XLA 2.8 environment also runs the actual compiler
on **eight XLA CPU software devices**: FP32/BF16 sharded head gradients match
the oracle; successive changed-label loss/backward execution compile counts are
`[1,0,0]`; a real tiny Qwen/PEFT/FSDPv2 checkpointed backward matches the CPU
reference. This is not emulated TPU hardware and supplies no TPU HBM or speed
measurement. The two explicit real-TPU tests must run on the user's hardware.

All three notebooks, the CUDA chunked loss/attention/debug helpers, shared SFT
wrappers, model/inference code and GRPO files are byte-for-byte unchanged from
the patch base. With the new flag disabled, stripping its guarded additions
from the three modified existing Python files yields the original ASTs. No
GPU model/optimizer/kernel logic has been rewritten.
