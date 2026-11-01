# TPU static-shape generation (patch 0062)

Scope: **rollout and evaluation text generation on TPU**. Opt-in (`--tpu-generation-chunk-tokens N`); GPU/CPU never set it and keep
`model.generate` untouched.

## Why

`model.generate` grows the KV cache and the attention mask by one position per token. Under PyTorch/XLA every new shape is a new
graph, so a 1,000-token reply would compile ~1,000 programs (and every new prompt length compiles a new prefill). On a Kaggle TPU,
with no interactive session, that is the difference between a run and a queue-wasting stall.

## Design

| Piece | How it stays static |
|---|---|
| KV cache | One `StaticCache(C)` per call. `C` is the smallest of `--tpu-generation-tiers` (default 8192, 16384, 32768) that holds `prompt + max_new_tokens`. Nothing is truncated: a request that fits no tier raises `GenerationTooLong` and the rollout is dropped and counted |
| Prefill | Left-padded prompt fed in chunks of `--tpu-generation-chunk-tokens` (default 512). Chunk offset, rotary positions and masks are **device tensors**, so every chunk after the first reuses one graph |
| Decode | One graph. Cache write index, rotary position and mask derive from device counters; one `[batch]` token tensor is read back per step to detect EOS |
| Sampling | On device: greedy, temperature, top-k, top-p (Gumbel-max over the top-k values). `temperature` from the call, `top_k`/`top_p` from the model's `generation_config`, like `generate` |
| Vision | Same canonical static slots as training (`--vision-slots`). Real images first, unused slots are zero patches that are encoded and ignored; the tool images a turn needs are gathered into the prompt by index, so the vision graph is one shape. Identical rows (the K copies of a no-tools group) share one vision pass |
| Batch | Rows are padded to a multiple of the core count by repeating row 0 (per-step cost is weight traffic, the same for 1 or 8 rows), so the batch shards cleanly over the mesh |
| Hybrid Qwen3.5 | Left padding is safe because the recurrent state is still zero where the padding sits and padded positions are zeroed before the linear-attention layers; padded queries attend only to themselves so no softmax row is empty; M-RoPE positions come from the model's own `get_rope_index` on the host |

`generate_local` is the single choke point (rollouts, evaluation, the API pool): when the model carries the config it dispatches to
`tpu_generate_local`, a drop-in with the same keyword arguments and return shape. Each turn re-prefills (no cross-turn cache reuse on
TPU), so `past_key_values` must be `None`.

## Programs and the start-up warm-up

Per `(batch rows, tier)`: first prefill chunk, later prefill chunk, decode, prefill sampler. XLA settles on its final program set after
the third chunk and third decode step (the first programs see freshly created cache buffers), so the warm-up uses at least 4 chunks and
6 decode steps. After that **nothing depends on prompt length, `max_new_tokens`, image pixels, or the sampling seed**
(measured on 8 virtual XLA-CPU devices: 0 new compilations).

`--tpu-generation-warmup` (default on) runs one real vision prompt per tier and batch size (`1` and `--group-size`) at start-up,
including a prefill that fills the largest tier. A compile or HBM problem therefore appears in the first minutes of the run report
(`docs/TPU_DIAGNOSTICS.md`, phase `tpu_generation_warmup`), not after hours in the queue. A tier that fails is **disabled**
(`[TPU GEN WARMUP FAILED]`), so prompts that need it are dropped and counted instead of ending the run; if every tier fails, the run
stops with the exception text.

## What is dropped, and how it is counted

`GenerationTooLong` (prompt + `--max-new-tokens` beyond the largest tier) and `SlotOverflow` (more tool images than `--vision-slots`):
- with tools: that one rollout is dropped (its siblings continue);
- no tools: the whole K-batch is dropped;
- logged per step as `n_dropped_generation`, printed in the step line, and summarised by `scripts/summarize_grpo_log.py`.
Token budgets are never lowered (`--max-new-tokens` is what you pass).

## Notebook (GRPO, TPU profile)

```
TPU_GEN_CHUNK_TOKENS = 512
TPU_GEN_TIERS = (8192, 16384, 32768)
```

## What the tests prove, and what they do not

Proved (CPU eager and 8 virtual XLA-CPU devices): token-for-token equality with stock `generate` for single prompts of several
lengths, left-padded unequal batches, prompts longer than one chunk, image prompts with an empty slot, mixed image/text rows, K
identical rows with batch padding, EOS stopping and padding; the same equality through PEFT + FSDPv2 on the mesh; zero recompiles for
a new prompt length / `max_new_tokens` / pixels / sampling seed; sampling statistics (top-k/top-p/temperature); drop-and-count.

Not proved (needs the TPU, read it from the warm-up lines and the run report): that a 32k prefill and its attention fit a 16 GiB
chip (no fused TPU SDPA kernel is assumed, so a chunk attends to the whole `C`-sized cache with a mask), real decode throughput with
FSDP-sharded weights (each step gathers the weights; expect tens of milliseconds per token per layer-gather, slow by design),
the compile time of each tier, and TPU-only lowering of the reference Gated-DeltaNet loop.
