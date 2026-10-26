# What output-head loss chunking does, and why it preserves SFT

This describes the implementation in `dental_agent/training/chunked_loss.py`. It is supervised-token **output-head chunking**, not sequence truncation, independent context windows, vocabulary sampling, an approximate softmax, or a change to the trace dataset. The training sequence limit remains 16,384 tokens.

## 1. SFT predicts a distribution at each supervised position

For a tokenized multimodal trace, the causal transformer computes hidden states `H` of shape `[B,T,D]`. With teacher forcing, position `t` sees the ground-truth prefix and predicts token `t+1`. The causal mask prevents future-token access. Images, system prompts, tool outputs and earlier assistant tokens still participate in context according to the model's architecture and causal mask.

The model does not first generate a complete guessed trace and then compare two strings. Taking an argmax to produce a guessed token would discard distribution information and is not differentiable in the way needed for this loss. Instead, the language-model head produces a vector of vocabulary scores for every supervised prediction:

\[
z_t = W h_t + b, \qquad z_t \in \mathbb{R}^{V}.
\]

The target `y_t` is the single correct next-token ID. For one supervised token, cross-entropy is

\[
\ell_t = -\log p_\theta(y_t\mid x_{\leq t})
        = -z_{t,y_t} + \log \sum_{v=1}^{V}\exp(z_{t,v}).
\]

This explains why a single target ID does not imply that only its score matters: the denominator compares the correct token against every vocabulary alternative. Stable implementations use log-sum-exp rather than directly exponentiating arbitrarily large values. PyTorch `cross_entropy` takes raw logits and integer target IDs; it combines the needed normalization and negative log-likelihood internally. Do not apply a separate softmax first.

For example, if the next correct token is the token corresponding to `36`, assigning it score 8 is insufficient evidence of a good prediction: another token may have score 12, or all competitors may be much lower. The loss depends on the distribution, not just the correct token's raw score. Tokenization is model-specific: text such as `36` may also consist of more than one token.

For an unweighted supervised position, the logit gradient is

\[
\frac{\partial\ell_t}{\partial z_{t,v}}
= p_\theta(v\mid x_{\leq t}) - \mathbf{1}[v=y_t].
\]

Even when the output head is frozen during LoRA, its weights transmit this gradient to hidden states and then to trainable adapters. The head need not itself be trainable for full-vocabulary probabilities to matter.

## 2. Why the stock path allocates so much

A general-purpose `ForConditionalGeneration` forward supports returning logits for callers as well as computing a loss. Its usual implementation projects **all sequence positions** through `lm_head` before applying ignore-index masking in the loss. A label of `-100` usually suppresses the corresponding loss contribution; it does not automatically suppress the earlier head projection.

That yields a logits tensor `[B,T,V]`. For this checkpoint, `V=248,320`, `D=4,096`, and batch size is 1. A 16,384-position FP32 logits tensor alone occupies

\[
\frac{16384\times248320\times4}{2^{30}} = 15.15625\ \mathrm{GiB}.
\]

A 16-bit head output is half that, but converting it to FP32 for cross-entropy adds another large allocation, and backward and loss internals need additional workspace/saved values. Actual peak memory depends on the implementation and tensor lifetimes; a fixed multiplier such as "exactly three times logits" is not a reliable measurement.

Earlier descriptions of "sending thousands of logits back" were imprecise. The observed original failure was a large allocation on GPU 1 inside `lm_head`. It was not a network transfer to a server or inherently a transfer to CPU. Layer placement may require inter-GPU transfers, but this patch transfers small hidden-state chunks/targets to the output head and a scalar loss back, rather than returning a complete `[T,V]` output to the training loop.

Exact full-vocabulary cross-entropy needs the normalization over all vocabulary scores in some form; it does **not** require every sequence position's full score vector to be stored simultaneously. More specialized fused linear-cross-entropy methods can also tile the vocabulary dimension and reduce logits materialization further. That is a possible future optimization, not what this patch currently implements.

## 3. The implemented transformation

1. Run the full multimodal transformer body once, obtaining `[B,T,D]` hidden states. Preserve inputs such as `mm_token_type_ids`, image grids and pixel values.
2. Shift labels: hidden position `t` predicts `labels[t+1]`.
3. Select positions whose shifted label is not `-100`. This keeps all supervised assistant positions, including valid tool-call text. Existing malformed-turn masks are preserved.
4. Divide the selected hidden rows into groups of at most `C=loss_chunk_tokens` rows.
5. For each group, calculate `lm_head(hidden_chunk)`, FP32 cross-entropy with `reduction='sum'`, and add the scalar result.
6. Divide the summed loss by the **total supervised-token count**, not by the number of chunks.
7. Use non-reentrant activation checkpointing around each head/loss group during training. Its large saved intermediates are discarded and recomputed as needed during backward. The checkpoint inputs and hidden-state graph are retained; detaching them would break training.

Checkpointing is essential to bound training memory: a Python loop that computes chunk losses but retains every chunk's logits graph until backward can still accumulate a large memory footprint. Separately, the transformer layers already use gradient checkpointing. Output-head checkpointing does not mean that the entire transformer is rerun once per output chunk.

The body and output head may live on different GPUs. The implementation explicitly aligns each head input and target with the head, and moves each scalar NLL to the loss accumulation device. This was missing from the early chunk helper.

## 4. Equality of the objective and gradient

Let `S` be the set of supervised prediction positions and let `S_1,...,S_M` be a partition of it into chunks. Then

\[
L = \frac{1}{|S|}\sum_{t\in S}\ell_t
  = \frac{1}{|S|}\sum_{j=1}^{M}\sum_{t\in S_j}\ell_t.
\]

Differentiation is linear, so in exact arithmetic

\[
\nabla_\theta L
= \frac{1}{|S|}\sum_j\sum_{t\in S_j}\nabla_\theta\ell_t.
\]

Every selected position retains its **original full-context hidden state** and its **entire vocabulary**. No context boundary is inserted at the chunk boundary. Masked context positions can still receive indirect gradients through their influence on supervised positions. Excluding their own output-head loss is already the intended ignore-index objective.

The final chunk is often shorter. Taking a plain mean of chunk means would overweight that final short chunk. This implementation sums token losses and divides by the total token count. The outer trainer also normalizes accumulated gradients by the total shifted supervised-token count across microbatches, rather than treating unequal-length traces as equal-size token batches.

The code does not perform an optimizer step between loss chunks. Batch size, accumulation schedule, token masking, learning rate and the optimizer are separate controls.

## 5. Does chunk size change model quality?

**It does not intentionally change the learning objective or remove information.** There is no mathematical reason to expect better or worse quality merely from selecting 128 versus 1,024 output rows per chunk while holding everything else fixed.

**It is not a promise of identical trained weights or identical clinical metrics.** Floating-point summation order, matrix multiplication shapes/kernel selection, BF16/FP16 rounding and nondeterministic GPU reductions can differ. Checkpointing must also reproduce stochastic operations correctly; the loss helper uses PyTorch's default RNG-preserving checkpoint behavior. Current head computation is a linear projection plus cross-entropy, with no stochastic dropout in the head itself.

Changing FP32 frozen embeddings/head to BF16/FP16 is a separate numerical change from chunking. NF4 QLoRA versus BF16 LoRA is another separate change. So are FlashAttention versus another attention kernel, the T4 precision choice, and changing gradient accumulation. Do not attribute all differences between two runs to `C` if these other settings also changed.

Evidence included here: tiny real Qwen3.5 CPU tests compare loss and gradients against the stock path, including multimodal projector LoRA, uneven final chunks and multiple chunk sizes. User-provided T4 logs show finite gradients and a real adapter update. Neither proves equivalence of downstream dental diagnosis metrics. For a controlled academic comparison, hold data order, seeds, masks, optimizer/schedule, accumulation, numerical precision and attention backend fixed; compare loss and gradient errors on the same batch, then compare held-out metrics across repeated seeds when resources permit.

Suggested methods wording:

> We computed full-vocabulary causal cross-entropy over supervised assistant positions using checkpointed output-head chunks of C positions. Transformer context and the set of supervised target tokens were unchanged. Chunk losses were summed and normalized by the total supervised-token count. This changes memory scheduling rather than the intended objective; finite-precision implementations may differ numerically.

Report the actual `C`, base-weight/compute precision, GPU models, attention implementation and software versions. Do not describe this patch as truncated BPTT, sampled softmax, vocabulary pruning, a clinical performance improvement, or the Cut Cross-Entropy algorithm.

## 6. Chunk size, chunk count, memory and speed

The CLI sets **tokens per chunk**, not number of chunks. If there are `K` supervised positions, `M=ceil(K/C)`.

For the observed sample with `K=4,905` and vocabulary size `248,320`:

| C: rows per chunk | Number of chunks M | One FP32 `[C,V]` array, GiB |
|---:|---:|---:|
| 128 | 39 | 0.118 |
| 256 | 20 | 0.237 |
| 512 | 10 | 0.474 |
| 1,024 | 5 | 0.947 |
| 2,048 | 3 | 1.895 |
| 4,096 | 2 | 3.789 |

These are calculated sizes for **one** array, not measured total VRAM. Head output, FP32 conversion, CE intermediates/gradients, model/adapter/optimizer state, transformer activations, attention workspace and allocator reservation also occupy memory.

Smaller `C` lowers peak output-loss memory and causes more calls/recomputation overhead. Larger `C` uses more memory and may improve GEMM efficiency and reduce Python/kernel overhead. Speed need not improve monotonically. The total supervised vocabulary work is broadly similar, plus checkpoint recomputation; chunking does not remove all output-layer compute.

Use the observed-safe 128 setting for the current T4 run. The final notebook offers 256 on T4, 1,024 for RTX4090 QLoRA and 2,048 for the 48-GB BF16-LoRA workstation as **larger candidates to test**, not measured guarantees. Increase only on the next diagnostic run; do not interrupt an otherwise useful training run just to change this.

## Sources and implementation references

- [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
- [PyTorch logsumexp](https://docs.pytorch.org/docs/stable/generated/torch.logsumexp.html)
- [PyTorch checkpointing](https://docs.pytorch.org/docs/stable/checkpoint)
- [Cut Your Losses in Large-Vocabulary Language Models](https://arxiv.org/abs/2411.09009): related fused-loss work, not the implementation shipped here.
- Project implementation: `dental_agent/training/chunked_loss.py`; integration: `scripts/train_sft.py`; evidence: `tests/test_chunked_loss.py` and the session summary in `SFT_GPU_RUNBOOK.md`.
