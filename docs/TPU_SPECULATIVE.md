# TPU speculative decoding (patches 0064, 0065)

vLLM has no TPU path for this model here, so speculation runs inside the static-shape generator of patch 0062
(`docs/TPU_GENERATION.md`), in the same process as every other TPU evaluation. GPU/CPU paths, flags, result files and identities are
unchanged; the vLLM benchmark (`scripts/bench/spec_decode_bench.py`) is untouched and still serves the GPU profiles.

## What runs

| Condition | TPU | How |
|---|---|---|
| `baseline` | yes | the ordinary 0062 decode loop |
| `ngram_2`, `ngram_3`, `ngram_5` | yes | prompt-lookup drafts, greedy, exact verification (names and parameters mirror vLLM's `method=ngram`: `num_speculative_tokens=k`, lookup 1..k) |
| `mtp_1`, `mtp_2`, `mtp_3` | yes, patch 0065 | the checkpoint's native MTP head, rebuilt in `dental_agent/model/tpu_mtp.py` (Hugging Face drops the `mtp.*` tensors); `mtp_k` = `k` chained draft tokens |

Entry points: `scripts/bench/spec_decode_bench_tpu.py` (same JSON schema, same equivalence gate as the vLLM benchmark), the
`speculative` kind of `NotebookEvaluations` on the `tpu` profile, and `Speculative_Decoding.ipynb` with
`EVAL_TARGET_PROFILES = HARDWARE_PROFILE = "tpu"`. The TPU job needs no vLLM and no pre-merged checkpoint: the adapter is merged
into the base weights in memory, exactly as for the other TPU evaluations.

## Algorithm

One decode round feeds a fixed block of `W = k + 1` tokens: the last emitted token followed by `k` draft tokens (a shorter draft is
padded by repeating its last token; padding can only be rejected). The target model's own greedy token after each fed token is
`t_0 .. t_k`; with `a` the number of leading drafts equal to the model's choice, the round emits `t_0 .. t_a` (1 to `W` tokens). With
no draft (no earlier occurrence of the context suffix) an ordinary single-token step runs. Nothing about the round depends on `a`
on the host: it is a device scalar, so there is **one verification program per `k`** and one readback per round.

**Why this is not "just batch the drafts".** Qwen3.5 is hybrid. Its Gated-DeltaNet layers carry a recurrent state and a convolution
window that a rejected draft would corrupt, and `transformers` can only crop the cache of full-attention layers. The verification
forward therefore (1) runs the linear layers with the per-token recurrent rule that single-token decode uses (not the chunked rule),
recording the state after every one of the `W` tokens and the conv window, and (2) in the same XLA graph commits the state of the
last accepted token and rewinds every full-attention write position, from device scalars. Repetition penalty is applied position by
position (each position sees the drafts before it as accepted), which is what sequential decoding would do for the accepted prefix.

Cache tiers must hold `max_new_tokens + k` (the last round may write `k` rejected positions); a request that fits only without the
overrun is refused with the ordinary `GenerationTooLong` (recorded and counted as in 0063, never truncated).

## Native MTP head (`mtp_k`)

Head (one full-attention layer + dense MLP, its own KV cache; same architecture vLLM serves): for the pair (target hidden state `h_i` after
the final norm, embedding of the token `x_{i+1}` that followed it) it predicts `x_{i+2}`:
`fc(concat[pre_fc_norm_embedding(embed(x_{i+1})), pre_fc_norm_hidden(h_i)])` -> decoder layer (rotary position of `h_i`) -> `norm` ->
the target's own `lm_head`. Draft 1 comes from the last verified pair, draft `i` from (previous head output, previous draft) at the next
position, `k` times. The head's cache is filled during prefill (pairs of every prompt position) and refilled after each verified block
(pairs `(h_j, embed(t_j))` for the accepted `j`, the rest rewound), so a round is one program: verify `k + 1` tokens, roll the hybrid state
back, refill the head cache, draft `k` more; drafts never visit the host.

Weights: `mtp.*` tensors of the local base snapshot (the LoRA adapter never touches them), loaded **strictly** by name and shape into the
head built from the text config; any mismatch stops the benchmark before the first episode and prints missing / unexpected / wrong-shape
tensors. The names (`fc`, `norm`, `pre_fc_norm_hidden`, `pre_fc_norm_embedding`, `layers.0.{input_layernorm, self_attn.{q,k,v,o}_proj,
q_norm, k_norm, post_attention_layernorm, mlp.{gate,up,down}_proj}`) come from the public vLLM/MLX implementations, **not** from the real
9B checkpoint, which could not be opened offline: the first TPU run is also the first load of the real head.

What is verified offline: tokens are always the target's own greedy tokens (equality with the baseline for a randomly initialised head,
partial and full acceptances via an oracle seam, EOS, boundaries, repetition penalty, images); every head output of a decode (prefill
fill, refill, rewind, chain) equals running the head over the whole sequence at once; zero new XLA programs after settling. What is not:
that vLLM wires the pairing / hidden-state choice exactly like this. A deviation can only lower the acceptance rate (visible in the
benchmark's `acceptance_rate`), never change an output. If acceptance for `mtp_1` is near zero, suspect the head wiring first.

## Exactness is checked, not assumed

Equality with non-speculative decoding is proven on CPU (eager) and on 8 virtual XLA-CPU devices through PEFT + FSDPv2 for many
prompts, partial and full acceptances, EOS inside an accepted block, `max_new_tokens` boundaries, repetition penalty and an image
prompt with an empty slot. On a TPU the full-attention layers see a `W`-wide query block instead of one query, and XLA may reduce
that in a different order, so a last-bit difference in a near-tied logit is possible. The benchmark therefore gates every
speculative condition against the baseline of the same run on: decoded text of every turn, tool actions / arguments / outcomes,
`final_answer`, format validity and pooled exact F1. A divergent condition is kept as evidence and the run ends `failed_or_divergent`.
An episode refused under speculation but not under the baseline (prompt within `k` tokens of a tier) shows up there too.

## Measurements

Per request (from the generator, `stats["last_call"]`): prompt tokens, generated tokens, prefill seconds, decode seconds, drafted and
accepted draft tokens, verification rounds and plain steps. Episodes carry the same `measurements` block as the vLLM benchmark
(`tokens_per_second`, `acceptance_rate`, ...). Every condition is preceded by `--warmup-episodes` (default 2) full dev episodes that
are excluded from timing, because each `k` compiles its own verification program during the first rounds and XLA settles after the
third. `*_tpu_generation_report.json` records compile-time-relevant counters and refusals. TPU timings are TPU timings: do not pool them
with GPU serving numbers.

## What the tests prove, and what they do not

Proved offline: CPU equality (23 n-gram + 24 MTP tests), XLA-CPU equality and **zero new XLA programs** across prompt lengths, draft lengths and
acceptance patterns once settled (`tests/test_tpu_speculative_xla_cpu.py`; the counter used is `UncachedCompile`, because
`CompileTime` also counts per-execution handle creation), coordinator identity / command / parser wiring.

Not provable offline: the speed-up. A verification round costs one forward over `W` tokens (plus `W` sequential recurrent steps in
every linear layer) and, under FSDP, one gather of the weights, the same as a single decode step; the gain is the number of accepted
tokens per round, which depends on how repetitive the tool-use transcripts are. Prefill memory at 32k/48k, the lowering of the
per-token Gated-DeltaNet loop on a real TPU and sharding of the real 9B model are likewise only observable on the TPU.
