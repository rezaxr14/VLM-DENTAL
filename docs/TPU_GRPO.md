# TPU GRPO (patch 0061): chunked scoring, extra vision slots, merger training

Scope: the TPU **policy update and scoring**. Rollout generation on TPU is unchanged here (still dynamic shapes; patch 0062).
GPU/CPU paths are untouched: every new behaviour is TPU-only and behind explicit flags.

## What changed

| Area | Before | Now (TPU only) |
|---|---|---|
| Log-prob scoring | Full FP32 `[rows, T, V]` logits: ~15 GiB per row at 16k, ~30 GiB at 32k (V = 248,320) | `--tpu-logprob-chunk-tokens N`: N positions per row at a time in one XLA while-loop (0.237 GiB per 256-chunk), frozen-head recomputed backward |
| Vision projector (merger) | Frozen unconditionally on TPU (the SFT-trained merger LoRA stopped learning) | Trained iff the policy adapter carries merger LoRA, the same rule as the CUDA branch |
| Activation memory | No gradient checkpointing on TPU (CUDA-only block) | Enabled with the TPU SFT settings (`use_reentrant=True`, `preserve_rng_state=False`) when `--gradient-checkpointing` |
| Tool images over the slot budget | `ValueError` from the collator: a multi-hour run ends | Rollout dropped and counted (`n_dropped_slot_overflow`); peak demand per family logged |
| `--vision-slots` | "must equal SFT" | May be larger than SFT's; extra slots are empty and masked. Preflight reports a larger budget as INFO, a smaller one as WARN |

## Scoring maths

`dental_agent/training/tpu_chunked_logprobs.py` mirrors `tpu_chunked_loss.py` (SFT) but returns a per-token value, because GRPO's
gradient depends on a per-token weight (PPO ratio, advantage, KL), not on one scalar:

- forward: `lp[b,t] = z[b,t,y] - logsumexp(z[b,t,:])`, `z = frozen_head(hidden)` in FP32, one chunk at a time, written into a fixed `[B,T]` buffer;
- backward, with `g` the incoming per-token gradient: `dz = (onehot(y) - softmax(z)) * g`, `dh = dz @ W`, again one chunk at a time.

Worked check (a 3-token case): `z = [0, ln 3]`, target `y = 1`: `softmax = [0.25, 0.75]`, `lp = ln 0.75 = -0.2877`; with `g = 2`:
`dz = ([0, 1] - [0.25, 0.75]) * 2 = [-0.5, +0.5]`. The CPU tests compare values and gradients of the full PPO-clip + k3-KL objective with stock FP32 cross-entropy.

The installer wraps the model forward **before FSDP wrapping**; only forwards that pass `score_labels` change. `generate()` and every
other forward are untouched. `token_log_probs` uses it automatically on XLA tensors when it is installed (GPU/CPU branches unchanged).
Requires `--spmd --fsdp --pad-vision-to-slots`, a frozen plain `nn.Linear` `lm_head`, and `torch_xla.experimental.scan` (checked before the 9B weights load).

## Notebook settings (GRPO notebook, TPU profile)

```
TARGET_SEQ_LEN = 32768            # same as the SFT
VISION_SLOTS = (8, 16, 6)         # 11,904 vision tokens; text room 20,864 at 32k (SFT used 5 10 4)
TPU_LOGPROB_CHUNK_TOKENS = 256
```

`(8, 16, 6)` is a first guess, not a measurement. Every update forward encodes all 30 slots (vision-encoder compute is ~59% above the
SFT's 19 slots) and the budget cannot be tuned without data, so the log records, per step, `slot_peak_full/crop/compare` (highest image
count any rollout demanded, dropped ones included) and `n_dropped_slot_overflow`. `python scripts/summarize_grpo_log.py <log>` prints
`Vision slots: N rollouts dropped over budget; peak demand FULL a, CROP b, COMPARE c`. Lower the budget if the peaks stay far below it
(faster), raise it if rollouts are dropped (more learning signal kept).

## What the CPU/XLA-CPU tests prove, and what they do not

Proved: numerical equality of log-probs and gradients with the stock path (fp32 and bf16, masked rows, chunk sizes that do not divide
T), a compact while-loop with no full `[B,T,V]` buffer in the lowered HLO, **no recompilation when only label values change**, the
installer through PEFT + FSDPv2 + reentrant checkpointing on 8 virtual XLA-CPU devices, and drop-and-count of over-budget rollouts.
Not proved (needs the TPU; read it from the run report, `docs/TPU_DIAGNOSTICS.md`): that a 32k policy update fits a 16 GiB chip, the
real compile time, and TPU-only lowering. A GRPO policy forward is a training forward, so it carries the same attention-memory risk as the SFT.
