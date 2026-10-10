# Runbook: SFT and GRPO, GPU path and TPU path

Run the commands of one section top to bottom. The notebooks (`notebooks/VLM_Dental_Colab_SFT.ipynb`, `VLM_Dental_Colab_GRPO.ipynb`) generate exactly these commands. Nothing is chosen for you: precision, view mode and sequence length are always explicit flags.

Replace `<repo>` with your Hugging Face model repo. Commands are one line each (PowerShell and bash).

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
python scripts/compute_exact_trace_lengths.py --canonical-resize --recompute
```

**A3. See how many traces fit each ceiling, and write the 4 longest traces as a probe set**
```
python scripts/select_longest_traces.py --traces data/traces/train_cot_traces_dentex.jsonl --manifest data/traces/trace_token_lengths.json --top 4 --output data/traces/probe_longest.jsonl
```

**A4. Smoke test** (tiny random model; catches crashes and library-version problems, not memory)
```
python scripts/smoke_test_sft.py --traces data/traces/train_cot_traces_dentex.jsonl -- --canonical-resize --precision qlora
```
Expect `[SMOKE] PASSED`. Stop and report the traceback if it fails. The same check exists for GRPO and evaluation (4-bit, as on a 24 GB card):
```
python scripts/smoke_test_grpo.py --traces data/traces/train_cot_traces_dentex.jsonl --track with_tools --eval-precision qlora -- --canonical-resize
```
Expect `[SMOKE-GRPO] PASSED`.

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

**A7. Full SFT** (use the same `--precision` and `--max-seq-len` as A5; other hyperparameters as set in the notebook)
```
python scripts/train_sft.py --track with_tools --stage dentex_alone --canonical-resize --precision qlora --lora-linear-attn --mask-malformed-turns --max-seq-len 16384 --epochs 3 --batch-size 1 --gradient-accumulation-steps 8 --eval-strategy epoch --save-strategy epoch --hf-repo <repo> --push-every-steps 50
```
The adapter is saved to `data/models/qwen3_5_9b_sft_with_tools_dentex_alone_<precision>` (for example `..._qlora`).

**A8. GRPO** (same `--canonical-resize` and `--precision` as SFT). GRPO does not look for the precision-suffixed folder, so pass the SFT adapter explicitly:
```
python scripts/run_grpo.py --track with_tools --sft-stage dentex_alone --sft-model-dir data/models/qwen3_5_9b_sft_with_tools_dentex_alone_qlora --dataset dentex --group-size 4 --epochs 2 --lr 5e-6 --kl-beta 0.04 --clip-eps 0.2 --temperature 0.7 --canonical-resize --precision qlora --hf-repo <repo> --push-every-steps 25
```
GRPO on a GPU has no sequence-length ceiling: rollout length is set by the agent loop, and memory is dominated by generation. Watch `nvidia-smi` through the first step. If it runs out of memory, report the step it reached.

**A9. Evaluate at original size** (no `--canonical-resize`, whichever way the model was trained)
```
python scripts/evaluate_models.py --condition sft_with_tools --adapter-path data/models/qwen3_5_9b_sft_with_tools_dentex_alone_qlora --dataset dentex --split test --precision qlora
python scripts/evaluate_models.py --condition grpo_with_tools --adapter-path data/models/qwen3_5_9b_grpo_with_tools_k4_dentex_alone/grpo-with_tools-final --dataset dentex --split test --precision qlora
```
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

**B5. Choose the sequence length**

| Result | What to do |
|--------|------------|
| 16384 compiles and runs | Keep it, or try a longer length to keep more traces: repeat B4 with `--max-seq-len 18432`, then `20480` (steps of 2048). Use the largest length that completes; B3 shows how many traces each keeps. |
| 16384 fails with `RESOURCE_EXHAUSTED` | Do not reduce cores. Try a smaller slot budget if B3 shows it still covers every trace (for example `--vision-slots 4 8 3`; this lowers the static vision tokens, so a shorter `--max-seq-len` keeps the same text capacity). Otherwise the next option is tiered static lengths (several graphs), which is not implemented yet. |

The same `--max-seq-len`, `--vision-slots`, `--triangular-shim` and `--lora-linear-attn` setting must be used in warmup and SFT (the LoRA targets change the compiled graph), and the same view/shape flags in GRPO.

**B6. Full SFT**
```
python scripts/train_sft.py --track with_tools --stage dentex_alone --spmd --fsdp --num-cores 8 --canonical-resize --pad-vision-to-slots --vision-slots 5 10 4 --triangular-shim --lora-linear-attn --mask-malformed-turns --max-seq-len 16384 --epochs 3 --batch-size 1 --eval-strategy epoch --save-strategy epoch --hf-repo <repo> --push-every-steps 50
```
`--gradient-accumulation-steps` defaults to 1 on multi-core TPU (effective batch 8). Output: `data/models/qwen3_5_9b_sft_with_tools_dentex_alone_bf16`.

**B7. GRPO** (same view and shape flags as B6)
```
python scripts/run_grpo.py --track with_tools --sft-stage dentex_alone --sft-model-dir data/models/qwen3_5_9b_sft_with_tools_dentex_alone_bf16 --dataset dentex --num-cores 8 --spmd --fsdp --canonical-resize --pad-vision-to-slots --max-seq-len 16384 --vision-slots 5 10 4 --triangular-shim --group-size 4 --epochs 2 --lr 5e-6 --kl-beta 0.04 --clip-eps 0.2 --temperature 0.7 --hf-repo <repo> --push-every-steps 25
```
The policy update runs at the same static shapes as SFT. Rollout generation does not: its shapes still vary and will recompile on XLA (see `docs/CANONICAL_VISION_AND_TPU_PIPELINE.md` section 7). Rollouts longer than `--max-seq-len` are excluded from the update and counted.

**B8. Evaluate** on a GPU or CPU machine, at original size, as in A9.

---

## Section C. Troubleshooting

**Both paths**
- `[DATASET MASK] ... filtered out N overlength traces` is larger than expected, or a warning says the manifest was ignored: the manifest was measured with a different `--canonical-resize` setting. Re-run `compute_exact_trace_lengths.py` with the same setting as training. The summary header prints the image mode; DENTEX lengths several times larger than expected mean `--canonical-resize` was left off.
- `[DATASET] N malformed assistant turns ... masked from the loss`: expected. Those turns (empty, truncated or mis-nested JSON that the agent loop rejected) stay as context and are not training targets. `--no-mask-malformed-turns` trains on them.
- A `[TOOL IMAGE]` warning means an image had no matching tool call and a default crop was rendered. It should not appear on the current traces; send the warning text if it does.
- GRPO reward is flat and `kl` stays 0.0000 for many steps: check that `--temperature` is above 0 (rollouts of one image must differ) and that the SFT adapter was found (the log prints `[SFT-REF] ...`). With a fresh adapter an early `kl` of 0.0000 is normal.
- Any crash: run the matching smoke test (A4) on the same machine first; it separates library-version problems from data problems.

**GPU path**
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
