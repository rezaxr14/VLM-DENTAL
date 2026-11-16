# Model size: Qwen3.5-9B (default) or Qwen3.5-4B

`DENTAL_MODEL_SIZE=9b|4b` (unset = `9b`, behaviour unchanged). One size per run.

**In the notebooks the hardware profile name is the only selector.** `HARDWARE_PROFILE = "tpu"` is Qwen3.5-9B exactly as before; `"tpu_4b"` is the same recipe with Qwen3.5-4B; every GPU profile has the same `_4b` form (`dual_t4_4b`, `dual_t4_fp16_4b`, `quadro_rtx8000_4b`, ...). Two named variants carry their weight layout: `dual_t4_ddp_fp16` (one full fp16 replica per T4, torchrun) and `tpu_4b_dp` (a full bf16 copy per v5e core instead of FSDPv2 shards). The notebook derives the model size, the `--data-parallel` / `--replicate-weights` launch flags and the checkpoint names from that one name (`dental_agent.utils.model_size.select_profile`); the environment variable is only what the scripts read.

| | 9B | 4B |
|---|---|---|
| Hub id / local folder | `Qwen/Qwen3.5-9B` / `Qwen3.5-9B` | `Qwen/Qwen3.5-4B` / `Qwen3.5-4B` |
| Local-folder env var | `QWEN35_9B_LOCAL_DIR` | `QWEN35_4B_LOCAL_DIR` |
| Artifact prefix | `qwen3_5_9b_*` | `qwen3_5_4b_*` |
| Text layers / hidden / MLP | 32 / 4096 / 12288 | 32 / 2560 / 9216 |
| Layer pattern, heads, KV heads, head_dim, linear-attn dims | identical | identical |
| Embeddings | untied | tied |
| Vision depth / hidden / FFN / merger out | 27 / 1152 / 4304 / 4096 | 24 / 1024 / 4096 / 2560 |
| MTP head | 1 layer | 1 layer |
| 16-bit weights | about 18 GiB | about 9.5 GiB (**estimate** from the shapes, not measured) |

What it changes: Hub id, local folder, local-dir env var, checkpoint / Hub / result names, the 16-bit memory guard, the GRPO activation
estimate, the disk plan. A 4B run with a 9B folder (or the reverse) is refused before any weight is read (`qwen_folder_problems`
compares `text_config.hidden_size`). A 4B run never searches the size-less legacy `checkpoints/sft_*` folder (9B only).

What it does not change: LoRA targets (matched by module name, same model class), vision slots and token budgets (patch 16, merge 2:
the same tokens per image), tools, prompts, the harness, the trace generator (`GENERATOR_MODEL` stays a separate setting).

KV-cache and linear-attention state per token are the same size in both models (same KV heads, head_dim and linear dims). The 4B frees
weight memory and shrinks per-token hidden / MLP activations; it does not make attention cheaper per token.

Every result of a 4B run is a separate cohort: baseline, SFT, GRPO (K = 1, 4, 8, 16) and the harness runs must be produced at 4B.
Measure before relying on any speed or fit claim: peak memory and step time at your real `TARGET_SEQ_LEN`.
