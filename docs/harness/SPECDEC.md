# Speculative decoding and serving efficiency (Phase 3)

## Evidence and compatibility report — 2026-10-28

| Question | Evidence / status |
| --- | --- |
| Installed serving version in the build environment | Neither vLLM nor SGLang installed; no CUDA device exposed. `probe_environment.py` records actual versions on the notebook machine. No 4090 fit or speed has been measured here. |
| Qwen3.5-9B vision + native MTP | The official vLLM recipe describes the dense multimodal 9B model and native MTP (recipe minimum vLLM 0.17.0). The public model config has one MTP layer and its index includes `mtp.*`. This establishes documented support, not hardware qualification of our dental workloads. |
| N-gram speculation | The vLLM speculative-decoding guide documents `method=ngram`, draft token count, and prompt lookup bounds. Its exact interaction with this multimodal, hybrid-attention checkpoint must pass our startup/equivalence probe. |
| LoRA + speculation | vLLM documents backbone LoRA and experimental tower/connector support for some multimodal models. That is not a guarantee for every module adapted by this repository or every speculation method/version. Qwen3.5 multi-LoRA/MTP routing is discussed in an upstream RFC. This benchmark deliberately uses **merged** BF16 weights; it does not claim that all unmerged combinations are supported or unsupported. |
| Fine-tuning native MTP | Speculators documents Qwen3.5-9B MTP-only training and stitching. Current preprocessing accepts on-policy multimodal conversations via the render endpoint. Its generic response-regeneration CLI rejects multimodal presets; this project therefore regenerates real tool episodes with its existing multimodal loop. Exact render/extraction behavior remains version-gated. |
| 4090 memory | Official index total is 19,306,216,416 bytes (~17.98 GiB). Analytical full-attention KV lower bound at 16,384 tokens, batch 1, BF16 is 0.5 GiB from the public config. Neither number includes GDN state, visual/prefill activations, workspaces, CUDA graphs, fragmentation or MTP overhead. Fit on 24 GB is **unverified**. The notebook does not shorten token budgets or silently quantize to force fit. |

Primary sources (checked 2026-10-28):

- https://recipes.vllm.ai/Qwen/Qwen3.5-9B
- https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/config.json
- https://huggingface.co/Qwen/Qwen3.5-9B/raw/main/model.safetensors.index.json
- https://docs.vllm.ai/en/latest/features/speculative_decoding/
- https://docs.vllm.ai/en/latest/features/speculative_decoding/mtp/
- https://docs.vllm.ai/en/latest/features/lora/
- https://github.com/vllm-project/vllm/issues/44826 (proposal, not a support guarantee)
- https://docs.vllm.ai/en/latest/features/per_request_metrics/
- https://github.com/vllm-project/vllm/blob/main/docs/features/speculative_decoding/acceptance_metrics.md
- https://docs.vllm.ai/projects/speculators/en/v0.8.1/user_guide/tutorials/train/
- https://docs.vllm.ai/projects/speculators/en/latest/reference/speculators/data_generation/preprocessing/
- https://docs.vllm.ai/projects/speculators/en/latest/cli/response_regeneration/

## Hardware profiles

| Profile | Inference experiment | Serving dtype | GPUs / TP |
| --- | --- | --- | --- |
| `dual_t4` | QLoRA FP16 rules/skills; merged 16-bit speculative target | FP16 | 2 / TP=2 |
| `rtx4090_qlora` | QLoRA BF16 rules/skills; merged 16-bit speculative target | BF16 | 1 / TP=1 |
| `rtxpro6000_48gb` | BF16 LoRA, 48 GB card | BF16 | 1 / TP=1 |
| `workstation48_lora` | Alias for the existing RTX Pro checkpoint identity | BF16 | 1 / TP=1 |

RTX Pro retains `workstation48_lora_bf16` as the published SFT/GRPO tag. The TPU rules/skills profile remains available; this CUDA vLLM benchmark does not claim TPU or Turing speculation qualification. T4 uses FP16 serving, because it has no native BF16 support. Its optional offline MTP trainer selects explicit FP32 hidden-state/autocast compute: Speculators documents BF16 and FP32 training, not FP16 gradient-scaled training (https://docs.vllm.ai/projects/speculators/en/latest/cli/train/). 4090 and RTX Pro profiles use BF16 training compute. The actual vLLM/CUDA/version combination must pass startup and equivalence; an unsupported mode is recorded as failed, never silently replaced.

## Reproduce after training

Use `notebooks/VLM_Dental_Colab_Speculative_Decoding.ipynb` in a **separate runtime**, not alongside the user's active training jobs. It contains isolated checkout/auth/profile/adapter selection, separate serving environments, environment probe, CPU merge and byte audit, planned commands, dev benchmark, red-pending table generation, optional on-policy MTP training, before/after acceptance report, and export. The candidate pins vLLM 0.31.0 / Speculators 0.8.1 are explicit, not a claim of local compatibility; keep the actual version report and server logs. Scripts do not alter repository dependency specifications.

```
python scripts/bench/probe_environment.py --vllm-executable /path/to/vllm --output paper/generated/specdec_environment.json
python scripts/bench/merge_adapter.py --base /path/to/base --adapter /path/to/trained/adapter --output /new/merged
python scripts/bench/check_mtp.py --base /path/to/base --checkpoint /new/merged --output paper/generated/mtp_merge_audit.json
python scripts/bench/spec_decode_bench.py --checkpoint /new/merged --vllm-executable /path/to/vllm \
  --dev-split data/splits/dentex_dev.json --limit 2 --output paper/generated/spec_decode_bench.json
python scripts/paper/generate_serving_table.py --input paper/generated/spec_decode_bench.json
```

The benchmark's one command serializes baseline, MTP N=1,2,3 and ngram k=2,3,5. Each uses the same target checkpoint, greedy sampling, seed, system prompt, tool execution, geometry, budgets, warmup and fixed dev image list. Ngram k controls draft count and maximum lookup window (minimum 1). The benchmark cannot use the test split. Do a tiny dev smoke before a full dev run. An absent baseline fails the run; a failed speculative condition is recorded and subsequent conditions are attempted. Unsupported configurations are not silently relabeled as baseline.

## Checkpoint hygiene

The merge is from the original 16-bit base into a **new** directory on CPU, merging the selected SFT adapter or GRPO `grpo_policy` only. The original QLoRA run remains untouched. HF loaders may discard native MTP tensors they do not implement: `preserve_mtp` copies those original serialized tensor bytes into a dedicated safetensors shard and rebuilds the index. The audit requires the same MTP tensor set, dtype, shape and SHA-256 bytes. Presence alone does not prove a trained head is useful; absence may cause a serving error or unsupported initialization rather than any guaranteed acceptance rate. No invented zero-percent claim is made.

Merged BF16 inference is a different numerical target from the earlier 4-bit inference. The equivalence gate is against the **same merged target's non-speculative baseline**, not a claim of identity to the prior QLoRA serving path.

## Measurements and correctness

Per episode JSON records generated token counts from server usage, total wall time, request wall time, server prefill/decode phase time, tool time, draft/accepted counts, acceptance rate, sampled whole-device GPU memory, complete request metrics, and raw/action/final output. Exact request payload hashes and server commands are retained. Server version, local packages, hardware and checkpoint file identity are recorded. Warmups are excluded. Sampling GPU memory every 0.2 s may miss transient peaks; it is not allocator telemetry or per-model memory. Exclusive server ownership is required for Prometheus counter deltas to be attributable.

Prefill is read from the server's prefill-time counter, never equated to TTFT. Decode uses the phase counter or the documented per-request decode field. TTFT is retained separately. Missing counters remain null. HTTP transport, lossless PNG encoding, metric scraping and loop overhead are included in total episode latency; tool time is separate. Acceptance aggregates accepted/drafted tokens, not a mean of request rates. The table's prefill/decode shares exclude tool/network time. All comparisons use the same instrumentation.

The conservative greedy gate requires exact decoded text across all turns, identical tool calls/arguments/outcomes, identical final answer, valid formats and unchanged pooled dev exact F1. Divergences are saved and return exit code 2. The chat endpoint exposes no generated token IDs in this implementation, so the report explicitly marks token-ID equality unverified. Do not claim token-ID equality from re-tokenizing text. Failed/divergent conditions produce red todo table cells rather than claimed speedups.

## Optional MTP fine-tuning

1. Start the official Speculators `scripts/launch_vllm.py` wrapper against the merged dental target with `--target-layer-ids 32` in the serving environment. The notebook only stops the server process group it owns; it never uses `pkill`.
2. `regenerate_train.py` selects **train** images represented by canonical with-tools trace files, excludes fixed dev membership, and re-executes the existing unhinted model/tool loop. It does not expose trace ground truth or teacher responses. Output is on-policy multimodal conversations plus provenance; incomplete/length-truncated episodes are excluded and counted.
3. `mtp_finetune.py` plans (default) or executes (explicit `--execute`) the official `prepare-data`, `generate-offline-data`, `train --speculator-type mtp`, and `stitch-mtp` commands. It validates provenance. On one GPU, run prepare/extract while the owned server is alive, stop it, then run offline train/stitch. No base SFT or GRPO training is started. No image stripping fallback is permitted if a version cannot render/extract multimodal states.
4. The stitch audit checks that all non-MTP target bytes remain unchanged and that MTP tensors remain present. Repeat the identical benchmark against the stitched checkpoint and use `compare_acceptance.py` to report before/after rates only with paired settings and successful equivalence/F1 gates.

Actual fine-tuning, device compatibility, acceptance improvement, dev F1 and serving speed are **[pending]**. Build-time tests use tiny serialized tensors, synthetic images/scripted responses and a local test HTTP server. They verify plumbing, not physical hardware performance. The generated table is not automatically inserted into the frozen experiment section.
