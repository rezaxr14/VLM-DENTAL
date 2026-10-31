# TPU run report and the 32k length probe

Kaggle TPU runs here are commit-and-run only: no interactive session, hours of queue, and the failure is read afterwards.
`dental_agent/training/tpu_diagnostics.py` makes every run leave evidence. It is **off unless `--tpu-report-dir` is passed**
(the TPU notebooks pass it; GPU profiles never do). It imports no `torch_xla` itself and never raises into training.

## What a run leaves in `--tpu-report-dir`

| File | Content |
|---|---|
| `environment.json` | package versions (torch, torch_xla, libtpu, transformers, peft...), relevant env vars, host RAM, argv |
| `events.jsonl` | begin/end/mark per stage with seconds, host RSS, XLA compile-count delta, device memory |
| `snapshots.jsonl` | XLA compile count and time, `aten::` CPU-fallback counters, device memory (steps 1-3, then every `--tpu-report-step-snapshot-every`) |
| `heartbeat.jsonl` | host RSS / free RAM / current stage and step every `--tpu-report-heartbeat-seconds` (host only, no XLA calls) |
| `tpu_info.log` | periodic `tpu-info` output: the only source of true per-chip HBM |
| `stacks.txt` | Python stacks of all threads on native abort, on SIGTERM, and every `--tpu-report-stack-seconds` |
| `alerts.txt` | e.g. "XLA recompiled after warm-up" |
| `xla_metrics_*.txt` | full torch_xla metrics report at failure/finish |
| `final.json` | `ok` / `crashed` (stage + traceback) / `terminated` |

With `--tpu-report-hub-repo` (an existing repo; never created) the JSON/JSONL/TXT/LOG files are pushed every
`--tpu-report-upload-seconds` and at the end, under `--tpu-report-hub-subfolder`. Weights (`*.safetensors`, `*.pt`) are never uploaded.

## Reading a dead run

1. **`final.json` exists**: `status` and `stage` say what failed; `RESOURCE_EXHAUSTED` in `error` means HBM, and `tpu_info.log` shows the peak before it.
2. **No `final.json`**: the process was killed without Python getting control (queue timeout, OOM-killer, hard abort). Read the last line of `heartbeat.jsonl` (stage, step, host RSS) and the last block of `stacks.txt` (where the main thread was blocked: a compile, a data load, a collective).
3. **Slow but alive**: `snapshots.jsonl` compile count must stop growing after step 3. If it grows, `alerts.txt` names it; the shape that varies is the bug (a non-static input).
4. `aten::_local_scalar_dense` in the fallback counters is benign (loss `.item()`). Other `aten::` entries are ops running on host and worth investigating.

## AOT ladder and 32k

The TPU notebooks set `TARGET_SEQ_LEN = 32768` (GPU profiles stay at 16384) and warm up `TPU_AOT_BUCKETS = (16384, 32768)`: one
static graph per length, ascending, each wrapped in its own report stage (`aot_bucket_16384`, `aot_bucket_32768`). A failure at
32768 after 16384 succeeded tells you the largest length that fits a chip; then lower `TARGET_SEQ_LEN` and the ladder together
(the ladder must be ascending, unique and contain the target).

Arithmetic at vocabulary 248,320 (tensor sizes, not measured HBM):

| | 16,384 | 32,768 |
|---|---:|---:|
| full FP32 logits `[1,L,V]` | 15.16 GiB | 30.3 GiB |
| 256-position chunk (chunked loss) | 0.237 GiB | 0.237 GiB |
| text room after 7,488 vision tokens | 8,896 | 25,280 |

The chunked loss removes the logits term, but attention-score memory grows 4x if SDPA falls to the math path, and activations
grow linearly. Whether 32k fits a 16 GiB chip is only answerable on the TPU; the ladder is built to answer it in one run.

## Testing

`tests/test_tpu_diagnostics.py` runs anywhere. `tests/test_tpu_diagnostics_xla_cpu.py` needs
`PJRT_DEVICE=CPU CPU_NUM_DEVICES=8 RUN_XLA_CPU_TESTS=1` and **must run in its own process** (every XLA-CPU test file calls
`xr.use_spmd()` and two of them in one process segfault). XLA-CPU proves compile and recompile behaviour, not HBM fit.
