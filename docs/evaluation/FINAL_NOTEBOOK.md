# Final paper evaluation on 0048

Open `notebooks/VLM_Dental_Colab_Final_Evaluation.ipynb` after SFT/GRPO. Apply 0050 to
0048 / `57386da582177fda87191a2a28ed6e717ab7e9f6`; **0049 is not required**.
The notebook can clone the pinned base and apply an uploaded 0050 patch. It never pulls/resets
an existing checkout, changes old runners, starts training or kills other jobs.

Configure the profile, stage, selected K, paths and unique run ID in one block. Hub tags follow
the shared artifact names: dual_t4_fp16, rtx4090_bf16, workstation48_lora_bf16, tpu_bf16.
The Hub revision is resolved once; only exact evaluation adapters are fetched, not optimizers.
Local overrides are explicit and hashed. Four independently trained SFT/GRPO × tools/no-tools
adapters are required for the complete paper comparison. GRPO uses grpo_policy, not reference.

Start MODE=dev and SMOKE_LIMIT=2 under a smoke ID. Then use a new ID and SMOKE_LIMIT=None for
the full dev matrix and probes. K-sweep evaluates already-trained checkpoints and does not pick
a winner. After selecting K, evaluate its full dev matrix and freeze the selection. MODE=test
then requires the same checkpoint/code/software/budgets and all official test images. Partial
test runs are rejected. Never regenerate the committed dev split or tune on the test set.

The shared aggregator produces pooled exact/tooth F1, paired whole-image bootstrap intervals,
per-class/healthy/pathological results, complete-case agreement, format failures, costs and
localization. The supplementary calibration report keeps confidence coverage/raw pairs and
uses the shared ECE; its legacy exclusion of confidence zero is reported, not silently changed.
Healthy-image invention is distinct from finding-level FP/(TP+FP). No result means pending.

Rules/skills and speculative serving stay dev-only and off by default. All four harness contexts
share a checkpoint. Speculation compares one merged target across baseline/MTP/lookup and requires
its decoded-output/F1 gates; missing counters stay null. Use the companion speculative notebook
for a compatible separate serving environment and optional MTP-only training. This final notebook
only compares measured before/after files and starts no such training.

The native evaluator has no sharded 9B TPU generation. It cannot treat 8×16-GB cores as pooled HBM;
the notebook stops before loading this target on TPU. CPU/TPU aggregation works with
AGGREGATE_ONLY=True, the original evidence profile, and AGGREGATE_SOURCE_DIR pointing to a saved
phase folder with run_config.json/run_identity.json/cohort.json and JSONLs. CUDA profiles request
T4 FP16 QLoRA, 4090 BF16 QLoRA and 48-GB BF16 inference. Fit/support/speed require real runtime runs.

Identity records include adapter and YOLO hashes, code/software/dev split, image hashes and
base metadata/index plus shard sizes. They do not claim a full hash of every 19-GB base tensor.
Known training-metadata mismatches are rejected; missing historical metadata is not fabricated as
proof of dev exclusion. Archive the real metadata and assess gaps before submission.

Native matrix/probes resume matching completed episodes. Malformed/duplicate JSONL or incomplete
cohorts cannot be called final. Preserve an interrupted raw file before explicit trailing-record
repair. Harness/serving failures retain evidence and need fresh run output names.

The output ZIP includes episodes, transcripts, logs/commands, tables/plots, hashes/metadata, analysis
source and the notebook. Download from Kaggle Outputs or the Colab download cell. Optional Hub
upload is off and targets evaluations/final/<profile>/<stage>/<run>/ only. No Git commits/pushes.
Existing frontier benchmarks can be imported with original prompt/denominator; no API is called.
Specialist AP remains distinct from finding F1. 0051 adds optional paper plotting/building; primary
LaTeX fragments remain review candidates and do not silently rewrite the frozen paper table.
