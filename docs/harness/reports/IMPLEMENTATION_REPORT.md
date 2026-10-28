# Harness implementation report — 2026-10-28

Author: Reza Nadimi <rezaxr14@gmail.com>. Base: f97354e (profile routing / patch 0045).

## Delivered sequence

1. `harness/rules`: versioned extracted contract, loader, default-OFF exact renderer, reward/verifier/default-weight implementation inventory and drift tests, discrepancy overview, saved synthetic regression fixtures. All core, optional judge, legacy and composite reward definitions are inventoried; legacy discrepancies are reported without changing active math. The three distinct no-tools prompts remain unchanged rather than being silently unified.
2. `harness/skills`: five sourced procedural skills, default-OFF index/on-demand and full-injection modes, shared-dispatcher harness action outside the eight diagnostic tools, optional episode event logs, paired dev experiment runner, full Rules + Skills notebook. User-requested review badges were removed from all final skill/rule/notebook source. Sources and actionable instructions remain.
3. `harness/specdec`: merged-checkpoint MTP byte preservation/audit, read-only capability probe, managed serial vLLM benchmark, strict decoded-output/tool/final-answer equivalence gates, honest phase timing and missing metrics, MTP on-policy training-data regeneration and official offline trainer/stitch workflow, before/after acceptance comparison, pending-aware Serving efficiency table generator, full Speculative Decoding notebook.

## Runtime preservation

The normal agent's defaults are unchanged: callers do not opt into a harness, reply backend or timing observer. New trajectory metadata is omitted with the harness off. SFT/GRPO training scripts and implementations, trace generator, core/legacy/judge/composite rewards, eight-tool registry, original evaluation scripts and `paper/sections/04_experiments.tex` are byte-identical to the base. No checkpoint weights, user processes, training data or test images were modified. No test records/images/metrics were used to construct skills or rules. No model/dataset weights were downloaded during development.

The protocol proposal is `docs/harness/proposals/frozen_protocol_amendment.patch`. It is NOT applied. The pending serving table is NOT inserted into the frozen paper section. Nothing was pushed.

## Profiles

Both notebooks include dual T4, RTX 4090, RTX Pro 6000 48GB, and the existing workstation alias with matching SFT/GRPO tags. Rules/skills also retain TPU. The CUDA speculative benchmark explicitly refuses TPU rather than pretending it selects a TPU backend.

| Hardware | Checkpoint tag | Rules/skills | Speculative serving | Optional offline MTP training |
| --- | --- | --- | --- | --- |
| Dual T4 | dual_t4_fp16 | QLoRA FP16 | FP16, two GPUs / TP=2 | Explicit FP32 hidden-state/autocast compute |
| RTX 4090 | rtx4090_bf16 | QLoRA BF16 | BF16, one GPU | BF16 |
| RTX Pro 6000 48GB | workstation48_lora_bf16 | BF16 LoRA | BF16, one GPU | BF16 |
| TPU | tpu_bf16 | BF16 inference profile | Not implemented in CUDA benchmark | Not claimed |

The RTX Pro spelling is `rtxpro6000_48gb`; `workstation48_lora` remains a tag-compatible alias. No 96GB assumption is made. All token/context budgets are explicit and never reduced to force memory fit. Turing/vLLM speculative compatibility remains a real startup/equivalence test, not a claimed qualification.

## Flags and defaults

- `HarnessConfig.rules_from_file=False`.
- `HarnessOptions.skills=False`, `skills_mode='off'`; modes `index_only` (index + on-demand) and `full_inject` are explicit.
- `run_agent(..., harness=None, reply_fn=None, tool_observer=None)` leaves existing entry points on their original path.
- The new dev runner explicitly selects conditions; no existing runner enables them.
- Speculative decoding exists only in the new serving scripts/notebook; normal inference is unchanged.
- MTP fine-tuning plans commands unless `--execute` is supplied. Notebook flags for installs/merge/fine-tuning are explicit; fine-tuning defaults OFF.

## Validation and limits

- Final focused tests: 52 passed, including golden prompt variants, default-OFF loop equivalence, real parser/dispatcher with scripted generation, skill loader/index/on-demand/full logging, synthetic dev episode plumbing, train/test provenance rejection, MTP serialized bytes, HTTP metric parsing, lossless PNG transport, divergence gates, CLI help and all requested notebook profiles; existing reward/tool and shared-loop tests are included.
- Both complete notebooks compile. `verify_paper.py` passes with the existing unreferenced-label warning. Protected source/protocol byte checks pass.
- Full suite was attempted after each phase. Initial collection failed on missing dependencies. After installing test dependencies in this isolated environment, one full pass completed with 633 passed, 13 failed, 7 skipped. A later bounded full rerun timed out at 240 seconds; it is not represented as passing.
- The 13 failing tests were rerun on the untouched base and modified checkout under the same updated dependencies: both report the same 10 failures and 3 passes. The first dependency failure (Google SDK) and the two notebook syntax/dependency cases were resolved by dependency installation. Remaining baseline failures are listed below; they were not fixed by changing active training/evaluation code.
- No physical GPU/TPU, vLLM/SGLang execution, trained dental checkpoint benchmark, MTP fine-tuning or clinical outcome measurement occurred here. Serving speed, device fit, acceptance before/after, and dev F1 remain pending in generated artifacts. The synthetic traces are plumbing regressions, not an archived real-trace clinical verifier audit.

## Remaining baseline failures

- `tests/test_dentex_normal_loader.py::TestDentexNormalLoader::test_dentex_normal_local_loading`
- `tests/test_fsdp_wrapper.py::test_setup_hardware_never_switches_requested_precision`
- `tests/test_hf_checkpoint_sync.py::test_mock_upload_checkpoint_to_hf`
- `tests/test_k_sweep.py::test_sweep_resume_skips_a_finished_k_and_passes_every_setting`
- `tests/test_local_base_model.py::test_every_model_id_the_notebook_passes_is_the_resolved_one[VLM_Dental_Colab_SFT.ipynb]`
- `tests/test_local_base_model.py::test_sft_manifest_warmup_launch_and_eval_cells_receive_the_resolved_model_id`
- `tests/test_local_base_model.py::test_zero_shot_provider_loads_the_local_folder_while_the_model_name_stays_the_hub_id`
- `tests/test_sft_hub_notebook.py::test_full_configuration_and_training_command_match_actual_parser[dual_t4-qlora-fp16-256]`
- `tests/test_sft_hub_notebook.py::test_full_configuration_and_training_command_match_actual_parser[rtx4090_qlora-qlora-bf16-1024]`
- `tests/test_tool_ablation.py::test_runner_runs_only_with_tools_cells_for_an_ablation_and_uses_tagged_paths`

## Reproduce

```
OMP_NUM_THREADS=2 python -m pytest tests/test_harness_rules.py tests/test_harness_skills.py tests/test_harness_inference.py tests/test_harness_specdec.py tests/test_unified_agent_loop.py tests/test_rewards.py tests/test_tools.py -q
python -m pytest -q
python scripts/paper/verify_paper.py
```

Use the two new notebooks in isolated runtimes after training. Full commands and primary serving references are in `SKILLS.md` and `SPECDEC.md`. New outputs use fresh experiment paths; no existing experiment is overwritten. The benchmark records exact commands, environment, settings, paired IDs, request hashes, errors and divergences. All unsupported/missing measurements remain null or red todo; none is filled with invented numbers.
