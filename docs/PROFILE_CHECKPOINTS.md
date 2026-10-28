# Profile checkpoint routing (0045)

Applies to upstream f0a27d9 (includes 0044). Author: Reza Nadimi <rezaxr14@gmail.com>.

GRPO profile tags match SFT: dual_t4_fp16, rtx4090_bf16, workstation48_lora_bf16, tpu_bf16. The existing rtx4090 GRPO profile is an alias for SFT's rtx4090_qlora identity. Hardware/runtime numerical settings are unchanged.

The GRPO configuration downloads only the exact sft/qwen3_5_9b_sft_<track>_<stage>_<precision>__<tag> reference, failing if absent. This never falls back to another precision or profile. GRPO starts a new policy from SFT when its own namespace has no saved GRPO checkpoint, or resumes its own saved state when present. Single and sweep runs use grpo/qwen3_5_9b_grpo_<track>_k<K>_<stage>_<precision>__<tag>, with matching local folders and preflight checks. Untagged CLI callers retain legacy naming. A completed local GRPO run is skipped by existing completion logic.

Evaluation cells in SFT and GRPO use profile/stage-specific result folders. GRPO matrix evaluation defaults to the selected trained track plus both base conditions; evaluate the other trained track by selecting its TRACK and rerunning configuration. One adapter cannot represent both trained tracks. K-sweep adapters and log paths use the same tagged names as training.

Local zero-shot remains base-model inference, with no SFT adapter or optimizer resume. Its JSONL resume files now live under data/evaluations/local_zero_shot/<profile-tag>. The original unquantized T4 baseline is distinct: dual_t4_unquantized_fp16. Other local profiles match the tags above. API benchmark commands/settings are unchanged. Existing untagged zero-shot results are not silently imported into a new profile directory; new runs resume records only inside their selected profile directory.

Switching profile selects another experiment. This does not transfer an experiment between hardware or verify compute dtype against legacy metadata. To resume, keep profile, track, stage, K, data and training configuration consistent. Keep SFT's optimizer/scheduler/state files for SFT resume and complete GRPO checkpoints for GRPO resume.

Validation: all code cells of the three edited notebooks compile after IPython transformation; three dependency-light routing tests pass (distinct namespaces, legacy naming, exact snapshot selection/no fallback, unsafe tags). Patch applies to a clean worktree of the stated base. Full training/inference, remote Hub operations and physical hardware behavior have not been executed here.

Apply the patch to the source repository used by notebook clone cells before a fresh Kaggle Commit & Run. Reimporting a notebook alone does not update cloned backend scripts.
