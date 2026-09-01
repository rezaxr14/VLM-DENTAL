# Local Command Execution & Bandwidth Discipline Rules

## 1. Zero Unprompted Local Execution (CRITICAL)
- **Rule:** The agent is STRICTLY PROHIBITED from autonomously executing local shell/terminal commands (`run_command`) that download large files, trigger long-running training/evaluation scripts, or consume local CPU/GPU/network resources without explicit, prior user consultation and approval.
- **Rule:** Never attempt to download remote model weights, dataset zip files, or run cross-validation evaluations on the local development machine unless explicitly commanded by the user.

## 2. Cloud-First Execution Invariant (CRITICAL)
- **Rule:** All heavy computations, YOLO model training, multi-fold cross-validation, and held-out benchmark evaluations MUST be delegated to **Google Colab** / cloud GPUs.
- **Rule:** When an evaluation or benchmark needs to run, provide a self-contained, copy-pasteable Colab cell snippet rather than executing it locally in the background.

## 3. Bandwidth Preservation Invariant (CRITICAL)
- **Rule:** Do NOT waste user network bandwidth by triggering multi-hundred-megabyte or gigabyte downloads locally.
- **Rule:** If local inspection of remote repository contents is needed, use lightweight metadata queries (`HfApi.list_repo_files` or `list_datasets`) rather than `snapshot_download` or bulk file downloads.

## 4. Mandatory Pre-Execution Consultation
- Before invoking any non-trivial command on the user's local terminal:
  1. Clearly state what command needs to run and why.
  2. Ask for explicit user confirmation before launching.
