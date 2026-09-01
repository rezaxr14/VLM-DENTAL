---
name: local-command-discipline
description: Strict protocol enforcing zero unprompted local command execution, cloud-first execution delegation, and bandwidth preservation in VLM-DENTAL.
---

# Local Command Execution & Bandwidth Discipline Skill

## Core Principles

1. **Zero Unprompted Execution**:
   - Never launch background downloads, heavy training, or local evaluations autonomously.
   - Do not invoke `run_command` on heavy scripts without prior consultation and explicit user confirmation.

2. **Cloud-First Delegation**:
   - Heavy tasks (YOLO training, multi-fold cross-validation, held-out target benchmark evaluations, large dataset zip extractions) belong in **Google Colab**.
   - Provide clean, robust, copy-pasteable Colab cells for cloud execution.

3. **Bandwidth Preservation**:
   - Never initiate multi-hundred megabyte or gigabyte downloads on the user's local machine.
   - For repository inspections, use lightweight metadata queries (`HfApi.list_repo_files`) instead of bulk file fetching.

## Checklist Before Calling `run_command`

- [ ] Is this command downloading large remote files (model checkpoints, dataset images)? $\rightarrow$ **STOP. Delegate to Colab.**
- [ ] Is this command running a heavy evaluation or training script? $\rightarrow$ **STOP. Delegate to Colab.**
- [ ] Did the user explicitly ask to run this exact command locally right now? $\rightarrow$ If NO, ask for confirmation first.
- [ ] Is it a lightweight git status/inspection command? $\rightarrow$ Allowed only when directly serving the user's immediate request.
