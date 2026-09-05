---
trigger: always_on
---

# VLM-DENTAL Master Architecture & Rules Index

This repository contains the full agentic training, trace generation, and grounding pipeline for Vision-Language Models on dental panoramic radiographs.

All agents must adhere to the modular rules in `.agents/rules/`:

1. **[FDI Notation & Clinical Annotation Semantics](dentex_and_fdi_notation.md)**:
   - FDI two-digit notation (11-48) is mandatory; `dentex_row_to_fdi()` in `dentex.py` is the single source of truth for DENTEX 0-index conversion.
   - Set-level multi-finding completeness (`match_multi_findings()`) is required; never truncate with `.iloc[0]`.
   - Never guess unverified dataset annotations; use honest `NotImplementedError` stops.

2. **[Agent Tools & Model Architecture](agent_tools_and_models.md)**:
   - All tools must be registered in `dental_agent/tools/registry.py` and `dental_agent/agent/tool_dispatch.py`.
   - Dynamic real-time tool execution via LangGraph loop (`langgraph_loop.py`).
   - Unified `Qwen/Qwen3.5-9B` training backbone; zero hardcoded models in Python code (`.env` single source of truth).

3. **[Trace Generation & Verification Pipeline](trace_pipeline_and_api.md)**:
   - Canonical verified trace path: `data/traces/train_cot_traces.jsonl`.
   - Decoupled generation and verification phases with resume tracking.
   - Fail-fast API error handling with explicit `IGNORE_429=true` opt-in.
   - Zero real API calls for automated tests.

4. **[Defensive Coding & Planning Invariants](defensive_coding_and_planning.md)**:
   - Strict planning mode enforcement (`implementation_plan.md` required before any code modification).
   - Preservation of working functionality (no deleting capabilities or parameters).
   - 5-point adversarial self-critique (path normalization, null safety, caching, mutability, determinism).
   - Surgical asset downloads with local cache verification.

5. **[Git Commit Discipline](git_commit_discipline.md)**:
   - Zero unauthorized/automatic git commits.
   - Single-commit squashing when commits are explicitly requested.
   - Zero `git push` by agents.

6. **[Local Command & Bandwidth Discipline](local_command_and_bandwidth_discipline.md)**:
   - Zero unprompted local command execution; mandatory consultation before launching commands.
   - Cloud-first delegation: heavy training, cross-validation, and evaluations belong in Google Colab.
   - Zero bandwidth waste: no downloading multi-gigabyte models or datasets to the local development machine.

7. **[Implementation Plan Versioning & Backup](implementation_plan_backup_discipline.md)**:
   - Mandatory plan archiving: before modifying or rewriting ANY implementation plan (even for a single line), always save the full prior version to the IDE session directory (`<appDataDir>/brain/<conversation-id>/plans/`).
   - Never overwrite active plans in-place, and never pollute the project repository with backup plans.

8. **[Visual Verification Discipline](visual_verification_discipline.md)**:
   - Zero-hallucination image checking: before claiming an image contains generated visual elements (bounding boxes, charts, etc.), the agent MUST visually verify the image itself using the `view_file` tool.
   - Never assume an image was rendered correctly just because the code executed without errors.

9. **[Token Constraints & Hyperparameter Invariants](token_and_hyperparameter_invariants.md)**:
   - Zero tampering with user token limits or budgets (`LOCAL_MAX_TOKENS`, `max_tokens`, 16384 headroom).
   - Never lower, clamp, or alter user-specified token headroom or dependency versions (`transformers>=5.0.0`).

