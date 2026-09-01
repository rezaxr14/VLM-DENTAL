# Implementation Plan Versioning & Backup Invariant Rules

## 1. Mandatory Pre-Modification Plan Backup (CRITICAL)
- **Rule:** Before modifying, updating, or rewriting ANY implementation plan (`implementation_plan.md` or any architectural/feature plan) — **even for changing a single line** — the agent MUST FIRST save a full copy of the existing plan to a persistent, versioned backup file.
- **Rule:** Plan backups MUST be stored **OUTSIDE the project working tree** in the IDE artifact/session directory (`<appDataDir>/brain/<conversation-id>/plans/` or `<appDataDir>/brain/<conversation-id>/scratch/`).
- **Rule:** NEVER save temporary or backup implementation plans into the project repository (e.g., `docs/plans/`, `src/`, etc.) to prevent repository pollution and noise.

## 2. Backup File Naming Convention
- All backups in `<appDataDir>/brain/<conversation-id>/plans/` must be saved with descriptive names or timestamps:
  - `<appDataDir>/brain/<conversation-id>/plans/implementation_plan_<topic>_<YYYYMMDD_HHMMSS>.md`
  - `<appDataDir>/brain/<conversation-id>/plans/implementation_plan_<topic>_v<N>.md`

## 3. History Preservation & Audit Trail
- Every modification to an implementation plan must explicitly state:
  1. The path in the IDE session storage where the previous version was backed up.
  2. The exact diff or reason for the update.
- If a user requests restoring a previous plan, retrieve it directly from `<appDataDir>/brain/<conversation-id>/plans/` without data loss.
