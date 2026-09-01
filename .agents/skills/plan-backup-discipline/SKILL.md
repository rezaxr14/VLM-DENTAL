---
name: plan-backup-discipline
description: Strict protocol enforcing mandatory backup and versioning of implementation plans outside the project tree before any modification, edit, or overwrite in VLM-DENTAL.
---

# Implementation Plan Backup & Versioning Skill

## Overview
This skill mandates that no implementation plan is ever modified, updated, or overwritten in-place without first saving the current version to a persistent historical backup file stored **outside the project repository**.

## Step-by-Step Workflow Before Modifying Any Plan

1. **Inspect Existing Plan**:
   - Check if `implementation_plan.md` (or the active plan artifact) already exists and contains content.

2. **Create Versioned Backup Outside the Project Repository**:
   - Copy the exact, complete content of the existing plan to:
     `<appDataDir>/brain/<conversation-id>/plans/implementation_plan_<topic>_<timestamp>.md` or `<appDataDir>/brain/<conversation-id>/plans/implementation_plan_<topic>_v<N>.md`
   - Ensure the directory `<appDataDir>/brain/<conversation-id>/plans/` exists.
   - **DO NOT** save backup plans inside the project tree (`docs/plans/`, etc.).

3. **Perform the Update**:
   - Only after the backup is confirmed written to IDE session storage, apply edits or overwrite `implementation_plan.md`.

4. **Document the Backup**:
   - Explicitly note the backup path in the IDE session directory where the prior version was preserved.

## Pre-Edit Checklist

- [ ] Does an existing implementation plan exist?
- [ ] Has the current plan been saved to `<appDataDir>/brain/<conversation-id>/plans/` outside the project repository?
- [ ] Is the project working tree kept clean without plan clutter?
