---
name: git-commit-discipline
description: Strict protocol enforcing zero unauthorized git commits, single-commit squashing, and user-exclusive git push policy in VLM-DENTAL.
---

# Git Commit Discipline Skill

This skill enforces strict version control behavior across all coding sessions in `VLM-DENTAL`.

---

## 1. Zero Unauthorized Commits Invariant
* **Rule**: Never run `git commit` autonomously after modifying code, creating artifacts, or running unit tests.
* **Invariant**: All edits must remain uncommitted in the working tree for the user to review.

---

## 2. Single-Commit Squashing Protocol
When the user explicitly commands a commit:
1. Ensure all modified files and verified tests are clean.
2. If previous unpushed commits exist that need squashing, use:
   ```bash
   git reset --soft origin/main
   git commit -m "<type>(<scope>): <clear descriptive message>"
   ```
3. Never create multiple intermediate micro-commits during a single session.

---

## 3. User-Exclusive Git Push
* **Rule**: `git push` is strictly forbidden for the agent. Only the user executes pushes to `origin/main` or remote repositories.
