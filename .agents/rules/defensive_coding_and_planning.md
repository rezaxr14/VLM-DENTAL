---
trigger: always_on
---

# Defensive Coding, Preservation & Planning Invariants

## 1. Preservation of Working Functionality (CRITICAL)
- **Rule:** Altering, enhancing, and combining functions is encouraged, but deleting working functionality or dropping existing capabilities/flags/notebook cells is strictly forbidden (Rule 14).
- Always ensure existing pipelines remain backward-compatible.

## 2. Mandatory Adversarial Code Self-Critique (CRITICAL)
Proactively audit all code changes across 5 defensive vectors before completion:
1. **Cross-Platform Path Normalization**: Normalize backslashes (`\`) and slashes (`/`).
2. **Null & Key Safety**: Guard against missing dictionary keys and `None` returns (`(val or "")[:N]`).
3. **I/O & Search Bottlenecks**: Use in-memory memoization caches (`_RESOLVED_PATH_CACHE`) to avoid repeating $O(N)$ filesystem scans.
4. **Mutability Side-Effects**: Deep copy mutable nested structures before modifying values in-place.
5. **Concurrency & Determinism**: Sort collections converted from sets (`sorted(set(...))`) before chunking.

## 3. Local Asset Re-Use & Surgical Download Invariant (CRITICAL)
- **Rule:** NEVER download assets from remote repositories (HF Hub, cloud URLs) if files exist locally on disk.
- **Rule:** When downloading is strictly required, compute the exact missing delta (`needed_ids = [id for id in target_ids if not exists_locally(id) and id not in completed_ids]`) and download only those IDs.
- **Rule:** Load completed items (`load_completed_ids`) BEFORE triggering downloads.

## 4. Strict Planning Mode Enforcement (CRITICAL)
- **Rule:** Before making any modifications to the codebase (touching a single line of project code, running modifying scripts, etc.), agents MUST FIRST generate an `implementation_plan.md` artifact and explicitly await user approval.
- Executing code-modifying tools prior to receiving explicit approval on the plan is strictly forbidden.
