---
name: code-critique-and-hardening
description: >-
  Systematic guide and checklist for defensive coding, non-destructive refactoring,
  cross-platform compatibility, and adversarial code self-critique in VLM-DENTAL.
---

# Code Critique & Hardening Guide

Use this skill whenever implementing new tools, data loaders, training loops, evaluation baselines, or verification pipelines to ensure zero regressions, robust cross-environment execution, and hardened code quality.

---

## 1. Non-Destructive Refactoring Protocol
When adding new features or fixing existing bugs:
1. **Never Delete Working Functionality**: Modifying, improving, and combining existing functions is encouraged, but deleting working functionality or dropping existing parameters/flags/cells is strictly prohibited (Rule 14).
2. **Inventory Callers & Dependencies**: Inspect `scripts/`, `notebooks/`, and `tests/` before refactoring core modules to understand all dependent workflows.
3. **Backward-Compatible Signatures**: Always supply sensible defaults for new parameters (e.g. `data_dir: str | Path | None = None`).
4. **Preserve Specialized Workflows**: Keep distinct pipelines (e.g., `--no-tools` baseline vs tool-based, local generator vs remote API pool) properly separated without cross-polluting behavior.

---

## 2. The 5-Point Adversarial Audit Checklist
Before finalizing any implementation, actively audit the code against these five vulnerability vectors:

### A. Cross-Platform Path Normalization
- **Risk**: Windows paths contain backslashes (`\`), while POSIX/Linux systems use forward slashes (`/`). Using `Path().name` or `os.path.basename` on a Windows path string while running in Linux/Colab fails to extract the filename.
- **Defensive Pattern**:
  ```python
  raw_str = str(image_path).strip() if image_path is not None else ""
  fname = raw_str.replace("\\", "/").split("/")[-1] if raw_str else f"{image_id}.png"
  stem = fname.rsplit(".", 1)[0] if "." in fname else fname
  ```

### B. Null, Key, & Type Safety
- **Risk**: Missing keys in dictionary lookups or `None` values returned by external APIs crash pipelines when accessed directly or sliced.
- **Defensive Pattern**:
  ```python
  # Safe string slicing:
  reason_str = (reason or "")[:60]

  # Safe dictionary lookup:
  val = record.get("key", default_value)
  ```

### C. I/O & Search Bottlenecks (Memoization)
- **Risk**: Scanning disk directories or globbing thousands of files per trace in a large loop causes major performance bottlenecks.
- **Defensive Pattern**:
  ```python
  # Use in-memory memoization cache for path/dataset lookups
  _RESOLVED_PATH_CACHE: dict[tuple[str, int, str], Path] = {}
  ```

### D. Mutability & In-Place Side-Effects
- **Risk**: Shallow copying lists of dictionaries (`list(messages)`) leaves internal dictionaries referenced, meaning in-place mutations alter original data.
- **Defensive Pattern**:
  ```python
  # Explicitly deep copy or re-instantiate dicts before modifying
  repaired_messages = list(trajectory.get("messages", []))
  if isinstance(repaired_messages[i], dict):
      repaired_messages[i] = dict(repaired_messages[i])
      repaired_messages[i]["content"] = new_content
  ```

### E. Concurrency & Slicing Determinism
- **Risk**: Python `set` iteration order is non-deterministic across processes due to hash randomization. Passing an unsorted set conversion to slice chunkers causes worker collisions.
- **Defensive Pattern**:
  ```python
  sorted_ids = sorted(set(image_ids))
  ```

---

## 3. Mandatory Automated Test Coverage
Every new helper, data resolver, or pipeline extension must include unit tests:
1. **Direct Success Test**: Tests the expected happy path with standard inputs.
2. **Fallback / Cross-Environment Test**: Simulates foreign file paths (e.g. Kaggle/Windows paths on Linux).
3. **Edge-Case / Invalid Input Test**: Verifies graceful handling of `None`, empty strings, and malformed inputs without exceptions.

---

## 4. Resource & Bandwidth Optimization (Surgical Downloads)
Whenever writing code that resolves or downloads dataset assets:
1. **Check Local Disk First**:
   ```python
   if local_path and os.path.exists(str(local_path)):
       return local_path
   ```
2. **Pre-Filter Completed & Existing IDs Before Network Fetch**:
   ```python
   # Load completed records first:
   completed_ids = load_completed_ids(output_path, only_successful=True)
   
   # Identify only remaining, missing IDs:
   needed_ids = [
       img_id for img_id in target_ids
       if img_id not in completed_ids and not is_cached_locally(img_id)
   ]
   
   # Only fetch the exact delta:
   if needed_ids:
       local_paths_map = download_slice(needed_ids, repo_id=repo_id, cache_dir=cache_dir)
   ```
3. **Memoize Lookups**: Cache resolved file paths in `_RESOLVED_PATH_CACHE` to eliminate redundant disk probes.

