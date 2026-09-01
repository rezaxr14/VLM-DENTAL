"""
Git sync for parallel Colab/Kaggle trace-generation workers.

Multiple independent Colab sessions can run scripts/run_trace_gen.py at the
same time, each assigned a disjoint slice of image IDs via
--total-slices/--slice-index/--slice-seed (see that script's docstring --
this partitioning already existed before this module; it's what makes the
sync below safe). Because each worker's slice is disjoint by construction,
every worker's locally-generated trace lines are for DIFFERENT image_ids
than every other worker's -- so when synced back to the same shared JSONL
file, they are always pure, non-overlapping *appends*, never edits to the
same line. This is exactly the case git's merge machinery handles natively
and reliably: a three-way merge where both sides only added new lines
concatenates both sides' additions without a conflict, provided nobody
edits or reorders existing lines. This module leans on that property
rather than reinventing a custom merge -- but never trusts it blindly (see
the conflict handling in sync_and_push below).

WHAT sync_and_push DOES ON EACH CALL:
  1. `git add` the given paths, then commit locally IF there's something to
     commit (no-op is fine -- a call can also just be "push whatever was
     already committed but failed to push last time", see step 4).
  2. `git fetch origin`, then `git merge --no-edit origin/<branch>` into
     the current branch -- NOT rebase. A plain merge is the more standard
     idiom for "many independent writers converging on a shared branch",
     and its 3-way semantics are the most direct match for "both sides
     only appended new lines" (as opposed to rebase's per-commit patch
     replay, which is a less direct fit for this exact scenario).
  3. If the merge reports a REAL conflict (should only happen from
     operator error -- e.g. two workers accidentally given the same
     --slice-index, or someone hand-editing a trace file mid-run) --
     ABORT the merge immediately and return False with a clear message.
     This module NEVER attempts automatic conflict resolution on training
     data -- same "honest stop, don't guess" principle as the dataset
     loaders (see .agents/rules/vlm_dental.md Rule 12). A silently
     mis-resolved conflict here could duplicate, drop, or corrupt training
     traces in a way nothing downstream would ever catch.
  4. `git push`. If rejected (another worker pushed between this worker's
     fetch and its own push -- a genuine race, not a conflict), loop back
     to step 2 and retry, up to max_retries times with a short backoff.
     This step always runs, even if step 1 found nothing new to commit --
     handles the case where a *previous* sync call's commit succeeded but
     its push didn't (e.g. a transient network drop), leaving an unpushed
     local commit that this call should still try to deliver.

WHAT THIS DOES NOT DO: decide when to sync, or what to sync -- that's the
caller's job (scripts/run_trace_gen.py calls this every --git-sync-every
generated traces, and once more at the end of each session/mode). This
module also never touches anything outside the exact paths it's given --
it will not accidentally commit unrelated local changes (model
checkpoints, scratch files, etc.) sitting in the working tree, since `git
add`/`git commit` are both invoked with an explicit `--` path list, not `-A`.

SCOPE NOTE, relative to .agents/rules/vlm_dental.md Rule 11 ("never git
push/commit"): that rule governs IDE coding agents (Claude Code,
Antigravity) editing this repo -- it's about an agent never pushing code
changes on the user's behalf without being asked. This module is a
different thing: a pipeline feature the user explicitly opts into (by
setting GITHUB_TOKEN and passing --git-sync-every), running under the
user's own Colab session, touching only the exact trace-data paths it's
told to sync. Building or maintaining this module doesn't conflict with
Rule 11; an IDE agent using this module's existence as precedent to justify
pushing unrelated repo changes on the user's behalf would be a misreading
of both this docstring and that rule.

AUTHENTICATION: reads GITHUB_TOKEN from the environment (see .env.example,
already documented there as "GitHub (Auto-push verified traces from
Colab)" before this module existed). If set, auth is passed as a one-off
`-c http.<url>.extraheader=...` flag on each git invocation -- never
persisted to .git/config via `git remote set-url`, and never printed in any
log line -- so a Colab session's token doesn't linger in repo state or
console scrollback after the session ends.
"""

from __future__ import annotations

import base64
import os
import subprocess
import time
from pathlib import Path


def _run_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
    )


def _repo_root() -> Path:
    result = _run_git(["rev-parse", "--show-toplevel"], cwd=Path.cwd())
    if result.returncode != 0:
        raise RuntimeError(f"not inside a git repository ({result.stderr.strip()})")
    return Path(result.stdout.strip())


def _auth_args() -> list[str]:
    """One-off auth flag for this invocation only -- see module docstring
    for why this is preferred over a persisted `git remote set-url`."""
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        return []
    basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
    return ["-c", f"http.https://github.com/.extraheader=AUTHORIZATION: basic {basic}"]


def ensure_git_identity(repo_root: Path) -> None:
    """Set user.name/user.email for this repo if not already configured --
    needed because Colab containers are fresh every session, so a global
    git identity from a previous session doesn't carry over. Reads
    GIT_AUTHOR_NAME/GIT_AUTHOR_EMAIL from the environment if set; falls
    back to a clearly-machine-attributed default otherwise, so commits
    made by this module are always easy to tell apart from a human's
    manual commits when reading the log later.
    """
    if not _run_git(["config", "user.name"], cwd=repo_root).stdout.strip():
        name = os.environ.get("GIT_AUTHOR_NAME", "vlm-dental-trace-gen-bot")
        _run_git(["config", "user.name", name], cwd=repo_root)
    if not _run_git(["config", "user.email"], cwd=repo_root).stdout.strip():
        email = os.environ.get("GIT_AUTHOR_EMAIL", "trace-gen-bot@vlm-dental.local")
        _run_git(["config", "user.email", email], cwd=repo_root)


def check_for_duplicate_ids(path: str | Path) -> dict[tuple[str, int], int]:
    """Scan a trace JSONL file for (dataset, image_id) keys appearing more
    than once -- the one risk `merge=union` (see .gitattributes) doesn't
    protect against, since a union merge has no concept of what counts as
    a semantic duplicate for this file's content, it just unions lines.

    Returns a dict of {(dataset, image_id): count} for every key seen more
    than once; empty dict means no duplicates found. Purely diagnostic --
    does not modify the file or decide anything on its own. Expected to
    only ever find something here from operator error (e.g. two parallel
    workers accidentally given the same --slice-index), since correct
    --total-slices/--slice-index/--slice-seed usage guarantees disjoint
    image_id assignment per dataset across workers -- see
    scripts/run_trace_gen.py's docstring.
    """
    import json
    from collections import Counter

    path = Path(path)
    if not path.exists() or path.suffix.lower() != ".jsonl":
        return {}

    counts: Counter[tuple[str, int]] = Counter()
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue
            if not isinstance(record, dict) or "image_id" not in record:
                continue
            try:
                key = (str(record.get("dataset", "dentex")), int(record["image_id"]))
                counts[key] += 1
            except (ValueError, TypeError):
                continue

    return {k: v for k, v in counts.items() if v > 1}


def sync_and_push(
    paths: list[str | Path],
    commit_message: str,
    branch: str | None = None,
    max_retries: int = 5,
    retry_backoff_seconds: float = 5.0,
) -> bool:
    """Commit the given paths (if changed) and push, pulling in any other
    workers' concurrent pushes first. See module docstring for the full
    step-by-step and why a plain merge (not rebase) is used.

    Returns True on success (including "nothing new, nothing to do" -- that
    is not a failure). Returns False if a real conflict was hit (aborted,
    never auto-resolved) or retries were exhausted on repeated push races.
    NEVER raises for ordinary git/network failures -- callers (the
    trace-gen loop) should treat a False return as "will retry next sync
    interval", not a reason to crash an otherwise-healthy generation
    session over a sync hiccup.
    """
    try:
        repo_root = _repo_root()
    except RuntimeError as e:
        print(f"[git-sync] {e} -- skipping sync.")
        return False

    ensure_git_identity(repo_root)
    auth_args = _auth_args()

    if branch is None:
        branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root).stdout.strip() or "main"

    rel_paths = [str(Path(p)) for p in paths]

    add_result = _run_git(["add", "--", *rel_paths], cwd=repo_root)
    if add_result.returncode != 0:
        print(f"[git-sync] git add failed: {add_result.stderr.strip()}")
        return False

    if _run_git(["status", "--porcelain", "--", *rel_paths], cwd=repo_root).stdout.strip():
        commit_result = _run_git(["commit", "-m", commit_message, "--", *rel_paths], cwd=repo_root)
        if commit_result.returncode != 0:
            print(f"[git-sync] git commit failed: {commit_result.stderr.strip()}")
            return False
        print(f"[git-sync] Committed: {commit_message}")
    # No `else: return True` here on purpose -- there may be a commit from a
    # PREVIOUS sync call that was made successfully but never got pushed
    # (e.g. a transient network drop between commit and push). Always fall
    # through to fetch/merge/push so that case still gets delivered.

    for attempt in range(1, max_retries + 1):
        fetch_result = _run_git([*auth_args, "fetch", "origin", branch], cwd=repo_root)
        if fetch_result.returncode != 0:
            print(f"[git-sync] fetch failed (attempt {attempt}/{max_retries}): {fetch_result.stderr.strip()}")
            time.sleep(retry_backoff_seconds)
            continue

        merge_result = _run_git(["merge", "--no-edit", f"origin/{branch}"], cwd=repo_root)
        if merge_result.returncode != 0:
            # Could be a real conflict, or a non-zero exit for some other
            # transient reason -- check for actual conflict markers before
            # treating this as fatal.
            conflicted = _run_git(["diff", "--name-only", "--diff-filter=U"], cwd=repo_root).stdout.strip()
            if conflicted:
                print(f"[git-sync] MERGE CONFLICT on: {conflicted}")
                print(
                    "[git-sync] Aborting merge -- will NOT auto-resolve a conflict on training "
                    "data. This should only happen from operator error (e.g. two workers given "
                    "the same --slice-index, or a trace file hand-edited mid-run). Resolve by "
                    "hand, then re-run."
                )
                _run_git(["merge", "--abort"], cwd=repo_root)
                return False
            print(
                f"[git-sync] merge reported an error with no conflict markers -- treating as "
                f"transient (attempt {attempt}/{max_retries}): {merge_result.stderr.strip()}"
            )
            time.sleep(retry_backoff_seconds)
            continue

        push_result = _run_git([*auth_args, "push", "origin", f"HEAD:{branch}"], cwd=repo_root)
        if push_result.returncode == 0:
            print(f"[git-sync] Pushed successfully (attempt {attempt}/{max_retries}).")
            for p in rel_paths:
                dupes = check_for_duplicate_ids(repo_root / p)
                if dupes:
                    print(
                        f"[git-sync] WARNING: {p} has {len(dupes)} (dataset, image_id) pair(s) "
                        f"appearing more than once after merge: {dupes}. This union-merges lines "
                        "without deduplicating -- likely two workers were given overlapping "
                        "--slice-index assignments. Not blocking the push (data isn't corrupted, "
                        "just duplicated), but worth checking your worker configuration and "
                        "deduplicating before this feeds SFT training."
                    )
            return True

        last_line = push_result.stderr.strip().splitlines()[-1] if push_result.stderr.strip() else "(no output)"
        print(f"[git-sync] push rejected, likely a race with another worker (attempt {attempt}/{max_retries}): {last_line}")
        time.sleep(retry_backoff_seconds)

    print(f"[git-sync] Exhausted {max_retries} retries -- giving up for this sync interval. "
          "Local commit is safe on disk and will be retried on the next sync call.")
    return False
