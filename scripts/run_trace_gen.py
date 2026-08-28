#!/usr/bin/env python3
"""
Autonomous CoT Trace Generator & Verifier for DENTEX dataset.

Operates in two independent modes:

  --mode generate   Runs the LangGraph loop for each image, writing raw traces
                    to ``train_cot_traces_unverified.jsonl`` as fast as hardware
                    allows (no rate limit when GENERATOR_PROVIDER=local).

  --mode verify     Reads unverified traces, verifies each via the ProviderPool
                    (external API with strict rate pacing), and promotes
                    passing traces to ``train_cot_traces.jsonl``.

Both modes support resume — they track processed image IDs so they can be
interrupted and restarted without data loss.

Usage:
    python scripts/run_trace_gen.py --mode generate
    python scripts/run_trace_gen.py --mode verify
    python scripts/run_trace_gen.py --mode generate --max-images 10
    python scripts/run_trace_gen.py --status-only

Running multiple parallel Colab/Kaggle workers (see dental_agent/training/git_sync.py
for the mechanics, and .gitattributes for the merge=union rule this relies on):
    # Colab instance 1 of 3 (--slice-seed MUST match across all instances):
    python scripts/run_trace_gen.py --mode generate --total-slices 3 --slice-index 1 \
        --slice-seed 42 --git-sync-every 5
    # Colab instance 2 of 3:
    python scripts/run_trace_gen.py --mode generate --total-slices 3 --slice-index 2 \
        --slice-seed 42 --git-sync-every 5
    # Colab instance 3 of 3:
    python scripts/run_trace_gen.py --mode generate --total-slices 3 --slice-index 3 \
        --slice-seed 42 --git-sync-every 5
Requires GITHUB_TOKEN set in .env on each instance (already documented there).

Generating baseline #3's no-tools SFT training data (dentex-agentic-vlm-
proposal.md §6) instead of the main tool-based traces -- same --dataset,
--total-slices etc. all still apply, reads/writes separate _no_tools files:
    python scripts/run_trace_gen.py --mode generate --no-tools
    python scripts/run_trace_gen.py --mode verify --no-tools
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Ensure dental_agent is importable when running standalone
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dental_agent.config import load_config, load_env
from dental_agent.data.dentex import load_dentex_dataset
from dental_agent.data.tufts import load_tufts_dataset
from dental_agent.training.api_pool import (
    verify_local_server_health,
    RPDLimitExhausted,
)
import dental_agent.training.trace_generation as tg
from dental_agent.training.trace_generation import (
    generate_only,
    generate_only_no_tools,
    verify_pending,
    repair_pending,
    clean_unverified_traces,
)
from dental_agent.training.git_sync import sync_and_push
from dental_agent.utils.serialization import to_jsonable


# ---------------------------------------------------------------------------
# Output path defaults
# ---------------------------------------------------------------------------

DEFAULT_UNVERIFIED = "data/traces/train_cot_traces_unverified.jsonl"
DEFAULT_VERIFIED = "data/traces/train_cot_traces.jsonl"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_completed_ids(output_path: Path, only_successful: bool = False) -> set[int]:
    """Read existing output file and extract processed image IDs.

    By default (only_successful=False) counts every record with an image_id,
    which is what verify_pending wants (it filters by status itself). Pass
    only_successful=True for the generation resume check specifically —
    otherwise an image that failed once (e.g. hit a transient truncation
    error) gets marked "completed" and is silently skipped forever, even
    after a fix that would let it succeed on retry.
    """
    completed: set[int] = set()
    if not output_path.exists():
        return completed

    with open(output_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if "image_id" not in record:
                    continue
                if only_successful and record.get("status") != "unverified":
                    continue
                completed.add(int(record["image_id"]))
            except Exception:
                pass

    return completed


def append_trace(output_path: Path, trace_record: dict[str, Any]) -> None:
    """Atomically append a single generated trace to output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = to_jsonable(trace_record)
    with open(output_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(serializable) + "\n")


def print_banner(mode: str, completed_count: int, total_images: int) -> None:
    print("\n" + "=" * 70, flush=True)
    print(f"DENTAL AGENT: AUTONOMOUS CoT TRACE {'GENERATOR' if mode == 'generate' else 'VERIFIER'}", flush=True)
    print("=" * 70, flush=True)

    if mode == "generate":
        if tg.GENERATOR_PROVIDER == "local":
            print(f"* Generator          : LOCAL vLLM ({tg.GENERATOR_MODEL}) — no rate limit", flush=True)
        else:
            import os
            prefix = tg.GENERATOR_PROVIDER.upper().replace('_NIM', '')
            cd = os.environ.get(f"{prefix}_COOLDOWN_SECONDS", "None")
            rpd = os.environ.get(f"{prefix}_RPD_LIMIT", "None")
            print(f"* Generator Provider : {tg.GENERATOR_PROVIDER.upper()} ({tg.GENERATOR_MODEL})", flush=True)
            print(f"* Generator Limits   : {cd}s cooldown, {rpd} RPD cap", flush=True)
        print(f"* Output             : {DEFAULT_UNVERIFIED}", flush=True)
    else:
        import os
        prefix = tg.VERIFIER_PROVIDER.upper().replace('_NIM', '')
        cd = os.environ.get(f"{prefix}_COOLDOWN_SECONDS", "None")
        rpd = os.environ.get(f"{prefix}_RPD_LIMIT", "None")
        print(f"* Verifier Provider  : {tg.VERIFIER_PROVIDER.upper()} ({tg.VERIFIER_MODEL})", flush=True)
        print(f"* Verifier Limits    : {cd}s cooldown, {rpd} RPD cap", flush=True)
        print(f"* Input              : {DEFAULT_UNVERIFIED}", flush=True)
        print(f"* Output             : {DEFAULT_VERIFIED}", flush=True)

    print(f"* Dataset Progress   : {completed_count} / {total_images} images", flush=True)
    print("=" * 70 + "\n", flush=True)


# ---------------------------------------------------------------------------
# Generate mode
# ---------------------------------------------------------------------------

def run_generate(args: argparse.Namespace, cfg: Any) -> None:
    """Dispatches to _run_generate_for_dataset once per comma-separated
    --dataset name (default: a single "dentex", unchanged behavior). Each
    dataset gets its own output file, since resumability tracking
    (load_completed_ids) keys on numeric image id -- letting two datasets
    share one output file would risk a false "already done" skip if they
    happen to have images with the same numeric id (exactly the collision
    prepare_yolo_dataset.py's dataset_tag exists to prevent on the YOLO
    side; the fix here is the same idea, applied by giving each dataset
    its own file rather than tagging records within a shared one)."""
    dataset_list = [d.strip() for d in args.dataset.split(",") if d.strip()]
    if not dataset_list:
        dataset_list = ["dentex"]

    explicit_output = args.output  # None unless the user passed --output themselves
    for dataset_name in dataset_list:
        if explicit_output:
            # User gave an explicit path -- honor it exactly as before, even
            # if it means multiple datasets share one file (their choice).
            output_path = Path(explicit_output)
        elif len(dataset_list) > 1:
            output_path = Path(str(DEFAULT_UNVERIFIED).replace(".jsonl", f"_{dataset_name}.jsonl"))
        else:
            # Single dataset, no explicit output -- exactly today's default,
            # unchanged, so existing single-dataset invocations are unaffected.
            output_path = Path(DEFAULT_UNVERIFIED)

        if args.no_tools and not explicit_output:
            # Separate file from the main tool-based traces -- these feed
            # baseline #3's SFT stage (proposal §6), never the main system's,
            # so they must never land in the same file. Suffix applied after
            # the dataset-name suffix above (if any), not instead of it, so
            # --no-tools with multiple --dataset names still gets one file
            # per dataset, just also tagged _no_tools.
            output_path = Path(str(output_path).replace(".jsonl", "_no_tools.jsonl"))

        if len(dataset_list) > 1:
            print(f"\n{'=' * 70}\nDataset: {dataset_name}\n{'=' * 70}")
        _run_generate_for_dataset(args, cfg, dataset_name, output_path)


def _run_generate_for_dataset(args: argparse.Namespace, cfg: Any, dataset_name: str, output_path: Path) -> None:
    """Run the LangGraph generation loop for one dataset, writing unverified traces."""

    # Load dataset -- dispatches on dataset_name (default: dentex, unchanged
    # behavior). tufts currently raises NotImplementedError with a clear
    # message until its tooth-position/diagnosis mapping is filled in
    # (see dental_agent/data/tufts.py's module docstring) -- that's
    # intentional, not a bug in this dispatch.
    if dataset_name == "tufts":
        imgs_df, annots_df, cats_df = load_tufts_dataset(data_dir=cfg.data_dir)
    else:
        imgs_df, annots_df, cats_df = load_dentex_dataset(
            data_dir=cfg.data_dir, split_name=args.split
        )

    annotated_ids = set(annots_df["image_id"].unique())
    eligible_imgs = imgs_df[imgs_df["id"].isin(annotated_ids)]

    if args.total_slices > 1:
        from dental_agent.data.slicing import get_slice_ids
        slice_ids = get_slice_ids(eligible_imgs["id"].tolist(), args.total_slices, args.slice_index, args.slice_seed)
        eligible_imgs = eligible_imgs[eligible_imgs["id"].isin(slice_ids)]

    if dataset_name == "tufts":
        repo_id = os.environ.get("TUFTS_IMAGES_REPO")
    else:
        repo_id = os.environ.get("DENTEX_IMAGES_REPO")
    if repo_id:
        print(f"Fetching targeted images from {repo_id}...")
        if dataset_name == "tufts":
            from dental_agent.data.tufts import download_tufts_slice as _download_slice
        else:
            from dental_agent.data.dentex import download_dentex_slice as _download_slice
        local_paths_map = _download_slice(eligible_imgs["id"].tolist(), repo_id=repo_id, cache_dir=cfg.data_dir)
        def _update_path(row):
            pid = row["id"]
            if pid in local_paths_map and local_paths_map[pid] is not None:
                return str(local_paths_map[pid])
            return row["local_path"]
        # Update the original imgs_df so downstream functions like generate_only see the new paths
        imgs_df["local_path"] = imgs_df.apply(_update_path, axis=1)
        # Re-filter eligible_imgs based on the updated imgs_df
        eligible_imgs = imgs_df[imgs_df["id"].isin(eligible_imgs["id"])]
        
    eligible_imgs = eligible_imgs[eligible_imgs["local_path"].notna() & (eligible_imgs["local_path"] != "None")]

    total_eligible = len(eligible_imgs)
    completed_ids = load_completed_ids(output_path, only_successful=True)
    remaining_imgs = eligible_imgs[~eligible_imgs["id"].isin(completed_ids)]

    slice_info = f" (Slice {args.slice_index}/{args.total_slices})" if args.total_slices > 1 else ""
    print(f"Targeting{slice_info}: {len(eligible_imgs)} eligible images.")
    print_banner("generate", len(completed_ids), total_eligible)

    if args.status_only:
        print("Status check complete. Run without --status-only to begin generation.")
        return

    if remaining_imgs.empty:
        print(f"[DONE] All {total_eligible} images in split '{args.split}' have already been generated!")
        print(f"Unverified traces: {output_path}")
        return

    todo_images = remaining_imgs.to_dict(orient="records")
    if args.max_images is not None:
        todo_images = todo_images[: args.max_images]

    print(f"Starting generation: {len(todo_images)} image(s) (Pacing delay: {args.pacing_delay}s)...")
    print(f"Output: {output_path}\n")

    generated_in_session = 0
    failed_in_session = 0
    since_last_sync = 0
    session_start_time = time.time()

    for idx, img_record in enumerate(todo_images, start=1):
        image_id = int(img_record["id"])
        image_file = os.path.basename(str(img_record.get("local_path", "")))

        print(f"[{idx}/{len(todo_images)}] Generating Image ID {image_id} ({image_file})...", flush=True)

        try:
            # Health check for local vLLM
            if tg.GENERATOR_PROVIDER == "local":
                health_retries = 0
                while not verify_local_server_health(timeout=5.0):
                    health_retries += 1
                    if health_retries > 24:
                        raise RuntimeError("Local vLLM server is unresponsive for > 2 minutes. Aborting.")
                    print(f"  vLLM unresponsive. Waiting 5s... ({health_retries}/24)")
                    time.sleep(5)

            t0 = time.time()
            if args.no_tools:
                result = generate_only_no_tools(
                    image_id=image_id,
                    images_df=imgs_df,
                    annots_df=annots_df,
                    categories_df=cats_df,
                    max_tokens=args.max_tokens or 2048,
                )
            else:
                result = generate_only(
                    image_id=image_id,
                    images_df=imgs_df,
                    annots_df=annots_df,
                    categories_df=cats_df,
                    max_turns=args.max_turns,
                    max_tool_calls=args.max_tool_calls,
                    max_tokens_per_turn=args.max_tokens,
                    min_turns=args.min_turns,
                    turns_per_finding_buffer=args.turns_per_finding_buffer,
                    context_trim_threshold=args.context_trim_threshold,
                    perturb_small_probability=args.perturb_small_prob,
                    perturb_big_probability=args.perturb_big_prob,
                    max_blobs_per_turn=args.max_blobs_per_turn,
                    max_padding_turns=args.max_padding_turns,
                    max_identical_repeats=args.max_identical_repeats,
                )
            elapsed = time.time() - t0

            if result is None:
                print(f"  [SKIP] No valid image/annotations for ID {image_id}", flush=True)
                continue

            # Tagged so verify_pending (shared across datasets) can key
            # dedup on (dataset, image_id) rather than bare image_id --
            # without this, a DENTEX image already verified would silently
            # cause a same-numbered Tufts image to be skipped as "already
            # done" forever, since both datasets number from a low integer.
            result["dataset"] = dataset_name
            append_trace(output_path, result)
            status = result.get("status", "unknown")

            if status == "unverified":
                generated_in_session += 1
                print(f"  [OK] Generated in {elapsed:.1f}s (Total: {len(completed_ids) + generated_in_session})", flush=True)
            else:
                failed_in_session += 1
                reason = result.get("failure_reason", "unknown")
                print(f"  [FAIL] {reason} ({elapsed:.1f}s)", flush=True)

            since_last_sync += 1
            if args.git_sync_every > 0 and since_last_sync >= args.git_sync_every:
                slice_tag = f"slice {args.slice_index}/{args.total_slices} " if args.total_slices > 1 else ""
                try:
                    sync_and_push(
                        [output_path],
                        f"trace-gen: {slice_tag}dataset={dataset_name} +{since_last_sync} traces "
                        f"(session total {generated_in_session + failed_in_session})",
                    )
                except Exception as e:
                    # A sync hiccup should never take down an otherwise-healthy
                    # generation session -- local progress is already safely on
                    # disk regardless of push success (see append_trace above).
                    print(f"  [git-sync] unexpected error, continuing generation: {e}", flush=True)
                since_last_sync = 0

            if idx < len(todo_images) and args.pacing_delay > 0:
                time.sleep(args.pacing_delay)


        except RPDLimitExhausted as e:
            print(f"\n\n[DAILY LIMIT REACHED] {e}")
            print("Generator API usage limit reached. Progress saved; resume later.")
            break
        except RuntimeError as e:
            if "Hard stop" in str(e):
                print(f"\n\n[API ERROR] {e}")
                print("Generator hit an API error and stopped per No Retries rule.")
                print("Progress saved; resume later.")
                break
            print(f"  [ERROR] Error on Image ID {image_id}: {e}")
            failed_in_session += 1
            time.sleep(2.0)
        except KeyboardInterrupt:
            print("\n\n[PAUSED] Session paused by user.")
            break
        except Exception as e:
            print(f"  [ERROR] Error on Image ID {image_id}: {e}")
            failed_in_session += 1
            time.sleep(2.0)

    # Session Summary
    total_time = time.time() - session_start_time
    total_generated = len(load_completed_ids(output_path, only_successful=True))

    if args.git_sync_every > 0:
        # Final sync regardless of the since_last_sync counter, so a session
        # ending mid-interval (RPD exhaustion, Ctrl-C, or just finishing with
        # a remainder under the threshold) never leaves work stranded only
        # on this Colab instance's local disk.
        slice_tag = f"slice {args.slice_index}/{args.total_slices} " if args.total_slices > 1 else ""
        try:
            sync_and_push(
                [output_path],
                f"trace-gen: {slice_tag}dataset={dataset_name} session end "
                f"(+{generated_in_session + failed_in_session} this session)",
            )
        except Exception as e:
            print(f"  [git-sync] unexpected error during final sync: {e}", flush=True)

    from dental_agent.training.api_pool import _TRACKER
    generator_provider = os.environ.get("GENERATOR_PROVIDER", "local")

    print("\n" + "=" * 70)
    print("GENERATION SESSION SUMMARY")
    print("=" * 70)
    print(f"* Generated this session  : {generated_in_session}")
    print(f"* Failed this session     : {failed_in_session}")
    print(f"* Total in unverified file: {total_generated} / {total_eligible} images")
    print(f"* API State               : {_TRACKER.get_stats(generator_provider)}")
    print(f"* Session Duration        : {total_time / 60:.1f} minutes")
    print(f"* Output File             : {output_path}")
    print("=" * 70 + "\n")

    print("NEXT STEP: Run verification to promote traces:")
    print(f"  python scripts/run_trace_gen.py --mode verify --dataset {dataset_name} --split {args.split}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Verify mode
# ---------------------------------------------------------------------------

def run_verify(args: argparse.Namespace, cfg: Any) -> None:
    """Dispatches to _run_verify_for_dataset once per comma-separated
    --dataset name, each reading its own unverified file (matching what
    run_generate wrote per dataset) -- but all writing into the SAME
    shared verified_path. Verified traces are self-contained (each record
    carries its own image_path and ground truth), so combining them from
    multiple datasets into one file for SFT training is safe; only the
    unverified/resumability-tracking side needed to stay per-dataset."""
    dataset_list = [d.strip() for d in args.dataset.split(",") if d.strip()]
    if not dataset_list:
        dataset_list = ["dentex"]

    explicit_output = args.output
    for dataset_name in dataset_list:
        if explicit_output:
            unverified_path = Path(explicit_output)
        elif len(dataset_list) > 1:
            unverified_path = Path(str(DEFAULT_UNVERIFIED).replace(".jsonl", f"_{dataset_name}.jsonl"))
        else:
            unverified_path = Path(DEFAULT_UNVERIFIED)

        if args.no_tools and not explicit_output:
            # Must match run_generate's --no-tools suffixing exactly, or
            # verify mode would read the wrong (or a nonexistent) file.
            unverified_path = Path(str(unverified_path).replace(".jsonl", "_no_tools.jsonl"))

        if len(dataset_list) > 1:
            print(f"\n{'=' * 70}\nDataset: {dataset_name}\n{'=' * 70}")
        _run_verify_for_dataset(args, cfg, unverified_path)


def _run_verify_for_dataset(args: argparse.Namespace, cfg: Any, unverified_path: Path) -> None:
    """Read one dataset's unverified traces and verify them via external API verifiers."""
    verified_path = Path(DEFAULT_VERIFIED)
    if args.no_tools:
        # Separate verified file too -- baseline #3's SFT data must never
        # mix with the main system's (see run_generate's --no-tools handling
        # and generate_only_no_tools's docstring). verify_pending itself
        # doesn't care which trace kind it's verifying (see that function's
        # docstring in trace_generation.py) -- only the file it's pointed at
        # needs to be kept separate, which this achieves.
        verified_path = Path(str(verified_path).replace(".jsonl", "_no_tools.jsonl"))

    # Count totals for banner
    unverified_ids = load_completed_ids(unverified_path)
    verified_ids = load_completed_ids(verified_path)

    if args.total_slices > 1:
        from dental_agent.data.slicing import get_slice_ids
        slice_ids = get_slice_ids(list(unverified_ids), args.total_slices, args.slice_index, args.slice_seed)
        unverified_ids = unverified_ids & set(slice_ids)

    print_banner("verify", len(verified_ids), len(unverified_ids))

    if args.status_only:
        pending = len(unverified_ids - verified_ids)
        print(f"Pending verification: {pending} trace(s)")
        print("Run without --status-only to begin verification.")
        return

    print(f"Starting verification pass...")
    print(f"Input:  {unverified_path}")
    print(f"Output: {verified_path}\n")

    session_start = time.time()
    result = verify_pending(
        unverified_path=unverified_path,
        verified_path=verified_path,
        max_repairs=args.max_repairs,
        total_slices=args.total_slices,
        slice_index=args.slice_index,
        slice_seed=args.slice_seed,
        pacing_delay=args.pacing_delay,
        max_images=args.max_images,
        provider=os.environ.get("VERIFIER_PROVIDER"),
        model=os.environ.get("VERIFIER_MODEL"),
    )

    total_time = time.time() - session_start
    total_verified = len(load_completed_ids(verified_path))

    if args.git_sync_every > 0 and result["verified"] > 0:
        # verify_pending runs to completion in one call with no natural
        # per-record hook, unlike generate mode's per-image loop -- so
        # --git-sync-every is a simple on/off flag here (any nonzero value),
        # not a fine-grained interval. verified_path is the SAME shared file
        # across all datasets/workers (see run_verify's docstring) -- still
        # safe for the same reason as the per-worker unverified files:
        # verify_pending only ever appends newly-promoted lines, never edits
        # existing ones, so concurrent verify-mode workers' pushes are pure
        # appends too (see .gitattributes' merge=union rule).
        try:
            sync_and_push(
                [verified_path],
                f"trace-verify: +{result['verified']} verified, +{result['rejected']} rejected this run",
            )
        except Exception as e:
            print(f"  [git-sync] unexpected error during sync: {e}", flush=True)
    
    from dental_agent.training.api_pool import _TRACKER
    verifier_provider = os.environ.get("VERIFIER_PROVIDER", "local")

    print("\n" + "=" * 70)
    print("VERIFICATION SESSION SUMMARY")
    print("=" * 70)
    print(f"* Pending at start   : {result['pending']}")
    print(f"* Verified this run  : {result['verified']}")
    print(f"* Rejected this run  : {result['rejected']}")
    print(f"* Total verified     : {total_verified}")
    print(f"* API State          : {_TRACKER.get_stats(verifier_provider)}")
    print(f"* Session Duration   : {total_time / 60:.1f} minutes")
    print(f"* Verified File      : {verified_path}")
    print("=" * 70 + "\n")


def run_clean(args: argparse.Namespace, cfg: Any) -> None:
    dataset_name = (args.dataset.split(",")[0] if "," in args.dataset else args.dataset).strip().lower()
    suffix = "_no_tools" if args.no_tools else ""
    unverified_path = Path(args.output or f"data/traces/train_cot_traces_{dataset_name}{suffix}_unverified.jsonl")
    
    if not unverified_path.exists() and Path(f"data/traces/train_cot_traces{suffix}_unverified.jsonl").exists():
        unverified_path = Path(f"data/traces/train_cot_traces{suffix}_unverified.jsonl")
        
    print(f"\nScanning and cleaning trace file: {unverified_path}...")
    stats = clean_unverified_traces(unverified_path, backup=True, purge_failed=getattr(args, "purge_failed", True))
    print("=" * 60)
    print("TRACE CLEANING COMPLETE")
    print("=" * 60)
    print(f"* Valid Traces Retained      : {stats['kept']}")
    print(f"* Corrupted XML/Blobs Purged : {stats['corrupted']}")
    print(f"* Failed Traces Purged       : {stats['failed']}")
    print("=" * 60 + "\n")


def run_repair(args: argparse.Namespace, cfg: Any) -> None:
    dataset_name = (args.dataset.split(",")[0] if "," in args.dataset else args.dataset).strip().lower()
    suffix = "_no_tools" if args.no_tools else ""
    unverified_path = Path(args.output or f"data/traces/train_cot_traces_{dataset_name}{suffix}_unverified.jsonl")
    verified_path = Path(DEFAULT_VERIFIED if not args.no_tools else "data/traces/train_cot_traces_no_tools.jsonl")

    if not unverified_path.exists() and Path(f"data/traces/train_cot_traces{suffix}_unverified.jsonl").exists():
        unverified_path = Path(f"data/traces/train_cot_traces{suffix}_unverified.jsonl")

    session_start = time.time()
    
    # Auto-resolve default verifier if unset
    prov = args.verifier_provider or os.environ.get("VERIFIER_PROVIDER")
    mod = args.verifier_model or os.environ.get("VERIFIER_MODEL")
    if not prov:
        prov = "gemini" if args.no_tools else "openrouter"
        mod = "gemini-3.5-flash-lite" if args.no_tools else "minimax/minimax-m3:free"
        
    result = repair_pending(
        unverified_path=unverified_path,
        verified_path=verified_path,
        provider=prov,
        model=mod,
        total_slices=args.total_slices,
        slice_index=args.slice_index,
        slice_seed=args.slice_seed,
        pacing_delay=args.pacing_delay,
        max_images=args.max_images,
    )

    total_time = time.time() - session_start
    total_verified = len(load_completed_ids(verified_path))

    if args.git_sync_every > 0 and result["repaired_and_promoted"] > 0:
        try:
            sync_and_push(
                [verified_path],
                f"trace-repair: +{result['repaired_and_promoted']} repaired & promoted",
            )
        except Exception as e:
            print(f"  [git-sync] unexpected error during sync: {e}", flush=True)

    print("\n" + "=" * 70)
    print("VERIFIER SELF-REPAIR SESSION SUMMARY")
    print("=" * 70)
    print(f"* Unverified at start : {result['pending_repair']}")
    print(f"* Repaired & Promoted : {result['repaired_and_promoted']}")
    print(f"* Still Unverified    : {result['still_unverified']}")
    print(f"* Total Verified File : {total_verified}")
    print(f"* Session Duration    : {total_time / 60:.1f} minutes")
    print(f"* Output File         : {verified_path}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous CoT Trace Generator & Verifier"
    )
    parser.add_argument("--total-slices", type=int, default=1,
        help="Split the dataset into this many equal, randomized chunks (default: 1, i.e. no slicing).")
    parser.add_argument("--slice-index", type=int, default=1,
        help="Which slice (1-indexed) this run processes. Must be between 1 and --total-slices.")
    parser.add_argument("--slice-seed", type=int, default=42,
        help="Seed for the random slice partition. Keep identical across all parallel Colab instances.")
    parser.add_argument("--no-tools", action="store_true",
        help="Generate/verify SFT training traces for baseline #3 (dentex-agentic-vlm-"
             "proposal.md §6: 'Full agent without tool access... isolates the contribution "
             "of tools') instead of the main tool-based system: single-turn, no LangGraph "
             "tool loop, no ToolRegistry -- see dental_agent/training/trace_generation.py's "
             "generate_no_tools_trajectory. Reads/writes separate _no_tools-suffixed files "
             "(both unverified and verified) so these never mix into the main system's SFT "
             "training set. Does NOT affect --mode verify's use of verify_pending, which is "
             "reused unmodified for both trace kinds. Do not confuse this with baseline #1's "
             "ZERO_SHOT_PROMPT (evaluation/baselines.py, no training involved) or baseline "
             "#3's own GRPO-rollout-time NO_TOOLS_SYSTEM_PROMPT (agent/loop.py) -- this flag "
             "is specifically for generating that baseline's Stage 1 SFT data.")
    parser.add_argument("--git-sync-every", type=int, default=0,
        help="Push generated/verified traces to the shared git repo every N traces (default: 0 = "
             "disabled, no git operations attempted). Designed for running --total-slices > 1 "
             "across multiple parallel Colab/Kaggle instances: each worker pulls in the others' "
             "already-pushed traces before pushing its own, safely, since --slice-index guarantees "
             "disjoint image IDs per worker (concurrent appends to the same file, never edits to "
             "the same line -- see dental_agent/training/git_sync.py and .gitattributes' merge=union "
             "rule for exactly why this is safe rather than assumed). In generate mode this is a "
             "periodic checkpoint interval (every N images); in verify mode it's a simple on/off "
             "flag -- sync once after the verify pass completes, regardless of the number given. "
             "Requires GITHUB_TOKEN in .env (already documented there) to authenticate the push; "
             "without it, sync attempts will fail (loudly, but non-fatally -- generation continues, "
             "just without a successful push) since the repo's origin remote is normally "
             "unauthenticated for write access.")
    parser.add_argument("--generator-provider", type=str, default=None,
        help="Override the GENERATOR_PROVIDER from .env (e.g. 'groq', 'nvidia', 'local')")
    parser.add_argument("--generator-model", type=str, default=None,
        help="Override the GENERATOR_MODEL from .env (e.g. 'qwen/qwen3.6-27b')")
    parser.add_argument("--verifier-provider", type=str, default=None,
        help="Override the VERIFIER_PROVIDER from .env (e.g. 'groq', 'nvidia', 'openrouter', 'gemini')")
    parser.add_argument("--verifier-model", type=str, default=None,
        help="Override the VERIFIER_MODEL from .env (e.g. 'meta/muse-glimmer-30b')")
    parser.add_argument(
        "--mode",
        type=str,
        default="generate",
        choices=["generate", "verify", "repair", "clean"],
        help="Operational mode: 'generate' (traces), 'verify' (strict pass), 'repair' (intelligent verifier repair), 'clean' (purge corruption). Default: generate",
    )
    parser.add_argument(
        "--clean-corrupted",
        action="store_true",
        help="Scan and purge historical multi-blob / XML artifact traces and failed generations before processing",
    )
    parser.add_argument(
        "--purge-failed",
        action="store_true",
        default=True,
        help="Purge generation_failed entries during cleaning so image IDs can be cleanly regenerated without duplicates",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Override output file path. Defaults based on mode: generate → train_cot_traces_unverified.jsonl, verify → reads from unverified",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="DENTEX split to process (validation or train) -- ignored for --dataset tufts",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="dentex",
        help="Which dataset(s) to generate traces from. Comma-separated for multiple in one "
             "run (e.g. --dataset dentex,tufts) -- each gets its own output file, since "
             "resumability tracking keys on numeric image id and two datasets could otherwise "
             "collide. Defaults to dentex (unchanged behavior). tufts currently raises "
             "NotImplementedError until its tooth-position/diagnosis mapping is filled in -- "
             "see dental_agent/data/tufts.py.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Candidate traces to generate per example (default: 1)",
    )
    parser.add_argument(
        "--max-images",
        "--max-traces",
        dest="max_images",
        type=int,
        default=None,
        help="Maximum images to attempt in this session (default: all remaining). "
             "'--max-traces' is an alias for the same thing -- use whichever name you prefer.",
    )
    parser.add_argument(
        "--pacing-delay",
        type=float,
        default=1.5,
        help="Safe delay (seconds) between successive trace calls (default: 1.5s)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=25,
        help="Ceiling on the per-image turn budget (generate mode only, default: 25). The "
             "actual per-image budget is dynamic: max(--min-turns, n_findings + "
             "--turns-per-finding-buffer), capped at this value -- verify the true max "
             "finding count in your dataset and raise this if it's not comfortably above that.",
    )
    parser.add_argument(
        "--max-tool-calls",
        type=int,
        default=50,
        help="Hard graph limit for the total number of tool calls permitted before failing (default: 50).",
    )
    parser.add_argument(
        "--min-turns",
        type=int,
        default=5,
        help="Floor on the per-image turn budget, used even for single-finding images (default: 5).",
    )
    parser.add_argument(
        "--turns-per-finding-buffer",
        type=int,
        default=5,
        help="Per-image turn budget = max(--min-turns, n_findings + this), capped at --max-turns (default: 5).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Max tokens per generator turn (generate mode only, default: fallback to .env). Raise this if "
             "you still see 'Unparseable model output' failures with no JSON at all in the raw "
             "output -- that's the model's thinking getting cut off, not a formatting mistake.",
    )
    parser.add_argument(
        "--image-max-dim",
        type=int,
        default=None,
        help="Max image dimension for API payload scaling (CLI argument takes strict precedence over .env). Set to 0 to send unscaled images.",
    )
    parser.add_argument(
        "--perturb-small-prob",
        type=float,
        default=0.25,
        help="Probability of applying a small synthetic bounding box perturbation during trace gen (default: 0.25)",
    )
    parser.add_argument(
        "--perturb-big-prob",
        type=float,
        default=0.30,
        help="Probability of applying a large synthetic bounding box perturbation during trace gen (default: 0.30)",
    )
    parser.add_argument(
        "--max-blobs-per-turn",
        type=int,
        default=2,
        help="Maximum action-bearing JSON blocks in a single raw LLM response before failing fast as a multi-blob dump (default: 2)",
    )
    parser.add_argument(
        "--max-padding-turns",
        type=int,
        default=3,
        help="Maximum consecutive filler/padding thought turns before terminating the trace (default: 3)",
    )
    parser.add_argument(
        "--max-identical-repeats",
        type=int,
        default=3,
        help="Maximum consecutive identical tool calls before terminating the trace as a loop (default: 3)",
    )
    parser.add_argument(
        "--context-trim-threshold",
        type=int,
        default=None,
        help="Estimated token threshold at which older tool image results are trimmed to preserve context budget (default: provider env fallback)",
    )
    parser.add_argument(
        "--max-repairs",
        type=int,
        default=1,
        help="Maximum repair attempts by the verifier on trace rejection (default: 1)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Re-generate traces that failed in previous runs instead of skipping them",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only display current pool status and progress without processing",
    )
    parser.add_argument(
        "--ignore-429",
        action="store_true",
        help="Ignore 429 rate limit errors and retry up to 10 times",
    )
    parser.add_argument(
        "--ignore-api-errors",
        action="store_true",
        help="Ignore all API errors (including rate limits) and retry up to 10 times (overrides rule)",
    )
    args = parser.parse_args()
    if args.total_slices > 1 and not (1 <= args.slice_index <= args.total_slices):
        parser.error('slice_index must be >= 1 and <= total_slices')
    return args

def interactive_prompt(role: str) -> str:
    print(f"\n[?] Missing {role.upper()} provider. Select from list:")
    choices = ["local", "groq", "nvidia_nim", "openrouter", "gemini"]
    for i, p in enumerate(choices, 1):
        print(f"  {i}. {p}")
    
    while True:
        try:
            val = input(f"Enter number (1-{len(choices)}): ").strip()
            idx = int(val) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            print("Invalid choice. Try again.")
        except (ValueError, EOFError):
            print("Invalid input. Try again.")

def main() -> None:
    load_env()
    cfg = load_config()
    args = parse_args()
    
    import os
    if args.ignore_429:
        os.environ["IGNORE_429"] = "true"
    if args.ignore_api_errors:
        os.environ["IGNORE_API_ERRORS"] = "true"
    import dental_agent.training.trace_generation as tg

    if args.generator_provider:
        os.environ["GENERATOR_PROVIDER"] = args.generator_provider
    elif args.mode == "generate" and not os.environ.get("GENERATOR_PROVIDER"):
        os.environ["GENERATOR_PROVIDER"] = interactive_prompt("generator")
        
    tg.GENERATOR_PROVIDER = os.environ.get("GENERATOR_PROVIDER", "local")
        
    if args.generator_model:
        os.environ["GENERATOR_MODEL"] = args.generator_model
    if "GENERATOR_MODEL" in os.environ:
        tg.GENERATOR_MODEL = os.environ["GENERATOR_MODEL"]
        
    if args.verifier_provider:
        os.environ["VERIFIER_PROVIDER"] = args.verifier_provider
    elif args.mode == "verify" and not os.environ.get("VERIFIER_PROVIDER"):
        os.environ["VERIFIER_PROVIDER"] = interactive_prompt("verifier")
        
    tg.VERIFIER_PROVIDER = os.environ.get("VERIFIER_PROVIDER", "local")
        
    if args.verifier_model:
        os.environ["VERIFIER_MODEL"] = args.verifier_model
    if "VERIFIER_MODEL" in os.environ:
        tg.VERIFIER_MODEL = os.environ["VERIFIER_MODEL"]

    if args.image_max_dim is not None:
        g_prefix = tg.GENERATOR_PROVIDER.upper().replace('_NIM', '')
        v_prefix = tg.VERIFIER_PROVIDER.upper().replace('_NIM', '')
        os.environ[f"{g_prefix}_IMAGE_MAX_DIM"] = str(args.image_max_dim)
        os.environ[f"{v_prefix}_IMAGE_MAX_DIM"] = str(args.image_max_dim)

    if getattr(args, "clean_corrupted", False) or args.mode == "clean":
        run_clean(args, cfg)
        if args.mode == "clean":
            return

    if args.mode == "generate":
        run_generate(args, cfg)
    elif args.mode == "verify":
        run_verify(args, cfg)
    elif args.mode == "repair":
        run_repair(args, cfg)


if __name__ == "__main__":
    main()
