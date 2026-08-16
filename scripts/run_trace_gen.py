#!/usr/bin/env python3
"""
Autonomous CoT Trace Generator & Verifier for DENTEX dataset.

Operates in two independent modes:

  --mode generate   Runs the LangGraph loop for each image, writing raw traces
                    to ``train_cot_traces_unverified.jsonl`` as fast as hardware
                    allows (no rate limit when GENERATOR_PROVIDER=local).

  --mode verify     Reads unverified traces, verifies each via the ProviderPool
                    (external API round-robin with rate limits), and promotes
                    passing traces to ``train_cot_traces.jsonl``.

Both modes support resume — they track processed image IDs so they can be
interrupted and restarted without data loss.

Usage:
    python scripts/run_trace_gen.py --mode generate
    python scripts/run_trace_gen.py --mode verify
    python scripts/run_trace_gen.py --mode generate --max-images 10
    python scripts/run_trace_gen.py --status-only
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
from dental_agent.training.api_pool import (
    verify_local_server_health,
    RPDLimitExhausted,
)
import dental_agent.training.trace_generation as tg
from dental_agent.training.trace_generation import (
    generate_only,
    verify_pending,
)
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
    """Run the LangGraph generation loop, writing unverified traces."""
    output_path = Path(args.output or DEFAULT_UNVERIFIED)

    # Load dataset
    imgs_df, annots_df, cats_df = load_dentex_dataset(
        data_dir=cfg.data_dir, split_name=args.split
    )

    annotated_ids = set(annots_df["image_id"].unique())
    eligible_imgs = imgs_df[imgs_df["id"].isin(annotated_ids)]

    if args.total_slices > 1:
        from dental_agent.data.slicing import get_slice_ids
        slice_ids = get_slice_ids(eligible_imgs["id"].tolist(), args.total_slices, args.slice_index, args.slice_seed)
        eligible_imgs = eligible_imgs[eligible_imgs["id"].isin(slice_ids)]

    repo_id = os.environ.get("DENTEX_IMAGES_REPO")
    if repo_id:
        print(f"Fetching targeted images from {repo_id}...")
        from dental_agent.data.dentex import download_dentex_slice
        local_paths_map = download_dentex_slice(eligible_imgs["id"].tolist(), repo_id=repo_id, cache_dir=cfg.data_dir)
        def _update_path(row):
            pid = row["id"]
            if pid in local_paths_map and local_paths_map[pid] is not None:
                return str(local_paths_map[pid])
            return row["local_path"]
        # Use .copy() to avoid SettingWithCopyWarning
        eligible_imgs = eligible_imgs.copy()
        eligible_imgs["local_path"] = eligible_imgs.apply(_update_path, axis=1)
        
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
            result = generate_only(
                image_id=image_id,
                images_df=imgs_df,
                annots_df=annots_df,
                categories_df=cats_df,
                max_turns=args.max_turns,
                max_tokens_per_turn=args.max_tokens,
            )
            elapsed = time.time() - t0

            if result is None:
                print(f"  [SKIP] No valid image/annotations for ID {image_id}", flush=True)
                continue

            append_trace(output_path, result)
            status = result.get("status", "unknown")

            if status == "unverified":
                generated_in_session += 1
                print(f"  [OK] Generated in {elapsed:.1f}s (Total: {len(completed_ids) + generated_in_session})", flush=True)
            else:
                failed_in_session += 1
                reason = result.get("failure_reason", "unknown")
                print(f"  [FAIL] {reason} ({elapsed:.1f}s)", flush=True)

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
    print(f"  python scripts/run_trace_gen.py --mode verify --split {args.split}")
    print("=" * 70 + "\n")


# ---------------------------------------------------------------------------
# Verify mode
# ---------------------------------------------------------------------------

def run_verify(args: argparse.Namespace, cfg: Any) -> None:
    """Read unverified traces and verify them via external API verifiers."""
    unverified_path = Path(args.output or DEFAULT_UNVERIFIED)
    verified_path = Path(DEFAULT_VERIFIED)

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
    )

    total_time = time.time() - session_start
    total_verified = len(load_completed_ids(verified_path))
    
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
        choices=["generate", "verify"],
        help="Operational mode: 'generate' (LangGraph traces) or 'verify' (cross-family verification). Default: generate",
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
        help="DENTEX split to process (validation or train)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1,
        help="Candidate traces to generate per example (default: 1)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum images to process in this session (default: all remaining)",
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
        default=8,
        help="Max tool calls allowed before a trace is abandoned (generate mode only, default: 8)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=4096,
        help="Max tokens per generator turn (generate mode only, default: 4096). Raise this if "
             "you still see 'Unparseable model output' failures with no JSON at all in the raw "
             "output -- that's the model's thinking getting cut off, not a formatting mistake.",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only display current pool status and progress without processing",
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

    if args.mode == "generate":
        run_generate(args, cfg)
    elif args.mode == "verify":
        run_verify(args, cfg)


if __name__ == "__main__":
    main()
