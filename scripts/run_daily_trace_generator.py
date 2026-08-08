#!/usr/bin/env python3
"""
Autonomous Daily CoT Trace Generator for DENTEX dataset.

Designed to be run independently throughout the day from any machine.
Safely rotates across the configured multi-model Gemini key pool,
respects rate limits (5 RPM / key, 40 RPD / key), appends traces
line-by-line in JSONL format, resumes automatically from where it left off,
and provides Git synchronization instructions for multi-machine workflows.

Usage:
    python scripts/run_daily_trace_generator.py
    python scripts/run_daily_trace_generator.py --max-images 10
    python scripts/run_daily_trace_generator.py --status-only
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

from dental_agent.config import load_config, load_env
from dental_agent.data.dentex import load_dentex_dataset
from dental_agent.training.api_pool import (
    AllKeysExhaustedToday,
    get_gemini_pool,
)
from dental_agent.training.trace_generation import (
    GENERATOR_MODEL,
    GENERATOR_PROVIDER,
    VERIFIER_MODEL,
    VERIFIER_PROVIDER,
    build_trace_example,
)
from dental_agent.utils.serialization import to_jsonable


def load_completed_ids(output_path: Path) -> set[int]:
    """Read existing output file and extract all processed image IDs."""
    completed: set[int] = set()
    if not output_path.exists():
        return completed

    if output_path.suffix == ".jsonl":
        with open(output_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if "image_id" in record:
                        completed.add(int(record["image_id"]))
                except Exception:
                    pass
    elif output_path.suffix == ".json":
        try:
            with open(output_path, "r", encoding="utf-8", errors="replace") as f:
                data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "image_id" in item:
                            completed.add(int(item["image_id"]))
        except Exception:
            pass

    return completed


def append_trace(output_path: Path, trace_record: dict[str, Any]) -> None:
    """Atomically append a single generated trace to output file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = to_jsonable(trace_record)

    if output_path.suffix == ".jsonl":
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(serializable) + "\n")
    elif output_path.suffix == ".json":
        data = []
        if output_path.exists():
            try:
                with open(output_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = []
        data.append(serializable)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)


def print_banner(pool: Any, completed_count: int, total_images: int) -> None:
    print("\n" + "=" * 70, flush=True)
    print("DENTAL AGENT: AUTONOMOUS DAILY CoT TRACE GENERATOR", flush=True)
    print("=" * 70, flush=True)
    print(f"* Active Keys in Pool : {len(pool.keys)} Gemini API keys", flush=True)
    print(f"* Configured Models   : {pool.models}", flush=True)
    print(f"* Primary Generator   : {GENERATOR_PROVIDER}/{GENERATOR_MODEL}", flush=True)
    print(f"* Groundedness Judge  : {VERIFIER_PROVIDER}/{VERIFIER_MODEL}", flush=True)
    print(f"* Dataset Progress    : {completed_count} / {total_images} images completed", flush=True)
    print("=" * 70 + "\n", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Autonomous Daily Synthetic CoT Trace Generator"
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="data/traces/cot_traces_aim1.jsonl",
        help="Path to output JSONL file (default: data/traces/cot_traces_aim1.jsonl)",
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
        "--status-only",
        action="store_true",
        help="Only display current key pool status and dataset progress without generating traces",
    )
    return parser.parse_args()


def main() -> None:
    load_env()
    cfg = load_config()
    args = parse_args()

    output_path = Path(args.output)
    pool = get_gemini_pool()

    # Load dataset
    imgs_df, annots_df, cats_df = load_dentex_dataset(
        data_dir=cfg.data_dir, split_name=args.split
    )

    # Filter to images with valid local paths and annotations
    valid_imgs = imgs_df[imgs_df["local_path"].notna()]
    annotated_ids = set(annots_df["image_id"].unique())
    eligible_imgs = valid_imgs[valid_imgs["id"].isin(annotated_ids)]

    total_eligible = len(eligible_imgs)
    completed_ids = load_completed_ids(output_path)
    remaining_imgs = eligible_imgs[~eligible_imgs["id"].isin(completed_ids)]

    print_banner(pool, len(completed_ids), total_eligible)

    # Display key pool status table
    status_df = pool.status()
    print("--- KEY POOL CAPACITY STATUS ---")
    print(status_df.to_string(index=False))
    print("-" * 70 + "\n")

    if args.status_only:
        print("Status check complete. Run without --status-only to begin trace generation.")
        return

    if remaining_imgs.empty:
        print(f"[DONE] All {total_eligible} images in split '{args.split}' have already been processed!")
        print(f"Traces are saved in: {output_path}")
        return

    todo_images = remaining_imgs.to_dict(orient="records")
    if args.max_images is not None:
        todo_images = todo_images[: args.max_images]

    print(f"Starting session: Processing {len(todo_images)} image(s) (Pacing delay: {args.pacing_delay}s)...")
    print(f"Output destination: {output_path}\n")

    generated_in_session = 0
    verified_in_session = 0
    session_start_time = time.time()

    for idx, img_record in enumerate(todo_images, start=1):
        image_id = int(img_record["id"])
        image_file = os.path.basename(str(img_record.get("local_path", "")))

        print(f"[{idx}/{len(todo_images)}] Processing Image ID {image_id} ({image_file})...", end=" ", flush=True)

        try:
            t0 = time.time()
            example = build_trace_example(
                image_id=image_id,
                images_df=imgs_df,
                annots_df=annots_df,
                categories_df=cats_df,
                k=args.k,
            )
            elapsed = time.time() - t0

            n_v = example.get("n_verified", 0)
            append_trace(output_path, example)

            generated_in_session += 1
            verified_in_session += n_v

            status_str = f"[OK] Verified ({n_v} trace)" if n_v > 0 else "[WARN] Unverified"
            print(f"{status_str} in {elapsed:.1f}s (Total saved: {len(completed_ids) + generated_in_session})", flush=True)

            # Respect safe inter-request pacing
            if idx < len(todo_images) and args.pacing_delay > 0:
                time.sleep(args.pacing_delay)

        except AllKeysExhaustedToday as e:
            print(f"\n\n[DAILY LIMIT REACHED] {e}")
            print("All configured Gemini API keys have exhausted their daily quotas for today.")
            print("The script has saved all progress cleanly and will resume automatically on the next run.")
            break
        except KeyboardInterrupt:
            print("\n\n[PAUSED] Session paused by user.")
            break
        except Exception as e:
            print(f"[ERROR] Error on Image ID {image_id}: {e}")
            time.sleep(2.0)

    # Session Summary
    total_time = time.time() - session_start_time
    total_saved = len(load_completed_ids(output_path))

    print("\n" + "=" * 70)
    print("SESSION SUMMARY")
    print("=" * 70)
    print(f"* Images Processed this session : {generated_in_session}")
    print(f"* Grounded Traces Generated    : {verified_in_session}")
    print(f"* Total Saved Dataset Size     : {total_saved} / {total_eligible} images")
    print(f"* Session Duration             : {total_time / 60:.1f} minutes")
    print(f"* Output File                  : {output_path}")
    print("=" * 70)

    print("\nGIT SYNC INSTRUCTIONS:")
    print("  Before starting on another machine : git pull")
    print(f"  To synchronize this session's data : git add {output_path} && git commit -m \"Add {generated_in_session} synthetic CoT traces\" && git push")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
