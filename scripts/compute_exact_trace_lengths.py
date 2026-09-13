#!/usr/bin/env python3
"""
scripts/compute_exact_trace_lengths.py

Offline exact token length computation for VLM-DENTAL SFT traces.

Pre-computes the exact, single-digit token count for every trace in the curriculum
using authentic base images, real tool executions (zoom_crop, denoise, window_level,
contralateral_compare, enhance_contrast), chat templates, and vision processor.

Outputs data/traces/trace_token_lengths.json which is synced to Hugging Face Hub.
On Kaggle/Colab, DentalSFTDataset loads this manifest to filter overlength traces
instantly (O(1) lookup) without running CPU vision processing or opening images.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

# Ensure repository root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv
load_dotenv(repo_root / ".env")

from transformers import AutoProcessor
from dental_agent.training.sft import DentalSFTDataset


CANONICAL_CURRICULUM_FILES = [
    "train_cot_traces_dentex.jsonl",
    "train_cot_traces_dentex_no_tools.jsonl",
    "train_cot_traces_tufts.jsonl",
    "train_cot_traces_tufts_no_tools.jsonl",
    "train_cot_traces_tufts_all.jsonl",
    "train_cot_traces_tufts_all_no_tools.jsonl",
    "train_cot_traces_healthy_dentex.jsonl",
    "train_cot_traces_healthy_dentex_no_tools.jsonl",
    "train_cot_traces_healthy_tufts.jsonl",
    "train_cot_traces_healthy_tufts_no_tools.jsonl",
    "train_cot_traces.jsonl",
    "train_cot_traces_no_tools.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute exact single-digit token lengths for all SFT clinical traces.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B"),
        help="Model ID or local path for tokenizer/processor",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("DENTAL_AGENT_DATA_DIR", "data"),
        help="Base directory for datasets, images, and traces",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/traces/trace_token_lengths.json",
        help="Path to save output token lengths JSON manifest",
    )
    parser.add_argument(
        "--trace-files",
        type=str,
        nargs="+",
        default=None,
        help="Optional specific trace JSONL file paths to process",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of all traces instead of resuming from existing manifest",
    )
    parser.add_argument(
        "--upload-hf",
        action="store_true",
        help="Automatically upload the generated manifest to Hugging Face Hub",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict no-fallback mode: abort (raise) on any trace encode failure "
        "instead of writing the text+1772-per-image estimate. Required for 10K gating.",
    )
    return parser.parse_args()


def save_manifest(
    out_path: Path,
    model_id: str,
    all_lengths: List[int],
    lengths_by_image_id: Dict[str, int],
    lengths_by_file_and_id: Dict[str, int],
) -> Dict[str, Any]:
    if not all_lengths:
        return {}
    arr = np.array(all_lengths)
    stats = {
        "count": int(len(arr)),
        "min": int(np.min(arr)),
        "p25": int(np.percentile(arr, 25)),
        "p50": int(np.percentile(arr, 50)),
        "p75": int(np.percentile(arr, 75)),
        "p90": int(np.percentile(arr, 90)),
        "p95": int(np.percentile(arr, 95)),
        "p99": int(np.percentile(arr, 99)),
        "max": int(np.max(arr)),
    }
    compliance = {
        "32768": int(np.sum(arr <= 32768)),
        "24576": int(np.sum(arr <= 24576)),
        "16384": int(np.sum(arr <= 16384)),
        "10240": int(np.sum(arr <= 10240)),
        "8192": int(np.sum(arr <= 8192)),
    }

    from collections import defaultdict
    by_file = defaultdict(list)
    for k, v in lengths_by_file_and_id.items():
        fn = k.split("::")[0]
        by_file[fn].append(v)

    file_stats = {}
    for fn, f_lens in sorted(by_file.items()):
        f_arr = np.array(f_lens)
        file_stats[fn] = {
            "count": int(len(f_arr)),
            "min": int(np.min(f_arr)),
            "p50": int(np.percentile(f_arr, 50)),
            "p90": int(np.percentile(f_arr, 90)),
            "max": int(np.max(f_arr)),
            "le_10240": int(np.sum(f_arr <= 10240)),
            "le_8192": int(np.sum(f_arr <= 8192)),
        }

    manifest = {
        "_meta": {
            "model_id": model_id,
            "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_traces": len(all_lengths),
            "stats": stats,
            "compliance": compliance,
            "file_stats": file_stats,
        },
        "lengths_by_image_id": lengths_by_image_id,
        "lengths_by_file_and_id": lengths_by_file_and_id,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return {"stats": stats, "compliance": compliance, "file_stats": file_stats}


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    traces_dir = data_dir / "traces" if (data_dir / "traces").exists() else Path("data/traces")

    # Determine files to process
    if args.trace_files:
        target_files = [Path(p) for p in args.trace_files]
    else:
        target_files = []
        for fn in CANONICAL_CURRICULUM_FILES:
            fp = traces_dir / fn
            if fp.is_file() and fp.stat().st_size > 0:
                target_files.append(fp)

    if not target_files:
        print(f"[ERROR] No valid trace files found in {traces_dir}!", flush=True)
        sys.exit(1)

    print("=" * 80, flush=True)
    print("VLM-DENTAL: OFFLINE EXACT TRACE TOKEN LENGTH EXTRACTION", flush=True)
    print(f"* Model ID    : {args.model_id}", flush=True)
    print(f"* Data Dir    : {data_dir}", flush=True)
    print(f"* Output File : {args.output_file}", flush=True)
    print(f"* Trace Files : {len(target_files)} files to process", flush=True)
    for tf in target_files:
        print(f"    - {tf.name} ({tf.stat().st_size / 1024 / 1024:.2f} MB)", flush=True)
    print("=" * 80, flush=True)

    print(f"\n[PROCESSOR] Loading processor for model '{args.model_id}'...", flush=True)
    processor = None

    # 1. First check local Qwen 3.5 repository snapshot or HF cache snapshots to eliminate network latency
    candidate_paths = [
        repo_root / "data" / "qwen3_5_tokenizer",
        Path("data/qwen3_5_tokenizer"),
    ]
    import glob
    candidate_paths.extend([Path(p) for p in glob.glob(str(Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*"))])
    candidate_paths.extend([Path(p) for p in glob.glob("C:/Users/*/.cache/huggingface/hub/models--Qwen--Qwen3.5-9B/snapshots/*")])

    for cp in candidate_paths:
        if cp.is_dir() and (cp / "tokenizer.json").is_file():
            try:
                print(f"  [PROCESSOR] Found local Qwen 3.5 snapshot at: {cp}", flush=True)
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(str(cp), local_files_only=True, trust_remote_code=True)
                print(f"  [PROCESSOR] Successfully instantiated {processor.__class__.__name__} from local snapshot!", flush=True)
                break
            except Exception as e:
                print(f"  [PROCESSOR] Local snapshot loading from {cp} failed ({e}); checking next...", flush=True)
                processor = None

    # 2. Fallback to standard AutoProcessor if local snapshot not found
    if processor is None:
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    if hasattr(processor, "tokenizer") and processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    lengths_by_image_id: Dict[str, int] = {}
    lengths_by_file_and_id: Dict[str, int] = {}
    all_lengths: List[int] = []

    out_path = Path(args.output_file)
    if out_path.is_file() and not args.recompute:
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
                lengths_by_image_id = existing_manifest.get("lengths_by_image_id", {})
                lengths_by_file_and_id = existing_manifest.get("lengths_by_file_and_id", {})
                print(f"[RESUME] Loaded {len(lengths_by_file_and_id)} pre-computed traces from {out_path}", flush=True)
        except Exception:
            pass

    total_processed = 0

    for f_idx, tf in enumerate(target_files, 1):
        print(f"\n[{f_idx}/{len(target_files)}] Ingesting trace file: {tf.name}...", flush=True)
        track = "no_tools" if "no_tools" in tf.name else "with_tools"

        # Initialize dataset (loads raw records without max_seq_len filtering)
        ds = DentalSFTDataset(
            data_path=str(tf),
            processor=processor,
            track=track,
            data_dir=str(data_dir),
            max_seq_len=None,
        )

        n_samples = len(ds)
        print(f"  Loaded {n_samples} traces. Processing exact tokens (tool crops + encoding)...", flush=True)

        for s_idx in range(n_samples):
            rec = ds.records[s_idx]
            rec_id = str(rec.get("image_id", s_idx))
            ds_name = str(rec.get("dataset", "default"))
            qualified_key = f"{tf.name}::{ds_name}::{rec_id}"
            image_key = f"{ds_name}::{rec_id}"

            if qualified_key in lengths_by_file_and_id:
                exact_tokens = lengths_by_file_and_id[qualified_key]
                lengths_by_image_id[image_key] = exact_tokens
                all_lengths.append(exact_tokens)
                total_processed += 1
                continue

            # Fallback 1: Check legacy un-namespaced key for single-dataset files
            legacy_key = f"{tf.name}::{rec_id}"
            if tf.name not in ("train_cot_traces.jsonl", "train_cot_traces_no_tools.jsonl") and legacy_key in lengths_by_file_and_id:
                exact_tokens = lengths_by_file_and_id[legacy_key]
                lengths_by_image_id[image_key] = exact_tokens
                lengths_by_file_and_id[qualified_key] = exact_tokens
                all_lengths.append(exact_tokens)
                total_processed += 1
                continue

            # Fallback 2: Check matching record in the cohort split file
            cohort_fn = f"train_cot_traces_{ds_name}.jsonl" if "no_tools" not in tf.name else f"train_cot_traces_{ds_name}_no_tools.jsonl"
            cohort_k1 = f"{cohort_fn}::{ds_name}::{rec_id}"
            cohort_k2 = f"{cohort_fn}::{rec_id}"
            found_cohort_val = lengths_by_file_and_id.get(cohort_k1) or lengths_by_file_and_id.get(cohort_k2)
            if found_cohort_val is not None:
                exact_tokens = found_cohort_val
                lengths_by_image_id[image_key] = exact_tokens
                lengths_by_file_and_id[qualified_key] = exact_tokens
                all_lengths.append(exact_tokens)
                total_processed += 1
                continue

            try:
                # __getitem__ executes authentic tools, applies chat template, and calls processor
                enc = ds[s_idx]
                exact_tokens = int(enc["input_ids"].shape[1])
            except Exception as e:
                if getattr(args, "strict", False):
                    # Strict no-fallback mode: images were mandated, so any encode
                    # failure aborts instead of writing an estimate into the manifest.
                    raise RuntimeError(
                        f"[STRICT] Exact encode failed for {tf.name}::{ds_name}::{rec_id}: {e}. "
                        "Aborting (no text+1772 estimate allowed)."
                    ) from e
                print(f"  [WARNING] Error encoding sample {s_idx} ({rec_id}): {e}; falling back to estimate.", flush=True)
                # Fallback estimation: encode text + 1772 tokens per image
                raw_msgs = rec.get("messages", [])
                text_content = ""
                img_cnt = 0
                for m in raw_msgs:
                    c = m.get("content", "")
                    if isinstance(c, list):
                        for item in c:
                            if isinstance(item, dict):
                                if item.get("type") == "image":
                                    img_cnt += 1
                                elif item.get("type") == "text":
                                    text_content += item.get("text", "")
                    elif isinstance(c, str):
                        text_content += c
                t_tokens = len(processor.tokenizer.encode(text_content, add_special_tokens=False))
                exact_tokens = t_tokens + max(img_cnt, 1) * 1772

            lengths_by_image_id[image_key] = exact_tokens
            lengths_by_file_and_id[qualified_key] = exact_tokens
            all_lengths.append(exact_tokens)
            total_processed += 1

            if (s_idx + 1) % 50 == 0 or (s_idx + 1) == n_samples:
                print(f"  Processed {s_idx + 1}/{n_samples} traces (sample {rec_id} = {exact_tokens} tokens)...", flush=True)
                save_manifest(out_path, args.model_id, list(lengths_by_file_and_id.values()), lengths_by_image_id, lengths_by_file_and_id)

        # Checkpoint save after completing each trace file
        save_manifest(out_path, args.model_id, list(lengths_by_file_and_id.values()), lengths_by_image_id, lengths_by_file_and_id)
        print(f"  [CHECKPOINT] Manifest updated at {out_path} ({len(lengths_by_file_and_id)} traces total).", flush=True)

    res = save_manifest(out_path, args.model_id, list(lengths_by_file_and_id.values()), lengths_by_image_id, lengths_by_file_and_id)
    stats = res.get("stats", {})
    compliance = res.get("compliance", {})

    print("\n" + "=" * 80)
    print("EXACT TOKEN LENGTH EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"* Output File : {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    print(f"* Total Traces: {stats['count']}")
    print(f"* Min Length  : {stats['min']} tokens")
    print(f"* Median (P50): {stats['p50']} tokens")
    print(f"* 90th %ile   : {stats['p90']} tokens")
    print(f"* 95th %ile   : {stats['p95']} tokens")
    print(f"* 99th %ile   : {stats['p99']} tokens")
    print(f"* Max Length  : {stats['max']} tokens")
    print("-" * 80)
    print("COMPLIANCE AT STATIC SEQUENCE LENGTHS:")
    for ceiling, count in compliance.items():
        pct = (count / stats['count']) * 100
        print(f"  <= {int(ceiling):5d} tokens: {count:4d} / {stats['count']} ({pct:5.2f}%)")
    print("=" * 80)

    if args.upload_hf:
        print("\n[HF SYNC] Uploading manifest to Hugging Face Hub...")
        try:
            from scripts.sync_traces_hf import upload_traces
            upload_traces(source_dir=str(out_path.parent), files=[out_path.name])
            print("[HF SYNC] Successfully uploaded manifest to Hugging Face!")
        except Exception as e:
            print(f"[HF SYNC WARNING] Upload failed ({e}); you can upload manually via: python scripts/sync_traces_hf.py --upload")


if __name__ == "__main__":
    main()
