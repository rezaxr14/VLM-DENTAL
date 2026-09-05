#!/usr/bin/env python3
"""
Zero-Shot VLM Baseline Evaluation Runner (§6, Baseline #1).

Evaluates general-purpose Vision-Language Models (GPT-4o, Gemini 3.5/3.7, Claude 3.5/3.7,
NVIDIA NIM, Groq, OpenRouter, and local vLLM) on dental panoramic radiographs without
domain fine-tuning and without tool access.

Usage:
    python scripts/run_zero_shot.py --provider gemini --model gemini-3.5-flash-lite
    python scripts/run_zero_shot.py --provider nvidia_nim --model meta/muse-glimmer-30b
    python scripts/run_zero_shot.py --provider local --model Qwen/Qwen3.5-9B

Parallel workers (horizontal slicing with git sync):
    python scripts/run_zero_shot.py --provider gemini --total-slices 4 --slice-index 1 --git-sync-every 5
    python scripts/run_zero_shot.py --provider gemini --total-slices 4 --slice-index 2 --git-sync-every 5
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
from dental_agent.data.fdi_utils import row_to_fdi
from dental_agent.data.tufts import load_tufts_dataset
from dental_agent.data.slicing import get_slice_ids
from dental_agent.training.api_pool import (
    call_llm,
    verify_local_server_health,
    RPDLimitExhausted,
)
from dental_agent.evaluation.baselines import (
    ZERO_SHOT_PROMPT,
    parse_zero_shot_response,
    match_zero_shot_finding,
    majority_class_baseline_metrics,
)
from dental_agent.evaluation.metrics import (
    compute_evaluation_metrics,
    compute_diagnostic_metrics,
    expected_calibration_error,
    bootstrap_metric_ci,
    match_multi_findings,
    extract_predicted_findings,
    normalize_dental_diagnosis,
)
from dental_agent.evaluation.reporting import (
    generate_summary_table,
    generate_markdown_report,
)
from dental_agent.training.git_sync import sync_and_push
from dental_agent.utils.serialization import to_jsonable
from PIL import Image


# ---------------------------------------------------------------------------
# Output path defaults
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = "data/evaluations"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_completed_ids(output_path: Path, retry_empty: bool = True) -> set[int]:
    """Read existing output file and extract processed image IDs for resume.
    If retry_empty is True, excludes records where predictions are empty or format failed.
    Truncated records (finish_reason='length') are ALWAYS excluded regardless of retry_empty,
    because a cut-off response is definitionally incomplete and must be re-evaluated."""
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
                if "image_id" in record:
                    img_id = int(record["image_id"])
                    # Always re-run truncated responses — a cut-off reply is never valid
                    if record.get("finish_reason") == "length":
                        continue
                    if retry_empty:
                        preds = record.get("predictions", [])
                        format_ok = record.get("format_ok", False)
                        if not preds or not format_ok:
                            continue  # Treat as incomplete so it gets re-evaluated
                    completed.add(img_id)
            except Exception:
                pass
    return completed


def save_evaluation_record_atomic(output_path: Path, record: dict[str, Any]) -> None:
    """Save record to JSONL atomically, replacing any existing entry for this image_id in-place."""
    records_by_id: dict[int, dict[str, Any]] = {}
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if "image_id" in r:
                        records_by_id[int(r["image_id"])] = r
                except Exception:
                    pass

    records_by_id[int(record["image_id"])] = record

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(".tmp")
    with open(temp_path, "w", encoding="utf-8") as f:
        for r in records_by_id.values():
            f.write(json.dumps(r) + "\n")
    temp_path.replace(output_path)


def resolve_provider_and_model(args: argparse.Namespace) -> tuple[str, str]:
    """Resolve provider and model following precedence: CLI args > .env > sensible defaults."""
    provider = args.provider
    if not provider:
        provider = (
            os.environ.get("ZERO_SHOT_PROVIDER")
            or os.environ.get("VERIFIER_PROVIDER")
            or os.environ.get("GENERATOR_PROVIDER")
            or "gemini"
        )
    provider = provider.lower()

    model = args.model
    if not model:
        prefix = provider.upper().replace("_NIM", "")
        model = (
            os.environ.get(f"{prefix}_ZERO_SHOT_MODEL")
            or os.environ.get(f"{prefix}_VERIFIER_MODEL")
            or os.environ.get(f"{prefix}_GENERATOR_MODEL")
            or os.environ.get("GENERATOR_MODEL")
        )
        if not model:
            default_models = {
                "gemini": "gemini-3.5-flash-lite",
                "nvidia_nim": "meta/llama-3.2-11b-vision-instruct",
                "groq": "qwen/qwen3.6-27b",
                "openrouter": "google/gemma-4-31b-it:free",
                "local": "Qwen/Qwen3.5-9B",
                "transformers": "Qwen/Qwen3.5-9B",
                "local_hf": "Qwen/Qwen3.5-9B",
                "openai": "gpt-4o",
                "anthropic": "claude-3-7-sonnet-20250219",
            }
            model = default_models.get(provider, "gemini-3.5-flash-lite")

    return provider, model


def print_banner(provider: str, model: str, completed_count: int, total_images: int, output_path: Path) -> None:
    print("\n" + "=" * 70, flush=True)
    print("DENTAL AGENT: ZERO-SHOT VLM BASELINE EVALUATION (§6, Baseline #1)", flush=True)
    print("=" * 70, flush=True)
    print(f"* Provider         : {provider.upper()}", flush=True)
    print(f"* Model            : {model}", flush=True)
    print(f"* Output File      : {output_path}", flush=True)
    print(f"* Progress         : {completed_count} / {total_images} images", flush=True)
    print("=" * 70 + "\n", flush=True)


# ---------------------------------------------------------------------------
# Core Evaluation Loop
# ---------------------------------------------------------------------------

KNOWN_PROVIDERS: set[str] = {
    "gemini", "nvidia_nim", "groq", "openrouter", "local", "transformers", "local_hf", "openai", "anthropic"
}


def _is_rate_limit_or_fatal_error(e: Exception) -> bool:
    msg = str(e).lower()
    return (
        isinstance(e, RPDLimitExhausted)
        or "429" in msg
        or "rate limit" in msg
        or "too many requests" in msg
        or "rpd limit" in msg
        or "quota" in msg
        or "insufficient" in msg
        or "credit" in msg
    )


def run_zero_shot_evaluation(args: argparse.Namespace) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if getattr(args, "repetition_penalty", None) is not None:
        os.environ["TRANSFORMERS_REPETITION_PENALTY"] = str(args.repetition_penalty)
    load_env()
    cfg = load_config(args.config)

    # 1. Expand evaluation targets
    targets: list[tuple[str, str]] = []
    
    # Check if comma-separated list of items has provider:model syntax (e.g. "gemini:gemini-3.5-flash-lite,nvidia_nim:meta/muse-glimmer-30b")
    if args.model and any(":" in item and item.split(":", 1)[0].strip().lower() in KNOWN_PROVIDERS for item in args.model.split(",")):
        for item in args.model.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" in item and item.split(":", 1)[0].strip().lower() in KNOWN_PROVIDERS:
                p, m = item.split(":", 1)
                targets.append((p.strip().lower(), m.strip()))
            else:
                p, m = resolve_provider_and_model(argparse.Namespace(**{**vars(args), "model": item}))
                targets.append((p, m))
    else:
        raw_models = [m.strip() for m in args.model.split(",")] if args.model else [None]
        raw_providers = [p.strip() for p in args.provider.split(",")] if args.provider else [None]

        if len(raw_models) > 1 and len(raw_providers) == 1:
            for m in raw_models:
                p, resolved_m = resolve_provider_and_model(argparse.Namespace(**{**vars(args), "model": m, "provider": raw_providers[0]}))
                targets.append((p, resolved_m))
        elif len(raw_providers) > 1 and len(raw_models) == 1:
            for p in raw_providers:
                resolved_p, m = resolve_provider_and_model(argparse.Namespace(**{**vars(args), "model": raw_models[0], "provider": p}))
                targets.append((resolved_p, m))
        elif len(raw_models) > 1 and len(raw_providers) == len(raw_models):
            for p, m in zip(raw_providers, raw_models):
                resolved_p, resolved_m = resolve_provider_and_model(argparse.Namespace(**{**vars(args), "model": m, "provider": p}))
                targets.append((resolved_p, resolved_m))
        else:
            p, m = resolve_provider_and_model(args)
            targets.append((p, m))

    dataset_name = args.dataset.strip().lower()
    split_name = args.split.strip().lower()

    # Track providers that hit 429/quota limits to hard-skip all other models on that provider
    failed_providers: set[str] = set()

    for p_idx, (provider, model) in enumerate(targets, start=1):
        if provider in failed_providers:
            print(f"\n⏩ [SKIP-PROVIDER] Skipping target {p_idx}/{len(targets)}: {provider.upper()} ({model}) — provider '{provider}' encountered a 429/quota limit earlier.\n", flush=True)
            continue

        if len(targets) > 1:
            print("\n" + "#" * 70)
            print(f"EVALUATION TARGET {p_idx}/{len(targets)}: {provider.upper()} ({model})")
            print("#" * 70 + "\n")
        
        provider_failed = _run_zero_shot_for_target(args, cfg, dataset_name, split_name, provider, model)
        if provider_failed:
            failed_providers.add(provider)
            print(f"\n⚠️ [PROVIDER-LOCK] Marked provider '{provider}' as failed. All remaining models on '{provider}' will be skipped immediately.", flush=True)


def _run_zero_shot_for_target(
    args: argparse.Namespace,
    cfg: Any,
    dataset_name: str,
    split_name: str,
    provider: str,
    model: str,
) -> None:
    # Standardized, unambiguous output naming convention:
    # data/evaluations/zero_shot_{dataset}_{split}_{provider}_{model}.jsonl
    safe_model_tag = model.replace("/", "--").replace(":", "-")
    os.makedirs(args.output_dir, exist_ok=True)
    if args.output and len(getattr(args, "model", "").split(",")) <= 1:
        output_path = Path(args.output)
    else:
        output_path = Path(args.output_dir) / f"zero_shot_{dataset_name}_{split_name}_{provider}_{safe_model_tag}.jsonl"

    # 1. Load Dataset
    print(f"Loading {dataset_name.upper()} dataset (split='{split_name}')...")
    if dataset_name == "tufts":
        imgs_df, annots_df, cats_df = load_tufts_dataset(data_dir=cfg.data_dir)
    else:
        imgs_df, annots_df, cats_df = load_dentex_dataset(
            data_dir=cfg.data_dir, split_name=split_name
        )

    annotated_ids = set(annots_df["image_id"].unique())
    eligible_imgs = imgs_df[imgs_df["id"].isin(annotated_ids)]

    # 2. Horizontal Slicing
    if args.total_slices > 1:
        slice_ids = get_slice_ids(eligible_imgs["id"].tolist(), args.total_slices, args.slice_index, args.slice_seed)
        eligible_imgs = eligible_imgs[eligible_imgs["id"].isin(slice_ids)]

    # 3. Targeted Dynamic Slice Download if configured
    repo_id = os.environ.get("TUFTS_IMAGES_REPO" if dataset_name == "tufts" else "DENTEX_IMAGES_REPO")
    missing_mask = eligible_imgs["local_path"].isna() | ~eligible_imgs["local_path"].apply(lambda p: os.path.exists(str(p)) if p and str(p) != "None" else False)
    if repo_id and missing_mask.any():
        missing_ids = eligible_imgs[missing_mask]["id"].tolist()
        if missing_ids:
            print(f"Fetching {len(missing_ids)} missing images from {repo_id}...")
            if dataset_name == "tufts":
                from dental_agent.data.tufts import download_tufts_slice as _download_slice
            else:
                from dental_agent.data.dentex import download_dentex_slice as _download_slice
            local_paths_map = _download_slice(missing_ids, repo_id=repo_id, cache_dir=cfg.data_dir, split_name=args.split)
            def _update_path(row):
                pid = row["id"]
                if pid in local_paths_map and local_paths_map[pid] is not None:
                    return str(local_paths_map[pid])
                return row["local_path"]
            imgs_df["local_path"] = imgs_df.apply(_update_path, axis=1)
            eligible_imgs = imgs_df[imgs_df["id"].isin(eligible_imgs["id"])]

    eligible_imgs = eligible_imgs[eligible_imgs["local_path"].notna() & (eligible_imgs["local_path"] != "None")]

    if getattr(args, "fresh", False) and output_path.exists():
        output_path.unlink()
        print(f"  [FRESH] Removed existing output file: {output_path}")

    total_eligible = len(eligible_imgs)
    retry_empty = getattr(args, "retry_empty", True)
    completed_ids = load_completed_ids(output_path, retry_empty=retry_empty)
    remaining_imgs = eligible_imgs[~eligible_imgs["id"].isin(completed_ids)]

    if getattr(args, "start_image_id", None) is not None:
        remaining_imgs = remaining_imgs[remaining_imgs["id"] >= args.start_image_id]

    slice_info = f" (Slice {args.slice_index}/{args.total_slices})" if args.total_slices > 1 else ""
    print(f"Targeting{slice_info}: {len(eligible_imgs)} eligible images (Completed: {len(completed_ids)}, Remaining: {len(remaining_imgs)}).")
    print_banner(provider, model, len(completed_ids), total_eligible, output_path)

    if args.status_only:
        print("Status check complete. Run without --status-only to begin evaluation.")
        return

    DENTEX_DEFAULT_DIAGNOSES = {
        0: "Impacted",
        1: "Caries",
        2: "Periapical Lesion",
        3: "Deep Caries",
    }
    cat_lookup = (
        dict(zip(cats_df["id"], cats_df["name"]))
        if cats_df is not None and len(cats_df)
        else DENTEX_DEFAULT_DIAGNOSES
    )

    # Compute Majority Class Baseline if requested
    if args.run_majority_baseline:
        print("\n--- Computing Majority-Class Baseline Floor on Cohort ---")
        majority_metrics = majority_class_baseline_metrics(
            holdout_image_ids=eligible_imgs["id"].tolist(),
            annots_df=annots_df,
            categories_df=cats_df,
        )
        print(f"Majority Baseline FDI Accuracy    : {majority_metrics.get('fdi_accuracy', 0.0):.3f}")
        print(f"Majority Baseline Balanced Acc   : {majority_metrics.get('diagnosis_balanced_accuracy', 0.0):.3f}\n")

    if remaining_imgs.empty:
        print(f"[DONE] All {total_eligible} images have already been evaluated!")
        _print_final_summary(output_path, provider, model, args)
        return False

    if args.max_images is not None and args.max_images > 0:
        remaining_imgs = remaining_imgs.iloc[:args.max_images]

    print(f"Starting Zero-Shot evaluation: {len(remaining_imgs)} image(s) (Pacing delay: {args.pacing_delay}s)...")

    # Evaluation loop
    output_path.parent.mkdir(parents=True, exist_ok=True)
    evaluated_in_session = 0
    failed_in_session = 0
    since_last_sync = 0
    session_start_time = time.time()

    for idx, (_, img_record) in enumerate(remaining_imgs.iterrows(), start=1):
        image_id = int(img_record["id"])
        image_path = str(img_record.get("local_path", ""))
        image_file = os.path.basename(image_path)

        print(f"[{idx}/{len(remaining_imgs)}] Evaluating Image ID {image_id} ({image_file})...", flush=True)

        if not os.path.exists(image_path):
            print(f"  [WARN] Image file missing: {image_path}. Skipping.")
            continue

        try:
            # Respect pacing delay
            if args.pacing_delay > 0:
                time.sleep(args.pacing_delay)

            # Health check for local vLLM
            if provider == "local":
                health_retries = 0
                while not verify_local_server_health():
                    health_retries += 1
                    if health_retries > 24:
                        raise RuntimeError("Local vLLM server unresponsive > 2m. Aborting.")
                    print(f"  vLLM unresponsive. Waiting 5s... ({health_retries}/24)")
                    time.sleep(5)

            # Extract ALL ground truth findings for this image, dataset-aware (fdi_utils.row_to_fdi) --
            # this used to call dentex_row_to_fdi unconditionally, which would have silently
            # double-incremented every quadrant/tooth_position on the tufts branch above.
            anns = annots_df[annots_df["image_id"] == image_id]
            if anns.empty:
                print(f"  [WARN] No annotations for image {image_id}. Skipping.")
                continue

            gt_findings = []
            for _, ann_row in anns.iterrows():
                q, pos = row_to_fdi(ann_row)
                d_id = ann_row.get("category_id_3")
                try:
                    d_id_int = int(d_id)
                except (ValueError, TypeError):
                    d_id_int = None
                d_raw = cat_lookup.get(d_id, cat_lookup.get(d_id_int, DENTEX_DEFAULT_DIAGNOSES.get(d_id_int, "Caries")))
                gt_findings.append({
                    "quadrant": q,
                    "tooth_position": pos,
                    "diagnosis": normalize_dental_diagnosis(d_raw),
                    "raw_diagnosis": str(d_raw),
                })

            prefix = provider.upper().replace('_NIM', '')

            # Configuration Precedence: CLI Arguments > .env variables > Provider Defaults
            # 1. Image Max Dimension
            env_dim = os.environ.get(f"{prefix}_IMAGE_MAX_DIM")
            if args.image_max_dim is not None:
                dim_to_use = args.image_max_dim
            elif env_dim is not None and env_dim.strip():
                dim_to_use = int(env_dim)
            else:
                default_max_dims = {"GROQ": 640, "OPENROUTER": 1600}
                dim_to_use = default_max_dims.get(prefix, 0)

            image = Image.open(image_path).convert("RGB")
            if dim_to_use > 0:
                image.thumbnail((dim_to_use, dim_to_use), Image.Resampling.LANCZOS)

            # 2. Max Tokens
            env_tokens = os.environ.get(f"{prefix}_MAX_TOKENS")
            if args.max_tokens is not None:
                tokens_to_use = args.max_tokens
            elif env_tokens is not None and env_tokens.strip():
                tokens_to_use = int(env_tokens)
            else:
                default_max_tokens = {
                    "GROQ": 4096,
                    "LOCAL": 16384,
                    "TRANSFORMERS": 16384,
                    "LOCAL_HF": 16384,
                    "NVIDIA": 16384,
                    "GEMINI": 16384,
                    "OPENROUTER": 16384,
                    "OPENAI": 4096,
                    "ANTHROPIC": 4096,
                }
                tokens_to_use = default_max_tokens.get(prefix, 4096)

            # Call VLM with ZERO_SHOT_PROMPT (with reasoning & multi-finding guidelines)
            resp_out = call_llm(
                provider=provider,
                model=model,
                system_prompt="You are an expert dental radiologist analyzing panoramic dental radiographs. Provide concise reasoning and always complete your response with the final JSON object.",
                user_content=ZERO_SHOT_PROMPT,
                image=image,
                temperature=args.temperature,
                max_tokens=tokens_to_use,
                return_metadata=True,
                repetition_penalty=getattr(args, "repetition_penalty", None),
            )

            if isinstance(resp_out, tuple):
                raw_reply, meta = resp_out
            else:
                raw_reply, meta = resp_out, {}

            finish_reason = meta.get("finish_reason", "stop")
            usage_info = meta.get("usage", {})
            comp_tokens = usage_info.get("completion_tokens", 0)

            if finish_reason == "length":
                print(f"  ⚠️  [TRUNCATED] Model hit max_tokens limit ({tokens_to_use}) before completing reasoning!", flush=True)

            # Parse and match predictions against ALL ground truth findings
            parsed_raw = parse_zero_shot_response(raw_reply)
            pred_findings = extract_predicted_findings(parsed_raw)
            match_res = match_multi_findings(gt_findings, pred_findings)

            # Check format adherence
            format_ok = bool(parsed_raw is not None and (len(pred_findings) > 0 or isinstance(parsed_raw, (dict, list))))
            
            # Clinical summary indicators
            fdi_ok = match_res["fdi_tp"] > 0
            exact_match = match_res["exact_tp"] > 0
            all_exact_match = (match_res["exact_fn"] == 0 and match_res["exact_fp"] == 0)

            primary_matched = match_res["matched_pairs"][0]["pred"] if match_res["matched_pairs"] else (pred_findings[0] if pred_findings else None)
            confidence = primary_matched.get("confidence") if primary_matched else None

            # Formatted console display
            gt_str = ", ".join(f"FDI {g['quadrant']}{g['tooth_position']} ({g['diagnosis']})" for g in gt_findings)
            pred_str = ", ".join(f"FDI {p['quadrant']}{p['tooth_position']} ({p['diagnosis']})" for p in pred_findings) if pred_findings else "None"
            
            print(f"  GT ({len(gt_findings)}): {gt_str}")
            print(f"  Pred ({len(pred_findings)}): {pred_str}")
            if not pred_findings:
                if raw_reply:
                    preview = raw_reply.replace("\n", " ").strip()
                    if len(preview) > 140:
                        preview = preview[:140] + "..."
                    print(f"  ℹ️  [NO ABNORMALITIES REPORTED] Model output {len(raw_reply.split())} words (finish='{finish_reason}'). Preview: {preview}")
                else:
                    print(f"  ⚠️  [EMPTY API RESPONSE] Provider returned 0 content tokens (finish='{finish_reason}')")
            print(
                f"  Score: FDI Match: {match_res['fdi_tp']}/{len(gt_findings)} (P: {match_res['fdi_precision']:.2f}, R: {match_res['fdi_recall']:.2f}, F1: {match_res['fdi_f1']:.2f}) | "
                f"Exact Match: {match_res['exact_tp']}/{len(gt_findings)} (P: {match_res['exact_precision']:.2f}, R: {match_res['exact_recall']:.2f}, F1: {match_res['exact_f1']:.2f}) | "
                f"Closeness: {match_res['closeness_score']:.2f} (Spatial: {match_res['spatial_proximity']:.2f}, Diag: {match_res['diagnostic_similarity']:.2f})",
                flush=True,
            )

            record = to_jsonable({
                "image_id": image_id,
                "dataset": dataset_name,
                "split": args.split,
                "provider": provider,
                "model": model,
                "ground_truth": gt_findings,
                "predictions": pred_findings,
                "matched_pairs": match_res["matched_pairs"],
                "fdi_precision": match_res["fdi_precision"],
                "fdi_recall": match_res["fdi_recall"],
                "fdi_f1": match_res["fdi_f1"],
                "exact_precision": match_res["exact_precision"],
                "exact_recall": match_res["exact_recall"],
                "exact_f1": match_res["exact_f1"],
                "closeness_score": match_res["closeness_score"],
                "spatial_proximity": match_res["spatial_proximity"],
                "diagnostic_similarity": match_res["diagnostic_similarity"],
                "fdi_correct": fdi_ok,
                "quadrant_correct": fdi_ok,
                "tooth_position_correct": fdi_ok,
                "diagnosis_correct": exact_match,
                "exact_match": exact_match,
                "all_exact_match": all_exact_match,
                "final_answer": primary_matched,
                "raw_output": raw_reply,
                "format_ok": format_ok,
                "finish_reason": finish_reason,
                "confidence": confidence,
                "timestamp": time.time(),
            })

            # Save to JSONL atomically (replaces in-place if image_id was already present)
            save_evaluation_record_atomic(output_path, record)

            evaluated_in_session += 1
            since_last_sync += 1

            # Incremental Git Sync
            if args.git_sync_every > 0 and since_last_sync >= args.git_sync_every:
                print(f"  [git-sync] Syncing checkpoint after {since_last_sync} images...", flush=True)
                sync_and_push([str(output_path)], commit_message=f"eval(zero_shot): checkpoint {provider}/{model}")
                since_last_sync = 0

            if args.pacing_delay > 0:
                time.sleep(args.pacing_delay)

        except Exception as e:
            failed_in_session += 1
            if _is_rate_limit_or_fatal_error(e):
                print(f"\n🛑 [429 / RATE-LIMIT] Provider '{provider}' encountered a rate limit or quota exhaustion: {e}", flush=True)
                print(f"⏩ Hard-skipping provider '{provider}' and advancing immediately to the next provider...", flush=True)
                if args.git_sync_every > 0 and since_last_sync > 0:
                    sync_and_push([str(output_path)], commit_message=f"eval(zero_shot): checkpoint {provider}/{model}")
                return True
            
            print(f"  [ERROR] Image ID {image_id} failed on {provider}/{model}: {e}", flush=True)
            if not args.ignore_api_errors:
                print("Aborting. Use --ignore-api-errors to continue past single-image errors.")
                raise e

    # Final Sync
    if args.git_sync_every > 0 and since_last_sync > 0:
        print(f"\n[git-sync] Performing final session push...", flush=True)
        sync_and_push([str(output_path)], commit_message=f"eval(zero_shot): session complete {provider}/{model}")

    elapsed = time.time() - session_start_time
    print(f"\nSession finished in {elapsed:.1f}s. Evaluated: {evaluated_in_session}, Failed: {failed_in_session}")
    _print_final_summary(output_path, provider, model, args)
    return False


def _print_final_summary(output_path: Path, provider: str, model: str, args: argparse.Namespace) -> None:
    if not output_path.exists():
        return

    records = []
    with open(output_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass

    if not records:
        print("No evaluation records found.")
        return

    n = len(records)
    fmt_ok = sum(1 for r in records if r.get("format_ok"))
    fdi_ok = sum(1 for r in records if r.get("fdi_correct"))
    quad_ok = sum(1 for r in records if r.get("quadrant_correct"))
    pos_ok = sum(1 for r in records if r.get("tooth_position_correct"))
    diag_ok = sum(1 for r in records if r.get("diagnosis_correct"))
    exact_ok = sum(1 for r in records if r.get("exact_match"))

    mean_fdi_f1 = sum(r.get("fdi_f1", 1.0 if r.get("fdi_correct") else 0.0) for r in records) / n
    mean_exact_f1 = sum(r.get("exact_f1", 1.0 if r.get("exact_match") else 0.0) for r in records) / n
    mean_closeness = sum(r.get("closeness_score", 0.0) for r in records) / n
    mean_spatial = sum(r.get("spatial_proximity", 0.0) for r in records) / n
    mean_diag_sim = sum(r.get("diagnostic_similarity", 0.0) for r in records) / n

    confidences = [r["confidence"] for r in records if r.get("confidence") is not None]
    correctness = [int(r.get("exact_match", False)) for r in records if r.get("confidence") is not None]
    ece = expected_calibration_error(confidences, correctness) if len(confidences) >= 5 else 0.0

    point_em, em_low, em_high = bootstrap_metric_ci(
        records,
        lambda recs: sum(1 for r in recs if r.get("exact_match")) / len(recs) if recs else 0.0,
    )

    metrics_dict = {
        f"Zero-Shot ({provider}/{model})": {
            "format_adherence": fmt_ok / n,
            "quadrant_accuracy": quad_ok / n,
            "tooth_position_accuracy": pos_ok / n,
            "fdi_localization_accuracy": fdi_ok / n,
            "fdi_localization_f1": mean_fdi_f1,
            "pathology_accuracy": diag_ok / n,
            "pathology_macro_f1": diag_ok / n,
            "exact_match_accuracy": exact_ok / n,
            "exact_match_f1": mean_exact_f1,
            "closeness_score": mean_closeness,
            "spatial_proximity": mean_spatial,
            "diagnostic_similarity": mean_diag_sim,
            "exact_match_ci_95": [em_low, em_high],
            "ece": ece,
            "mean_tool_calls": 0.0,
            "total_samples": float(n),
        }
    }

    print("\n" + "=" * 70)
    print(f"BENCHMARK RESULTS: {provider.upper()} / {model} (n={n})")
    print("=" * 70)
    print(f"Format Compliance         : {fmt_ok / n * 100:.1f}% ({fmt_ok}/{n})")
    print(f"FDI Localization Accuracy : {fdi_ok / n * 100:.1f}% ({fdi_ok}/{n}) [Mean F1: {mean_fdi_f1:.3f}]")
    print(f"Pathology Diagnosis Acc   : {diag_ok / n * 100:.1f}% ({diag_ok}/{n})")
    print(f"Exact Match Accuracy      : {exact_ok / n * 100:.1f}% ({exact_ok}/{n}) [Mean F1: {mean_exact_f1:.3f}]")
    print(f"Exact Match 95% CI        : [{em_low * 100:.1f}%, {em_high * 100:.1f}%]")
    print(f"Continuous Closeness Score: {mean_closeness:.3f} (Spatial: {mean_spatial:.3f}, Diag Sim: {mean_diag_sim:.3f})")
    print(f"Expected Calibration Error: {ece:.4f}")
    print("=" * 70)

    summary_table = generate_summary_table(metrics_dict, table_format="github")
    print("\n" + summary_table + "\n")

    if getattr(args, "generate_report", True):
        safe_model_tag = model.replace('/', '--').replace(':', '-')
        report_path = Path("experiments") / f"zero_shot_report_{provider}_{safe_model_tag}.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        generate_markdown_report(metrics_dict, output_path=report_path)
        if args.git_sync_every > 0:
            sync_and_push([str(output_path), str(report_path)], commit_message=f"eval(zero_shot): final results & report {provider}/{model}")


# ---------------------------------------------------------------------------
# CLI Argument Parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Zero-Shot VLM Baseline Evaluation CLI (§6, Baseline #1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", "-c", default=None, help="Path to config YAML")
    parser.add_argument("--dataset", default="dentex", help="Dataset name ('dentex' or 'tufts')")
    parser.add_argument("--split", default="test", help="Dataset split ('test', 'validation', 'train')")
    parser.add_argument("--provider", default=None, help="Provider ('gemini', 'nvidia_nim', 'groq', 'openrouter', 'transformers', 'local', 'openai', 'anthropic')")
    parser.add_argument("--model", default=None, help="Model name / checkpoint identifier or comma-separated list")
    parser.add_argument("--suite", choices=["option7", "benchmark", "all"], default=None, help="Run pre-configured multi-model benchmark suite (7 NVIDIA NIM + 3 OpenRouter models)")
    
    # Slicing & Workers
    parser.add_argument("--total-slices", type=int, default=1, help="Total parallel slices across instances")
    parser.add_argument("--slice-index", type=int, default=1, help="1-based slice index for this instance")
    parser.add_argument("--slice-seed", type=int, default=42, help="Seed for deterministic dataset slicing")
    
    # Pacing, Limits & Sync
    parser.add_argument("--pacing-delay", type=float, default=1.5, help="Delay (seconds) between successive LLM requests")
    parser.add_argument("--git-sync-every", type=int, default=0, help="Push checkpoint to Git every N images (0 = disabled)")
    parser.add_argument("--max-images", type=int, default=None, help="Cap number of images to evaluate in this run")
    parser.add_argument("--start-image-id", "--start-id", type=int, default=None, help="Only evaluate images with ID >= start-image-id")
    
    # Output & Scaling
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for evaluation JSONL outputs")
    parser.add_argument("--output", default=None, help="Explicit output file path override")
    parser.add_argument("--image-max-dim", type=int, default=None, help="Max image dimension (0 = full resolution). Falls back to .env then provider defaults if omitted.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Repetition penalty for local/transformers generation (e.g. 1.10 to prevent reasoning loops)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Max tokens for VLM response (reasoning thought + findings JSON). Falls back to .env then provider defaults if omitted.")
    
    # Flags
    parser.add_argument("--fresh", "--overwrite", action="store_true", help="Start evaluation from scratch and overwrite existing output JSONL")
    parser.add_argument("--retry-empty", action=argparse.BooleanOptionalAction, default=True, help="Re-evaluate images whose previous run yielded 0 predictions (e.g. truncated).")
    parser.add_argument("--ignore-429", action="store_true", help="Opt into retrying 429 rate limit errors (up to 10 retries)")
    parser.add_argument("--ignore-api-errors", action="store_true", help="Continue evaluating remaining images on API errors")
    parser.add_argument("--status-only", action="store_true", help="Inspect slice progress and exit without calling APIs")
    parser.add_argument("--run-majority-baseline", action="store_true", help="Also compute majority-class baseline floor on cohort")
    parser.add_argument("--generate-report", action="store_true", default=True, help="Write markdown/LaTeX report in experiments/")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_zero_shot_evaluation(args)


if __name__ == "__main__":
    main()
