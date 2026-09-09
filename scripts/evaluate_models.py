#!/usr/bin/env python3
"""
Unified Model Evaluation CLI for VLM-DENTAL (§6, §20, §26).

Evaluates all 6 experimental conditions on the DENTEX test split:
  1. base_no_tools:   Base Qwen/Qwen3.5-9B, zero-shot single-turn (ZERO_SHOT_PROMPT)
  2. base_with_tools:  Base Qwen/Qwen3.5-9B, multi-turn agent loop with registered tools
  3. sft_no_tools:    Stage 1 SFT model, single-turn (ZERO_SHOT_PROMPT)
  4. sft_with_tools:   Stage 1 SFT model, multi-turn agent loop with tools
  5. grpo_no_tools:   Stage 2 GRPO RL policy, single-turn (ZERO_SHOT_PROMPT)
  6. grpo_with_tools:  Stage 2 GRPO RL policy, multi-turn agent loop with tools

Usage:
    # Re-use cached base zero-shot results:
    python scripts/evaluate_models.py --condition base_no_tools --reuse-cached

    # Run SFT with tools on test split:
    python scripts/evaluate_models.py --condition sft_with_tools --adapter-path data/models/qwen3_5_9b_sft_with_tools_dentex_alone

    # Run all available conditions:
    python scripts/evaluate_models.py --condition all --adapter-path data/models/qwen3_5_9b_sft_with_tools_dentex_alone

    # Smoke test on 3 images:
    python scripts/evaluate_models.py --condition base_no_tools --reuse-cached --limit 3

Hardware: Runs on Cloud TPU v5e-8 (torch_xla), CUDA GPU, or CPU.
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

repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from dotenv import load_dotenv
load_dotenv()

from dental_agent.config import load_config, load_env
from dental_agent.data.dentex import load_dentex_dataset
from dental_agent.data.fdi_utils import row_to_fdi
from dental_agent.agent.prompts import ZERO_SHOT_PROMPT, build_agent_system_prompt
from dental_agent.agent.parsing import parse_agent_json
from dental_agent.agent.tool_dispatch import execute_tool_call
from dental_agent.tools.registry import ToolRegistry
from dental_agent.evaluation.baselines import parse_zero_shot_response
from dental_agent.evaluation.metrics import (
    match_multi_findings,
    extract_predicted_findings,
    normalize_dental_diagnosis,
    expected_calibration_error,
    bootstrap_metric_ci,
)
from dental_agent.evaluation.reporting import generate_summary_table, generate_markdown_report
from dental_agent.utils.serialization import to_jsonable
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_CONDITIONS = [
    "base_no_tools", "base_with_tools",
    "sft_no_tools", "sft_with_tools",
    "grpo_no_tools", "grpo_with_tools",
]

DENTEX_DEFAULT_DIAGNOSES = {
    0: "Impacted",
    1: "Caries",
    2: "Periapical Lesion",
    3: "Deep Caries",
}

DEFAULT_OUTPUT_DIR = "data/evaluations"


# ---------------------------------------------------------------------------
# Hardware Setup
# ---------------------------------------------------------------------------

def setup_eval_hardware(precision: str = "bf16"):
    """Detect hardware: TPU / CUDA / CPU and return (device, dtype)."""
    import torch
    is_tpu = False
    try:
        import torch_xla.core.xla_model as xm
        device = xm.xla_device()
        is_tpu = True
        print(f"[HARDWARE] Cloud TPU device: {device}")
    except Exception:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            print(f"[HARDWARE] CUDA GPU: {torch.cuda.get_device_name(device)}")
        else:
            device = torch.device("cpu")
            print("[HARDWARE] Running on CPU.")

    dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)
    return device, dtype, is_tpu


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_eval_model(model_id: str, adapter_path: str | None, device, dtype, condition: str):
    """Load base model and optionally attach LoRA adapter for SFT/GRPO conditions.

    Returns (model, processor).
    """
    import torch
    from transformers import AutoProcessor
    from dental_agent.model.backbone import get_model_classes

    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    ModelClass = get_model_classes()

    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "auto"

    print(f"[MODEL] Loading base model: {model_id} (dtype={dtype})...")
    try:
        model = ModelClass.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        # Fallback for older transformers that use 'dtype' instead of 'torch_dtype'
        load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
        model = ModelClass.from_pretrained(model_id, **load_kwargs)

    # Attach LoRA adapter for SFT / GRPO conditions
    if adapter_path and condition.startswith(("sft_", "grpo_")):
        print(f"[MODEL] Attaching LoRA adapter from: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
        print("[MODEL] LoRA adapter merged successfully.")

    model.eval()

    # Move to device if not already mapped
    if not hasattr(model, "hf_device_map") or not model.hf_device_map:
        try:
            model = model.to(device)
        except Exception:
            pass  # Already on device via device_map="auto"

    return model, processor


# ---------------------------------------------------------------------------
# Ground Truth Extraction
# ---------------------------------------------------------------------------

def extract_ground_truth(image_id: int, annots_df, cats_df) -> list[dict[str, Any]]:
    """Extract all ground truth findings for an image using dataset-aware FDI conversion."""
    cat_lookup = (
        dict(zip(cats_df["id"], cats_df["name"]))
        if cats_df is not None and len(cats_df)
        else DENTEX_DEFAULT_DIAGNOSES
    )

    anns = annots_df[annots_df["image_id"] == image_id]
    gt_findings = []
    if not anns.empty:
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
    return gt_findings


# ---------------------------------------------------------------------------
# No-Tools Evaluation Engine (Single-Turn, ZERO_SHOT_PROMPT)
# ---------------------------------------------------------------------------

def run_no_tools_eval(
    model, processor, image: Image.Image, image_id: int, device,
) -> tuple[list[dict], str, bool]:
    """Run single-turn zero-shot evaluation using ZERO_SHOT_PROMPT.

    Returns (pred_findings, raw_output, format_ok).
    """
    from dental_agent.model.inference import generate_agent_reply

    messages = [
        {
            "role": "system",
            "content": "You are an expert dental radiologist analyzing panoramic dental radiographs. "
                       "Provide concise reasoning and always complete your response with the final JSON object.",
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": ZERO_SHOT_PROMPT},
            ],
        },
    ]

    raw_reply, prompt_len, gen_ids = generate_agent_reply(
        model, processor, messages, max_new_tokens=2048, return_ids=True,
    )

    parsed_raw = parse_zero_shot_response(raw_reply)
    pred_findings = extract_predicted_findings(parsed_raw)
    format_ok = bool(parsed_raw is not None and isinstance(parsed_raw, (dict, list)))
    return pred_findings, raw_reply, format_ok


# ---------------------------------------------------------------------------
# With-Tools Evaluation Engine (Multi-Turn Agent Loop)
# ---------------------------------------------------------------------------

def run_with_tools_eval(
    model, processor, image: Image.Image, image_id: int, device,
    registry: ToolRegistry, dataset: str = "dentex",
    max_turns: int = 15, max_tool_calls: int = 25,
) -> tuple[list[dict], str, bool, int, int]:
    """Run multi-turn agent loop with real dynamic tool execution.

    Returns (pred_findings, raw_output, format_ok, turn_count, tool_call_count).
    """
    from dental_agent.model.inference import generate_agent_reply

    system_prompt = build_agent_system_prompt(registry.format_tool_descriptions(), dataset=dataset)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {
                    "type": "text",
                    "text": f"Analyze this panoramic X-ray (image_id={image_id}). "
                            f"Identify any abnormal teeth and determine the diagnosis.",
                },
            ],
        },
    ]

    final_answer = None
    tool_call_count = 0
    turn_count = 0
    all_raw_outputs = []
    registered_names = {t.name for t in registry.list_tools()}

    for turn_idx in range(max_turns * 2):  # generous safety valve
        raw_reply, prompt_len, gen_ids = generate_agent_reply(
            model, processor, messages, max_new_tokens=1024, return_ids=True,
        )
        all_raw_outputs.append(raw_reply)
        turn_count += 1

        parsed = parse_agent_json(raw_reply)
        if not parsed:
            # Model produced unparseable output, nudge it
            messages.append({"role": "assistant", "content": raw_reply})
            messages.append({
                "role": "user",
                "content": "Your response could not be parsed as valid JSON. "
                           "Please respond with exactly one JSON object.",
            })
            if turn_count >= max_turns:
                break
            continue

        messages.append({"role": "assistant", "content": raw_reply})

        # Check for final_answer
        if "final_answer" in parsed:
            final_answer = parsed["final_answer"]
            break

        # Process tool calls
        tool_calls = parsed.get("tool_calls", [])
        # Support single tool_call format
        if not tool_calls and "tool" in parsed:
            tool_calls = [{"tool": parsed["tool"], "args": parsed.get("args", {})}]

        if not tool_calls:
            # No tool calls and no final answer -- force conclusion
            if turn_count >= max_turns:
                break
            messages.append({
                "role": "user",
                "content": "Please either call a tool or provide your final_answer.",
            })
            continue

        # Execute each tool call
        observations = []
        for tc in tool_calls:
            tool_name = tc.get("tool", "")
            tool_args = tc.get("args", {})

            if tool_name not in registered_names:
                observations.append({"type": "text", "text": f"Error: Unknown tool '{tool_name}'."})
                continue

            if tool_call_count >= max_tool_calls:
                observations.append({
                    "type": "text",
                    "text": f"Tool budget exhausted ({max_tool_calls} calls). Please provide your final_answer now.",
                })
                break

            try:
                result = execute_tool_call(registry, tool_name, tool_args, image)
                tool_call_count += 1

                if isinstance(result, Image.Image):
                    observations.append({"type": "image", "image": result})
                    observations.append({"type": "text", "text": f"[Tool Result: {tool_name} returned an image]"})
                else:
                    observations.append({"type": "text", "text": f"[Tool Result: {tool_name}] {json.dumps(result) if isinstance(result, (dict, list)) else str(result)}"})
            except Exception as e:
                observations.append({"type": "text", "text": f"[Tool Error: {tool_name}] {str(e)}"})
                tool_call_count += 1

        messages.append({"role": "user", "content": observations if observations else "Tool execution complete."})

        if tool_call_count >= max_tool_calls and final_answer is None:
            messages.append({
                "role": "user",
                "content": f"Tool budget exhausted ({max_tool_calls} calls). You MUST provide your final_answer now.",
            })

        if turn_count >= max_turns:
            break

    # Extract predictions from final_answer
    pred_findings = []
    format_ok = False
    if final_answer is not None:
        format_ok = True
        if isinstance(final_answer, list):
            pred_findings = extract_predicted_findings({"final_answer": final_answer})
        elif isinstance(final_answer, dict):
            pred_findings = extract_predicted_findings({"final_answer": [final_answer]})

    combined_raw = "\n---\n".join(all_raw_outputs)
    return pred_findings, combined_raw, format_ok, turn_count, tool_call_count


# ---------------------------------------------------------------------------
# Cached Results Loader
# ---------------------------------------------------------------------------

def load_cached_results(cached_path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    """Load pre-computed evaluation results from a JSONL file."""
    records = []
    if not cached_path.exists():
        print(f"[WARNING] Cached file not found: {cached_path}")
        return records

    with open(cached_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass

    if limit is not None and limit > 0:
        records = records[:limit]

    print(f"[CACHED] Loaded {len(records)} pre-computed results from {cached_path}")
    return records


# ---------------------------------------------------------------------------
# Summary & Reporting
# ---------------------------------------------------------------------------

def compute_condition_summary(records: list[dict[str, Any]], condition: str) -> dict[str, Any]:
    """Compute aggregate metrics from per-image evaluation records."""
    n = len(records)
    if n == 0:
        return {}

    fmt_ok = sum(1 for r in records if r.get("format_ok"))
    fdi_ok = sum(1 for r in records if r.get("fdi_correct"))
    exact_ok = sum(1 for r in records if r.get("exact_match"))

    mean_fdi_f1 = sum(r.get("fdi_f1", 0.0) for r in records) / n
    mean_exact_f1 = sum(r.get("exact_f1", 0.0) for r in records) / n
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

    # Tool usage metrics (for with_tools conditions)
    mean_tools = 0.0
    mean_turns = 0.0
    if "with_tools" in condition:
        tool_counts = [r.get("tool_calls", 0) for r in records]
        turn_counts = [r.get("turns", 0) for r in records]
        mean_tools = sum(tool_counts) / n if tool_counts else 0.0
        mean_turns = sum(turn_counts) / n if turn_counts else 0.0

    return {
        "condition": condition,
        "total_samples": n,
        "format_adherence": fmt_ok / n,
        "fdi_localization_accuracy": fdi_ok / n,
        "fdi_localization_f1": mean_fdi_f1,
        "exact_match_accuracy": exact_ok / n,
        "exact_match_f1": mean_exact_f1,
        "exact_match_ci_95": [em_low, em_high],
        "closeness_score": mean_closeness,
        "spatial_proximity": mean_spatial,
        "diagnostic_similarity": mean_diag_sim,
        "ece": ece,
        "mean_tool_calls": mean_tools,
        "mean_turns": mean_turns,
    }


def print_condition_banner(condition: str, n_images: int, output_path: Path) -> None:
    print("\n" + "=" * 70)
    print(f"VLM-DENTAL EVALUATION: {condition.upper()}")
    print("=" * 70)
    print(f"* Condition    : {condition}")
    print(f"* Total Images : {n_images}")
    print(f"* Output File  : {output_path}")
    print("=" * 70 + "\n")


def print_condition_summary(summary: dict[str, Any]) -> None:
    condition = summary.get("condition", "unknown")
    n = summary.get("total_samples", 0)
    print("\n" + "=" * 70)
    print(f"RESULTS: {condition.upper()} (n={n})")
    print("=" * 70)
    print(f"Format Compliance         : {summary.get('format_adherence', 0) * 100:.1f}%")
    print(f"FDI Localization Accuracy : {summary.get('fdi_localization_accuracy', 0) * 100:.1f}% [Mean F1: {summary.get('fdi_localization_f1', 0):.3f}]")
    print(f"Exact Match Accuracy      : {summary.get('exact_match_accuracy', 0) * 100:.1f}% [Mean F1: {summary.get('exact_match_f1', 0):.3f}]")
    ci = summary.get("exact_match_ci_95", [0, 0])
    print(f"Exact Match 95% CI        : [{ci[0] * 100:.1f}%, {ci[1] * 100:.1f}%]")
    print(f"Closeness Score           : {summary.get('closeness_score', 0):.3f} (Spatial: {summary.get('spatial_proximity', 0):.3f}, Diag: {summary.get('diagnostic_similarity', 0):.3f})")
    print(f"ECE                       : {summary.get('ece', 0):.4f}")
    if "with_tools" in condition:
        print(f"Mean Tool Calls           : {summary.get('mean_tool_calls', 0):.1f}")
        print(f"Mean Turns                : {summary.get('mean_turns', 0):.1f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main Evaluation Dispatcher
# ---------------------------------------------------------------------------

def evaluate_condition(
    condition: str,
    args: argparse.Namespace,
    imgs_df,
    annots_df,
    cats_df,
) -> list[dict[str, Any]]:
    """Evaluate a single condition and return per-image records."""
    dataset_name = args.dataset.strip().lower()
    split_name = args.split.strip().lower()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"eval_{dataset_name}_{split_name}_{condition}.jsonl"

    # For base_no_tools: check if we can reuse cached zero-shot results
    if condition == "base_no_tools" and args.reuse_cached:
        model_id = args.model_id.replace("/", "--").replace(":", "-")
        cached_path = output_dir / f"zero_shot_{dataset_name}_{split_name}_transformers_{model_id}.jsonl"
        if cached_path.exists():
            records = load_cached_results(cached_path, limit=args.limit)
            # Write re-tagged records to condition-specific output
            with open(output_path, "w", encoding="utf-8") as f:
                for r in records:
                    r["condition"] = condition
                    f.write(json.dumps(to_jsonable(r)) + "\n")
            return records
        else:
            print(f"[WARNING] --reuse-cached specified but cached file not found: {cached_path}")
            print("[WARNING] Falling back to live evaluation...")

    # Filter eligible images
    eligible_imgs = imgs_df[imgs_df["local_path"].notna() & (imgs_df["local_path"] != "None")].copy()
    eligible_imgs = eligible_imgs[eligible_imgs["local_path"].apply(lambda p: os.path.exists(str(p)) if p else False)]

    if args.limit is not None and args.limit > 0:
        eligible_imgs = eligible_imgs.iloc[:args.limit]

    print_condition_banner(condition, len(eligible_imgs), output_path)

    # Load model
    is_tools = "with_tools" in condition
    adapter_path = args.adapter_path if condition.startswith(("sft_", "grpo_")) else None
    device, dtype, is_tpu = setup_eval_hardware(args.precision)
    model, processor = load_eval_model(args.model_id, adapter_path, device, dtype, condition)

    registry = None
    if is_tools:
        registry = ToolRegistry.create_default()

    # Evaluation loop
    records = []
    for idx, (_, img_row) in enumerate(eligible_imgs.iterrows(), start=1):
        image_id = int(img_row["id"])
        image_path = str(img_row["local_path"])
        print(f"[{idx}/{len(eligible_imgs)}] Image ID {image_id}...", end=" ", flush=True)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"SKIP (cannot open: {e})")
            continue

        gt_findings = extract_ground_truth(image_id, annots_df, cats_df)

        if is_tools:
            pred_findings, raw_output, format_ok, turns, tools_used = run_with_tools_eval(
                model, processor, image, image_id, device,
                registry=registry, dataset=args.dataset,
                max_turns=args.max_turns, max_tool_calls=args.max_tool_calls,
            )
        else:
            pred_findings, raw_output, format_ok = run_no_tools_eval(
                model, processor, image, image_id, device,
            )
            turns = 1
            tools_used = 0

        # Multi-finding set-level matching (Rule 13)
        match_res = match_multi_findings(gt_findings, pred_findings)

        # Clinical correctness indicators
        if len(gt_findings) == 0:
            is_correct_normal = bool(len(pred_findings) == 0 and format_ok)
            fdi_ok = is_correct_normal
            exact_match = is_correct_normal
            all_exact_match = is_correct_normal
        else:
            fdi_ok = bool(match_res["fdi_tp"] > 0)
            exact_match = bool(match_res["exact_tp"] > 0)
            all_exact_match = bool(match_res["exact_fn"] == 0 and match_res["exact_fp"] == 0)

        primary_matched = match_res["matched_pairs"][0]["pred"] if match_res["matched_pairs"] else (pred_findings[0] if pred_findings else None)
        confidence = primary_matched.get("confidence") if primary_matched else None

        status = "✓" if exact_match else "✗"
        print(f"{status} GT:{len(gt_findings)} Pred:{len(pred_findings)} FDI_F1:{match_res['fdi_f1']:.2f} Exact_F1:{match_res['exact_f1']:.2f}", flush=True)

        record = to_jsonable({
            "image_id": image_id,
            "condition": condition,
            "dataset": args.dataset,
            "split": args.split,
            "model_id": args.model_id,
            "adapter_path": adapter_path or "",
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
            "exact_match": exact_match,
            "all_exact_match": all_exact_match,
            "format_ok": format_ok,
            "confidence": confidence,
            "turns": turns,
            "tool_calls": tools_used,
            "raw_output": raw_output,
            "timestamp": time.time(),
        })
        records.append(record)

        # Append incrementally to output JSONL
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # Clean up model to free memory before next condition
    del model, processor
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return records


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VLM-DENTAL: Unified Model Evaluation CLI (§6, §20, §26)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--condition",
        type=str,
        required=True,
        choices=ALL_CONDITIONS + ["all"],
        help="Evaluation condition to run. Use 'all' to run all available conditions sequentially.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B"),
        help="Base VLM model identifier or local directory",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Path or HF Hub repo for fine-tuned LoRA adapter (required for sft_* and grpo_* conditions)",
    )
    parser.add_argument("--dataset", type=str, default="dentex", help="Dataset name ('dentex')")
    parser.add_argument("--split", type=str, default="test", help="Dataset split ('test', 'validation')")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("DENTAL_AGENT_DATA_DIR", "data"),
        help="Base directory for datasets and images",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Evaluation output directory")
    parser.add_argument("--reuse-cached", action="store_true", help="For base_no_tools: load existing zero-shot results instead of recomputing")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick smoke tests")
    parser.add_argument("--max-turns", type=int, default=15, help="Maximum turns for multi-turn agent loop (with_tools conditions)")
    parser.add_argument("--max-tool-calls", type=int, default=25, help="Maximum tool calls per radiograph (with_tools conditions)")
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Numerical precision for model inference",
    )
    parser.add_argument("--config", "-c", default=None, help="Path to config YAML")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env()

    # Load dataset
    print(f"\n[DATA] Loading {args.dataset.upper()} dataset (split='{args.split}')...")
    imgs_df, annots_df, cats_df = load_dentex_dataset(data_dir=args.data_dir, split_name=args.split)
    print(f"[DATA] Loaded {len(imgs_df)} images, {len(annots_df)} annotations.\n")

    # Determine conditions to run
    if args.condition == "all":
        conditions = []
        for c in ALL_CONDITIONS:
            if c.startswith(("sft_", "grpo_")) and not args.adapter_path:
                print(f"[SKIP] {c}: no --adapter-path provided.")
                continue
            conditions.append(c)
    else:
        conditions = [args.condition]

    # Run evaluations
    all_summaries: dict[str, dict[str, Any]] = {}

    for cond in conditions:
        print(f"\n{'#' * 70}")
        print(f"# CONDITION: {cond.upper()}")
        print(f"{'#' * 70}\n")

        records = evaluate_condition(cond, args, imgs_df, annots_df, cats_df)
        summary = compute_condition_summary(records, cond)
        all_summaries[cond] = summary
        print_condition_summary(summary)

    # Generate comparative summary
    if len(all_summaries) > 1:
        print("\n" + "=" * 70)
        print("COMPARATIVE SUMMARY ACROSS CONDITIONS")
        print("=" * 70)

        # CSV export
        csv_path = Path(args.output_dir) / f"eval_{args.dataset}_{args.split}_summary.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "condition", "total_samples", "format_adherence",
                "fdi_localization_accuracy", "fdi_localization_f1",
                "exact_match_accuracy", "exact_match_f1",
                "closeness_score", "spatial_proximity", "diagnostic_similarity",
                "ece", "mean_tool_calls", "mean_turns",
            ])
            writer.writeheader()
            for s in all_summaries.values():
                row = {k: v for k, v in s.items() if k != "exact_match_ci_95"}
                writer.writerow(row)
        print(f"\n[OUTPUT] Summary CSV: {csv_path}")

    # Generate Markdown report
    report_path = Path(args.output_dir) / f"eval_{args.dataset}_{args.split}_report.md"
    metrics_dict = {}
    for cond, s in all_summaries.items():
        metrics_dict[cond] = {
            "format_adherence": s.get("format_adherence", 0),
            "fdi_localization_accuracy": s.get("fdi_localization_accuracy", 0),
            "fdi_localization_f1": s.get("fdi_localization_f1", 0),
            "exact_match_accuracy": s.get("exact_match_accuracy", 0),
            "exact_match_f1": s.get("exact_match_f1", 0),
            "exact_match_ci_95": s.get("exact_match_ci_95", [0, 0]),
            "closeness_score": s.get("closeness_score", 0),
            "spatial_proximity": s.get("spatial_proximity", 0),
            "diagnostic_similarity": s.get("diagnostic_similarity", 0),
            "ece": s.get("ece", 0),
            "mean_tool_calls": s.get("mean_tool_calls", 0),
            "total_samples": s.get("total_samples", 0),
        }
    generate_markdown_report(metrics_dict, output_path=report_path)
    print(f"[OUTPUT] Report: {report_path}")

    print(f"\n[COMPLETE] Evaluation finished for {len(all_summaries)} condition(s).")


if __name__ == "__main__":
    main()
