#!/usr/bin/env python3
"""
Unified Model Evaluation CLI for VLM-DENTAL (§6, §20, §26).

Evaluates all 6 experimental conditions on the DENTEX test split, through the SAME agent loop GRPO rolls out with
(dental_agent/agent/loop.py), so evaluation, rollouts and SFT share prompts, observation format and views:
  1. base_no_tools:   Base Qwen/Qwen3.5-9B, single turn, unified no-tools prompt (thought + final_answer)
  2. base_with_tools:  Base Qwen/Qwen3.5-9B, multi-turn agent loop with registered tools
  3. sft_no_tools:    Stage 1 SFT model, single turn, unified no-tools prompt
  4. sft_with_tools:   Stage 1 SFT model, multi-turn agent loop with tools
  5. grpo_no_tools:   Stage 2 GRPO RL policy, single turn, unified no-tools prompt
  6. grpo_with_tools:  Stage 2 GRPO RL policy, multi-turn agent loop with tools
(--reuse-cached for base_no_tools loads the earlier ZERO_SHOT_PROMPT run instead; its records are tagged
prompt="zero_shot_cached" and are not comparable to the unified-prompt conditions.)

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
from dental_agent.data.dev_split import assert_not_test_split, image_key, load_dev_keys
from dental_agent.agent.prompts import FIRST_USER_PROMPT, ZERO_SHOT_PROMPT, build_agent_system_prompt
from dental_agent.agent.parsing import parse_agent_json
from dental_agent.agent.tool_dispatch import execute_tool_call
from dental_agent.agent.loop import run_agent, run_agent_no_tools
from dental_agent.tools.registry import ToolRegistry
from dental_agent.evaluation.baselines import parse_zero_shot_response
from dental_agent.evaluation.metrics import (
    match_multi_findings,
    extract_predicted_findings,
    expected_calibration_error,
    bootstrap_metric_ci,
)
from dental_agent.evaluation.ground_truth import DENTEX_DEFAULT_DIAGNOSES, extract_ground_truth  # noqa: F401
from dental_agent.evaluation.reporting import generate_summary_table, generate_markdown_report
from dental_agent.evaluation.tool_ablation import ToolProbe, spec_from_args
from dental_agent.utils.serialization import to_jsonable
from dental_agent.utils.canonical import (
    add_letterbox_arg, family_for_tool, set_letterbox, to_canonical,
)
from PIL import Image

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ALL_CONDITIONS = [
    "base_no_tools", "base_with_tools",
    "sft_no_tools", "sft_with_tools",
    "grpo_no_tools", "grpo_with_tools",
]

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

    dtype = torch.bfloat16 if precision in ("bf16", "qlora") else (torch.float16 if precision == "fp16" else torch.float32)
    return device, dtype, is_tpu


# ---------------------------------------------------------------------------
# Model Loading
# ---------------------------------------------------------------------------

def load_eval_model(model_id: str, adapter_path: str | None, device, dtype, condition: str, quantize_4bit: bool = False):
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
    if quantize_4bit:
        from dental_agent.model.backbone import qlora_bnb_config
        # Same NF4 config and skip list (vision tower + lm_head stay 16-bit) the adapter was trained under.
        load_kwargs["quantization_config"] = qlora_bnb_config(dtype)

    print(f"[MODEL] Loading base model: {model_id} (dtype={dtype}, 4-bit={quantize_4bit})...")
    try:
        model = ModelClass.from_pretrained(model_id, **load_kwargs)
    except TypeError:
        # Fallback for older transformers that use 'dtype' instead of 'torch_dtype'
        load_kwargs["dtype"] = load_kwargs.pop("torch_dtype")
        model = ModelClass.from_pretrained(model_id, **load_kwargs)

    # Attach LoRA adapter for SFT / GRPO conditions
    if adapter_path and condition.startswith(("sft_", "grpo_")):
        # A GRPO checkpoint folder holds two adapters: 'grpo_policy' (trained) and 'reference' (frozen SFT copy) and has
        # no adapter at its root. Evaluating a GRPO checkpoint always means the trained policy.
        root = Path(adapter_path)
        if not (root / "adapter_config.json").exists() and (root / "grpo_policy" / "adapter_config.json").exists():
            adapter_path = str(root / "grpo_policy")
            print("[MODEL] GRPO checkpoint given: using its trained 'grpo_policy' adapter (not 'reference').")
        print(f"[MODEL] Attaching LoRA adapter from: {adapter_path}")
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        if quantize_4bit:
            print("[MODEL] LoRA adapter attached (not merged into the 4-bit base weights).")
        else:
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

# extract_ground_truth and DENTEX_DEFAULT_DIAGNOSES live in dental_agent/evaluation/ground_truth.py (imported above), so the
# ablation, batch and sweep runners use the very same definition of a finding.


# ---------------------------------------------------------------------------
# No-Tools Evaluation Engine (Single-Turn, ZERO_SHOT_PROMPT)
# ---------------------------------------------------------------------------

def run_no_tools_eval(
    model, processor, image: Image.Image, image_id: int, device,
    canonical_resize: bool = False,
) -> tuple[list[dict], str, bool]:
    """Run single-turn zero-shot evaluation using ZERO_SHOT_PROMPT.

    Returns (pred_findings, raw_output, format_ok).
    """
    from dental_agent.model.inference import generate_agent_reply

    if canonical_resize:
        image = to_canonical(image, "FULL")  # same view the SFT/GRPO policy was trained on

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
    canonical_resize: bool = False,
) -> tuple[list[dict], str, bool, int, int]:
    """Run multi-turn agent loop with real dynamic tool execution.

    Returns (pred_findings, raw_output, format_ok, turn_count, tool_call_count).
    """
    from dental_agent.model.inference import generate_agent_reply

    system_prompt = build_agent_system_prompt(registry.format_tool_descriptions(), dataset=dataset)

    # Tools execute on the NATIVE `image` (bbox args are native pixels); the model is shown canonical views.
    canonical = canonical_resize
    view_image = to_canonical(image, "FULL") if canonical else image

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": view_image},
                {"type": "text", "text": FIRST_USER_PROMPT},
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
                    if canonical:
                        result = to_canonical(result, family_for_tool(tool_name))
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
# Shared-loop Evaluation (what evaluate_condition runs)
# ---------------------------------------------------------------------------

def run_condition_trajectory(
    model, processor, images_df, image_id: int, registry: ToolRegistry | None, args: argparse.Namespace,
):
    """Run ONE image through the SAME loop GRPO rolls out with (``dental_agent.agent.loop``) and return the trajectory.

    Evaluation, GRPO rollouts and SFT therefore share the system prompt, the ``Result of <tool>:`` observation format,
    canonical views, KV-cache reuse and the per-turn token cap. Greedy decoding (temperature 0).
    """
    if registry is not None:
        return run_agent(
            image_id=image_id, images_df=images_df, model=model, processor=processor, registry=registry,
            max_tool_calls=args.max_tool_calls, verbose=False, canonical_resize=args.canonical_resize,
            temperature=0.0, dataset=args.dataset, max_new_tokens=args.max_new_tokens,
            max_turns=args.max_turns, malformed_retries=args.malformed_retries,
        )
    return run_agent_no_tools(
        image_id=image_id, images_df=images_df, model=model, processor=processor, verbose=False,
        canonical_resize=args.canonical_resize, temperature=0.0, max_new_tokens=args.max_new_tokens,
    )


def findings_from_trajectory(traj) -> list[dict[str, Any]]:
    """Predicted findings of a trajectory: a list answer, a single-finding dict, or nothing."""
    final_answer = traj.final_answer
    if isinstance(final_answer, list):
        return extract_predicted_findings({"final_answer": final_answer})
    if isinstance(final_answer, dict):
        return extract_predicted_findings({"final_answer": [final_answer]})
    return []


def run_agent_condition(
    model, processor, images_df, image_id: int, registry: ToolRegistry | None, args: argparse.Namespace,
) -> tuple[list[dict], str, bool, int, int]:
    """Evaluate one image with the shared loop (see ``run_condition_trajectory``).

    ``run_with_tools_eval`` / ``run_no_tools_eval`` above are the earlier stand-alone loops and are no longer called.

    Returns (pred_findings, raw_output, format_ok, turn_count, tool_call_count).
    """
    traj = run_condition_trajectory(model, processor, images_df, image_id, registry, args)
    raw_output = "\n---\n".join(str(t.get("raw_output", "")) for t in traj.turns)
    return findings_from_trajectory(traj), raw_output, bool(traj.format_ok), len(traj.turns), int(traj.tool_calls)


def transcript_for_record(traj) -> dict[str, Any]:
    """Compact, JSON-safe transcript stored in every record: all messages except the (constant) system prompt, with
    images replaced by ``<image>``, plus each turn's status and tool calls. Failure analysis and qualitative figures
    read it; the metrics never do."""
    messages = []
    for m in getattr(traj, "messages", None) or []:
        if m.get("role") == "system":
            continue
        content = m.get("content")
        if isinstance(content, str):
            text = content
        else:
            text = "\n".join(
                part.get("text", "") if part.get("type") == "text" else "<image>"
                for part in (content or []) if isinstance(part, dict)
            )
        messages.append({"role": m.get("role"), "content": text})
    turns = [
        {"turn": t.get("turn"), "status": t.get("status"), "tool_calls": t.get("tool_calls_this_turn", [])}
        for t in traj.turns
    ]
    return {"messages": messages, "turns": turns}


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
# Image set, settings fingerprint and resume
# ---------------------------------------------------------------------------

def load_eval_split(args: argparse.Namespace):
    """Images, annotations and categories of the split to evaluate.

    * ``test``: the official DENTEX validation set (50 images). The final numbers and nothing else.
    * ``dev``: the dev images carved out of the DENTEX TRAIN set (``--dev-split``, see scripts/make_dev_split.py), held
      out of SFT and GRPO. The group-size sweep and the ablations are measured here, never on ``test``.
    """
    split = args.split.strip().lower()
    if split == "dev":
        if not getattr(args, "dev_split", None):
            raise SystemExit("--split dev needs --dev-split (the file written by scripts/make_dev_split.py).")
        imgs_df, annots_df, cats_df = load_dentex_dataset(data_dir=args.data_dir, split_name="train")
        assert_not_test_split(imgs_df)
        keys = load_dev_keys(args.dev_split)
        imgs_df = imgs_df[[image_key(args.dataset, i) in keys for i in imgs_df["id"]]].reset_index(drop=True)
        if len(imgs_df) != len(keys):
            raise SystemExit(f"Dev split lists {len(keys)} images but only {len(imgs_df)} are in the DENTEX train data.")
        annots_df = annots_df[annots_df["image_id"].isin(set(imgs_df["id"]))]
        return imgs_df, annots_df, cats_df
    return load_dentex_dataset(data_dir=args.data_dir, split_name=args.split)


def eligible_images(imgs_df, limit: int | None = None):
    """Images with a readable local file (optionally the first ``limit``)."""
    eligible = imgs_df[imgs_df["local_path"].notna() & (imgs_df["local_path"] != "None")].copy()
    eligible = eligible[eligible["local_path"].apply(lambda p: os.path.exists(str(p)) if p else False)]
    if limit is not None and limit > 0:
        eligible = eligible.iloc[:limit]
    return eligible


def eval_settings(args: argparse.Namespace, condition: str, adapter_path: str | None) -> dict[str, Any]:
    """Everything that changes what a condition's numbers mean. Stored in every record; a resume must match it."""
    return {
        "condition": condition,
        "model_id": args.model_id,
        "adapter_path": adapter_path or "",
        "precision": args.precision,
        "canonical_resize": bool(args.canonical_resize),
        **({"canonical_letterbox": False} if args.canonical_resize and not getattr(args, "canonical_letterbox", True) else {}),   # absent = padded (legacy files stay resumable)
        "max_tool_calls": int(args.max_tool_calls),
        "max_turns": args.max_turns,
        "max_new_tokens": int(args.max_new_tokens),
        "malformed_retries": int(args.malformed_retries),
        "temperature": 0.0,
        "prompt": "unified",
    }


def prepare_output_file(output_path: Path, mode: str, settings: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Apply ``--existing-results`` to a result file. Returns the already-finished records (by image id) to keep.

    ``error``     refuse when the file already has records (the old behaviour silently appended duplicates);
    ``resume``    keep the finished images and run only the rest, provided they were produced with identical settings;
    ``overwrite`` start the file again.
    """
    if not output_path.is_file() or output_path.stat().st_size == 0:
        return {}
    if mode == "overwrite":
        output_path.unlink()
        return {}
    if mode != "resume":
        raise SystemExit(
            f"{output_path} already has results. Add --existing-results resume to continue it, or "
            "--existing-results overwrite to start it again."
        )
    done: dict[int, dict[str, Any]] = {}
    for rec in load_cached_results(output_path):
        done[int(rec["image_id"])] = rec
    for rec in done.values():
        old = rec.get("eval_settings")
        if old is None:
            raise SystemExit(f"{output_path} was written before settings were recorded; it cannot be resumed safely. Use --existing-results overwrite.")
        diff = {k: (old.get(k), v) for k, v in settings.items() if old.get(k) != v}
        if diff:
            raise SystemExit(f"{output_path} was produced with different settings {diff}; resuming would mix them. Use --existing-results overwrite.")
    return done


# ---------------------------------------------------------------------------
# Main Evaluation Dispatcher
# ---------------------------------------------------------------------------

def evaluate_condition(
    condition: str,
    args: argparse.Namespace,
    imgs_df,
    annots_df,
    cats_df,
    loaded: tuple | None = None,
) -> list[dict[str, Any]]:
    """Evaluate a single condition and return per-image records.

    ``loaded`` = (model, processor) reuses an already loaded model (the matrix runner loads each checkpoint once and
    runs all of its cells); it is then not reloaded and not freed here.
    """
    dataset_name = args.dataset.strip().lower()
    split_name = args.split.strip().lower()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Original-size results keep their historical file name; canonical-size results get a suffix so the two views of
    # one condition never share (and never append to) a file.
    size_tag = "_canonical" if args.canonical_resize else ""
    spec = spec_from_args(args)
    if spec.active and "with_tools" not in condition:
        raise SystemExit(f"Tool ablations (--remove-tools, --locate-shift-frac, --locate-scale) apply to with_tools conditions; got {condition}.")
    # An ablation gets its own file (tag after the size tag), so it never mixes with the unablated results.
    output_path = output_dir / f"eval_{dataset_name}_{split_name}_{condition}{size_tag}{spec.tag()}.jsonl"

    # For base_no_tools: check if we can reuse cached zero-shot results
    if condition == "base_no_tools" and args.reuse_cached:
        from dental_agent.utils.inputs import model_file_tag
        model_id = model_file_tag(args.model_id)     # a local QWEN35_9B_LOCAL_DIR folder maps to the Qwen--Qwen3.5-9B file
        cached_path = output_dir / f"zero_shot_{dataset_name}_{split_name}_transformers_{model_id}.jsonl"
        if cached_path.exists():
            records = load_cached_results(cached_path, limit=args.limit)
            # Write re-tagged records to condition-specific output
            with open(output_path, "w", encoding="utf-8") as f:
                for r in records:
                    r["condition"] = condition
                    r["prompt"] = "zero_shot_cached"  # ZERO_SHOT_PROMPT, not the unified no-tools prompt
                    f.write(json.dumps(to_jsonable(r)) + "\n")
            return records
        else:
            print(f"[WARNING] --reuse-cached specified but cached file not found: {cached_path}")
            print("[WARNING] Falling back to live evaluation...")

    eligible_imgs = eligible_images(imgs_df, args.limit)

    print_condition_banner(condition, len(eligible_imgs), output_path)

    is_tools = "with_tools" in condition
    adapter_path = args.adapter_path if condition.startswith(("sft_", "grpo_")) else None
    settings = eval_settings(args, condition, adapter_path)
    if spec.active:
        settings["ablation"] = spec.settings()
    done = prepare_output_file(output_path, getattr(args, "existing_results", "error"), settings)
    if done:
        print(f"[RESUME] {len(done)} image(s) already finished in {output_path.name}; running the remaining ones.")

    # Load model (unless the caller already did)
    if loaded is None:
        device, dtype, is_tpu = setup_eval_hardware(args.precision)
        model, processor = load_eval_model(args.model_id, adapter_path, device, dtype, condition, quantize_4bit=args.precision == "qlora")
    else:
        model, processor = loaded

    registry = None
    if is_tools:
        registry = ToolRegistry.create_default()

    # Evaluation loop
    records = []
    for idx, (_, img_row) in enumerate(eligible_imgs.iterrows(), start=1):
        image_id = int(img_row["id"])
        image_path = str(img_row["local_path"])
        if image_id in done:
            records.append(done[image_id])
            continue
        print(f"[{idx}/{len(eligible_imgs)}] Image ID {image_id}...", end=" ", flush=True)

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"SKIP (cannot open: {e})")
            continue

        gt_findings = extract_ground_truth(image_id, annots_df, cats_df)

        # With tools, every image runs on a per-image registry that logs the localization calls and applies the
        # requested ablation (none by default, in which case the tools behave exactly as before).
        probe = ToolProbe(registry, spec, image_id) if registry is not None else None
        t_start = time.time()
        traj = run_condition_trajectory(model, processor, eligible_imgs, image_id, probe.registry if probe else None, args)
        seconds = time.time() - t_start
        pred_findings = findings_from_trajectory(traj)
        raw_output = "\n---\n".join(str(t.get("raw_output", "")) for t in traj.turns)
        format_ok, turns, tools_used = bool(traj.format_ok), len(traj.turns), int(traj.tool_calls)

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
            "image_view": "canonical" if args.canonical_resize else "original",
            "canonical_letterbox": bool(getattr(args, "canonical_letterbox", True)) if args.canonical_resize else None,
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
            "prompt": "unified",
            "seconds": seconds,
            "eval_settings": settings,
            "trajectory": transcript_for_record(traj),
            "ablation": probe.summary(gt_findings, traj.turns) if probe else None,
            "timestamp": time.time(),
        })
        records.append(record)

        # Append incrementally to output JSONL
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    # Clean up model to free memory before next condition (a caller-owned model stays loaded)
    if loaded is None:
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
    parser.add_argument(
        "--split", type=str, default="test",
        help="'test' = the official DENTEX validation set (final numbers only); 'dev' = the dev images held out of training "
             "(needs --dev-split; group-size sweep and ablations are measured here).",
    )
    parser.add_argument(
        "--dev-split", type=str, default=None,
        help="Dev split JSON (scripts/make_dev_split.py). Used by --split dev.",
    )
    parser.add_argument(
        "--remove-tools", nargs="+", default=[], metavar="TOOL",
        help="Ablation (with_tools conditions): withhold these tools (e.g. nudge_crop). Results go to a file tagged with the "
             "ablation. Measures how much the trained agent depends on the tool.",
    )
    parser.add_argument(
        "--locate-shift-frac", type=float, default=0.0,
        help="Ablation: move every locate_tooth box by this fraction of its own size in a fixed pseudo-random direction "
             "(0 = untouched). Tests whether the agent corrects a wrong box with nudge_crop.",
    )
    parser.add_argument(
        "--locate-scale", type=float, default=1.0,
        help="Ablation: rescale every locate_tooth box about its centre (1 = untouched).",
    )
    parser.add_argument(
        "--ablation-seed", type=int, default=0,
        help="Seed of the perturbation direction (same seed = same wrong box for every condition and run).",
    )
    parser.add_argument(
        "--existing-results", type=str, default="error", choices=["error", "resume", "overwrite"],
        help="What to do when this condition's result file already has records: error (default), resume (skip finished "
             "images; settings must be identical) or overwrite (start again).",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ.get("DENTAL_AGENT_DATA_DIR", "data"),
        help="Base directory for datasets and images",
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Evaluation output directory")
    parser.add_argument("--reuse-cached", action="store_true", help="For base_no_tools: load existing zero-shot results instead of recomputing")
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit for quick smoke tests")
    parser.add_argument(
        "--max-turns", type=int, default=None,
        help="Maximum assistant turns (with_tools conditions). Default: --max-tool-calls + 15, the bound GRPO rollouts use. "
             "The SFT traces average 11 turns and run to 35, so a small cap truncates episodes the model was trained to finish.",
    )
    parser.add_argument(
        "--max-tool-calls", type=int, default=50,
        help="Maximum tool calls per radiograph (with_tools conditions); the same budget GRPO rollouts use. "
             "The SFT traces make 23 calls at the median and up to 71.",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048,
        help="Token cap of ONE assistant turn. Use the same value as GRPO (run_grpo.py --max-new-tokens).",
    )
    parser.add_argument(
        "--malformed-retries", type=int, default=0,
        help="How many unparseable replies are answered with an error message before the episode ends (with_tools "
             "conditions). 0 = the first one ends it, exactly as in GRPO rollouts.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32", "qlora"],
        help="Numerical precision for model inference. qlora = 4-bit NF4 base weights (CUDA only), the setting for 24 GB GPUs; "
             "nothing selects it for you.",
    )
    parser.add_argument("--config", "-c", default=None, help="Path to config YAML")
    parser.add_argument(
        "--canonical-resize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Show every evaluated condition aspect-preserving canonical views (FULL 1536x768 / CROP 256x384 / "
             "COMPARE 512x384, black-bar padded unless --no-canonical-letterbox). Pass exactly what the evaluated checkpoint "
             "was trained with.",
    )
    add_letterbox_arg(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_env()
    set_letterbox(args.canonical_letterbox)
    print(f"[EVAL] canonical_resize={args.canonical_resize}" + (f" canonical_letterbox={args.canonical_letterbox}" if args.canonical_resize else ""))
    if args.canonical_resize and args.canonical_letterbox:
        try:
            import torch
            if torch.cuda.is_available():
                print("[EVAL WARNING] canonical views are black-bar padded on a CUDA GPU. Intended only for a checkpoint that was "
                      "trained with bars (the TPU recipe); a GPU-trained checkpoint is evaluated with --no-canonical-letterbox "
                      "or --no-canonical-resize, exactly as it was trained.")
        except Exception:
            pass

    # Load dataset
    print(f"\n[DATA] Loading {args.dataset.upper()} dataset (split='{args.split}')...")
    imgs_df, annots_df, cats_df = load_eval_split(args)
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
