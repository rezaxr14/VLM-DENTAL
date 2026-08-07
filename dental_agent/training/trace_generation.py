"""
Aim 1: Synthetic Expert Diagnostic Demonstration Trace Generation & Cross-Family Verification (§15, §16).

Includes:
- Multi-candidate generation (`generate_trace`) with self-consistency
- Cross-family verifier (`verify_trace`)
- Single image pipeline (`build_trace_example`, `generate_expert_trace`)
- Batch production generator with resume and quota handling (`run_aim1_batch`, `generate_trace_dataset`)
- API call and daily cost estimator (`estimate_aim1_api_calls`)
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, Optional
from PIL import Image
import pandas as pd
from tqdm import tqdm

from dental_agent.agent.parsing import parse_agent_json
from dental_agent.tools.registry import ToolRegistry
from dental_agent.training.api_pool import (
    call_llm,
    get_gemini_pool,
    AllKeysExhaustedToday,
)
from dental_agent.utils.serialization import to_jsonable

GENERATOR_PROVIDER = "gemini"
VERIFIER_PROVIDER = "anthropic"
GENERATOR_MODEL = "gemini-2.5-flash"
VERIFIER_MODEL = "claude-3-5-sonnet-20241022"

GENERATOR_SYSTEM_PROMPT = (
    "You are generating TRAINING DATA for a dental radiograph analysis agent. Given an "
    "X-ray image and the KNOWN correct answer, write a plausible step-by-step reasoning "
    "trace a model could have produced to reach that answer, including where it would "
    "zoom in and what visual evidence it would cite. Include one zoom_crop tool call "
    "(with an approximate bbox around the finding) before the final answer. Follow this "
    "schema across multiple lines:\n"
    "<think>...</think>\n"
    '{"tool": "zoom_crop", "args": {"bbox": [x, y, w, h]}}\n'
    "<think>...</think>\n"
    '{"final_answer": {"quadrant": ..., "tooth_position": ..., "diagnosis": "...", "confidence": ...}}'
)

VERIFIER_SYSTEM_PROMPT = (
    "You are a strict verifier, not a rewriter. Given an X-ray image, the KNOWN correct "
    "answer, and a candidate reasoning trace, judge ONLY whether every claim in the trace "
    "is actually supported by the image and the known answer — reject any trace asserting "
    "something the image/label does not support, even if the final answer happens to be "
    'correct. Respond with exactly one JSON object: {"grounded": true/false, "reason": "..."}.'
)

TEACHER_SYSTEM_PROMPT = GENERATOR_SYSTEM_PROMPT


def generate_trace(
    image: Image.Image,
    ground_truth: dict[str, Any],
    k: int = 3,
    provider: str = GENERATOR_PROVIDER,
    model: str = GENERATOR_MODEL,
    call_llm_fn: Callable[..., str] = call_llm,
) -> list[str]:
    """Generate k candidate traces for one example (self-consistency, §5.2 step 3)."""
    user_content = f"Known correct answer: {json.dumps(ground_truth)}"
    return [
        call_llm_fn(provider, model, GENERATOR_SYSTEM_PROMPT, user_content, image=image)
        for _ in range(k)
    ]


def verify_trace(
    image: Image.Image,
    ground_truth: dict[str, Any],
    trace: str,
    provider: str = VERIFIER_PROVIDER,
    model: str = VERIFIER_MODEL,
    call_llm_fn: Callable[..., str] = call_llm,
) -> dict[str, Any]:
    """Verify one trace with a DIFFERENT model family than the generator (bias control)."""
    user_content = f"Known correct answer: {json.dumps(ground_truth)}\n\nCandidate trace:\n{trace}"
    raw = call_llm_fn(provider, model, VERIFIER_SYSTEM_PROMPT, user_content, image=image)
    parsed = parse_agent_json(raw)
    return parsed if parsed else {"grounded": False, "reason": "verifier output unparseable"}


def build_trace_example(
    image_id: int,
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame | None = None,
    k: int = 3,
    diag_col: str = "category_id_3",
    call_llm_fn: Callable[..., str] = call_llm,
) -> dict[str, Any] | None:
    """Full Aim 1 pipeline for one image: generate k candidates, verify each, keep only
    grounded traces."""
    matches = images_df[images_df["id"] == image_id]
    if matches.empty:
        return None
    row = matches.iloc[0]
    image_path = row.get("local_path")
    if not image_path or not os.path.exists(str(image_path)):
        return None

    image = Image.open(image_path).convert("RGB")
    anns = annots_df[annots_df["image_id"] == image_id]
    if anns.empty:
        return None

    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )
    ann0 = anns.iloc[0]
    ground_truth = {
        "quadrant": int(ann0.get("category_id_1", 1)),
        "tooth_position": int(ann0.get("category_id_2", 1)),
        "diagnosis": cat_lookup.get(ann0.get(diag_col), "Caries"),
        "bbox": list(ann0.get("bbox", [0, 0, 50, 50])),
    }

    candidates = generate_trace(image, ground_truth, k=k, call_llm_fn=call_llm_fn)
    verified = [
        t for t in candidates
        if verify_trace(image, ground_truth, t, call_llm_fn=call_llm_fn).get("grounded")
    ]

    return {
        "image_id": image_id,
        "ground_truth": ground_truth,
        "n_candidates": len(candidates),
        "n_verified": len(verified),
        "verified_traces": verified,
    }


def estimate_aim1_api_calls(n_images: int, k: int = 3) -> int:
    """Rough call-count estimate before spending real budget: k generator calls + k
    verifier calls per image (one verification per candidate trace). Also projects how
    many DAYS the generator side will take given the Gemini pool's actual configured
    daily budget."""
    calls_per_image = 2 * k
    total = n_images * calls_per_image
    generator_calls = n_images * k
    print(f"{n_images} images x {calls_per_image} calls/image (k={k} generate + k verify) "
          f"= ~{total} total API calls (~{generator_calls} generator, ~{generator_calls} verifier).")

    pool = get_gemini_pool()
    if pool.keys:
        daily_budget = len(pool.keys) * pool.rpd_limit
        days = -(-generator_calls // max(daily_budget, 1))  # ceiling division
        print(f"\nGenerator (Gemini): {len(pool.keys)} key(s) x {pool.rpd_limit} "
              f"calls/day (after safety margin) = {daily_budget} calls/day budget.")
        print(f"At that rate, {generator_calls} generator calls take ~{days} day(s), "
              f"assuming you re-run run_aim1_batch() once per day as each daily cap resets.")
    else:
        print("\nGEMINI_API_KEYS isn't set yet — cannot project days-to-complete until it is.")
    return total


def run_aim1_batch(
    image_ids: list[int],
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame | None = None,
    k: int = 3,
    cache_path: str | Path | None = None,
    resume: bool = True,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    diag_col: str = "category_id_3",
) -> list[dict[str, Any]]:
    """Scale-up run: generate + verify traces across many images, with incremental disk caching
    (resume=True survives a interrupted session) and exponential backoff retry.
    AllKeysExhaustedToday stops cleanly without retrying."""
    results: list[dict[str, Any]] = []
    done_ids: set[int] = set()

    if cache_path and resume and os.path.exists(str(cache_path)):
        with open(cache_path) as f:
            results = json.load(f)
        done_ids = {r["image_id"] for r in results}
        print(f"Resuming: {len(done_ids)} image(s) already processed in {cache_path}")

    todo = [i for i in image_ids if i not in done_ids]
    total_candidates = total_verified = 0

    for idx, image_id in enumerate(todo):
        result = None
        try:
            for attempt in range(max_retries):
                try:
                    result = build_trace_example(
                        image_id=image_id,
                        images_df=images_df,
                        annots_df=annots_df,
                        categories_df=categories_df,
                        k=k,
                        diag_col=diag_col,
                    )
                    break
                except AllKeysExhaustedToday:
                    raise
                except Exception as e:
                    wait = retry_delay * (2 ** attempt)
                    print(f"  image_id={image_id}: attempt {attempt + 1}/{max_retries} failed ({e}); retrying in {wait:.0f}s")
                    time.sleep(wait)
            else:
                print(f"  image_id={image_id}: giving up after {max_retries} attempts, skipping")
        except AllKeysExhaustedToday as e:
            if cache_path:
                with open(cache_path, "w") as f:
                    json.dump(to_jsonable(results), f, indent=2)
            print(f"\n{e}")
            print(f"Stopped after {idx}/{len(todo)} image(s) from this run ({len(results)} total saved to {cache_path}).")
            return results

        if result:
            results.append(to_jsonable(result))
            total_candidates += result.get("n_candidates", 0)
            total_verified += result.get("n_verified", 0)

        if cache_path:
            with open(cache_path, "w") as f:
                json.dump(to_jsonable(results), f, indent=2)

        if (idx + 1) % 10 == 0 or idx == len(todo) - 1:
            rate = total_verified / max(total_candidates, 1)
            print(f"  {idx + 1}/{len(todo)} done — verified rate so far: {rate:.1%}")

    print(f"\nAll {len(todo)} image(s) processed.")
    return results


def generate_expert_trace(
    image: Image.Image,
    ground_truth: dict[str, Any],
    call_llm_fn: Callable[..., str] = call_llm,
    teacher_provider: str = GENERATOR_PROVIDER,
    teacher_model: str = GENERATOR_MODEL,
    verifier_provider: str = VERIFIER_PROVIDER,
    verifier_model: str = VERIFIER_MODEL,
) -> dict[str, Any] | None:
    """Generate a single verified expert demonstration trace for an annotated image."""
    user_prompt = f"Ground Truth Finding: {json.dumps(ground_truth)}"

    teacher_output = call_llm_fn(
        provider=teacher_provider,
        model=teacher_model,
        system_prompt=TEACHER_SYSTEM_PROMPT,
        user_content=user_prompt,
        image=image,
        temperature=0.2,
    )

    teacher_parsed = parse_agent_json(teacher_output)
    if not teacher_parsed and not teacher_output.strip().startswith("[") and not "<think>" in teacher_output:
        return None

    verifier_user = (
        f"Ground Truth: {json.dumps(ground_truth)}\n\n"
        f"Candidate Teacher Trace:\n{teacher_output}"
    )
    verifier_output = call_llm_fn(
        provider=verifier_provider,
        model=verifier_model,
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        user_content=verifier_user,
        image=image,
        temperature=0.0,
    )

    verifier_parsed = parse_agent_json(verifier_output)
    if verifier_parsed and (verifier_parsed.get("approved") or verifier_parsed.get("grounded")):
        return {
            "ground_truth": ground_truth,
            "raw_trace": teacher_output,
            "verifier_reason": verifier_parsed.get("reason"),
            "status": "approved",
        }

    return None


def generate_trace_dataset(
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    output_jsonl: str | Path,
    n_samples: int = 20,
    seed: int = 42,
    diag_col: str = "category_id_3",
) -> list[dict[str, Any]]:
    """Batch-generate and save a synthetic trace dataset in JSONL format for SFT."""
    os.makedirs(os.path.dirname(os.path.abspath(output_jsonl)), exist_ok=True)
    cat_lookup = dict(zip(categories_df["id"], categories_df["name"])) if len(categories_df) else {}

    valid_images = images_df.dropna(subset=["local_path"]).sample(
        n=min(n_samples, len(images_df)), random_state=seed
    )

    traces = []
    with open(output_jsonl, "w") as out_f:
        for _, img_row in tqdm(valid_images.iterrows(), total=len(valid_images), desc="Generating traces"):
            img_id = img_row["id"]
            img_annots = annots_df[annots_df["image_id"] == img_id]
            if img_annots.empty:
                continue

            ann = img_annots.iloc[0]
            quad = int(ann.get("category_id_1", 1))
            pos = int(ann.get("category_id_2", 1))
            diag_id = ann.get(diag_col)
            diag_name = cat_lookup.get(diag_id, "Caries")

            gt = {
                "quadrant": quad,
                "tooth_position": pos,
                "bbox": list(ann.get("bbox", [0, 0, 50, 50])),
                "diagnosis": diag_name,
            }

            img = Image.open(img_row["local_path"]).convert("RGB")
            try:
                trace = generate_expert_trace(img, gt)
                if trace:
                    trace["image_id"] = img_id
                    trace["image_path"] = img_row["local_path"]
                    traces.append(trace)
                    out_f.write(json.dumps(to_jsonable(trace)) + "\n")
                    out_f.flush()
            except Exception as e:
                print(f"[TraceGen] Failed for image {img_id}: {e}")

    print(f"Successfully generated {len(traces)} verified traces saved to {output_jsonl}")
    return traces
