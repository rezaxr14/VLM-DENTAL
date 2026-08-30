"""
Baseline evaluation runners: Majority-class floor and Zero-shot commercial VLMs (§20).

Includes:
- Majority class baseline (`majority_class_baseline_metrics`)
- Zero-shot API baseline with incremental caching (`run_zero_shot_baseline`, `run_zeroshot_baseline`)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from PIL import Image
import pandas as pd
from tqdm import tqdm

from dental_agent.agent.prompts import ZERO_SHOT_PROMPT
from dental_agent.agent.parsing import parse_agent_json
from dental_agent.training.api_pool import call_llm
from dental_agent.data.fdi_utils import row_to_fdi
from dental_agent.rewards.components import reward_accuracy
from dental_agent.evaluation.metrics import compute_evaluation_metrics
from dental_agent.utils.serialization import to_jsonable


def majority_class_baseline_metrics(
    holdout_image_ids: list[int],
    annots_df: pd.DataFrame,
    holdout_ids: set[int] | None = None,
    categories_df: pd.DataFrame | None = None,
    diag_col: str = "category_id_3",
) -> dict[str, Any]:
    """Always predicts the single most common quadrant, tooth position, and diagnosis
    (measured on the training pool) — a naive floor for evaluation metrics."""
    if holdout_ids is None:
        holdout_ids = set(holdout_image_ids)

    train_annots = annots_df[~annots_df["image_id"].isin(holdout_ids)]
    if train_annots.empty:
        train_annots = annots_df

    # Majority FDI values computed from already-converted, dataset-aware
    # per-row FDI values (fdi_utils.row_to_fdi) rather than aggregating the
    # raw category_id_1/category_id_2 column and adding a hardcoded +1 --
    # that hardcoded +1 assumed DENTEX's 0-indexed convention unconditionally,
    # which would silently be wrong the moment this ran against Tufts'
    # already-1-indexed rows (or any future dataset with its own convention).
    if len(train_annots):
        fdi_pairs = [row_to_fdi(row) for _, row in train_annots.iterrows()]
        majority_quadrant = int(pd.Series([q for q, _ in fdi_pairs]).mode().iloc[0])
        majority_tooth = int(pd.Series([t for _, t in fdi_pairs]).mode().iloc[0])
    else:
        majority_quadrant, majority_tooth = 1, 1

    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )
    if diag_col in train_annots and not train_annots[diag_col].dropna().empty:
        majority_diag_id = train_annots[diag_col].mode().iloc[0]
        majority_diag = cat_lookup.get(majority_diag_id, str(majority_diag_id))
    else:
        majority_diag = "Caries"

    fake_results = []
    for image_id in holdout_image_ids:
        anns = annots_df[annots_df["image_id"] == image_id]
        if anns.empty:
            continue
        ann0 = anns.iloc[0]
        quadrant, tooth_position = row_to_fdi(ann0)
        gt = {
            "quadrant": quadrant,
            "tooth_position": tooth_position,
            "diagnosis": cat_lookup.get(ann0.get(diag_col), "Caries"),
        }
        fake_results.append(to_jsonable({
            "image_id": image_id,
            "ground_truth": gt,
            "final_answer": {
                "quadrant": majority_quadrant,
                "tooth_position": majority_tooth,
                "diagnosis": majority_diag,
                "confidence": 1.0,
            },
            "tool_calls": 0,
            "format_ok": True,
            "reward": 0.0,
            "reward_components": {},
        }))

    metrics = compute_evaluation_metrics(fake_results)
    print(f"Majority-class baseline (always predicts quadrant={majority_quadrant}, "
          f"tooth_position={majority_tooth}, diagnosis={majority_diag!r}):")
    print(f"  fdi_accuracy={metrics.get('fdi_accuracy', 0.0):.3f}  "
          f"balanced_accuracy={metrics.get('diagnosis_balanced_accuracy', 0.0):.3f}")
    return metrics


def parse_zero_shot_response(text: str) -> dict[str, Any] | list[dict[str, Any]] | None:
    """Robustly parse JSON response from zero-shot VLM evaluation.
    
    Handles:
    - {"findings": [{"quadrant": ..., "tooth_position": ..., "diagnosis": ..., "confidence": ...}, ...]}
    - {"final_answer": [{"quadrant": ..., ...}]} or {"final_answer": {"quadrant": ..., ...}}
    - Direct {"quadrant": ..., "tooth_position": ..., "diagnosis": ...}
    - Markdown code blocks (```json ... ```)
    - <think>...</think> reasoning tags
    """
    if not text or not isinstance(text, str):
        return None

    import re
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    if not cleaned or (len(cleaned) < 5 and "<think>" in text):
        # Fallback: model might have put JSON inside unclosed <think> or entire text
        cleaned = text

    # 1. Try markdown code block extraction
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    for block in reversed(code_blocks):
        try:
            parsed = json.loads(block.strip())
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            # Trailing comma cleanup
            sub_block = re.sub(r",\s*([\]}])", r"\1", block.strip())
            try:
                parsed = json.loads(sub_block)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except Exception:
                pass

    # 2. Extract balanced JSON objects from cleaned and full text
    candidates = []
    for source_text in (cleaned, text):
        stack = []
        start = -1
        in_string = False
        escape = False

        for i, char in enumerate(source_text):
            if escape:
                escape = False
                continue
            if char == '\\':
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if not in_string:
                if char in ('{', '['):
                    if not stack:
                        start = i
                    stack.append(char)
                elif char in ('}', ']'):
                    if stack:
                        opener = stack.pop()
                        if not stack and start != -1:
                            candidates.append(source_text[start:i + 1])
                            start = -1

    def _has_valid_dental_keys(d: Any) -> bool:
        if isinstance(d, list):
            return any(_has_valid_dental_keys(item) for item in d)
        if isinstance(d, dict):
            return any(k in d for k in ("findings", "final_answer", "quadrant", "tooth_position", "tooth", "fdi", "teeth"))
        return False

    for cand in reversed(candidates):
        try:
            parsed = json.loads(cand)
            if isinstance(parsed, (dict, list)):
                if _has_valid_dental_keys(parsed):
                    return parsed
                # If dict has only thought/reasoning/analysis, unwrap and parse thought text
                if isinstance(parsed, dict):
                    thought_parts = [
                        str(v) for k, v in parsed.items()
                        if isinstance(v, str) and k in ("thought", "reasoning", "analysis", "content", "response", "message", "text")
                    ]
                    if thought_parts:
                        thought_findings = extract_findings_from_reasoning_text(" ".join(thought_parts))
                        if thought_findings:
                            return {"findings": thought_findings}
                return parsed
        except Exception:
            sub_cand = re.sub(r",\s*([\]}])", r"\1", cand)
            try:
                parsed = json.loads(sub_cand)
                if isinstance(parsed, (dict, list)):
                    if _has_valid_dental_keys(parsed):
                        return parsed
                    if isinstance(parsed, dict):
                        thought_parts = [
                            str(v) for k, v in parsed.items()
                            if isinstance(v, str) and k in ("thought", "reasoning", "analysis", "content", "response", "message", "text")
                        ]
                        if thought_parts:
                            thought_findings = extract_findings_from_reasoning_text(" ".join(thought_parts))
                            if thought_findings:
                                return {"findings": thought_findings}
                    return parsed
            except Exception:
                pass

    # 3. Fallback to individual finding dict recovery (partial/truncated JSON)
    individual_dicts = []
    for cand_match in re.finditer(r"\{[^{}]*?(?:quadrant|tooth_position|diagnosis)[^{}]*?\}", text, re.IGNORECASE):
        try:
            d = json.loads(cand_match.group(0))
            if isinstance(d, dict) and ("quadrant" in d or "diagnosis" in d or "tooth_position" in d):
                individual_dicts.append(d)
        except Exception:
            sub_d = re.sub(r",\s*([\]}])", r"\1", cand_match.group(0))
            try:
                d = json.loads(sub_d)
                if isinstance(d, dict) and ("quadrant" in d or "diagnosis" in d or "tooth_position" in d):
                    individual_dicts.append(d)
            except Exception:
                pass
    if individual_dicts:
        return {"findings": individual_dicts}

    # 4. Fallback to parse_agent_json
    agent_res = parse_agent_json(text)
    if agent_res:
        return agent_res

    # 5. Fallback to reasoning text clinical finding extraction
    text_findings = extract_findings_from_reasoning_text(text)
    if text_findings:
        return {"findings": text_findings}

    return None


def extract_findings_from_reasoning_text(text: str) -> list[dict]:
    """Fallback extractor when a reasoning model (e.g. Qwen 3.6, MiniMax, Nemotron) produces
    rich clinical reasoning but is truncated or omits top-level JSON formatting.
    Uses midpoint token-distance association and local negation scope to avoid
    cross-tooth pathology contamination."""
    if not text or not isinstance(text, str):
        return []

    import re
    findings = []
    seen = set()

    # Process each bullet point / line independently
    raw_lines = [l.strip() for l in text.splitlines() if l.strip()]

    for line in raw_lines:
        line_lower = line.lower()

        # Global skip if line is purely stating normal status with no teeth abnormal
        if any(norm_term in line_lower for norm_term in [
            "all normal", "entirely normal", "intact and sound", "without active pathology",
            "no pathology detected", "no significant abnormality"
        ]):
            continue

        detected_teeth: list[tuple[int, int, float]] = []  # (quad, pos, char_mid)

        # 1. 2-digit FDI patterns (e.g. "Tooth 18", "FDI 48", "#38", "18", "tooth 26")
        for tm in re.finditer(r"(?:Tooth|FDI|tooth|#)?\s*\b([1-4])([1-8])\b", line, re.IGNORECASE):
            # Check if this tooth is just a reference tooth preceded by 'against', 'adjacent to', 'near'
            prefix = line[:tm.start()].lower()
            if any(prefix.rstrip().endswith(prep) for prep in [
                "against", "against tooth", "adjacent to", "adjacent to tooth",
                "next to", "next to tooth", "mesial to", "distal to", "near"
            ]):
                continue
            char_mid = (tm.start() + tm.end()) / 2.0
            detected_teeth.append((int(tm.group(1)), int(tm.group(2)), char_mid))

        # 2. Separated Quadrant and Tooth (e.g. "Quadrant 1 ... Tooth 8", "Quad 3, Position 6")
        for qtm in re.finditer(r"(?:Quadrant|Quad|Q)\s*([1-4])\b[^\n\r;]*?\b(?:Tooth|Position|#|\()?\s*(?:Tooth|Position|#)?\s*([1-8])\b", line, re.IGNORECASE):
            char_mid = (qtm.start() + qtm.end()) / 2.0
            detected_teeth.append((int(qtm.group(1)), int(qtm.group(2)), char_mid))

        for ptq in re.finditer(r"\b(?:Tooth|Position|#)\s*([1-8])\b[^\n\r;]*?\b(?:Quadrant|Quad|Q)\s*([1-4])\b", line, re.IGNORECASE):
            char_mid = (ptq.start() + ptq.end()) / 2.0
            detected_teeth.append((int(ptq.group(2)), int(ptq.group(1)), char_mid))

        # 3. Named quadrant patterns (e.g. "Upper Right ... tooth 8", "Lower Left ... tooth 6")
        named_quads = [
            (r"(?:Upper\s+Right|Maxillary\s+Right)\b[^\n\r;]*?\b(?:Tooth|Position|#|\()?\s*(?:Tooth|Position|#)?\s*([1-8])\b", 1),
            (r"(?:Upper\s+Left|Maxillary\s+Left)\b[^\n\r;]*?\b(?:Tooth|Position|#|\()?\s*(?:Tooth|Position|#)?\s*([1-8])\b", 2),
            (r"(?:Lower\s+Left|Mandibular\s+Left)\b[^\n\r;]*?\b(?:Tooth|Position|#|\()?\s*(?:Tooth|Position|#)?\s*([1-8])\b", 3),
            (r"(?:Lower\s+Right|Mandibular\s+Right)\b[^\n\r;]*?\b(?:Tooth|Position|#|\()?\s*(?:Tooth|Position|#)?\s*([1-8])\b", 4),
        ]
        for pattern, q_val in named_quads:
            for nq_match in re.finditer(pattern, line, re.IGNORECASE):
                pos_val = int(nq_match.group(1))
                char_mid = (nq_match.start() + nq_match.end()) / 2.0
                detected_teeth.append((q_val, pos_val, char_mid))

        if not detected_teeth:
            continue

        # Find all pathology mentions on this line
        path_matches = list(re.finditer(
            r"\b(deep\s+caries|caries|carious|periapical\s+lesion|apical\s+lesion|impacted(?:\s+wisdom|\s+tooth)?|impaction)\b",
            line,
            re.IGNORECASE
        ))

        for pm in path_matches:
            diag_text = pm.group(1).strip()
            diag_start = pm.start()
            diag_mid = (pm.start() + pm.end()) / 2.0

            # Local negation check: look back up to 35 characters before the pathology match
            lookback_window = line[max(0, diag_start - 35):diag_start].lower()
            if any(neg in lookback_window for neg in [
                "no ", "not ", "without ", "ruled out", "free of ", "no evidence of ", "no signs of "
            ]):
                continue  # This specific pathology is negated locally

            # Associate with the closest detected tooth on this line
            closest_tooth = min(detected_teeth, key=lambda t: abs(t[2] - diag_mid))
            q, pos, _ = closest_tooth
            key = (q, pos)

            if key not in seen:
                seen.add(key)
                findings.append({
                    "quadrant": q,
                    "tooth_position": pos,
                    "diagnosis": diag_text,
                    "confidence": None,
                })

    return findings


def match_zero_shot_finding(
    parsed_output: Any,
    gt_quadrant: int | None = None,
    gt_tooth_position: int | None = None,
) -> dict[str, Any] | None:
    """Extract and match the most relevant finding from a parsed zero-shot response.
    
    Normalizes:
    - {"findings": [...]} -> selected finding dict
    - {"final_answer": [...]} -> selected finding dict
    - {"final_answer": {...}} -> finding dict
    - {"quadrant": ..., "tooth_position": ..., "diagnosis": ...} -> finding dict
    """
    if parsed_output is None:
        return None

    findings_list: list[dict[str, Any]] = []

    if isinstance(parsed_output, list):
        findings_list = [f for f in parsed_output if isinstance(f, dict)]
    elif isinstance(parsed_output, dict):
        if "findings" in parsed_output and isinstance(parsed_output["findings"], list):
            findings_list = [f for f in parsed_output["findings"] if isinstance(f, dict)]
        elif "final_answer" in parsed_output:
            fa = parsed_output["final_answer"]
            if isinstance(fa, list):
                findings_list = [f for f in fa if isinstance(f, dict)]
            elif isinstance(fa, dict):
                return fa
        elif "quadrant" in parsed_output or "diagnosis" in parsed_output:
            return parsed_output

    if not findings_list:
        return parsed_output if isinstance(parsed_output, dict) else None

    # If ground truth tooth position is known, prioritize finding that matches the tooth
    if gt_quadrant is not None and gt_tooth_position is not None:
        for f in findings_list:
            if f.get("quadrant") == gt_quadrant and f.get("tooth_position") == gt_tooth_position:
                return f
        for f in findings_list:
            if f.get("quadrant") == gt_quadrant:
                return f

    # Fallback to the first finding
    return findings_list[0]


def run_zero_shot_baseline(
    image_ids: list[int],
    images_df: pd.DataFrame,
    annots_df: pd.DataFrame,
    categories_df: pd.DataFrame | None = None,
    provider: str = "openai",
    model: str = "gpt-4o",
    cache_path: str | Path | None = None,
    resume: bool = True,
    diag_col: str = "category_id_3",
    pacing_delay: float = 0.0,
    image_max_dim: int = 0,
    temperature: float = 0.0,
    on_result_callback: Any = None,
) -> list[dict[str, Any]]:
    """No fine-tuning, no tool access, single pass — zero-shot commercial VLM evaluation."""
    results: list[dict[str, Any]] = []
    done_ids: set[int] = set()

    if cache_path and resume and os.path.exists(str(cache_path)):
        with open(cache_path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    results.append(rec)
                    if "image_id" in rec:
                        done_ids.add(int(rec["image_id"]))
                except Exception:
                    pass
        if not results and os.path.exists(str(cache_path)):
            # Try plain json format fallback
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        results = data
                        done_ids = {int(r["image_id"]) for r in results if "image_id" in r}
            except Exception:
                pass

        print(f"Resuming: {len(done_ids)} image(s) already processed in {cache_path}")

    cat_lookup = (
        dict(zip(categories_df["id"], categories_df["name"]))
        if categories_df is not None and len(categories_df)
        else {}
    )

    import time
    for image_id in tqdm(image_ids, desc=f"ZeroShot ({provider}/{model})"):
        if image_id in done_ids:
            continue
        anns = annots_df[annots_df["image_id"] == image_id]
        if anns.empty:
            continue
        ann0 = anns.iloc[0]
        quadrant, tooth_position = row_to_fdi(ann0)
        ground_truth = {
            "quadrant": quadrant,
            "tooth_position": tooth_position,
            "diagnosis": cat_lookup.get(ann0.get(diag_col), "Caries"),
        }

        matches = images_df[images_df["id"] == image_id]
        if matches.empty:
            continue
        row = matches.iloc[0]
        image_path = row.get("local_path")
        if not image_path or not os.path.exists(str(image_path)):
            continue
        image = Image.open(image_path).convert("RGB")

        if pacing_delay > 0:
            time.sleep(pacing_delay)

        try:
            raw = call_llm(
                provider=provider,
                model=model,
                system_prompt="You are an expert dental radiologist analyzing panoramic dental radiographs.",
                user_content=ZERO_SHOT_PROMPT,
                image=image,
                temperature=temperature,
                max_tokens=2048,
            )
            parsed_raw = parse_zero_shot_response(raw)
            parsed_final = match_zero_shot_finding(parsed_raw, gt_quadrant=quadrant, gt_tooth_position=tooth_position)
            err_msg = None
        except Exception as e:
            raw = f"Error: {e}"
            parsed_raw = None
            parsed_final = None
            err_msg = str(e)

        reward_val = reward_accuracy({"final_answer": parsed_final}, ground_truth) if parsed_final else 0.0
        format_ok = bool(
            parsed_final
            and isinstance(parsed_final, dict)
            and "diagnosis" in parsed_final
            and "quadrant" in parsed_final
            and "tooth_position" in parsed_final
        )

        item = to_jsonable({
            "image_id": image_id,
            "ground_truth": ground_truth,
            "final_answer": parsed_final,
            "raw_output": raw,
            "tool_calls": 0,
            "format_ok": format_ok,
            "reward": reward_val,
            "reward_components": {"accuracy": reward_val},
            "error": err_msg,
        })
        results.append(item)
        done_ids.add(image_id)

        if on_result_callback is not None:
            try:
                on_result_callback(item)
            except Exception:
                pass

        if cache_path:
            # Append line to jsonl
            try:
                os.makedirs(os.path.dirname(str(cache_path)) or ".", exist_ok=True)
                with open(cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(item) + "\n")
            except Exception as e:
                print(f"Warning: Failed to write to cache {cache_path}: {e}")

    return results


def run_zeroshot_baseline(
    images_df: pd.DataFrame,
    provider: str = "openai",
    model: str = "gpt-4o",
    sample_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Legacy wrapper for running zero-shot on an images dataframe."""
    eval_images = images_df.dropna(subset=["local_path"])
    if sample_limit:
        eval_images = eval_images.head(sample_limit)

    results = []
    for _, row in tqdm(eval_images.iterrows(), total=len(eval_images), desc=f"ZeroShot {model}"):
        img_id = row["id"]
        image = Image.open(row["local_path"]).convert("RGB")
        try:
            raw_reply = call_llm(
                provider=provider,
                model=model,
                system_prompt="You are a dental radiologist.",
                user_content=ZERO_SHOT_PROMPT,
                image=image,
                temperature=0.0,
            )
            parsed_raw = parse_zero_shot_response(raw_reply)
            parsed = match_zero_shot_finding(parsed_raw)
        except Exception as e:
            raw_reply = f"Error: {e}"
            parsed = None

        results.append({
            "image_id": img_id,
            "raw_output": raw_reply,
            "final_answer": parsed,
            "tool_calls": 0,
            "format_ok": bool(parsed and "diagnosis" in parsed),
        })

    return results
