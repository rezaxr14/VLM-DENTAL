"""Evaluated records for external models, in EXACTLY the schema ``scripts/evaluate_models.evaluate_condition`` writes.

Because the schema is identical, ``evaluation.matrix.summarize_cell`` / ``paired_bootstrap_diff`` / ``build_summary`` and the
paper tables read these records without modification. Everything that is specific to a hosted model goes in the extra
``external`` block, which the matrix code ignores.

Two layers (see ``importer``): the TRANSPORT record is produced where the model runs (answers + usage, no ground truth, no
metrics); the EVALUATED record is built here, locally, from the transport record and the dataset's annotations.

``findings_from_trajectory`` and ``transcript_for_record`` mirror the helpers of the same name in
``scripts/evaluate_models.py`` (a script that needs the training stack to import); a parity test runs wherever that script
imports.
"""
from __future__ import annotations

import time
from types import SimpleNamespace
from typing import Any

from dental_agent.evaluation.ground_truth import extract_ground_truth
from dental_agent.external.cohort import scores_as_cohort
from dental_agent.evaluation.metrics import extract_predicted_findings, match_multi_findings
from dental_agent.utils.serialization import to_jsonable

LEGACY_SYSTEM_PROMPT = "You are an expert dental radiologist analyzing panoramic dental radiographs."


def findings_from_trajectory(traj) -> list[dict[str, Any]]:
    fa = traj.final_answer
    if isinstance(fa, list):
        return extract_predicted_findings({"final_answer": fa})
    if isinstance(fa, dict):
        return extract_predicted_findings({"final_answer": [fa]})
    return []


def transcript_for_record(traj) -> dict[str, Any]:
    messages = []
    for m in getattr(traj, "messages", None) or []:
        if m.get("role") == "system":
            continue
        content = m.get("content")
        if isinstance(content, str):
            text = content
        else:
            text = "\n".join(part.get("text", "") if part.get("type") == "text" else "<image>"
                             for part in (content or []) if isinstance(part, dict))
        messages.append({"role": m.get("role"), "content": text})
    turns = [{"turn": t.get("turn"), "status": t.get("status"), "tool_calls": t.get("tool_calls_this_turn", [])}
             for t in traj.turns]
    return {"messages": messages, "turns": turns}


def legacy_zero_shot_trajectory(image_id: int, reply_text: str):
    """Trajectory-shaped result for the ``ZERO_SHOT_PROMPT`` supplement (parsed like Baseline #1: ``findings`` or
    ``final_answer`` JSON, code fences and <think> tags tolerated)."""
    from dental_agent.evaluation.baselines import parse_zero_shot_response   # heavy import, only this condition needs it
    parsed = parse_zero_shot_response(reply_text)
    findings = extract_predicted_findings(parsed)
    ok = isinstance(parsed, (dict, list))
    return SimpleNamespace(image_id=image_id, turns=[{"turn": 0, "raw_output": reply_text, "parsed": parsed, "status": "final_answer" if ok else "unparseable_json"}],
                           tool_calls=0, final_answer=findings if ok else None, format_ok=ok, messages=[], harness_events=None)


def transport_record(*, image_id: int, condition: str, model_slug: str, traj, usage: list[dict[str, Any]], seconds: float,
                     identity: dict[str, Any], error: dict[str, Any] | None = None) -> dict[str, Any]:
    """What the model-side run produces: the answers and the usage, no annotations and no metrics."""
    return to_jsonable({
        "kind": "transport", "image_id": int(image_id), "condition": condition, "model_slug": model_slug,
        "identity": identity, "seconds": seconds, "usage": usage, "error": error,
        "final_answer": getattr(traj, "final_answer", None), "format_ok": bool(getattr(traj, "format_ok", False)),
        "turns": [{"turn": t.get("turn"), "raw_output": t.get("raw_output"), "status": t.get("status"),
                   "tool_calls": t.get("tool_calls_this_turn", [])} for t in getattr(traj, "turns", [])],
        "tool_calls": int(getattr(traj, "tool_calls", 0)),
        "transcript": transcript_for_record(traj) if getattr(traj, "messages", None) else {"messages": [], "turns": []},
        "harness_events": getattr(traj, "harness_events", None),
        "timestamp": time.time(),
    })


def usage_summary(usage: list[dict[str, Any]]) -> dict[str, Any]:
    """Totals with missing values kept as unknown (None), never 0."""
    def total(key):
        vals = [u.get(key) for u in usage]
        known = [v for v in vals if v is not None]
        return (sum(known) if known and len(known) == len(vals) else None), len(vals) - len(known)
    pt, pt_missing = total("prompt_tokens")
    ct, ct_missing = total("completion_tokens")
    cost, cost_missing = total("cost_usd")
    return {"requests": len(usage), "prompt_tokens": pt, "completion_tokens": ct, "cost_usd": cost,
            "n_unknown_prompt_tokens": pt_missing, "n_unknown_completion_tokens": ct_missing, "n_unknown_cost": cost_missing,
            "refused": sum(1 for u in usage if u.get("refused")), "empty": sum(1 for u in usage if u.get("empty")),
            "wall_seconds": sum(u.get("wall_seconds") or 0.0 for u in usage)}


def evaluated_record(transport: dict[str, Any], annots_df, cats_df, *, dataset: str, split: str,
                     settings: dict[str, Any], model_slug: str | None = None) -> dict[str, Any]:
    """Score one transport record locally (Rule 13: every finding, set-level matching) into the evaluate_condition schema."""
    image_id = int(transport["image_id"])
    condition = transport["condition"]
    gt = extract_ground_truth(image_id, annots_df, cats_df)
    traj = SimpleNamespace(final_answer=transport.get("final_answer"))
    pred = findings_from_trajectory(traj)
    format_ok = bool(transport.get("format_ok"))
    scored, extra = pred, {}
    if scores_as_cohort(settings):                # 250-image cohort: canonical class names + family_exact_*, as evaluate_condition does
        from dental_agent.evaluation.test250_scoring import score_cohort
        scored, extra = score_cohort(gt, pred)
    m = match_multi_findings(gt, scored)
    if len(gt) == 0:
        ok = bool(len(scored) == 0 and format_ok)
        fdi_ok = exact = all_exact = ok
    else:
        fdi_ok = bool(m["fdi_tp"] > 0)
        exact = bool(m["exact_tp"] > 0)
        all_exact = bool(m["exact_fn"] == 0 and m["exact_fp"] == 0)
    primary = m["matched_pairs"][0]["pred"] if m["matched_pairs"] else (pred[0] if pred else None)
    turns = transport.get("turns") or []
    usage = transport.get("usage") or []
    return to_jsonable({
        "image_id": image_id, "condition": condition, "dataset": dataset, "split": split,
        "model_id": model_slug or transport.get("model_slug"),
        "image_view": "canonical", "canonical_letterbox": False,   # GPU-profile views: no black bars (as evaluate_condition records it)
        "adapter_path": "",
        "ground_truth": gt, "predictions": pred, "matched_pairs": m["matched_pairs"],
        "fdi_precision": m["fdi_precision"], "fdi_recall": m["fdi_recall"], "fdi_f1": m["fdi_f1"],
        "exact_precision": m["exact_precision"], "exact_recall": m["exact_recall"], "exact_f1": m["exact_f1"],
        "closeness_score": m["closeness_score"], "spatial_proximity": m["spatial_proximity"],
        "diagnostic_similarity": m["diagnostic_similarity"],
        "fdi_correct": fdi_ok, "exact_match": exact, "all_exact_match": all_exact, "format_ok": format_ok,
        "confidence": primary.get("confidence") if primary else None,
        "turns": len(turns), "tool_calls": int(transport.get("tool_calls", 0)),
        "raw_output": "\n---\n".join(str(t.get("raw_output", "")) for t in turns),
        "prompt": "legacy_zero_shot" if condition.endswith("legacy_prompt") else "unified",
        "seconds": transport.get("seconds"), "eval_settings": settings,
        "trajectory": transport.get("transcript") or {"messages": [], "turns": []},
        "ablation": None, "timestamp": time.time(), **extra,
        "external": {"provider": "kaggle", "model_slug": model_slug or transport.get("model_slug"),
                     "transport_identity": transport.get("identity"), "usage": usage_summary(usage),
                     "harness_events": transport.get("harness_events"), "error": transport.get("error")},
    })
