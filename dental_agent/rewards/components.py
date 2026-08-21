"""
Individual reward component functions for dental agent evaluation and GRPO training (§5.5).
"""

from __future__ import annotations

from typing import Any, Mapping


def _iter_tool_calls(trajectory: Mapping[str, Any]):
    """Yield (tool_name, tool_args, tool_ok) for every tool call across a trajectory's
    turns. Reads the canonical `tool_calls_this_turn` shape both agent loops
    (langgraph_loop.py's trace-gen graph and loop.py's run_agent) now produce.

    NOTE: turns used to carry a single `parsed.get("tool")` field, and these
    reward functions used to read that directly. parse_agent_json normalizes
    every parsed response into a "tool_calls" list (popping the old flat
    "tool"/"args" keys in the process), so `parsed.get("tool")` is now
    PERMANENTLY empty/None on every turn, regardless of what the model
    actually did -- reading it silently produced a reward computed from zero
    tool calls, always, even on trajectories that used tools extensively.
    Read tool_calls_this_turn instead; it's what both loops actually populate.
    """
    for t in trajectory.get("turns", []):
        if not isinstance(t, dict):
            continue
        for call in t.get("tool_calls_this_turn", []) or []:
            if not isinstance(call, dict):
                continue
            yield call.get("tool_name"), call.get("tool_args", {}) or {}, bool(call.get("tool_ok"))


def _is_valid_final_answer(ans: Any) -> bool:
    """A valid final answer is a non-empty dict OR a non-empty list of dicts
    (prompts.py: "A patient may have multiple findings; your final answer
    must be a list covering all of them") -- not dict-only."""
    if isinstance(ans, dict):
        return True
    if isinstance(ans, list) and len(ans) > 0:
        return True
    return False


def reward_format(trajectory: Mapping[str, Any]) -> float:
    """Format adherence reward (R_format in §5.5).

    Returns 1.0 if the agent produced a valid final answer (a single-finding
    dict, or a non-empty list for multiple findings), 0.0 otherwise.
    """
    if trajectory.get("format_ok"):
        return 1.0
    return 1.0 if _is_valid_final_answer(trajectory.get("final_answer")) else 0.0


def reward_tool_validity(trajectory: Mapping[str, Any]) -> float:
    """Tool-call validity reward (R_tool in §5.5).

    Returns 1.0 if no tools were called, or the fraction of attempted tool calls
    that had valid arguments and succeeded without error.
    """
    calls = list(_iter_tool_calls(trajectory))
    if not calls:
        return 1.0
    valid_count = sum(1 for _, _, ok in calls if ok)
    return float(valid_count / len(calls))


def reward_efficiency(
    trajectory: Mapping[str, Any],
    max_calls: int = 50,
    calls_per_finding: int = 6,
) -> float:
    """Tool efficiency reward (R_efficiency in §5.5).

    Rewards using no more tool calls than the case genuinely needed, rather
    than a flat per-call penalty applied uniformly regardless of total count.
    The previous design deducted a fixed penalty per call with no reference
    to how many findings were actually being investigated -- a trajectory
    thoroughly investigating 5 findings (locate + zoom + occasional
    nudge_crop + contralateral_compare per finding, easily 20+ legitimate
    calls) would score far worse than one that superficially investigated 1,
    and it would have actively fought against the tiered-perturbation /
    nudge_crop teaching signal from trace-gen, since every corrective
    nudge_crop call cost exactly the same flat penalty as any other tool
    regardless of whether it was legitimately needed. It also silently
    ignored its own `max_calls` parameter -- accepted but never referenced
    anywhere in the function body.

    Reference budget scales with case complexity via how many distinct teeth
    were located (a reasonable proxy for "how much genuinely needed
    investigating"), computed from the trajectory's own locate_tooth calls
    rather than requiring a separate trajectory-level field, so this works
    regardless of which loop produced the trajectory. locate_tooth and
    nudge_crop are exempt from the per-call cost entirely -- finding a tooth
    and correcting a bad detection are exactly the behaviors this reward
    should not be fighting against. Only cost beyond the reference budget,
    and exact repeated calls (same tool + same args back-to-back, which
    genuinely gained no new information), reduce the score, and the drop-off
    past budget is smooth rather than a hard cliff.
    """
    calls = list(_iter_tool_calls(trajectory))
    if not calls:
        return 1.0

    EXEMPT_TOOLS = {"locate_tooth", "nudge_crop"}
    LOW_COST_TOOLS = {"fdi_label"}

    located_teeth: set[int] = set()
    for name, args, ok in calls:
        if name == "locate_tooth" and ok:
            tooth = args.get("tooth")
            if tooth is not None:
                try:
                    located_teeth.add(int(tooth))
                except (TypeError, ValueError):
                    pass

    n_findings = max(1, len(located_teeth))
    reference_budget = min(max_calls, n_findings * calls_per_finding)

    billable = 0.0
    prev_call: tuple[Any, Any] | None = None
    for name, args, _ok in calls:
        if name in EXEMPT_TOOLS:
            prev_call = (name, tuple(sorted(args.items())) if isinstance(args, dict) else args)
            continue
        cost = 0.2 if name in LOW_COST_TOOLS else 1.0
        current = (name, tuple(sorted(args.items())) if isinstance(args, dict) else args)
        if prev_call == current:
            cost *= 5.0  # identical tool+args as the immediately preceding call: no new information gained
        billable += cost
        prev_call = current

    if billable <= reference_budget:
        return 1.0

    overage = billable - reference_budget
    score = 1.0 / (1.0 + overage / max(1.0, reference_budget))
    return max(0.0, min(1.0, score))


def reward_accuracy(
    trajectory: Mapping[str, Any],
    ground_truth: Mapping[str, Any] | list[Mapping[str, Any]],
) -> float:
    """Graded diagnostic accuracy reward (R_accuracy in §5.5).

    Graded breakdown per finding:
    - +0.25: Correct dental quadrant (1-4)
    - +0.25: Correct tooth position (1-8)  -> total +0.50 for exact FDI tooth localization
    - +0.50: Correct pathology diagnosis class (Caries, Deep Caries, Periapical Lesion, Impacted Tooth)
    - Total: 1.0 for perfect FDI localization + disease diagnosis, per matched finding.

    Handles multiple findings on both sides (ground_truth and the predicted
    final_answer can each be a single dict or a list of dicts). Predictions
    are matched to ground-truth findings by greedy highest-pair-score-first
    assignment (one-to-one) using the same 0.25/0.25/0.50 rule per pair --
    case sizes here are small (a panoramic X-ray realistically has a handful
    of findings at most), so greedy matching is equivalent to optimal in the
    overwhelming majority of cases and avoids pulling in a full
    assignment-problem solver for this scale.

    The result is an F1-style harmonic mean of:
    - recall: total matched score / number of ground-truth findings (a
      missed finding contributes 0, so under-prediction is penalized)
    - precision: total matched score / number of predicted findings (an
      extra, unmatched prediction contributes 0, so hallucinating findings
      that aren't there is penalized too -- a pure recall-oriented average
      would let the model spam findings hoping to match ground truth for
      free, which is a bad incentive for a diagnostic reward)

    With exactly one ground-truth finding and one predicted finding this
    reduces to exactly the original single-pair score (verified in tests) --
    existing single-finding call sites are unaffected.
    """
    def _normalize_findings(x: Any) -> list[dict[str, Any]]:
        if isinstance(x, dict):
            return [x]
        if isinstance(x, list):
            return [f for f in x if isinstance(f, dict)]
        return []

    def _pair_score(pred: Mapping[str, Any], gt: Mapping[str, Any]) -> float:
        s = 0.0
        pq, gq = pred.get("quadrant"), gt.get("quadrant")
        if pq is not None and gq is not None:
            try:
                if int(pq) == int(gq):
                    s += 0.25
            except (TypeError, ValueError):
                pass
        pp, gp = pred.get("tooth_position"), gt.get("tooth_position")
        if pp is not None and gp is not None:
            try:
                if int(pp) == int(gp):
                    s += 0.25
            except (TypeError, ValueError):
                pass
        pd_, gd = str(pred.get("diagnosis", "")).strip().lower(), str(gt.get("diagnosis", "")).strip().lower()
        if pd_ and gd and pd_ == gd:
            s += 0.50
        return s

    preds = _normalize_findings(trajectory.get("final_answer"))
    gts = _normalize_findings(ground_truth)

    if not gts:
        # No ground-truth findings at all (shouldn't happen for DENTEX, which
        # always has at least one annotation per image, but handle it rather
        # than divide by zero): correct if the model also predicted nothing.
        return 1.0 if not preds else 0.0
    if not preds:
        return 0.0  # nothing predicted, every ground-truth finding missed

    pairs = sorted(
        ((_pair_score(p, g), pi, gi) for pi, p in enumerate(preds) for gi, g in enumerate(gts)),
        key=lambda x: x[0],
        reverse=True,
    )

    matched_pred: set[int] = set()
    matched_gt: set[int] = set()
    matched_total = 0.0
    for score, pi, gi in pairs:
        if pi in matched_pred or gi in matched_gt:
            continue
        matched_pred.add(pi)
        matched_gt.add(gi)
        matched_total += score

    recall = matched_total / len(gts)
    precision = matched_total / len(preds)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
