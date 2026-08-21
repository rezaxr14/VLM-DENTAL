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
    ground_truth: Mapping[str, Any],
) -> float:
    """Graded diagnostic accuracy reward (R_accuracy in §5.5).

    Graded breakdown per finding:
    - +0.25: Correct dental quadrant (1-4)
    - +0.25: Correct tooth position (1-8)  -> total +0.50 for exact FDI tooth localization
    - +0.50: Correct pathology diagnosis class (Caries, Deep Caries, Periapical Lesion, Impacted Tooth)
    - Total: 1.0 for perfect FDI localization + disease diagnosis.

    NOTE: this still assumes a single ground-truth finding per trajectory
    (ground_truth is a single {quadrant, tooth_position, diagnosis} dict, and
    a list final_answer is not matched against multiple ground-truth
    findings) -- flagged separately, not fixed here. Extending this to score
    multi-finding trajectories properly needs a real design decision on
    matching strategy and false-positive handling, not a quick patch
    alongside efficiency retuning.
    """
    ans = trajectory.get("final_answer")
    if isinstance(ans, list):
        # Multi-finding answer against single-finding ground truth: take the
        # first finding as a reasonable stand-in rather than scoring 0.0
        # outright, until the real multi-finding design lands.
        ans = ans[0] if ans and isinstance(ans[0], dict) else None
    if not isinstance(ans, dict):
        return 0.0

    score = 0.0
    gt_quad = ground_truth.get("quadrant")
    gt_pos = ground_truth.get("tooth_position")
    gt_diag = str(ground_truth.get("diagnosis", "")).strip().lower()

    # 1. Quadrant check (+0.25)
    pred_quad = ans.get("quadrant")
    if pred_quad is not None and gt_quad is not None and int(pred_quad) == int(gt_quad):
        score += 0.25

    # 2. Tooth position check (+0.25)
    pred_pos = ans.get("tooth_position")
    if pred_pos is not None and gt_pos is not None and int(pred_pos) == int(gt_pos):
        score += 0.25

    # 3. Pathology diagnosis check (+0.50)
    pred_diag = str(ans.get("diagnosis", "")).strip().lower()
    if pred_diag and gt_diag and pred_diag == gt_diag:
        score += 0.50

    return score
