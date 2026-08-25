# Trace-Gen Configuration Reference

Numeric parameters decided during the self-correcting-grounding work
(nudge_crop + reliable hints + tiered perturbation + tool-parameter control).
Written down here so they're easy to check against once trace generation has
actually run — if the resulting traces look too easy, too hard, or the model
isn't using a tool the way it's meant to, this is the file to come back to
and adjust, rather than re-deriving the reasoning from scratch.

## Grounding tool (locate_tooth / YOLOv8m)

- 5-fold cross-validation, mean mAP50 = 0.5820 ± 0.0076, mean mAP50-95 = 0.3379 ± 0.0067
- Best fold (fold 1): mAP50 = 0.5901, mAP50-95 = 0.3464, Precision = 0.5457, Recall = 0.8880
- This is the fold currently pointed to by GROUNDING_MODEL_PATH.
- (Superseded number, do not use: mAP50 ≈ 0.647 from an earlier training run — still
  incorrectly referenced in figures/visualization_prompts.md as of this file's writing.)

## Trace-gen Ground-Truth Grounding

- During trace generation, `locate_tooth` operates in **Ground-Truth Grounding mode**: if the requested tooth has a known ground-truth bounding box, the tool returns it directly (bypassing the real YOLO detector), guaranteeing the tooth is found. The tiered perturbation below is then applied to this perfect box before showing it to the model, preserving the need for `nudge_crop` without suffering YOLO's recall failures.
- If the model requests a tooth that is *not* in the ground truth (an exploratory call), the tool falls back to an unconstrained full-image YOLO inference search.
- The former `search_region_hint` mechanism was removed because feeding a cropped region into YOLO distorted the scale and destroyed the model's ability to find the tooth (0% recall).

## Trace-gen perturbation mechanism (what the model is actually shown)

Applied AFTER a real detection comes back, to what's displayed — never changes what's
logged internally as ground truth. Three tiers, rolled independently of the real
detector's accuracy on purpose (see langgraph_loop.py's _tool_node_factory docstring
for the full reasoning — short version: tying this to the current detector's real
error rate would make traces go stale every time the detector improves).

| Tier  | Probability | Offset magnitude (dx_frac / dy_frac, per axis) | Intent |
|-------|-------------|--------------------------------------------------|--------|
| none  | 0.45        | 0                                                  | Clean accept-and-proceed demonstration |
| small | 0.25        | Uniform(0.12, 0.28), random sign                   | Genuine judgment call — often fine given zoom_crop's padding, sometimes worth a precise nudge. Deliberately not resolved either way by the prompt. |
| big   | 0.30        | Uniform(0.45, 0.75), random sign                   | Sized to be visually self-evident without any verbal cue — a verbal "this needs correcting" hint would only exist during trace-gen, not at GRPO/inference time, so magnitude alone has to carry it. |

Both tiers reuse `nudge_crop`'s own shift/clamp math (`_synthetic_offset`), so every
perturbation shown is something `nudge_crop` could plausibly have produced or corrected.

`call_record["true_bbox"]` and `call_record["perturb_tier"]` are logged internally per
tool call whenever a perturbation fires (for later auditing — e.g. did the model
actually catch and correct big-tier misses?) — never shown to the model itself.

## Tool suite (8 registered, dental_agent/tools/registry.py)

zoom_crop (padding_frac), window_level (preset, or center/width override), locate_tooth,
fdi_label, denoise (method, strength 0.0-1.0), contralateral_compare (bbox, quadrant),
enhance_contrast (factor), nudge_crop (bbox, dx_frac, dy_frac, scale).

## Dynamic tool-call budget

- `run_trace_gen`'s `max_tool_calls` defaults to 50 (unchanged — was already the case).
- GRPO's `max_tool_calls` (`run_agent`, `collect_grpo_group`, `grpo_step`) now also defaults
  to 50, and is counted by actual tool CALLS, not turns (a turn can carry several calls).
  `combine_reward`'s efficiency ceiling is threaded through explicitly at the call site in
  `collect_grpo_group` so it always matches whatever budget the rollout actually used.

## Critical fix: GRPO couldn't dispatch tool calls at all

`parse_agent_json` normalizes every parsed response into a `tool_calls` list, popping the
legacy flat `tool`/`args` keys in the process. `run_agent` (loop.py) was still reading the
now-permanently-absent `parsed.get("tool")` -- every tool-call attempt during a GRPO
rollout hit "not recognized", regardless of what the model asked for. Fixed: `run_agent`
now reads `tool_calls`, executes multiple calls per turn, and produces the same canonical
`tool_calls_this_turn` shape trace-gen's loop does.

## reward_efficiency redesign

Old design: flat penalty per tool call, no reference to case complexity, and its own
`max_calls` parameter was accepted but never used anywhere in the function body. New
design: reference budget = 6 calls per distinct located tooth (derived from the
trajectory's own `locate_tooth` calls, not a separate field); `locate_tooth` and
`nudge_crop` are exempt from the per-call cost entirely; only genuine waste (exceeding
budget, or an exact repeated tool+args call back-to-back) costs anything.

## Multi-finding accuracy scoring

`reward_accuracy` previously assumed one ground-truth finding per image, and `train_grpo`
/ `sweep.py` both only ever took `.iloc[0]` of an image's annotations -- discarding every
other real finding an image had before the reward even saw it. Both fixed: ground truth is
now every annotation row for that image (a list), and `reward_accuracy` matches predicted
findings to ground-truth findings by greedy highest-pair-score-first assignment, scoring
the result as an F1-style harmonic mean of recall (missed findings count as 0) and
precision (hallucinated extra findings also count as 0 -- prevents free-riding by
over-predicting). Reduces to exactly the old single-pair score when there's one of each.
