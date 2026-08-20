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

## Trace-gen hint mechanism (search_region_hint)

- `hint_probability = 1.0` — search_region_hint is always applied when ground truth
  exists for the requested tooth. This gives the real detector the best chance of a
  correct result; it is NOT a synthetic "perfect answer" injection — what's shown is
  still real (hint-narrowed) YOLO inference, which can occasionally still miss.

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

- `run_trace_gen`'s `max_tool_calls` defaults to 50 (already the case before today's
  patches — no change needed there for "some images need more calls than others").
- GRPO's `max_tool_calls` still defaults to 4 (`grpo.py`) — this is the actual mismatch,
  not yet addressed. Next patch.
