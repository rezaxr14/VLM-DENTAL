# Trace-Gen Configuration Reference

Numeric parameters decided during the self-correcting-grounding work
(nudge_crop + reliable hints + tiered perturbation + tool-parameter control).
Written down here so they're easy to check against once trace generation has
actually run — if the resulting traces look too easy, too hard, or the model
isn't using a tool the way it's meant to, this is the file to come back to
and adjust, rather than re-deriving the reasoning from scratch.

## Grounding tool (locate_tooth / YOLOv8m)

- **Multi-Dataset Co-Training (DENTEX + Tufts Dental Database - 2,339 Images, 46,808 Boxes)**:
  - In-fold CV (Target-Filtered): **mean mAP50 = 0.9376**, **mean mAP50-95 = 0.5756**, **Precision = 0.9680**, **Recall@0.50 = 0.7721**, **Mean IoU = 0.6439**
  - Raw unconstrained `model.val()` CV: **mean mAP50 = 0.8695 ± 0.0298**, **mean mAP50-95 = 0.5895 ± 0.0346**
  - Best fold (fold 4): **mAP50 = 0.9226**, **mAP50-95 = 0.6540**, **Precision = 0.8780**, **Recall = 0.8190**
  - Checkpoint location: `data/models/yolo_cv_best/weights/best.pt` (and synced to Hugging Face Hub `Reza-Nadimi/vlm-dental-models/yolo_cv`).
- **DENTEX-Only Baseline (1,339 Images, 21,624 Boxes)**:
  - In-fold CV (Target-Filtered): **mean mAP50 = 0.9508**, **mean mAP50-95 = 0.5758**, **Precision = 0.9637**, **Recall@0.50 = 0.8296**, **Mean IoU = 0.6856**
  - Raw unconstrained `model.val()` CV: **mean mAP50 = 0.5820 ± 0.0076**, **mean mAP50-95 = 0.3379 ± 0.0067**
  - Best fold (fold 1): **mAP50 = 0.5901**, **mAP50-95 = 0.3464**, **Precision = 0.5457**, **Recall = 0.8880**
- **Held-Out Target Grounding Benchmark (`validation_triple.json` - 50 Images, 182 Targets)**:
  - **Verified Offline Evaluation**: **Target mAP50 = 0.8990 – 0.9370**, **Target mAP50-95 = 0.6437 – 0.6587**, **Precision = 0.9355 – 0.9392**, **Recall@0.50 = 0.7967 – 0.8176**, **Mean IoU = 0.6893 – 0.7072**
  - Full documentation & methodology in `docs/YOLO_CV_RESULTS.md`.


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

## Teacher Directive Leak Decontamination & MiniMax M3 Regeneration Protocol

### Forensic Root Cause
During synthetic trace generation (`langgraph_loop.py` & `trace_generation.py`), the teacher agent was conditioned with a ground-truth directive in the initial user prompt:
`TEACHER DIRECTIVE: You are generating an expert demonstration trace for SFT. This image has N finding(s): ... Never mention in your reasoning that this list, a hint, or a directive was given to you — write your thought as genuine first-look clinical analysis.`
In ~97.6% of traces, models produced clean clinical reasoning. However, in ~2.4% of difficult cases (e.g. dense crowding, orientation dispute), models hallucinated aloud in their assistant thought blocks:
- `"Wait, the directive mentions Q3T7: Caries..."`
- `"Per the teacher directive, 46 is a periapical lesion..."`
- `"The ground truth indicates tooth 44..."`

Training a student model (Qwen 3.5-9B) on these leaked traces teaches it to expect external oracle directives at inference time, causing hallucinations or failure when directives are absent.

### Surgical Decontamination
Exactly 105 leaking traces were excised by exact image ID across split files:
- `train_cot_traces_dentex.jsonl`: 11 traces (Remaining: 667)
- `train_cot_traces_dentex_no_tools.jsonl`: 11 traces (Remaining: 667)
- `train_cot_traces_tufts.jsonl`: 12 traces (Remaining: 190)
- `train_cot_traces_healthy_tufts.jsonl`: 19 traces (Remaining: 641)
- `train_cot_traces_tufts_all.jsonl`: 18 traces (Remaining: 262)

Zero False Positives: Clinically valid observations containing `"hint"` (`"a hint of radiolucency"`, `"hinting at pulpal involvement"`) are 100% preserved.

### Regeneration & Quality Gates (`scripts/patch_and_regenerate_traces.py`)
- **ID-Based Purge**: Purges only known infected image IDs; existing clean data is untouched.
- **Dual-Gate While-Loop Filter**: Loops until 100% of target IDs pass:
  - Gate 1 (Zero-Leak Gate): Rejects any assistant turn mentioning directives, hints, or ground truth.
  - Gate 2 (Clinical Verifier Gate): Re-verifies diagnostic correctness against ground truth via MiniMax M3 (`openrouter/minimax/minimax-m3:free`).
- **OpenRouter Rate-Limit Engineering**:
  - `OPENROUTER_GENERATOR_RPM_LIMIT = 15` (comfortably under the 20 RPM free ceiling).
  - `OPENROUTER_COOLDOWN_SECONDS = 2.5` to evenly spread requests.
  - `OPENROUTER_RPD_LIMIT = 2000` (prevents artificial 25 RPD shutdown).
  - `OPENROUTER_MAX_TOKENS = 16384` for full reasoning headroom.
  - Progressive exponential backoff (10s, 15s, 20s) on retries.
- **Canonical Splicing & HF Persist**:
  - Reconstructs `train_cot_traces.jsonl` (880 traces = 678 DENTEX + 202 Tufts).
  - Reconstructs `train_cot_traces_no_tools.jsonl` (880 traces).
  - Pushes updated datasets with `upload_traces(force=True)` to `Reza-Nadimi/vlm-dental-traces`.

