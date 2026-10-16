#!/usr/bin/env python3
"""Generate every data-derived asset for the VLM-DENTAL paper.

    python scripts/paper/generate_paper_assets.py \
        --traces-dir data/traces \
        --dentex-disease-json "data/dentex/DENTEX/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json" \
        --dentex-xrays "data/dentex/DENTEX/training_data/quadrant-enumeration-disease/xrays" \
        --tufts-radiographs data/tufts/Radiographs \
        --out-dir paper

Reads ONLY the verified canonical trace files (never the ``*unverified*`` ones)
and emits, under ``--out-dir``:

* ``generated/table_corpus.tex``         - corpus composition + 1:1 parity (with- vs no-tools)
* ``generated/table_tool_usage.tex``     - tool invocation counts over the with-tools corpus
* ``generated/table_nudge_tier.tex``     - the tiered-perturbation mechanism analysis:
                                          P(nudge_crop | tier) and IoU before/after correction
* ``figures/svg/fig_tool_usage.svg``     - data figure, matplotlib
* ``figures/svg/fig_nudge_tier.svg``     - data figure, matplotlib
* ``figures/svg/fig_perturbation_example.svg`` - real DENTEX radiograph with the ground-truth box,
                                          the perturbed box the teacher was shown, and the post-nudge box
* ``figures/pdf/*.pdf``                  - PDF twins of every SVG (matplotlib exports both; hand-authored
                                          diagrams in ``figures/svg/`` are converted with svglib when available)
* ``assets_summary.json``                - every number emitted, for checking against the prose

Only writes under ``--out-dir``. Never modifies trace files.

The nudge analysis walks each with-tools trace in order, pairs every executed
tool call with its observation (``Result of <tool>: {json}``), and for each
``locate_tooth`` return follows the working box forward: a ``nudge_crop`` whose
``bbox`` argument matches the current working box (within a 2px tolerance) is
applied with nudge_crop's own shift/clamp math, exactly as the agent loop would
have. ``IoU before`` is IoU(shown box, ground-truth box); ``IoU after`` is
IoU(post-nudge working box, ground-truth box). The window ends at the next
``locate_tooth`` call, so a nudge is attributed to the locate result it responds to.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# --------------------------------------------------------------------------
# Canonical cohorts (verified files only)
# --------------------------------------------------------------------------

WITH_TOOLS_FILES = {
    "dentex_pathology": "train_cot_traces_dentex.jsonl",
    "dentex_healthy": "train_cot_traces_healthy_dentex.jsonl",
    "tufts_overlap": "train_cot_traces_tufts.jsonl",
    "tufts_healthy": "train_cot_traces_healthy_tufts.jsonl",
    "tufts_all_diseases": "train_cot_traces_tufts_all.jsonl",
}

NO_TOOLS_FILES = {
    "dentex_pathology": "train_cot_traces_dentex_no_tools.jsonl",
    "dentex_healthy": "train_cot_traces_healthy_dentex_no_tools.jsonl",
    "tufts_overlap": "train_cot_traces_tufts_no_tools.jsonl",
    "tufts_healthy": "train_cot_traces_healthy_tufts_no_tools.jsonl",
    "tufts_all_diseases": "train_cot_traces_tufts_all_no_tools.jsonl",
}

COHORT_LABELS = {
    "dentex_pathology": "DENTEX pathology",
    "dentex_healthy": "DENTEX healthy (negative controls)",
    "tufts_overlap": "Tufts overlap pathology",
    "tufts_healthy": "Tufts healthy (negative controls)",
    "tufts_all_diseases": "Tufts 4-disease pathology",
}

TOOL_ORDER = [
    "locate_tooth",
    "nudge_crop",
    "zoom_crop",
    "fdi_label",
    "window_level",
    "enhance_contrast",
    "denoise",
    "contralateral_compare",
]

TIER_ORDER = ["none", "small", "big"]

# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def iou(a: list[float], b: list[float]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ax1, ay1, bx1, by1 = ax0 + aw, ay0 + ah, bx0 + bw, by0 + bh
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nudge_box(bbox: list[float], dx_frac: float, dy_frac: float, scale: float, size: tuple[int, int] | None) -> list[float]:
    """Re-apply ``tool_nudge_crop``'s math without needing the PIL image.

    ``size`` is (width, height); when unknown, clamping is skipped (only
    affects boxes pushed against an image border).
    """
    x, y, w, h = bbox
    cx, cy = x + w / 2.0 + dx_frac * w, y + h / 2.0 + dy_frac * h
    scale = max(0.1, scale)
    new_w, new_h = w * scale, h * scale
    if size is not None:
        new_w = min(new_w, float(size[0]))
        new_h = min(new_h, float(size[1]))
    new_x, new_y = cx - new_w / 2.0, cy - new_h / 2.0
    if size is not None:
        new_x = max(0.0, min(new_x, size[0] - new_w))
        new_y = max(0.0, min(new_y, size[1] - new_h))
    return [new_x, new_y, new_w, new_h]


def close(a: list[float], b: list[float], tol: float = 2.0) -> bool:
    return all(abs(float(x) - float(y)) <= tol for x, y in zip(a, b))


def parse_observation(msgs: list[dict], msg_idx: int) -> dict[str, list[dict | None]]:
    """Retained ``Result of <tool>:`` payloads in one observation message.

    Messages are laid out as system, then per turn: user observation, assistant
    reply — so turn ``ti``'s observation lives at ``msgs[2*ti + 1]``. Dropped
    image parts appear as ``[Earlier tool result omitted ...]`` placeholder
    texts, so results are pooled **per tool name** and matched to calls
    afterwards (locate_tooth by its tooth number, everything else in order) —
    never by raw list position.
    """
    pools: dict[str, list[dict | None]] = {}
    if msg_idx >= len(msgs):
        return pools
    content = msgs[msg_idx].get("content")
    if not isinstance(content, list):
        return pools
    for part in content:
        if part.get("type") != "text":
            continue
        txt = str(part.get("text", ""))
        if not txt.startswith("Result of "):
            continue
        head, _, tail = txt.partition(": ")
        name = head[len("Result of "):]
        payload = None
        if tail:
            try:
                payload = json.loads(tail)
            except (json.JSONDecodeError, ValueError):
                payload = None
        pools.setdefault(name, []).append(payload)
    return pools


# --------------------------------------------------------------------------
# Per-trace call walk
# --------------------------------------------------------------------------

BOX_ARG_TOOLS = {"zoom_crop", "nudge_crop", "contralateral_compare"}


def walk_calls(trace: dict):
    """Yield (call_record, payload_or_None) for every executed call, in order.

    ``payload`` is the parsed ``Result of <tool>:`` JSON when the observation
    was retained (``None`` when the message was truncated away, the tool
    returned an image, or the call failed). Call success is additionally
    available as ``call_record["tool_ok"]``. ``locate_tooth`` payloads are
    matched to their call by tooth number, which is exact; other tools are
    matched in observation order within the turn.
    """
    msgs = trace.get("messages", [])
    for ti, turn in enumerate(trace.get("turns", [])):
        calls = turn.get("tool_calls_this_turn", [])
        pools = parse_observation(msgs, 2 * ti + 1)
        # locate_tooth: match by tooth number first (exact, order-free).
        locate_pool = pools.pop("locate_tooth", [])
        used = [False] * len(locate_pool)
        for c in calls:
            payload = None
            if c.get("tool_name") == "locate_tooth":
                tooth = c.get("tool_args", {}).get("tooth")
                for k, pl in enumerate(locate_pool):
                    if not used[k] and pl is not None and pl.get("tooth") == tooth:
                        payload = pl
                        used[k] = True
                        break
            else:
                pool = pools.get(c.get("tool_name"), [])
                if pool:
                    payload = pool.pop(0)
            yield c, payload


def _center(b: list[float]) -> tuple[float, float]:
    return (b[0] + b[2] / 2.0, b[1] + b[3] / 2.0)


def _dist(c1, c2) -> float:
    return math.hypot(c1[0] - c2[0], c1[1] - c2[1])


def analyse_trace(trace: dict, size_lookup):
    """Nudge-vs-tier records + raw tool counts for one trace.

    Recovery of the box the teacher was SHOWN for a ``locate_tooth`` call:

    * **Perturbed calls** (``perturb_tier`` set): ground truth comes from the
      internal ``true_bbox`` audit field; the shown box comes from the retained
      observation payload (which, after the perturbation hook, is exactly the
      post-perturbation box) or, when that message was truncated, from the
      agent's own downstream ``bbox`` arguments (zoom/nudge/contralateral calls
      operate on exactly the box the agent was given), attributed by nearest
      centre to the most recent not-yet-recovered locate.
    * **Clean calls** for a tooth that has ground truth: GT-grounding returns
      the exact ground-truth box by construction, so shown = true and
      ``IoU before`` is 1.0 identically. Exploratory calls for teeth without
      ground truth are excluded (there is no reference box).

    Each attributed ``nudge_crop`` is classified by direction: *corrective* if
    its shift moved the working box toward the ground-truth centre (cosine of
    the move vs. the required displacement > 0.3), *away* if against it
    (cosine < -0.3), *lateral* otherwise (includes pure rescales). A record is
    "corrected" when it received at least one corrective nudge.
    """
    size = size_lookup(trace)
    gt_boxes = {}
    for f in trace.get("ground_truth") or []:
        q, p = f.get("quadrant"), f.get("tooth_position")
        if q is not None and p is not None and f.get("bbox"):
            gt_boxes[f"{int(q)}{int(p)}"] = [float(v) for v in f["bbox"]]

    events = list(walk_calls(trace))
    records: list[dict] = []
    tool_counts: Counter = Counter()

    # Pass 1: locate_tooth records (in event order).
    for i, (c, payload) in enumerate(events):
        tool_counts[c.get("tool_name", "unknown")] += 1
        if c.get("tool_name") != "locate_tooth" or not c.get("tool_ok"):
            continue
        tier = c.get("perturb_tier")
        tooth = c.get("tool_args", {}).get("tooth")
        gt_box = gt_boxes.get(str(tooth)) if tooth is not None else None
        if tier and c.get("true_bbox"):
            true_box = [float(v) for v in c["true_bbox"]]
            # The perturbation hook fires on every successful locate_tooth; the
            # reference box is the ground-truth box when the tooth is annotated
            # (GT-grounding mode) and the detector's own box otherwise.
            reference = "gt" if gt_box is not None and close(true_box, gt_box) else "detector"
        elif gt_box is not None:
            # Clean GT-grounded call: the tool returned the exact ground-truth box.
            true_box = list(gt_box)
            tier = "none"
            reference = "gt"
        else:
            continue  # exploratory locate with no reference box
        shown = None
        shown_src = None
        payload_box = None
        if payload is not None and isinstance(payload.get("bbox"), list):
            payload_box = [float(v) for v in payload["bbox"]]
            if tier != "none":
                shown = list(payload_box)  # post-perturbation box, exactly what was displayed
                shown_src = "payload"
        if tier == "none":
            # No perturbation fired: what was displayed IS the ground-truth box.
            shown = list(true_box)
            shown_src = "gt"
        records.append({
            "tier": tier,
            "reference": reference,
            "true": true_box,
            "shown": shown,
            "shown_src": shown_src,
            "payload_box": payload_box,
            "working": list(shown) if shown is not None else None,
            "n_nudges": 0,
            "n_corrective": 0,
            "n_away": 0,
            "event_idx": i,
            "tooth": tooth,
        })

    # Pass 2: attribute box-arg calls and follow nudge chains.
    for i, (c, payload) in enumerate(events):
        tool = c.get("tool_name")
        args = c.get("tool_args", {})
        if tool not in BOX_ARG_TOOLS:
            continue
        b = args.get("bbox")
        if not isinstance(b, list) or len(b) != 4:
            continue
        b = [float(v) for v in b]
        bcenter = _center(b)

        # (1) continue an existing working-box chain?
        hit = None
        best_d = math.inf
        for r in records:
            if r["working"] is not None and close(r["working"], b):
                d = _dist(_center(r["working"]), bcenter)
                if d < best_d:
                    hit, best_d = r, d

        # (2) otherwise, first response to a recent not-yet-recovered locate?
        if hit is None:
            candidates = [r for r in records if r["shown"] is None and r["event_idx"] < i]
            best_d = math.inf
            for r in candidates[-6:]:
                d = _dist(_center(r["true"]), bcenter)
                if d < best_d:
                    hit, best_d = r, d
            if hit is not None:
                # Reject absurd attributions (agent-asserted boxes far from any locate).
                span = max(hit["true"][2], hit["true"][3])
                if best_d > 2.0 * span:
                    hit = None
            if hit is not None:
                hit["shown"] = b
                hit["working"] = list(b)
                hit["shown_src"] = "args"

        if tool == "nudge_crop" and hit is not None:
            result_box = None
            if payload is not None and isinstance(payload.get("bbox"), list):
                result_box = [float(v) for v in payload["bbox"]]
            if result_box is None:
                result_box = nudge_box(
                    hit["working"] or b,
                    float(args.get("dx_frac", 0.0)),
                    float(args.get("dy_frac", 0.0)),
                    float(args.get("scale", 1.0)),
                    size,
                )
            # Direction of this nudge relative to the ground-truth centre.
            w = hit["working"] or b
            move = (float(args.get("dx_frac", 0.0)) * w[2], float(args.get("dy_frac", 0.0)) * w[3])
            need = (hit["true"][0] + hit["true"][2] / 2.0 - (w[0] + w[2] / 2.0),
                    hit["true"][1] + hit["true"][3] / 2.0 - (w[1] + w[3] / 2.0))
            mlen = math.hypot(*move)
            nlen = math.hypot(*need)
            if mlen < 1e-6 or nlen < 1e-6:
                kind = "lateral"  # pure rescale or already centred
            else:
                cos = (move[0] * need[0] + move[1] * need[1]) / (mlen * nlen)
                kind = "corrective" if cos > 0.3 else ("away" if cos < -0.3 else "lateral")
            hit["working"] = result_box
            hit["n_nudges"] += 1
            if kind == "corrective":
                hit["n_corrective"] += 1
            elif kind == "away":
                hit["n_away"] += 1

    clean_checked = []
    for r in records:
        shown = r["shown"] if r["shown"] is not None else r["true"]
        final = r["working"] if r["working"] is not None else shown
        r["shown_box"] = list(shown)
        r["final_box"] = list(final)
        r["iou_before"] = iou(shown, r["true"])
        r["iou_after"] = iou(final, r["true"])
        r["nudged"] = r["n_nudges"] > 0
        r["corrected"] = r["n_corrective"] > 0
        r["shown_recovered"] = r["shown"] is not None
        if r["tier"] == "none" and r.get("payload_box"):
            clean_checked.append(sum(abs(a - b) for a, b in zip(r["payload_box"], r["true"])))
        r.pop("working", None)
        r.pop("shown", None)
        r.pop("payload_box", None)
    return records, tool_counts, (sum(clean_checked) / len(clean_checked) if clean_checked else None)


# --------------------------------------------------------------------------
# Asset generation
# --------------------------------------------------------------------------


def build_size_lookups(args):
    """image_id -> (width, height), best effort per dataset."""
    dentex_sizes: dict[int, tuple[int, int]] = {}
    try:
        with open(args.dentex_disease_json, encoding="utf-8") as fh:
            coco = json.load(fh)
        for im in coco.get("images", []):
            dentex_sizes[int(im["id"])] = (int(im["width"]), int(im["height"]))
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"warn: could not load DENTEX sizes: {exc}")

    def dentex_lookup(trace: dict):
        return dentex_sizes.get(int(trace.get("image_id", -1)))

    tufts_sizes: dict[int, tuple[int, int]] = {}
    rad_dir = Path(args.tufts_radiographs)
    if rad_dir.is_dir():
        from PIL import Image

        for p in rad_dir.glob("*.JPG"):
            try:
                with Image.open(p) as im:
                    tufts_sizes[int(p.stem)] = im.size
            except (OSError, ValueError):
                continue

    def tufts_lookup(trace: dict):
        return tufts_sizes.get(int(trace.get("image_id", -1)))

    return {"dentex": dentex_lookup, "tufts": tufts_lookup}


def collect(traces_dir: Path, size_lookups):
    """Run the full analysis over every canonical cohort."""
    per_cohort: dict[str, dict] = {}
    nudge_records: list[dict] = []
    tool_totals: Counter = Counter()

    for cohort, fname in WITH_TOOLS_FILES.items():
        path = traces_dir / fname
        if not path.exists():
            print(f"warn: missing {path}; cohort {cohort} skipped")
            continue
        dataset = "tufts" if cohort.startswith("tufts") else "dentex"
        lookup = size_lookups[dataset]
        rows = read_jsonl(path)
        cohort_records: list[dict] = []
        tool_counts: Counter = Counter()
        turn_counts, call_counts = [], []
        clean_errs = []
        for r in rows:
            recs, tc, cerr = analyse_trace(r, lookup)
            for rec in recs:
                rec["image_id"] = r.get("image_id")
            cohort_records.extend(recs)
            tool_counts.update(tc)
            if cerr is not None:
                clean_errs.append(cerr)
            turn_counts.append(len(r.get("turns", [])))
            call_counts.append(sum(len(t.get("tool_calls_this_turn", [])) for t in r.get("turns", [])))
        nudge_records.extend(cohort_records)
        tool_totals.update(tool_counts)
        gt_lists = [r.get("ground_truth") or [] for r in rows]
        per_cohort[cohort] = {
            "file": fname,
            "traces": len(rows),
            "images": len({r.get("image_id") for r in rows}),
            "findings": sum(len(g) for g in gt_lists),
            "healthy_traces": sum(1 for g in gt_lists if len(g) == 0),
            "avg_turns": sum(turn_counts) / len(turn_counts) if turn_counts else 0.0,
            "avg_tool_calls": sum(call_counts) / len(call_counts) if call_counts else 0.0,
            "tool_counts": dict(tool_counts),
            "records": cohort_records,
            "clean_match_abs_err": sum(clean_errs) / len(clean_errs) if clean_errs else None,
        }
        print(f"  {cohort}: {len(rows)} traces, {len(cohort_records)} locate_tooth returns analysed")

    no_tools_counts = {}
    for cohort, fname in NO_TOOLS_FILES.items():
        path = traces_dir / fname
        no_tools_counts[cohort] = len(read_jsonl(path)) if path.exists() else None

    return per_cohort, nudge_records, tool_totals, no_tools_counts


def agg_nudge(records: list[dict]) -> dict:
    return {
        "overall": _agg_subset(records),
        "gt_anchored": _agg_subset([r for r in records if r["reference"] == "gt"]),
        "detector_anchored": _agg_subset([r for r in records if r["reference"] == "detector"]),
    }


def _agg_subset(records: list[dict]) -> dict:
    out = {}
    for tier in TIER_ORDER:
        rs = [r for r in records if r["tier"] == tier]
        nudged = [r for r in rs if r["nudged"]]
        corrected = [r for r in rs if r["corrected"]]
        recov = [r for r in rs if r["shown_recovered"]]
        before = [r["iou_before"] for r in recov]
        after = [r["iou_after"] for r in corrected]
        out[tier] = {
            "n": len(rs),
            "n_shown_recovered": len(recov),
            "recovery_rate": len(recov) / len(rs) if rs else 0.0,
            "nudge_rate": len(nudged) / len(rs) if rs else 0.0,
            "corrected_rate": len(corrected) / len(rs) if rs else 0.0,
            "iou_before_mean": sum(before) / len(before) if before else 0.0,
            "iou_before_sd": _sd(before),
            "iou_after_mean": sum(after) / len(after) if after else None,
            "iou_after_sd": _sd(after),
            "delta_mean": (sum(after) / len(after) - sum(before) / len(before)) if after else None,
            "n_nudges_total": sum(r["n_nudges"] for r in rs),
            "n_corrective_total": sum(r["n_corrective"] for r in rs),
            "n_away_total": sum(r["n_away"] for r in rs),
        }
    return out


def _sd(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


# --------------------------------------------------------------------------
# LaTeX emitters
# --------------------------------------------------------------------------


def latex_escape(s: str) -> str:
    return s.replace("%", "\\%").replace("_", "\\_")


def write_corpus_table(out_dir: Path, per_cohort: dict, no_tools_counts: dict):
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Cohort & With-tools & No-tools & Findings & Healthy & Turns & Tool calls \\\\",
        "\\midrule",
    ]
    tot_w = tot_n = tot_f = tot_h = 0
    for cohort in WITH_TOOLS_FILES:
        c = per_cohort.get(cohort)
        if not c:
            continue
        n = no_tools_counts.get(cohort)
        parity = "\\checkmark" if n == c["traces"] else "\\textbf{mismatch}"
        lines.append(
            f"{latex_escape(COHORT_LABELS[cohort])} & {c['traces']} & {n if n is not None else '--'}~{parity} & "
            f"{c['findings']} & {c['healthy_traces']} & {c['avg_turns']:.1f} & {c['avg_tool_calls']:.1f} \\\\"
        )
        tot_w += c["traces"]
        tot_n += n or 0
        tot_f += c["findings"]
        tot_h += c["healthy_traces"]
    lines += [
        "\\midrule",
        f"Total & {tot_w} & {tot_n}~\\checkmark & {tot_f} & {tot_h} & -- & -- \\\\",
        "\\bottomrule",
        "\\end{tabular}",
    ]
    (out_dir / "generated" / "table_corpus.tex").write_text("\n".join(lines), encoding="utf-8")
    return {"total_traces": tot_w, "total_no_tools": tot_n, "total_findings": tot_f, "total_healthy": tot_h}


def write_tool_table(out_dir: Path, tool_totals: Counter):
    total = sum(tool_totals.values()) or 1
    lines = ["\\begin{tabular}{lrr}", "\\toprule", "Tool & Calls & Share \\\\", "\\midrule"]
    for t in TOOL_ORDER:
        n = tool_totals.get(t, 0)
        lines.append(f"\\texttt{{{t}}} & {n} & {100.0 * n / total:.1f}\\% \\\\")
    other = sum(v for k, v in tool_totals.items() if k not in TOOL_ORDER)
    if other:
        lines.append(f"other & {other} & {100.0 * other / total:.1f}\\% \\\\")
    lines += ["\\midrule", f"Total & {total} & 100.0\\% \\\\", "\\bottomrule", "\\end{tabular}"]
    (out_dir / "generated" / "table_tool_usage.tex").write_text("\n".join(lines), encoding="utf-8")


def write_nudge_table(out_dir: Path, agg: dict):
    """Table over the GT-anchored subset (reference box = ground-truth box),
    which is the reading in which ``IoU`` is literal."""
    lines = [
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Tier & $n$ & P(nudge) & P(corrective) & IoU before & IoU after & $\\Delta$IoU \\\\",
        "\\midrule",
    ]
    for tier in TIER_ORDER:
        a = agg["gt_anchored"][tier]
        after = f"{a['iou_after_mean']:.2f} $\\pm$ {a['iou_after_sd']:.2f}" if a["iou_after_mean"] is not None else "--"
        delta = f"{a['delta_mean']:+.2f}" if a["delta_mean"] is not None else "--"
        lines.append(
            f"{tier} & {a['n']:,} & {100.0 * a['nudge_rate']:.1f}\\% & {100.0 * a['corrected_rate']:.1f}\\% & "
            f"{a['iou_before_mean']:.2f} & {after} & {delta} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "generated" / "table_nudge_tier.tex").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# Figures (matplotlib -> SVG + PDF)
# --------------------------------------------------------------------------


def plot_tool_usage(out_dir: Path, tool_totals: Counter):
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 9})
    items = [(t, tool_totals.get(t, 0)) for t in TOOL_ORDER]
    items.sort(key=lambda kv: kv[1])
    names = [t for t, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    bars = ax.barh(names, vals, color="#4C72B0")
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals) * 0.01, b.get_y() + b.get_height() / 2, f"{v:,}", va="center", fontsize=8)
    ax.set_xlabel("Tool calls in the verified with-tools corpus")
    ax.spines[["top", "right"]].set_visible(False)
    ax.margins(x=0.12)
    fig.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(out_dir / "figures" / ext / f"fig_tool_usage.{ext}")
    plt.close(fig)


def plot_nudge_tier(out_dir: Path, agg: dict):
    plt.rcParams.update({"svg.fonttype": "path", "font.size": 9})
    tiers = TIER_ORDER
    rates = [100.0 * agg["overall"][t]["corrected_rate"] for t in tiers]
    before = [agg["gt_anchored"][t]["iou_before_mean"] for t in tiers]
    after = [agg["gt_anchored"][t]["iou_after_mean"] or 0.0 for t in tiers]
    ns = [agg["overall"][t]["n"] for t in tiers]
    x = range(len(tiers))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 2.6))
    bars = ax1.bar(x, rates, color=["#999999", "#DD8452", "#C44E52"])
    for b, r, n in zip(bars, rates, ns):
        ax1.text(b.get_x() + b.get_width() / 2, r + 1.5, f"{r:.0f}%\n(n={n:,})", ha="center", fontsize=8)
    ax1.set_xticks(list(x), tiers)
    ax1.set_ylabel("P(corrective nudge_crop)")
    ax1.set_ylim(0, max(rates) * 1.35 + 2)
    ax1.set_title("Correction behavior by perturbation tier", fontsize=9)
    ax1.spines[["top", "right"]].set_visible(False)

    w = 0.36
    ax2.bar([i - w / 2 for i in x], before, w, label="box as shown", color="#999999")
    ax2.bar([i + w / 2 for i in x], after, w, label="after correction", color="#55A868")
    ax2.set_xticks(list(x), tiers)
    ax2.set_ylabel("IoU with ground-truth box")
    ax2.set_ylim(0, 1.05)
    ax2.set_title("Box quality before / after correction", fontsize=9)
    ax2.legend(fontsize=8, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("svg", "pdf"):
        fig.savefig(out_dir / "figures" / ext / f"fig_nudge_tier.{ext}")
    plt.close(fig)


def render_perturbation_example(out_dir: Path, per_cohort: dict, args):
    """Pick the best big-tier recovery on a locally available DENTEX image and draw it."""
    from PIL import Image

    try:
        with open(args.dentex_disease_json, encoding="utf-8") as fh:
            coco = json.load(fh)
        id_to_file = {int(im["id"]): im["file_name"] for im in coco.get("images", [])}
    except (OSError, json.JSONDecodeError, KeyError) as exc:
        print(f"warn: cannot render perturbation example: {exc}")
        return None

    xrays = Path(args.dentex_xrays)
    candidates = []
    recs = per_cohort.get("dentex_pathology", {}).get("records", [])
    # records need their image id: stored during collect() as 'image_id'
    for r in recs:
        if r.get("image_id") not in id_to_file:
            continue
        path = xrays / id_to_file[r["image_id"]]
        if not path.exists():
            continue
        if r["tier"] == "big" and r["nudged"] and r["iou_after"] >= 0.55:
            gain = r["iou_after"] - r["iou_before"]
            if gain > 0.15:
                candidates.append((gain, r, path))
    if not candidates:
        print("warn: no qualifying big-tier example found for the illustration figure")
        return None
    candidates.sort(key=lambda t: -t[0])
    gain, rec, img_path = candidates[0]

    img = Image.open(img_path).convert("RGB")
    scale = 1400.0 / img.width
    img_small = img.resize((1400, max(1, int(img.height * scale))))

    def sx(b):
        return [b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.6), gridspec_kw={"width_ratios": [2.1, 1.0]})
    ax1.imshow(img_small, cmap="gray")
    draw_box(ax1, sx(rec["true"]), "#2ca02c", "-", "ground-truth box")
    draw_box(ax1, sx(rec["shown_box"]), "#d62728", "--", "shown to teacher (perturbed)")
    draw_box(ax1, sx(rec["final_box"]), "#ff7f0e", (0, (1, 1.3)), "after nudge_crop", lw=3.0)
    ax1.set_title("DENTEX image with tiered perturbation", fontsize=9)
    ax1.axis("off")

    # zoom around the union of the three boxes with padding
    xs = [rec["true"][0], rec["shown_box"][0], rec["final_box"][0]]
    ys = [rec["true"][1], rec["shown_box"][1], rec["final_box"][1]]
    xe = [b[0] + b[2] for b in (rec["true"], rec["shown_box"], rec["final_box"])]
    ye = [b[1] + b[3] for b in (rec["true"], rec["shown_box"], rec["final_box"])]
    x0, y0 = max(0, min(xs) - 160), max(0, min(ys) - 160)
    x1, y1 = min(img.width, max(xe) + 160), min(img.height, max(ye) + 160)
    crop = img.crop((int(x0), int(y0), int(x1), int(y1)))
    ax2.imshow(crop, cmap="gray")

    def cz(b):
        return [b[0] - x0, b[1] - y0, b[2], b[3]]

    draw_box(ax2, cz(rec["true"]), "#2ca02c", "-", None)
    draw_box(ax2, cz(rec["shown_box"]), "#d62728", "--", None)
    draw_box(ax2, cz(rec["final_box"]), "#ff7f0e", (0, (1, 1.3)), None, lw=3.0)
    ax2.set_title(f"zoom (IoU {rec['iou_before']:.2f} $\\to$ {rec['iou_after']:.2f})", fontsize=9)
    ax2.axis("off")

    handles = [
        plt.Line2D([], [], color="#2ca02c", ls="-", lw=2, label="ground-truth box"),
        plt.Line2D([], [], color="#d62728", ls="--", lw=2, label="shown (perturbed)"),
        plt.Line2D([], [], color="#ff7f0e", ls=(0, (1, 1.3)), lw=3.0, label="after nudge_crop"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    for ext in ("svg", "pdf"):
        fig.savefig(out_dir / "figures" / ext / f"fig_perturbation_example.{ext}", dpi=200)
    plt.close(fig)
    return {
        "image_id": rec["image_id"],
        "file": id_to_file[rec["image_id"]],
        "tooth_gain": gain,
        "iou_before": rec["iou_before"],
        "iou_after": rec["iou_after"],
    }


def draw_box(ax, b, color, ls, label, lw=2.0):
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((b[0], b[1]), b[2], b[3], fill=False, color=color, ls=ls, lw=lw, label=label))


# --------------------------------------------------------------------------
# SVG -> PDF for hand-authored diagrams
# --------------------------------------------------------------------------


def convert_diagrams(out_dir: Path):
    svg_dir = out_dir / "figures" / "svg"
    pdf_dir = out_dir / "figures" / "pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    try:
        from svglib.svglib import svg2rlg
        from reportlab.graphics import renderPDF
    except ImportError:
        print("warn: svglib not installed; hand-authored SVG diagrams not converted to PDF")
        return
    for svg in sorted(svg_dir.glob("*.svg")):
        pdf = pdf_dir / (svg.stem + ".pdf")
        if pdf.exists() and pdf.stat().st_mtime >= svg.stat().st_mtime:
            continue
        try:
            drawing = svg2rlg(str(svg))
            if drawing is not None:
                renderPDF.drawToFile(drawing, str(pdf))
                print(f"  converted {svg.name} -> {pdf.name}")
        except Exception as exc:  # noqa: BLE001
            print(f"warn: failed to convert {svg.name}: {exc}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--traces-dir", default="data/traces")
    ap.add_argument("--dentex-disease-json", default="data/dentex/DENTEX/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json")
    ap.add_argument("--dentex-xrays", default="data/dentex/DENTEX/training_data/quadrant-enumeration-disease/xrays")
    ap.add_argument("--tufts-radiographs", default="data/tufts/Radiographs")
    ap.add_argument("--out-dir", default="paper")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "generated").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "svg").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures" / "pdf").mkdir(parents=True, exist_ok=True)

    print("Collecting corpus statistics and nudge-tier analysis ...")
    size_lookups = build_size_lookups(args)
    per_cohort, nudge_records, tool_totals, no_tools_counts = collect(Path(args.traces_dir), size_lookups)

    agg = agg_nudge(nudge_records)

    corpus_totals = write_corpus_table(out_dir, per_cohort, no_tools_counts)
    write_tool_table(out_dir, tool_totals)
    write_nudge_table(out_dir, agg)
    plot_tool_usage(out_dir, tool_totals)
    plot_nudge_tier(out_dir, agg)
    example_meta = render_perturbation_example(out_dir, per_cohort, args)
    convert_diagrams(out_dir)

    summary = {
        "corpus": {k: {kk: vv for kk, vv in v.items() if kk != "records"} for k, v in per_cohort.items()},
        "corpus_totals": corpus_totals,
        "tool_totals": dict(tool_totals),
        "nudge_by_tier": agg,
        "perturbation_example": example_meta,
    }
    (out_dir / "assets_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nWrote assets to {out_dir}/ (tables, figures, assets_summary.json)")


if __name__ == "__main__":
    main()
