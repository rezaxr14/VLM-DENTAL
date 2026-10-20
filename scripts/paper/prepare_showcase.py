"""Prepare the showcase radiographs + correction episodes used by the paper figures.

Input : the folder of curated pairs (01_flagship_big_offset/, 02_..., each with
        image_full.jpg, annotated.png, crop_before/after.png, trace.json) mined
        from the verified corpus (data/traces, 1,847 with-tools traces).
Output: paper/figures/assets/showcase/<id>/{image.jpg,crop_before.jpg,crop_after.jpg}
        and showcase.json (native-coordinate boxes, correction chain, metrics).

    python scripts/paper/prepare_showcase.py --pairs path/to/send_to_claude

The raw DENTEX/Tufts images are not in the repository, so the committed
outputs of this script are the (small, downscaled) inputs of
`build_diagrams.py`; rerun it only when the curated pairs change.

Scale recovery: the pairs were downscaled to <= 1800 px wide, and the native
size is not stored. It is recovered by fitting the scale at which the trace's
own ground-truth boxes line up with the green boxes drawn in annotated.png
(validated on image 624: recovers its known 2808 px width exactly; Tufts
images are not downscaled, scale 1.0).
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "paper" / "figures" / "assets" / "showcase"

# folder -> (short id, role in the paper)
PAIRS = {
    "01_flagship_big_offset": "flagship",
    "02_three_step_reinspect": "three_step",
    "03_breadth_15_findings": "breadth",
    "04_deep_caries_13_findings": "deep_caries",
    "05_tufts_periapical": "tufts",
    "06_healthy_negative_control": "healthy",
    "08_small_tier_judgment": "small_tier",
}
FOCUS = {"flagship": 48, "three_step": 16, "breadth": 25, "deep_caries": 36, "tufts": 36, "small_tier": 16}
MAX_W = 1800
IMG_W = 1100  # stored width of the radiograph
CROP_H = 360
CTX_W = 720  # stored width of the per-episode context view

# ---- vendored from the miner (_select_traces.py): result parsing + IoU ----------------
RESULT_RE = re.compile(r"^Result of (\w+):\s*(.*)$", re.S)


def iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def offset_frac(box, gt):
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    gx, gy = gt[0] + gt[2] / 2, gt[1] + gt[3] / 2
    return math.hypot((cx - gx) / gt[2], (cy - gy) / gt[3])


def parse_trace(t):
    msgs = t["messages"]
    events, thoughts = [], []
    for i, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        try:
            parsed = json.loads(m["content"]) if isinstance(m["content"], str) else m["content"]
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(parsed, dict) and parsed.get("thought"):
            thoughts.append(parsed["thought"])
        calls = parsed.get("tool_calls") or []
        results = []
        if i + 1 < len(msgs) and msgs[i + 1]["role"] == "user" and isinstance(msgs[i + 1]["content"], list):
            for part in msgs[i + 1]["content"]:
                if part.get("type") != "text":
                    continue
                mm = RESULT_RE.match(part["text"].strip())
                if mm:
                    try:
                        payload = json.loads(mm.group(2)) if mm.group(2) else None
                    except json.JSONDecodeError:
                        payload = None
                    results.append((mm.group(1), payload))
        queues = defaultdict(list)
        for tool, payload in results:
            queues[tool].append(payload)
        taken = Counter()
        for call in calls:
            name = call.get("tool") or call.get("tool_name")
            args = call.get("args") or call.get("tool_args") or {}
            q = queues.get(name, [])
            payload = q[taken[name]] if taken[name] < len(q) else None
            taken[name] += 1
            events.append((name, args, payload))
    return events, thoughts


# ---- scale recovery ---------------------------------------------------------------------
def fit_scale(folder: Path, gt_boxes) -> float:
    ann = np.array(Image.open(folder / "annotated.png").convert("RGB")).astype(int)
    r, g, b = ann[..., 0], ann[..., 1], ann[..., 2]
    m = ((g > r + 30) & (g > b + 30) & (g > 100)).astype(np.uint8)
    m[:130, :420] = 0  # legend
    H, W = m.shape

    def score(k):
        s = 0
        for x, y, w, h in gt_boxes:
            X0, Y0, X1, Y1 = (int(round(v * k)) for v in (x, y, x + w, y + h))
            if X1 >= W or Y1 >= H:
                return -1
            for cx in (X0, X1):
                s += m[max(Y0, 0):Y1, max(cx - 1, 0):cx + 2].sum()
            for cy in (Y0, Y1):
                s += m[max(cy - 1, 0):cy + 2, max(X0, 0):X1].sum()
        return s

    ks = np.arange(0.50, 1.06, 0.0005)
    return round(float(ks[int(np.argmax([score(k) for k in ks]))]), 4)


def chain_for(t, focus):
    """(first returned box, nudge outputs, nudge args) for the focus tooth.

    Same attribution rule as the miner (_select_traces.analyse): a nudge belongs to
    the tooth whose last known box it overlaps (IoU > 0.5), else to the nearest
    ground-truth box by centre.
    """
    events, _ = parse_trace(t)
    gt = {f["quadrant"] * 10 + f["tooth_position"]: [float(x) for x in f["bbox"]] for f in t["ground_truth"]}
    first, last = {}, {}
    chain, args_chain = defaultdict(list), defaultdict(list)
    for name, args, res in events:
        if name == "locate_tooth":
            tooth = int(args.get("tooth", 0))
            if isinstance(res, dict) and "bbox" in res:
                box = [float(x) for x in res["bbox"]]
                first.setdefault(tooth, box)
                last[tooth] = box
        elif name == "nudge_crop":
            inp = [float(x) for x in (args.get("bbox") or [])]
            out = [float(x) for x in res["bbox"]] if isinstance(res, dict) and "bbox" in res else None
            if not inp or out is None:
                continue
            best, bi = None, 0.0
            for tt, kb in last.items():
                v = iou(inp, kb)
                if v > bi:
                    best, bi = tt, v
            target = best if bi > 0.5 else None
            if target is None:
                cx, cy = inp[0] + inp[2] / 2, inp[1] + inp[3] / 2
                target = min(gt, key=lambda tt: (cx - (gt[tt][0] + gt[tt][2] / 2)) ** 2 + (cy - (gt[tt][1] + gt[tt][3] / 2)) ** 2)
            chain[target].append(out)
            args_chain[target].append({k: args.get(k) for k in ("dx_frac", "dy_frac", "scale") if k in args})
            last[target] = out
    return first.get(focus), chain.get(focus, []), args_chain.get(focus, [])


def save_jpg(src: Path, dst: Path, width=None, height=None, q=86):
    im = Image.open(src).convert("RGB")
    if width and im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    if height and im.height > height:
        im = im.resize((round(im.width * height / im.height), height), Image.LANCZOS)
    im.save(dst, quality=q, optimize=True)
    return im.size


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", required=True, type=Path)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    meta = {}
    for folder, pid in PAIRS.items():
        d = args.pairs / folder
        t = json.loads((d / "trace.json").read_text(encoding="utf-8"))
        full = Image.open(d / "image_full.jpg")
        gt = [{"fdi": f["quadrant"] * 10 + f["tooth_position"], "diagnosis": f["diagnosis"], "bbox": [float(x) for x in f["bbox"]]} for f in t["ground_truth"]]
        boxes = [g["bbox"] for g in gt]
        k = 1.0 if full.width < MAX_W else (fit_scale(d, boxes) if boxes else None)
        item = {
            "folder": folder,
            "cohort": t["dataset"],
            "image_id": t["image_id"],
            "scale": k,
            "native_w": round(full.width / k) if k else None,
            "native_h": round(full.height / k) if k else None,
            "n_findings": len(gt),
            "n_turns": len(t["turns"]),
            "n_calls": t["tool_calls"],
            "ground_truth": gt,
            "final_answer": t.get("final_answer", []),
        }
        pdir = OUT / pid
        pdir.mkdir(exist_ok=True)
        item["image_size"] = list(save_jpg(d / "image_full.jpg", pdir / "image.jpg", width=IMG_W))
        focus = FOCUS.get(pid)
        if focus:
            init_box, chain, args_chain = chain_for(t, focus)
            gbox = next(g["bbox"] for g in gt if g["fdi"] == focus)
            item["focus"] = {
                "fdi": focus,
                "diagnosis": next(g["diagnosis"] for g in gt if g["fdi"] == focus),
                "gt_box": gbox,
                "returned_box": init_box,
                "nudges": chain,
                "nudge_args": args_chain,
                "iou_before": round(iou(init_box, gbox), 3),
                "iou_after": round(iou(chain[-1], gbox), 3) if chain else None,
                "offset_before": round(offset_frac(init_box, gbox), 3),
                "offset_after": round(offset_frac(chain[-1], gbox), 3) if chain else None,
                "dx_frac": round((init_box[0] - gbox[0]) / gbox[2], 3),
                "dy_frac": round((init_box[1] - gbox[1]) / gbox[3], 3),
            }
            # "context" view: the episode (returned box, expert box, last nudge) with margin,
            # cut from the full-resolution 1800 px copy; region stored in native coordinates.
            allb = [init_box, gbox] + chain
            x0 = min(b[0] for b in allb)
            y0 = min(b[1] for b in allb)
            x1 = max(b[0] + b[2] for b in allb)
            y1 = max(b[1] + b[3] for b in allb)
            mw, mh = 0.45 * (x1 - x0), 0.30 * (y1 - y0)
            rx0, ry0 = max(0.0, x0 - mw), max(0.0, y0 - mh)
            rx1, ry1 = min(float(item["native_w"]), x1 + mw), min(float(item["native_h"]), y1 + mh)
            ctx = full.convert("RGB").crop(tuple(int(round(v * k)) for v in (rx0, ry0, rx1, ry1)))
            if ctx.width > CTX_W:
                ctx = ctx.resize((CTX_W, round(ctx.height * CTX_W / ctx.width)), Image.LANCZOS)
            ctx.save(pdir / "context.jpg", quality=88, optimize=True)
            item["focus"]["context_region"] = [round(rx0, 1), round(ry0, 1), round(rx1, 1), round(ry1, 1)]
            item["focus"]["context_size"] = [ctx.width, ctx.height]
            for nm in ("crop_before", "crop_after"):
                if (d / f"{nm}.png").exists():
                    item["focus"][nm + "_size"] = list(save_jpg(d / f"{nm}.png", pdir / f"{nm}.jpg", height=CROP_H))
        meta[pid] = item
        print(f"{pid:12s} scale={k} native={item['native_w']}x{item['native_h']} findings={item['n_findings']}", item.get("focus", {}).get("iou_before"), item.get("focus", {}).get("iou_after"))
    (OUT / "showcase.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
