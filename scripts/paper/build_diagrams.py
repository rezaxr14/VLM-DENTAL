"""Build the three redesigned paper diagrams with embedded real radiograph assets.

Generates (into paper/figures/svg/, then converts to PDF):
- fig_traces_comparison.svg : paired with-tools vs no-tools traces for DENTEX
  image 624, with a real radiograph panel showing the actual box trajectory
  parsed from the released trace observations.
- fig_trace_generation.svg / fig_trace_verification.svg : generate (stage 1), then verify -> decontaminate/release (stages 2-3)
  pipeline with icons, mini-visuals and data-driven bars.
- fig_architecture.svg      : system architecture with real image assets,
  numbered flow badges and orthogonal connectors.

All numbers are measured corpus statistics; all radiograph pixels come from
the local DENTEX copy (image 624 = train_318.png). No downloads (Rule 16).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagram_common import (  # noqa: E402
    ASSET_DIR,
    BOXES_624,
    C_BLUE,
    C_BLUE_D,
    C_BLUE_FILL,
    C_GREEN,
    C_GREEN_D,
    C_GREEN_FILL,
    C_MUT,
    C_ORANGE,
    C_ORANGE_D,
    C_ORANGE_L,
    C_RED,
    C_SAND_D,
    C_SAND_FILL,
    C_SUB,
    C_TXT,
    CROP_FINAL_REGION,
    INSET_REGION,
    MONO,
    PAPER_DIR,
    SVG_DIR,
    arrow,
    badge,
    b64_of,
    icon_check,
    icon_db,
    icon_doc,
    icon_gear,
    icon_img,
    icon_magnifier,
    icon_shield,
    load_assets,
    load_showcase,
    native_box,
    scaled_box,
    svg,
    PILL_GREEN,
    PILL_GREY,
    PILL_ORANGE,
)
from diagram_common import R as RC
from diagram_common import T as TC
from diagram_common import L as LC
from diagram_common import IMG as IMGC


def _x(v):
    return f"{v:.1f}".rstrip("0").rstrip(".")


# shorthand wrappers that format floats nicely
def T(x, y, s, **kw):
    return TC(_x(x), _x(y), s, **kw)


def R(x, y, w, h, **kw):
    return RC(_x(x), _x(y), _x(w), _x(h), **kw)


def IMG(x, y, w, h, b64):
    return IMGC(_x(x), _x(y), _x(w), _x(h), b64)


def L(x1, y1, x2, y2, **kw):
    return LC(_x(x1), _x(y1), _x(x2), _x(y2), **kw)


# ---------------------------------------------------------------------------
# Figure 1: traces comparison (with tools vs without), real trajectory panel
# ---------------------------------------------------------------------------


def build_traces_evidence(a):
    """Evidence trail on the real radiograph: returned boxes, 4.5x inset, final crop."""
    W0, H0 = a["native_w"], a["native_h"]
    gt, loc = BOXES_624["gt"], BOXES_624["loc"]
    n1, n2, n3 = BOXES_624["n1"], BOXES_624["n2"], BOXES_624["n3"]
    W = 640
    warn: list[str] = []
    t = _checked(warn)
    p = []
    ox, oy, ow, oh = 10, 24, 372, 174
    p.append(IMG(ox, oy, ow, oh, a["opg"]))
    p.append(R(ox, oy, ow, oh, rx=2, stroke="#666666", sw=1.1, fill="none"))
    for box, color, sw, dash in (
        (gt, "#2ca02c", 1.7, None), (loc, C_BLUE, 1.7, None), (n1, "#FFC287", 1.1, None),
        (n2, "#FF9A4D", 1.2, None), (n3, C_ORANGE, 1.7, None),
    ):
        bx, by, bw, bh = scaled_box(box, (0, 0, W0, H0), ox, oy, ow, oh)
        p.append(R(bx, by, bw, bh, rx=0, fill="none", stroke=color, sw=sw, dash=dash))
    ih, iw, cw = 226, 96, 67
    ix, iy, cx = 414, 24, 536
    ssx, ssy = ow / W0, oh / H0  # INSET_REGION is (x0, y0, x1, y1) in native pixels
    rx_ = ox + INSET_REGION[0] * ssx
    ry_ = oy + INSET_REGION[1] * ssy
    rw_ = (INSET_REGION[2] - INSET_REGION[0]) * ssx
    rh_ = (INSET_REGION[3] - INSET_REGION[1]) * ssy
    p.append(R(rx_, ry_, rw_, rh_, rx=0, fill="none", stroke="#9A9A9A", sw=0.8))
    p.append(L(rx_ + rw_, ry_, ix, iy, color="#C4C4C4", sw=0.9))
    p.append(L(rx_ + rw_, ry_ + rh_, ix, iy + ih, color="#C4C4C4", sw=0.9))
    p.append(t(ix, iy - 7, "4.5× zoom", size=9.0, fill="#444444", anchor="start", maxw=96))
    p.append(IMG(ix, iy, iw, ih, a["inset"]))
    p.append(R(ix, iy, iw, ih, rx=4, stroke="#666666", sw=1.2, fill="none"))
    for box, color, sw, dash in (
        (gt, "#2ca02c", 1.8, None), (loc, C_BLUE, 1.8, None), (n1, "#FFC287", 1.1, None),
        (n2, "#FF9A4D", 1.3, None), (n3, C_ORANGE, 2.0, None),
    ):
        bx, by, bw, bh = scaled_box(box, INSET_REGION, ix, iy, iw, ih)
        p.append(R(bx, by, bw, bh, rx=0, fill="none", stroke=color, sw=sw, dash=dash))
    wx = ix + 0.538 * iw
    p.append(arrow([(wx, iy + 0.941 * ih), (wx, iy + 0.163 * ih)], color=C_ORANGE, sw=1.5))
    p.append(t(cx, iy - 7, "final crop · turn 10", size=9.0, fill="#444444", anchor="start", maxw=94))
    p.append(IMG(cx, iy, cw, ih, a["crop_final"]))
    p.append(R(cx, iy, cw, ih, rx=4, stroke="#666666", sw=1.2, fill="none"))
    ly = oy + oh + 22
    p.append(L(10, ly - 3, 34, ly - 3, color=C_BLUE, sw=2.6))
    p.append(t(40, ly, "locate_tooth(34) box", size=9.2, anchor="start", maxw=150))
    p.append(L(206, ly - 3, 214, ly - 3, color="#FFC287", sw=2.6))
    p.append(L(214, ly - 3, 222, ly - 3, color="#FF9A4D", sw=2.6))
    p.append(L(222, ly - 3, 230, ly - 3, color=C_ORANGE, sw=2.6))
    p.append(t(236, ly, "nudges 1-3, light to dark", size=9.2, anchor="start", maxw=146))
    p.append(L(10, ly + 15, 34, ly + 15, color="#2ca02c", sw=2.6))
    p.append(t(40, ly + 18, "ground-truth box", size=9.2, anchor="start", maxw=150))
    p.append(t(236, ly + 18, "coords as returned:", size=9.2, fill=C_SUB, anchor="start", maxw=146))
    p.append(t(10, ly + 38, f"{_fmt_box(loc, False)} - {_fmt_box(n1, False)}", size=8.6, fill="#555555", anchor="start", font=MONO, maxw=372))
    p.append(t(10, ly + 51, f"- {_fmt_box(n2, False)} - {_fmt_box(n3, False)}", size=8.6, fill="#555555", anchor="start", font=MONO, maxw=372))
    if warn:
        print("  traces_evidence text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, max(iy + ih, ly + 51) + 12, "".join(p))


def build_traces_comparison(a):
    """No-tools (left) vs with-tools (right) trace for the same image + pairing band."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    W = 640
    warn: list[str] = []
    t = _checked(warn)
    p = []
    by0, bh = 10, 468
    # ---------------- left: no-tools trace ----------------
    bx0, bw = 10, 254
    bcx = bx0 + bw / 2
    p.append(R(bx0, by0, bw, bh, fill="#FBFBFB", stroke="#BBBBBB", sw=1.2))
    p.append(t(bcx, by0 + 22, "No-tools trace (baseline)", size=11.5, weight="bold", fill="#444444", maxw=bw - 16))
    p.append(t(bcx, by0 + 36, "1 turn · 0 tool calls · hindsight description", size=8.8, fill="#666666", maxw=bw - 16))
    p.append(R(bx0 + 12, by0 + 46, bw - 24, 40, fill="#E8EEF7", stroke=C_BLUE, sw=1.1))
    p.append(t(bcx, by0 + 62, "user: panoramic X-ray, tool-free prompt", size=9.0, weight="bold", fill=C_BLUE_D, maxw=bw - 36))
    p.append(t(bcx, by0 + 77, "“identify any abnormal teeth and diagnose”", size=8.6, fill="#444444", maxw=bw - 36))
    p.append(arrow([(bcx, by0 + 86), (bcx, by0 + 98)], color="#888888", sw=1.3))
    tb_y, tb_h = by0 + 100, 218
    p.append(R(bx0 + 12, tb_y, bw - 24, tb_h, fill="#FFFFFF", stroke="#CCCCCC", sw=1.1))
    p.append(t(bcx, tb_y + 19, "thought (single turn)", size=9.8, weight="bold", maxw=bw - 36))
    paras = [
        "“On systematic review of this panoramic radiograph, I begin by orienting to the patient’s right and left sides as marked. The image shows a mixed dentition with multiple developing permanent teeth ...",
        "Focusing on the lower-left quadrant (Q3), I examine the posterior region ... I can see a radiolucent area involving the occlusal/cervical portion of the tooth, with loss of normal enamel contour and a darker zone consistent with a carious lesion ...”",
    ]
    yy = tb_y + 38
    for para in paras:
        for ln in _wrap(para, 8.6, bw - 50):
            p.append(t(bx0 + 24, yy, ln, size=8.6, fill="#333333", anchor="start", maxw=bw - 46))
            yy += 12
        yy += 6
    yy += 6
    for ln in ("One-shot description from the full image:", "no crop, no second look, nothing checkable", "against the pixels it actually saw."):
        p.append(t(bx0 + 24, yy, ln, size=8.4, fill="#666666", anchor="start", style="italic", maxw=bw - 46))
        yy += 12
    fy = tb_y + tb_h + 12
    p.append(arrow([(bcx, tb_y + tb_h), (bcx, fy)], color="#888888", sw=1.3))
    p.append(R(bx0 + 12, fy + 2, bw - 24, 42, fill="#F2F8F2", stroke=C_GREEN, sw=1.3))
    p.append(t(bcx, fy + 19, "final answer", size=9.8, weight="bold", fill=C_GREEN_D, maxw=bw - 36))
    p.append(t(bcx, fy + 35, "Q3T4 · Caries · confidence 0.82", size=9.0, fill="#333333", maxw=bw - 36))
    ty_ = fy + 54
    th_ = by0 + bh - ty_ - 12
    p.append(R(bx0 + 12, ty_, bw - 24, th_, fill="#FBF6F0", stroke="#DD8452", sw=1.1))
    p.append(t(bcx, ty_ + 17, "What this arm can teach", size=9.6, weight="bold", fill="#8A5A2E", maxw=bw - 36))
    p.append(t(bx0 + 24, ty_ + 33, "- output format + clinical vocabulary", size=8.8, fill="#444444", anchor="start", maxw=bw - 46))
    p.append(t(bx0 + 24, ty_ + 46, "- quadrant-survey habit", size=8.8, fill="#444444", anchor="start", maxw=bw - 46))
    p.append(t(bcx, ty_ + 64, "No evidence trail to verify or correct.", size=8.8, weight="bold", fill=C_RED, maxw=bw - 36))

    # ---------------- right: with-tools trace ----------------
    cx0, cw0 = 274, 356
    ccx = cx0 + cw0 / 2
    p.append(R(cx0, by0, cw0, bh, fill="#FCFCFD", stroke=C_BLUE, sw=1.4))
    p.append(t(ccx, by0 + 22, "With-tools trace (main policy)", size=11.5, weight="bold", fill=C_BLUE_D, maxw=cw0 - 16))
    p.append(t(ccx, by0 + 36, "23 turns · 22 real tool calls · 3 corrective nudges", size=8.8, fill="#666666", maxw=cw0 - 16))
    rows = [
        ("T0", "window_level", "survey the full field at high contrast", 0),
        ("T1", "fdi_label", "orient to FDI notation first", 0),
        ("T2", "locate_tooth(34)", f"box {_fmt_box(BOXES_624['loc'])} (GT-grounded detector)", 0),
        ("T3", "zoom_crop", "“shows only bone and root area ... too low”", 0),
        ("T4", "nudge_crop #1", f"dy -0.3 · scale 0.6 -> {_fmt_box(BOXES_624['n1'])}", 1),
        ("T6", "nudge_crop #2", f"dy -0.5 · scale 0.7 -> {_fmt_box(BOXES_624['n2'])}", 1),
        ("T8", "nudge_crop #3", f"dy -0.6 · scale 0.8 -> {_fmt_box(BOXES_624['n3'])}", 1),
        ("T9-10", "zoom + denoise", "“Excellent! Now I can clearly see tooth 34”", 0),
        ("T14+", "compare + enhance", "contralateral check, contrast pass, re-verify", 0),
    ]
    ry = by0 + 46
    for turn, name, sub, hot in rows:
        fill = C_ORANGE_L if hot else "#FFFFFF"
        stroke = "#F0B27A" if hot else "#DDDDDD"
        p.append(R(cx0 + 12, ry, cw0 - 24, 31, rx=5, fill=fill, stroke=stroke, sw=1.0))
        p.append(R(cx0 + 12, ry, 4, 31, rx=2, fill=C_ORANGE if hot else C_BLUE, stroke="none", sw=0))
        tcol = "#B45A04" if hot else C_BLUE_D
        p.append(t(cx0 + 24, ry + 19, turn, size=8.8, weight="bold", fill=tcol, anchor="start", maxw=36))
        p.append(t(cx0 + 64, ry + 13, name, size=9.4, weight="bold", fill=C_TXT, anchor="start", maxw=cw0 - 88))
        p.append(t(cx0 + 64, ry + 25, sub, size=8.4, fill="#555555", anchor="start", maxw=cw0 - 88))
        ry += 35
    fa = ry + 2
    p.append(R(cx0 + 12, fa, cw0 - 24, 44, rx=6, fill="#F2F8F2", stroke=C_GREEN, sw=1.4))
    p.append(t(ccx, fa + 19, "final answer (turn 22) · Q3T4 · Caries · conf 0.82", size=9.4, weight="bold", fill=C_GREEN_D, maxw=cw0 - 36))
    p.append(t(ccx, fa + 35, "matches ground truth - passed strict verification", size=8.8, fill="#555555", maxw=cw0 - 36))
    sm = fa + 52
    p.append(R(cx0 + 12, sm, cw0 - 24, by0 + bh - sm - 12, rx=6, fill="#F7F9FC", stroke="#C9D4E4", sw=1.0))
    p.append(t(ccx, sm + 20, "22 real tool calls · 3 corrective · every observation", size=8.8, fill=C_BLUE_D, maxw=cw0 - 36))
    p.append(t(ccx, sm + 33, "executed against the actual radiograph", size=8.8, fill=C_BLUE_D, maxw=cw0 - 36))

    # ---------------- bottom band ----------------
    dy = by0 + bh + 10
    p.append(R(10, dy, 620, 98, fill="#F7F8FA", stroke="#DDDDDD", sw=1.1))
    p.append(t(22, dy + 22, "What the exact 1:1 pairing isolates", size=11.5, weight="bold", anchor="start", maxw=290))
    p.append(R(318, dy + 8, 300, 22, rx=11, fill="#EEF2FA", stroke=C_BLUE, sw=1.1))
    p.append(t(468, dy + 23, "5 cohorts × 2 arms · 1,847 + 1,847 · 3,694 verified traces", size=8.6, weight="bold", fill=C_BLUE_D, maxw=288))
    cols = [
        (22, dy + 48, ["same image · same prompt template · same ground truth -", "only the tool channel differs"]),
        (322, dy + 48, ["any SFT delta is attributable to demonstrated tool use,", "not to data volume or prompt wording"]),
        (22, dy + 76, ["with-tools traces carry 44,350 executed tool calls;", "the no-tools arm carries zero"]),
        (322, dy + 76, ["per-cohort counts match exactly (678/27/660/202/280", "on each side) - parity by construction"]),
    ]
    for bx, by_, lines in cols:
        for i, ln in enumerate(lines):
            p.append(t(bx, by_ + i * 12, ln, size=8.8, fill="#333333", anchor="start", maxw=292))
    if warn:
        print("  traces_comparison text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, dy + 108, "".join(p))


# ---------------------------------------------------------------------------
# Figure 2: multi-stage trace synthesis (generate -> verify -> release)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Figure 2: multi-stage trace synthesis (generate -> verify -> release)
# ---------------------------------------------------------------------------


def _band(p, t, y, h, fill, stroke, pill_fill, pill_text, pill_w, right_text, right_col, pill_right=False):
    """Stage band with a header pill (left, or right when connectors must pass on the left)."""
    p.append(R(10, y, 620, h, fill=fill, stroke=stroke, sw=1.2))
    px = 630 - pill_w if pill_right else 10
    p.append(R(px, y, pill_w, 22, rx=6, fill=pill_fill, stroke="none", sw=0))
    p.append(t(px + pill_w / 2, y + 15, pill_text, size=8.4, weight="bold", fill="#FFFFFF", maxw=pill_w - 12))
    if pill_right:
        p.append(t(22, y + 15, right_text, size=8.4, fill=right_col, anchor="start", maxw=620 - pill_w - 30))
    else:
        p.append(t(618, y + 15, right_text, size=8.4, fill=right_col, anchor="end", maxw=620 - pill_w - 30))


def _card_title(p, t, x, y, w, icon, text, size=10.2):
    p.append(icon)
    p.append(t(x + 38, y + 24, text, size=size, weight="bold", anchor="start", maxw=w - 48))


def build_trace_generation(a):
    """Stage 1: ground-truth-directed generation loop -> unverified pool (640 units)."""
    W = 640
    warn: list[str] = []
    t = _checked(warn)
    p = []
    G = "#999999"
    ya = 10
    ca = ya + 34
    cha = 222
    _band(p, t, ya, 34 + cha + 12 + 112, C_BLUE_FILL, "#C7D3E8", C_BLUE, "STAGE 1 · GENERATION", 176, "provider family A: generators", C_BLUE_D)

    ix, iw = 20, 142
    p.append(R(ix, ca, iw, cha))
    _card_title(p, t, ix, ca, iw, icon_db(ix + 12, ca + 10), "Expert labels")
    fl_ = _showcase()["flagship"]
    fw_, fh_ = _fit(*fl_["meta"]["image_size"], 122, 62)
    p.append(IMG(ix + 10, ca + 36 + (62 - fh_) / 2, fw_, fh_, fl_["image"]))
    _gt_overlay(p, fl_["meta"], ix + 10, ca + 36 + (62 - fh_) / 2, fw_, fh_, sw=0.8)
    p.append(R(ix + 10, ca + 36 + (62 - fh_) / 2, fw_, fh_, rx=2, stroke="#666666", sw=1.0, fill="none"))
    p.append(t(ix + 10, ca + 114, "DENTEX · Tufts loaders", size=8.6, fill=C_SUB, anchor="start", maxw=iw - 16))
    p.append(t(ix + 10, ca + 127, "FDI conversion at ingest", size=8.6, fill=C_SUB, anchor="start", maxw=iw - 16))
    p.append(R(ix + 10, ca + 142, 122, 52, fill="#F0F7F0", stroke=C_GREEN, sw=1.1))
    p.append(t(ix + 71, ca + 161, "ground truth", size=8.6, weight="bold", fill=C_GREEN_D, maxw=112))
    p.append(t(ix + 71, ca + 173, "per image", size=8.6, weight="bold", fill=C_GREEN_D, maxw=112))
    p.append(t(ix + 71, ca + 186, f"FDI {fl_['meta']['focus']['fdi']} · {fl_['meta']['focus']['diagnosis']} · bbox", size=8.2, fill=C_GREEN_D, maxw=112))
    p.append(arrow([(ix + iw, ca + 110), (ix + iw + 16, ca + 110)], color=G, sw=1.5))

    lx, lw = ix + iw + 18, 440
    p.append(R(lx, ca, lw, cha))
    p.append(t(lx + 12, ca + 22, "Ground-truth-directed LangGraph loop", size=10.2, weight="bold", anchor="start", maxw=lw - 24))
    py = ca + 32
    for off, wd, label in ((12, 58, "thought"), (82, 68, "tool call"), (162, 82, "observation")):
        p.append(R(lx + off, py, wd, 22, rx=11, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
        p.append(t(lx + off + wd / 2, py + 15, label, size=8.6, fill=C_BLUE_D, maxw=wd - 8))
    p.append(arrow([(lx + 71, py + 11), (lx + 81, py + 11)], color=G, sw=1.1))
    p.append(arrow([(lx + 151, py + 11), (lx + 161, py + 11)], color=G, sw=1.1))
    p.append(arrow([(lx + 203, py + 22), (lx + 203, py + 42), (lx + 41, py + 42), (lx + 41, py + 23)], color="#B9C6DC", sw=1.2))
    p.append(t(lx + 122, py + 37, "repeat every turn", size=8.4, fill="#4F6690", maxw=140))
    # zoom monitor (right of the pills)
    mx, my, mw, mh = lx + 252, ca + 30, 180, 94
    cwid, chid = fl_["meta"]["focus"]["crop_before_size"]
    ctw = 64 * cwid / chid
    p.append(R(mx, my, mw, mh, rx=4, fill="#FAFBFD", stroke="#DDDDDD", sw=1.0))
    p.append(IMG(mx + 10, my + 8, ctw, 64, fl_["crop_before"]))
    p.append(R(mx + 10, my + 8, ctw, 64, rx=1, stroke="#888888", sw=0.8, fill="none"))
    p.append(t(mx + mw / 2, my + 32, "nudge", size=8.2, fill=C_ORANGE_D, maxw=44))
    p.append(arrow([(mx + 10 + ctw + 8, my + 44), (mx + mw - 10 - ctw - 8, my + 44)], color=C_ORANGE, sw=1.6))
    p.append(IMG(mx + mw - 10 - ctw, my + 8, ctw, 64, fl_["crop_after"]))
    p.append(R(mx + mw - 10 - ctw, my + 8, ctw, 64, rx=1, stroke="#888888", sw=0.8, fill="none"))
    p.append(t(mx + mw / 2, my + 86, "before / after (real episode)", size=8.2, fill="#666666", maxw=mw - 8))
    # generation pool chips
    gy = ca + 140
    p.append(t(lx + 12, gy, "generation pool:", size=8.6, fill=C_SUB, anchor="start", maxw=140))
    cx_ = lx + 12
    for cwid, label in ((108, "Gemini 3.5 Flash Lite"), (66, "MiniMax M3"), (66, "NVIDIA NIM"), (66, "local vLLM")):
        p.append(R(cx_, gy + 6, cwid, 17, rx=8.5, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
        p.append(t(cx_ + cwid / 2, gy + 18, label, size=8.2, fill=C_BLUE_D, maxw=cwid - 8))
        cx_ += cwid + 8
    qy = ca + 172
    p.append(R(lx + 12, qy, lw - 24, 40, rx=6, fill=C_ORANGE_L, stroke="#F0B27A", sw=1.1))
    p.append(t(lx + 22, qy + 16, "tiered display perturbation, independent of the true box:", size=8.6, fill=C_ORANGE_D, anchor="start", maxw=lw - 44))
    p.append(t(lx + 22, qy + 30, "45% small · 25% big · 30% none - keeps accept-vs-nudge demonstrable", size=8.6, fill=C_ORANGE_D, anchor="start", maxw=lw - 44))

    # unverified pool (row 2)
    uy = ca + cha + 26
    uh = 96
    p.append(arrow([(lx + lw / 2, ca + cha), (lx + lw / 2, uy - 1)], color=G, sw=1.5))
    p.append(R(20, uy, 600, uh))
    _card_title(p, t, 20, uy, 600, icon_doc(32, uy + 10), "Unverified trace pool")
    p.append(R(32, uy + 34, 300, 52, rx=4, fill="#F7F8FA", stroke="#E3E3E3", sw=1.0))
    f_ = fl_["meta"]["focus"]
    a0_ = f_["nudge_args"][0]
    snip = ['thought "... the box sits about one', '         arch too high"', f"call nudge_crop(dx={a0_['dx_frac']:+.2f}, dy={a0_['dy_frac']:+.2f})", f"obs  bbox -> {_fmt_box(f_['nudges'][0])}"]
    yy = uy + 46
    for ln in snip:
        p.append(t(40, yy, ln, size=7.8, fill="#333333", anchor="start", font=MONO, maxw=284))
        yy += 12
    p.append(R(346, uy + 14, 262, 22, rx=11, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(t(477, uy + 29, "44,350 executed tool calls", size=8.8, weight="bold", fill=C_BLUE_D, maxw=250))
    p.append(R(346, uy + 42, 262, 22, rx=11, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(t(477, uy + 57, "2,292 corrective nudges", size=8.8, weight="bold", fill=C_BLUE_D, maxw=250))
    p.append(t(477, uy + 82, "no training value until verified", size=8.6, fill=C_SUB, maxw=250))
    if warn:
        print("  trace_generation text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, ya + 34 + cha + 12 + 112 + 10, "".join(p))


def build_trace_verification(a):
    """Stages 2-3: strict verification + repair loop, dual-gate decontamination, release."""
    W = 640
    warn: list[str] = []
    t = _checked(warn)
    p = []
    G = "#999999"
    # ===================== stage 2 =====================
    yb = 10
    cb = yb + 46
    chb = 200
    hb = 46 + chb + 12
    _band(p, t, yb, hb, C_GREEN_FILL, "#BFDCC6", PILL_GREEN, "STAGE 2 · VERIFICATION", 176, "provider family B - never the generator pool", C_GREEN_D)

    vx, vw = 20, 176
    p.append(R(vx, cb, vw, chb))
    _card_title(p, t, vx, cb, vw, icon_shield(vx + 14, cb + 12), "Strict verifier")
    p.append(t(vx + 12, cb + 44, "staged cross-family check per trace:", size=8.2, fill=C_SUB, anchor="start", maxw=vw - 20))
    for iy_, l1, l2 in (
        (cb + 66, "diagnosis set-matches the full", "multi-finding ground truth"),
        (cb + 96, "evidence trail consistent - no", "claims on healthy tissue"),
        (cb + 126, "no directive leakage in CoT", None),
        (cb + 144, "format + FDI notation valid", None),
    ):
        p.append(icon_check(vx + 12, iy_ - 5.5))
        p.append(t(vx + 26, iy_, l1, size=8.6, anchor="start", maxw=vw - 34))
        if l2:
            p.append(t(vx + 26, iy_ + 12, l2, size=8.6, anchor="start", maxw=vw - 34))
    p.append(t(vx + 12, cb + 176, "strict rate pacing via the", size=8.2, fill=C_SUB, anchor="start", maxw=vw - 20))
    p.append(t(vx + 12, cb + 188, "provider pool", size=8.2, fill=C_SUB, anchor="start", maxw=vw - 20))

    rx_, rw_ = 248, 150
    gmid = (vx + vw + rx_) / 2
    p.append(arrow([(vx + vw, cb + 74), (rx_ - 2, cb + 74)], color=C_RED, sw=1.6))
    p.append(t(gmid, cb + 54, "REJECT", size=8.2, weight="bold", fill=C_RED, maxw=46))
    p.append(t(gmid, cb + 65, "· 92", size=8.2, weight="bold", fill=C_RED, maxw=46))
    p.append(R(rx_, cb, rw_, chb))
    _card_title(p, t, rx_, cb, rw_, icon_gear(rx_ + 14, cb + 12), "Repair loop")
    p.append(t(rx_ + 12, cb + 44, "editor rewrites the trace,", size=8.2, fill=C_SUB, anchor="start", maxw=rw_ - 18))
    p.append(t(rx_ + 12, cb + 56, "keeping the valid CoT prefix:", size=8.2, fill=C_SUB, anchor="start", maxw=rw_ - 18))
    for ry_, txt in ((cb + 82, "fix tool-arg / FDI slips"), (cb + 102, "drop phantom claims"), (cb + 122, "excise directive leakage")):
        p.append(R(rx_ + 12, ry_ - 5, 5, 5, rx=1, fill=C_ORANGE, stroke="none", sw=0))
        p.append(t(rx_ + 24, ry_, txt, size=8.6, anchor="start", maxw=rw_ - 32))
    p.append(t(rx_ + 12, cb + 168, "re-verified after every", size=8.2, fill=C_SUB, anchor="start", maxw=rw_ - 18))
    p.append(t(rx_ + 12, cb + 180, "repair", size=8.2, fill=C_SUB, anchor="start", maxw=rw_ - 18))
    p.append(arrow([(rx_, cb + 138), (vx + vw + 2, cb + 138)], color="#7FA98B", sw=1.6))
    p.append(t(gmid, cb + 156, "re-verify", size=8.2, weight="bold", fill=C_GREEN_D, maxw=48))
    p.append(t(gmid, cb + 167, "until pass", size=8.2, weight="bold", fill=C_GREEN_D, maxw=48))

    ax, aw = rx_ + rw_ + 40, 182
    p.append(arrow([(rx_ + rw_, cb + 96), (ax - 2, cb + 96)], color="#AAAAAA", sw=1.4))
    p.append(t((rx_ + rw_ + ax) / 2, cb + 88, "logged", size=8.0, fill=C_SUB, maxw=38))
    p.append(R(ax, cb, aw, chb))
    _card_title(p, t, ax, cb, aw, icon_magnifier(ax + 18, cb + 18, r=4.5), "First-pass audit")
    p.append(t(ax + 12, cb + 46, "92 rejections on 3,694 traces", size=8.8, weight="bold", anchor="start", maxw=aw - 20))
    p.append(t(ax + 12, cb + 58, "(97.5% pass)", size=8.8, weight="bold", anchor="start", maxw=aw - 20))
    for by_, label, bw_, bc, cnt, cc in (
        (cb + 82, "hallucinated findings on", 110, C_RED, "66", C_RED),
        (cb + 118, "contradicted the ground truth", 26, "#DD8452", "15", C_RED),
        (cb + 150, "drift left uncorrected", 19, C_ORANGE, "11", C_ORANGE_D),
    ):
        p.append(t(ax + 12, by_, label, size=8.2, fill="#555555", anchor="start", maxw=aw - 20))
        if by_ == cb + 82:
            p.append(t(ax + 12, by_ + 11, "healthy tissue", size=8.2, fill="#555555", anchor="start", maxw=aw - 20))
            by_ += 11
        p.append(R(ax + 12, by_ + 6, bw_, 9, rx=2, fill=bc, stroke="none", sw=0))
        p.append(t(ax + 12 + bw_ + 6, by_ + 14, cnt, size=8.4, weight="bold", fill=cc, anchor="start"))
    p.append(t(ax + 12, cb + 182, "category 3 exists because", size=8.0, fill=C_ORANGE_D, anchor="start", maxw=aw - 20))
    p.append(t(ax + 12, cb + 192, "perturbation pressure", size=8.0, fill=C_ORANGE_D, anchor="start", maxw=aw - 20))

    # ===================== stage 3 =====================
    yc = yb + hb + 26
    cc_ = yc + 34
    chc = 176
    hc = 34 + chc + 12
    _band(p, t, yc, hc, C_SAND_FILL, "#E5D9B8", PILL_ORANGE, "STAGE 3 · DECONTAMINATION + RELEASE", 192, "only clean traces enter the corpus", C_SAND_D, pill_right=True)
    p.append(arrow([(190, cb + chb), (190, cc_ - 1)], color=C_GREEN, sw=1.6))
    p.append(t(200, yc + 15, "PASS · 97.5% first-pass", size=8.2, fill=C_GREEN_D, anchor="start", maxw=96))
    gx, gw = 20, 190
    cx_, cw_ = 236, 190
    px_ = cx_ + 96
    p.append(arrow([(px_, cb + chb), (px_, cc_ - 1)], color="#7FA98B", sw=1.3))
    p.append(t(px_ + 8, yc + 15, "PASS", size=8.2, fill=C_GREEN_D, anchor="start", maxw=40))

    p.append(R(gx, cc_, gw, chc))
    _card_title(p, t, gx, cc_, gw, icon_shield(gx + 14, cc_ + 12, color=C_SAND_D), "Dual-gate decontamination", size=8.8)
    for i, ln in enumerate(("Gate 1 - regex sweep for directive", "phrasings leaked into the CoT")):
        p.append(t(gx + 12, cc_ + 46 + i * 12, ln, size=8.4, anchor="start", maxw=gw - 18))
    for i, ln in enumerate(("Gate 2 - flagged traces re-verified", "by family B before any action")):
        p.append(t(gx + 12, cc_ + 78 + i * 12, ln, size=8.4, anchor="start", maxw=gw - 18))
    p.append(R(gx + 12, cc_ + 104, 150, 22, rx=11, fill="#FDEEEE", stroke=C_RED, sw=1.0))
    p.append(t(gx + 87, cc_ + 119, "105 traces excised", size=8.6, weight="bold", fill="#8A2E33", maxw=140))
    p.append(t(gx + 12, cc_ + 148, "0 false positives on clinical", size=8.2, fill=C_GREEN_D, anchor="start", maxw=gw - 18))
    p.append(t(gx + 12, cc_ + 160, '"hint of"', size=8.2, fill=C_GREEN_D, anchor="start", maxw=gw - 18))
    p.append(arrow([(gx + gw, cc_ + 90), (cx_ - 2, cc_ + 90)], color=G, sw=1.5))
    p.append(t((gx + gw + cx_) / 2, cc_ + 82, "clean", size=8.0, fill=C_SUB, maxw=24))

    p.append(R(cx_, cc_, cw_, chc))
    _card_title(p, t, cx_, cc_, cw_, icon_db(cx_ + 12, cc_ + 10, color=C_GREEN_D), "Released corpus")
    p.append(R(cx_ + 12, cc_ + 40, 78, 14, rx=2, fill=C_BLUE, stroke="none", sw=0))
    p.append(t(cx_ + 51, cc_ + 68, "1,847 with-tools", size=8.2, fill=C_BLUE_D, maxw=82))
    p.append(R(cx_ + 100, cc_ + 40, 78, 14, rx=2, fill=C_GREEN, stroke="none", sw=0))
    p.append(t(cx_ + 139, cc_ + 68, "1,847 no-tools", size=8.2, fill=C_GREEN_D, maxw=82))
    p.append(t(cx_ + 12, cc_ + 90, "exact per-cohort 1:1 parity", size=8.2, fill=C_SUB, anchor="start", maxw=cw_ - 18))
    p.append(t(cx_ + 12, cc_ + 102, "(5 cohorts)", size=8.2, fill=C_SUB, anchor="start", maxw=cw_ - 18))
    p.append(t(cx_ + 12, cc_ + 124, "3,694 verified traces", size=8.6, anchor="start", maxw=cw_ - 18))
    p.append(t(cx_ + 12, cc_ + 137, "5,144 findings · 687 healthy images", size=8.4, anchor="start", maxw=cw_ - 18))
    p.append(t(cx_ + 12, cc_ + 150, "44,350 tool calls", size=8.6, anchor="start", maxw=cw_ - 18))
    p.append(arrow([(cx_ + cw_, cc_ + 90), (cx_ + cw_ + 24, cc_ + 90)], color=G, sw=1.5))
    p.append(t(cx_ + cw_ + 12, cc_ + 82, "clean", size=8.0, fill=C_SUB, maxw=24))

    rlx, rlw = cx_ + cw_ + 26, 168
    p.append(R(rlx, cc_, rlw, chc))
    _card_title(p, t, rlx, cc_, rlw, icon_doc(rlx + 12, cc_ + 10), "Release")
    p.append(t(rlx + 12, cc_ + 46, "public on the Hugging Face Hub", size=8.4, anchor="start", maxw=rlw - 18))
    for i, ln in enumerate(("per-call audit: perturbation", "tier, shown vs true bbox,", "offsets")):
        p.append(t(rlx + 12, cc_ + 64 + i * 12, ln, size=8.4, anchor="start", maxw=rlw - 18))
    p.append(R(rlx + 12, cc_ + 106, rlw - 24, 22, rx=11, fill="#EEF2FA", stroke=C_BLUE, sw=1.0))
    p.append(t(rlx + rlw / 2, cc_ + 121, "every number regenerable", size=8.4, weight="bold", fill=C_BLUE_D, maxw=rlw - 32))
    p.append(t(rlx + 12, cc_ + 148, "scripts/paper rebuilds all", size=8.2, fill=C_SUB, anchor="start", maxw=rlw - 18))
    p.append(t(rlx + 12, cc_ + 160, "figures and tables in this paper", size=8.2, fill=C_SUB, anchor="start", maxw=rlw - 18))
    if warn:
        print("  trace_verification text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, yc + hc + 10, "".join(p))


# ---------------------------------------------------------------------------
# Figure 3: architecture
# ---------------------------------------------------------------------------


def build_architecture(a):
    """Agent-loop architecture: input, policy <-> tools, answer, evaluation.

    First half of the old combined architecture figure (the offline
    data/trace/training stack is now `build_architecture_training`). 640-unit
    canvas so labels print at ~6.3-7.5 pt; badges 4/5/6 keep the numbered
    path of the training figure.
    """
    W = 640
    warn: list[str] = []
    t = _checked(warn)
    p = []
    G = "#777777"
    # ---- row 1: native input + dispatcher notes ----
    sc_ = _showcase()["flagship"]
    p.append(t(10, 22, "input · native-resolution OPG", size=9.2, weight="bold", anchor="start", maxw=300))
    p.append(IMG(10, 34, 300, 141, _cover_b64(sc_["image"], 300 / 141)))
    p.append(R(10, 34, 300, 141, rx=3, stroke="#666666", sw=1.2, fill="none"))
    p.append(t(10, 190, "every tool executes at full resolution", size=8.8, fill=C_SUB, anchor="start", maxw=300))
    nx, nw = 330, 300
    p.append(R(nx, 34, nw, 141, rx=8, fill=C_BLUE_FILL, stroke="#C7D3E8", sw=1.2))
    for i, (ln, col) in enumerate((
        ("shared dispatcher - every image-consuming tool", C_BLUE_D),
        ("runs on base_image; the crop chain never compounds", C_BLUE_D),
        (None, None),
        ("input geometry: TPU = canonical views + static", C_BLUE_D),
        ("slots (no recompile); GPU = native so far -", C_BLUE_D),
        ("decision pending; tools always run native", C_BLUE_D),
    )):
        if ln:
            p.append(t(nx + 14, 58 + i * 14, ln, size=8.8, fill=col, anchor="start", maxw=nw - 24))

    # ---- row 2: policy <-> tool node ----
    c0, ch = 222, 172
    px0, pw = 62, 176
    tx0, tw = 278, 352
    p.append(arrow([(150, 175), (150, c0 - 1)], color=G, sw=1.5))
    p.append(arrow([(294, 175), (294, c0 - 1)], color="#AAAAAA", sw=1.4))
    p.append(t(302, 204, "base_image", size=8.6, fill=C_SUB, anchor="start", maxw=70))
    # adapters arrive from the training stack
    ay = c0 + 70
    p.append(arrow([(10, ay), (px0 - 1, ay)], color="#7FA98B", sw=1.6))
    p.append(badge(4, 28, ay + 15))
    p.append(t(36, ay - 7, "adapters", size=8.4, fill=C_GREEN_D, maxw=46))
    p.append(R(px0, c0, pw, ch, rx=8, stroke=C_BLUE, sw=1.5))
    p.append(t(px0 + pw / 2, c0 + 24, "Policy", size=11.5, weight="bold", fill=C_BLUE_D))
    p.append(t(px0 + pw / 2, c0 + 46, "Qwen3.5-9B", size=10.2, weight="bold", maxw=pw - 14))
    p.append(t(px0 + pw / 2, c0 + 68, "reasoning node", size=8.8, fill=C_SUB, maxw=pw - 14))
    p.append(t(px0 + pw / 2, c0 + 81, "thought + JSON tool calls", size=8.8, fill=C_SUB, maxw=pw - 14))
    p.append(t(px0 + pw / 2, c0 + 106, "LoRA adapters:", size=8.8, fill=C_SUB, maxw=pw - 14))
    p.append(t(px0 + pw / 2, c0 + 119, "SFT · GRPO (pluggable)", size=8.8, fill=C_SUB, maxw=pw - 14))
    p.append(R(tx0, c0, tw, ch, rx=8, stroke=C_BLUE, sw=1.5))
    p.append(t(tx0 + tw / 2, c0 + 24, "Tool node", size=11.5, weight="bold", fill=C_BLUE_D))
    cwid, chid = sc_["meta"]["focus"]["crop_after_size"]
    lw_ = 120 * cwid / chid
    p.append(IMG(tx0 + 14, c0 + 36, lw_, 120, sc_["crop_after"]))
    p.append(R(tx0 + 14, c0 + 36, lw_, 120, rx=1, stroke="#888888", sw=0.9, fill="none"))
    p.append(t(tx0 + 14 + lw_ / 2, c0 + 168, "live crop", size=8.4, fill=C_SUB))
    tt = tx0 + 14 + lw_ + 14
    p.append(t(tt, c0 + 48, "8 registered tools", size=9.4, weight="bold", anchor="start", maxw=tw - 78))
    for i, line in enumerate(("zoom · window · denoise", "enhance · contrast", "locate · nudge", "bilateral compare · fdi")):
        p.append(t(tt, c0 + 66 + 13 * i, line, size=8.8, fill=C_SUB, anchor="start", maxw=tw - 78))
    p.append(t(tt, c0 + 130, "continuous parameters:", size=8.8, fill=C_SUB, anchor="start", maxw=tw - 78))
    p.append(t(tt, c0 + 143, "strength · center · width", size=8.8, fill=C_SUB, anchor="start", maxw=tw - 78))
    gx0, gx1 = px0 + pw, tx0
    p.append(arrow([(gx0, c0 + 60), (gx1, c0 + 60)], color=C_BLUE, sw=1.6))
    p.append(t((gx0 + gx1) / 2, c0 + 52, "call", size=8.6, fill=C_BLUE_D, maxw=36))
    p.append(arrow([(gx1, c0 + 112), (gx0, c0 + 112)], color="#8DA2C4", sw=1.6))
    p.append(t((gx0 + gx1) / 2, c0 + 104, "obs", size=8.6, fill="#4F6690", maxw=36))

    # ---- row 3: final answer ----
    a0 = c0 + ch + 34
    p.append(arrow([(150, c0 + ch), (150, a0 - 1)], color=G, sw=1.5))
    p.append(t(160, c0 + ch + 21, "on answer / budget", size=8.6, fill=C_SUB, anchor="start", maxw=140))
    p.append(R(10, a0, 620, 56, rx=8, fill="#F2F8F2", stroke=C_GREEN, sw=1.4))
    p.append(t(320, a0 + 20, "final answer - findings JSON", size=9.8, weight="bold", fill=C_GREEN_D, maxw=596))
    fa_ = next(e for e in sc_["meta"]["final_answer"] if e["quadrant"] * 10 + e["tooth_position"] == sc_["meta"]["focus"]["fdi"])
    p.append(t(320, a0 + 41, "[" + json.dumps(fa_, separators=(",", ":")) + ", ...]", size=8.4, fill=C_BLUE_D, font=MONO, maxw=596))

    # ---- row 4: evaluation ----
    e0 = a0 + 56 + 34
    eh = 132
    p.append(arrow([(320, a0 + 56), (320, e0 - 1)], color=G, sw=1.5))
    p.append(badge(5, 300, a0 + 56 + 17))
    p.append(t(336, a0 + 56 + 20, "per-image findings", size=8.6, fill=C_SUB, anchor="start", maxw=140))
    p.append(R(10, e0, 620, eh))
    p.append(icon_magnifier(32, e0 + 20, r=4.5))
    p.append(t(50, e0 + 24, "Evaluation", size=11.5, weight="bold", anchor="start"))
    p.append(t(24, e0 + 48, "set-matched precision / recall / F1", size=9.2, anchor="start", maxw=270))
    p.append(t(24, e0 + 61, "over multi-finding predictions", size=8.8, fill=C_SUB, anchor="start", maxw=270))
    p.append(t(24, e0 + 80, "95% bootstrap confidence intervals", size=9.2, anchor="start", maxw=270))
    p.append(t(24, e0 + 104, "heavy runs on GPU or TPU", size=8.8, fill=C_SUB, anchor="start", maxw=270))
    p.append(t(24, e0 + 117, "(target decision pending - see input geometry)", size=8.8, fill=C_SUB, anchor="start", maxw=270))
    p.append(R(316, e0 + 12, 302, eh - 24, rx=6, fill=C_BLUE_FILL, stroke="#C7D3E8", sw=1.0))
    p.append(t(328, e0 + 30, "probes (mechanism evidence):", size=9.0, weight="bold", fill=C_BLUE_D, anchor="start", maxw=280))
    for i, ln in enumerate(("- tool removal: with vs without", "- perturbation: shown-box vs true-box", "- nudge budget K sweep")):
        p.append(t(328, e0 + 47 + i * 13, ln, size=8.8, fill=C_BLUE_D, anchor="start", maxw=280))
    p.append(t(328, e0 + 94, "(results pending, reported as open cells)", size=8.6, fill=C_SUB, anchor="start", maxw=280))
    # dev-split signal back to training
    p.append(arrow([(560, e0 + eh), (560, e0 + eh + 26)], color=G, sw=1.4))
    p.append(badge(6, 538, e0 + eh + 13))
    p.append(t(526, e0 + eh + 17, "dev-split signal to training", size=8.6, fill=C_SUB, anchor="end", maxw=240))
    if warn:
        print("  architecture text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, e0 + eh + 34, "".join(p))


def build_architecture_training(a):
    """Offline stack: data layer -> trace synthesis -> SFT -> GRPO -> adapters."""
    W = 640
    warn: list[str] = []
    t = _checked(warn)
    p = []
    G = "#999999"
    # ---- row 1: data layer -> trace synthesis ----
    y1, h1 = 10, 262
    dx, dw = 10, 300
    p.append(R(dx, y1, dw, h1))
    p.append(icon_db(dx + 14, y1 + 12))
    p.append(t(dx + 38, y1 + 24, "Data layer", size=11.5, weight="bold", anchor="start"))
    tx = dx + 14
    p.append(t(tx, y1 + 48, "2 datasets in use: DENTEX · Tufts", size=9.2, anchor="start", maxw=dw - 24))
    p.append(t(tx, y1 + 62, "(8 more catalogued as candidates - pending)", size=8.8, fill=C_SUB, anchor="start", maxw=dw - 24))
    p.append(t(tx, y1 + 82, "loaders: DENTEX full · Tufts partial (1 open question)", size=8.8, fill=C_SUB, anchor="start", maxw=dw - 24))
    p.append(t(tx, y1 + 95, "FDI notation conversion at ingest", size=8.8, fill=C_SUB, anchor="start", maxw=dw - 24))
    dc_ = _showcase()["deep_caries"]
    tw_, th_ = _fit(*dc_["meta"]["image_size"], 130, 61)
    p.append(IMG(tx, y1 + 110, tw_, th_, dc_["image"]))
    _gt_overlay(p, dc_["meta"], tx, y1 + 110, tw_, th_, sw=0.9)
    p.append(R(tx, y1 + 110, tw_, th_, rx=2, stroke="#666666", sw=1.0, fill="none"))
    p.append(t(tx + 65, y1 + 185, "panoramic radiograph + per-tooth", size=8.4, fill=C_SUB, maxw=140))
    p.append(t(tx + 65, y1 + 196, "boxes + diagnoses", size=8.4, fill=C_SUB, maxw=140))
    p.append(R(dx + 154, y1 + 110, 132, 61, fill="#F0F7F0", stroke=C_GREEN, sw=1.1))
    p.append(t(dx + 220, y1 + 130, "multi-finding ground truth", size=8.4, weight="bold", fill=C_GREEN_D, maxw=124))
    p.append(t(dx + 220, y1 + 146, "1-7 findings per image", size=8.6, fill=C_GREEN_D, maxw=124))
    p.append(t(dx + 220, y1 + 160, "full set - never truncated", size=8.6, fill=C_GREEN_D, maxw=124))
    p.append(R(tx, y1 + 218, dw - 28, 24, rx=12, fill="#EEF2FA", stroke=C_BLUE, sw=1.0))
    p.append(t(dx + dw / 2, y1 + 234, "grounding model: YOLOv8m · 5-fold CV · mAP50 ~ 0.87", size=8.8, fill=C_BLUE_D, maxw=dw - 36))

    sx, sw_ = 330, 300
    p.append(arrow([(dx + dw, y1 + h1 / 2), (sx - 1, y1 + h1 / 2)], color=G, sw=1.5))
    p.append(badge(1, (dx + dw + sx) / 2, y1 + h1 / 2 - 14))
    p.append(R(sx, y1, sw_, h1, rx=8, stroke=C_BLUE, sw=1.4))
    p.append(icon_doc(sx + 14, y1 + 10))
    p.append(t(sx + 38, y1 + 22, "Trace synthesis (offline)", size=11.5, weight="bold", fill=C_BLUE_D, anchor="start"))
    flow = [
        (C_BLUE_FILL, "#C7D3E8", C_BLUE_D, "family A generates with-tools traces", "+ paired no-tools traces"),
        (C_ORANGE_L, "#F0B27A", C_ORANGE_D, "tiered perturbation of displayed boxes", "(nudge behavior stays demonstrable)"),
        (C_GREEN_FILL, "#BFDCC6", C_GREEN_D, "family B verifies + repairs every trace", "staged · cross-provider · multi-finding"),
        (C_SAND_FILL, "#E5D9B8", C_SAND_D, "dual-gate decontamination", "regex sweep + re-verification"),
    ]
    fy = y1 + 34
    for fill, stroke, col, l1, l2 in flow:
        p.append(R(sx + 12, fy, sw_ - 24, 32, rx=6, fill=fill, stroke=stroke, sw=1.0))
        p.append(t(sx + sw_ / 2, fy + 14, l1, size=9.0, fill=col, maxw=sw_ - 34))
        p.append(t(sx + sw_ / 2, fy + 26, l2, size=8.4, fill=col, maxw=sw_ - 34))
        p.append(arrow([(sx + sw_ / 2, fy + 32), (sx + sw_ / 2, fy + 42)], color=G, sw=1.2))
        fy += 42
    p.append(R(sx + 12, fy, sw_ - 24, 36, rx=6, fill="#EEF2FA", stroke=C_BLUE, sw=1.2))
    p.append(t(sx + sw_ / 2, fy + 15, "corpus: 1,847 + 1,847 · 3,694 traces", size=9.0, weight="bold", fill=C_BLUE_D, maxw=sw_ - 34))
    p.append(t(sx + sw_ / 2, fy + 28, "exact per-cohort parity", size=8.4, fill=C_BLUE_D, maxw=sw_ - 34))
    p.append(t(sx + 14, y1 + h1 - 8, "details in the trace-synthesis figure (all quotes, boxes and stats are real)", size=8.0, fill=C_SUB, anchor="start", maxw=sw_ - 24))

    # ---- row 2: training ----
    y2 = y1 + h1 + 36
    h2 = 214
    p.append(R(10, y2, 620, h2, rx=8, stroke=C_BLUE, sw=1.4))
    p.append(icon_gear(32, y2 + 20))
    p.append(t(50, y2 + 24, "Training", size=11.5, weight="bold", fill=C_BLUE_D, anchor="start"))
    # corpus -> SFT elbow
    sfx = 22
    p.append(arrow([(sx + sw_ / 2, y1 + h1), (sx + sw_ / 2, y1 + h1 + 14), (sfx + 90, y1 + h1 + 14), (sfx + 90, y2 + 36)], color="#7FA98B", sw=1.6))
    p.append(badge(2, sx + sw_ / 2 - 70, y1 + h1 + 14))
    cy = y2 + 38
    ch2 = 110
    sfw, gw, ow = 184, 248, 138
    p.append(R(sfx, cy, sfw, ch2, rx=6, fill=C_BLUE_FILL, stroke="#C7D3E8", sw=1.0))
    p.append(t(sfx + 12, cy + 20, "1 · SFT on verified traces", size=9.4, weight="bold", fill=C_BLUE_D, anchor="start", maxw=sfw - 20))
    p.append(t(sfx + 12, cy + 38, "behavior cloning of the tool loop", size=8.6, fill=C_SUB, anchor="start", maxw=sfw - 20))
    p.append(t(sfx + 12, cy + 51, "(1,847 with-tools traces)", size=8.6, fill=C_SUB, anchor="start", maxw=sfw - 20))
    gx = sfx + sfw + 26
    p.append(arrow([(sfx + sfw, cy + ch2 / 2), (gx - 1, cy + ch2 / 2)], color=G, sw=1.3))
    p.append(t(sfx + sfw + 13, cy + ch2 / 2 - 12, "then", size=8.4, fill=C_SUB, maxw=26))
    p.append(badge(3, sfx + sfw + 13, cy + ch2 / 2 + 14, r=7.5))
    p.append(R(gx, cy, gw, ch2, rx=6, fill=C_BLUE_FILL, stroke="#C7D3E8", sw=1.0))
    p.append(t(gx + 12, cy + 20, "2 · GRPO on fresh rollouts", size=9.4, weight="bold", fill=C_BLUE_D, anchor="start", maxw=gw - 20))
    p.append(t(gx + 12, cy + 38, "same loop, live tool execution", size=8.6, fill=C_SUB, anchor="start", maxw=gw - 20))
    p.append(t(gx + 12, cy + 51, "group-relative advantage", size=8.6, fill=C_SUB, anchor="start", maxw=gw - 20))
    px = gx + 12
    for cw, label in ((50, "accuracy"), (44, "format"), (50, "tool use"), (58, "efficiency")):
        p.append(R(px, cy + 60, cw, 17, rx=8.5, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
        p.append(t(px + cw / 2, cy + 72, label, size=8.0, fill=C_BLUE_D, maxw=cw - 4))
        px += cw + 6
    p.append(t(gx + 12, cy + 94, "efficiency term exempts locate_tooth + nudge_crop", size=8.2, fill=C_ORANGE_D, anchor="start", maxw=gw - 20))
    ox = gx + gw + 18
    p.append(arrow([(gx + gw, cy + ch2 / 2), (ox - 1, cy + ch2 / 2)], color=G, sw=1.3))
    p.append(R(ox, cy, ow, ch2, rx=6, fill="#F0F7F0", stroke=C_GREEN, sw=1.1))
    p.append(t(ox + ow / 2, cy + 22, "output:", size=9.4, weight="bold", fill=C_GREEN_D, maxw=ow - 16))
    p.append(t(ox + ow / 2, cy + 36, "LoRA adapters", size=9.4, weight="bold", fill=C_GREEN_D, maxw=ow - 16))
    p.append(t(ox + ow / 2, cy + 56, "loaded by the policy", size=8.6, fill=C_SUB, maxw=ow - 16))
    p.append(t(ox + ow / 2, cy + 68, "at inference", size=8.6, fill=C_SUB, maxw=ow - 16))
    ny = cy + ch2 + 16
    p.append(t(sfx, ny, "rollouts execute the real dispatcher - same code path as trace generation", size=8.4, fill=C_SUB, anchor="start", maxw=590))
    p.append(t(sfx, ny + 12, "checkpoint selection on the dev split (test set evaluated once, at the end)", size=8.4, fill=C_SUB, anchor="start", maxw=590))

    # ---- row 3: stubs pointing at the agent-loop figure ----
    y3 = y2 + h2 + 34
    stub_h = 44
    ex, ew = 22, 270
    p.append(R(ex, y3, ew, stub_h, rx=8, fill="#FAFAFA", stroke="#B5B5B5", sw=1.2))
    p.append(t(ex + ew / 2, y3 + 19, "Evaluation (agent-loop figure)", size=9.2, weight="bold", fill="#555555", maxw=ew - 16))
    p.append(t(ex + ew / 2, y3 + 33, "dev-split signal returns to training", size=8.4, fill=C_SUB, maxw=ew - 16))
    p.append(arrow([(ex + ew / 2, y3), (ex + ew / 2, y2 + h2 + 1)], color=G, sw=1.4))
    p.append(badge(6, ex + ew / 2, y3 - 17))
    qx, qw = 338, 280
    p.append(R(qx, y3, qw, stub_h, rx=8, fill="#FAFAFA", stroke="#B5B5B5", sw=1.2))
    p.append(t(qx + qw / 2, y3 + 19, "Policy + tool loop (agent-loop figure)", size=9.2, weight="bold", fill="#555555", maxw=qw - 16))
    p.append(t(qx + qw / 2, y3 + 33, "adapters loaded, findings evaluated", size=8.4, fill=C_SUB, maxw=qw - 16))
    p.append(arrow([(ox + ow / 2, cy + ch2), (ox + ow / 2, y3 - 1)], color="#7FA98B", sw=1.6))
    p.append(badge(4, ox + ow / 2, y3 - 17))
    p.append(t(10, y3 + stub_h + 24, "numbered path: 1 data -> traces · 2 corpus -> SFT · 3 SFT -> GRPO · 4 adapters -> policy ·", size=8.4, fill=C_SUB, anchor="start", maxw=620))
    p.append(t(10, y3 + stub_h + 36, "5 findings -> eval (agent-loop figure) · 6 dev-split signal", size=8.4, fill=C_SUB, anchor="start", maxw=620))
    if warn:
        print("  architecture_training text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, y3 + stub_h + 48, "".join(p))


# ---------------------------------------------------------------------------
# Figure 4: corpus composition (per-cohort, measured)
# ---------------------------------------------------------------------------


def _fmt_box(b, spaces=True):
    """[x, y, w, h] rounded to ints, as the tools return them in the traces."""
    sep = ", " if spaces else ","
    return "[" + sep.join(str(int(round(v))) for v in b) + "]"


def _checked(warn):
    """Return a T() wrapper that records labels wider than their box (maxw)."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    def t(x, y, s_, size=9.0, fill=C_TXT, anchor="middle", weight=None, maxw=None, font=None, style=None):
        face = "Courier" if font else ("Helvetica-Bold" if weight == "bold" else "Helvetica")
        if maxw is not None and stringWidth(s_, face, size) > maxw:
            warn.append(f"overflow {stringWidth(s_, face, size):.0f}>{maxw}: {s_!r}")
        return TC(x, y, s_, size=size, fill=fill, anchor=anchor, weight=weight, font=font, style=style)

    return t


def _wrap(text, size, maxw, weight=None):
    """Greedy word-wrap using Helvetica metrics (matches the SVG font stack)."""
    from reportlab.pdfbase.pdfmetrics import stringWidth

    face = "Helvetica-Bold" if weight == "bold" else "Helvetica"
    lines, cur = [], ""
    for word in text.split():
        trial = f"{cur} {word}".strip()
        if cur and stringWidth(trial, face, size) > maxw:
            lines.append(cur)
            cur = word
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def build_corpus_composition(a):
    """Corpus composition on a 640-unit canvas (labels print at ~6.3-7.5 pt)."""
    W = 640
    warn: list[str] = []
    t = _checked(warn)
    p = []
    p.append(TC(W / 2, 22, "Corpus composition: five cohorts, two exact arms", size=13, weight="bold"))
    p.append(t(W / 2, 38, "all counts measured from the released trace files (per-cohort 1:1 parity by construction)", size=9.2, fill=C_SUB, maxw=600))

    p.append(R(10, 50, 620, 262, fill="#FCFCFD", stroke="#DDDDDD", sw=1.2))
    hy = 74
    p.append(t(24, hy, "cohort", size=8.8, fill=C_SUB, anchor="start"))
    p.append(t(190, hy, "with-tools traces (bar = cohort size)", size=8.8, fill=C_SUB, anchor="start", maxw=210))
    p.append(t(412, hy, "traces", size=8.8, fill=C_SUB, anchor="start"))
    p.append(t(464, hy, "findings", size=8.8, fill=C_SUB, anchor="start"))
    p.append(t(522, hy, "healthy", size=8.8, fill=C_SUB, anchor="start"))
    p.append(t(574, hy, "avg calls", size=8.8, fill=C_SUB, anchor="start", maxw=52))

    summ = json.loads((PAPER_DIR / "assets_summary.json").read_text(encoding="utf-8"))
    corpus, totals = summ["corpus"], summ["corpus_totals"]
    cohort_rows = [
        ("dentex · pathology", "dentex_pathology", C_BLUE),
        ("dentex · healthy", "dentex_healthy", C_GREEN),
        ("tufts · periapical overlap", "tufts_overlap", C_BLUE),
        ("tufts · healthy", "tufts_healthy", C_GREEN),
        ("tufts · all diseases", "tufts_all_diseases", C_BLUE),
    ]
    rows = []
    for label, key, color in cohort_rows:
        c = corpus[key]
        rows.append((label, c["traces"], f"{c['findings']:,}" if c["findings"] else "-", str(c["healthy_traces"]), c["avg_tool_calls"], color))
    # the stated totals must follow from the measured cohorts
    assert sum(c["traces"] for c in corpus.values()) == totals["total_traces"] == totals["total_no_tools"]
    assert sum(c["findings"] for c in corpus.values()) == totals["total_findings"]
    assert sum(c["healthy_traces"] for c in corpus.values()) == totals["total_healthy"]
    n_calls = round(sum(c["avg_tool_calls"] * c["traces"] for c in corpus.values()))
    turns = [c["avg_turns"] for c in corpus.values()]
    calls_pt = [c["avg_tool_calls"] for c in corpus.values()]
    scale = 205.0 / max(c["traces"] for c in corpus.values())
    y = 84
    for name, n, findings, healthy, calls, color in rows:
        bw = n * scale
        p.append(t(24, y + 17, name, size=9.4, weight="bold", anchor="start", maxw=160))
        p.append(R(190, y + 5, max(bw, 2), 17, rx=3, fill=color, stroke="none", sw=0))
        p.append(t(412, y + 17, f"{n}", size=9.2, weight="bold", anchor="start"))
        p.append(t(464, y + 17, findings, size=9.2, anchor="start"))
        p.append(t(522, y + 17, healthy, size=9.2, anchor="start"))
        p.append(t(574, y + 17, f"{calls:.1f}", size=9.2, anchor="start"))
        y += 32

    classes = [("caries", 2189), ("deep caries", 578), ("impacted teeth", 604), ("periapical lesions", 789), ("further Tufts classes", 353)]
    n_unique = sum(n for _, n in classes)
    ty = 246
    p.append(R(22, ty - 8, 596, 52, rx=6, fill="#F7F8FA", stroke="#E3E3E3", sw=1.0))
    p.append(t(34, ty + 8, "totals", size=9.6, weight="bold", anchor="start"))
    p.append(t(84, ty + 8, f"{totals['total_traces']:,} with-tools + {totals['total_no_tools']:,} no-tools · exact parity · {n_calls:,} executed tool calls", size=9.0, anchor="start", maxw=526))
    p.append(t(84, ty + 21, f"{totals['total_findings']:,} findings ({n_unique:,} unique) · {totals['total_healthy']:,} healthy images", size=9.0, anchor="start", maxw=526))
    p.append(t(84, ty + 34, f"with-tools arm: {round(min(turns))}-{round(max(turns))} turns and {round(min(calls_pt))}-{round(max(calls_pt))} real tool calls per trace", size=8.6, fill=C_SUB, anchor="start", maxw=526))

    ly = 304
    p.append(L(24, ly - 3, 46, ly - 3, color=C_BLUE, sw=3))
    p.append(t(52, ly, "pathology cohorts (findings-bearing)", size=8.8, anchor="start", maxw=190))
    p.append(L(262, ly - 3, 284, ly - 3, color=C_GREEN, sw=3))
    p.append(t(290, ly, "healthy cohorts (negative controls)", size=8.8, anchor="start", maxw=190))

    # unique findings by class
    py0 = 326
    p.append(R(10, py0, 620, 154, fill="#FCFCFD", stroke="#DDDDDD", sw=1.2))
    p.append(t(24, py0 + 22, "unique findings by class", size=10, weight="bold", anchor="start"))
    cy = py0 + 44
    for name, n in classes:
        bw = 300.0 * n / max(c for _, c in classes)
        p.append(t(24, cy, name, size=8.8, anchor="start", maxw=104))
        p.append(R(134, cy - 8, bw, 10, rx=2, fill=C_BLUE, stroke="none", sw=0))
        p.append(t(134 + bw + 6, cy, f"{n:,}", size=8.8, weight="bold", anchor="start"))
        cy += 17
    p.append(t(24, py0 + 134, f"{n_unique:,} unique findings; {totals['total_findings']:,} annotations total (the {corpus['tufts_overlap']['traces']}-image Tufts overlap cohort", size=8.6, fill=C_SUB, anchor="start", maxw=596))
    p.append(t(24, py0 + 146, f"is contained in the {corpus['tufts_all_diseases']['traces']}-image all-diseases cohort)", size=8.6, fill=C_SUB, anchor="start", maxw=596))
    if warn:
        print("  corpus_composition text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, py0 + 162, "".join(p))


# ---------------------------------------------------------------------------
# Figure 5: dataset catalog (two in use, eight candidates)
# ---------------------------------------------------------------------------


def _check_dataset_catalog(in_use: dict[str, int], candidates: list[tuple[str, str]]) -> None:
    """Fail the build if the figure's dataset sizes / diagnosis flags drift from
    dental_agent/data/dataset_catalog.py (the single source of truth)."""
    import importlib.util

    path = Path(__file__).resolve().parents[2] / "dental_agent" / "data" / "dataset_catalog.py"
    spec = importlib.util.spec_from_file_location("_dscat", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_dscat"] = mod  # dataclasses need the module registered
    spec.loader.exec_module(mod)
    cat = {d.name: d for d in mod.PANORAMIC_DATASETS}
    alias = {  # figure label -> catalog name
        "DENTEX": "DENTEX",
        "Tufts Dental Database": "Tufts Panoramic Dataset",
        "Dental Radiography": "Dental Radiography",
        "DNS (UFBA-UESC)": "DNS (Detection, Numbering, and Segmentation)",
        "Segmented Mandibles": "Panoramic Dental X-rays With Segmented Mandibles",
        "TL-pano": "TL-pano",
        "TK Tooth Number Code": "TK_Tooth_Number_Code",
        "Tunisia Panoramic": "Panoramic Dental Xray Dataset",
        "Panoramic-Paraguay": "Panoramic-Paraguay",
        "Panoramic-Caries-Seg": "Panoramic-Caries-Segmentation",
    }
    problems = []
    for label, n in list(in_use.items()) + [(c[0], int(c[1].replace(",", ""))) for c in candidates]:
        entry = cat.get(alias[label])
        if entry is None or entry.n_images != n:
            problems.append(f"{label}: figure says {n}, catalog says {getattr(entry, 'n_images', 'missing')}")
    n_diag = sum(cat[alias[c[0]]].has_diagnosis_labels for c in candidates)
    if n_diag != 1:
        problems.append(f"figure claims 1 of 8 candidates is diagnosis-labeled; catalog has {n_diag}")
    if problems:
        raise ValueError("fig_datasets drifted from dataset_catalog.py:\n  " + "\n  ".join(problems))


def build_dataset_catalog(a):
    """Dataset catalog on a 640-unit canvas: in-use cards side by side, 2x4 candidates."""
    W = 640
    warn: list[str] = []
    t = _checked(warn)
    p = []
    p.append(TC(W / 2, 22, "The data layer: two sources in use, eight candidates on file", size=13, weight="bold"))
    p.append(t(W / 2, 37, "a candidate is promoted only after its annotation semantics are verified -", size=9.2, fill=C_SUB, maxw=610))
    p.append(t(W / 2, 49, "two loaders deliberately stop at one open question each", size=9.2, fill=C_SUB, maxw=610))

    # ---- in use ----
    y0 = 60
    p.append(R(10, y0, 620, 232, fill=C_GREEN_FILL, stroke=C_GREEN, sw=1.4))
    p.append(R(22, y0 + 10, 62, 18, rx=9, fill=PILL_GREEN, stroke="none", sw=0))
    p.append(t(53, y0 + 23, "IN USE", size=8.8, weight="bold", fill="#FFFFFF"))
    p.append(t(618, y0 + 23, "feeds the released corpus", size=8.8, fill=C_GREEN_D, anchor="end"))

    cy = y0 + 38
    # DENTEX card
    p.append(R(22, cy, 298, 184))
    p.append(icon_db(34, cy + 10))
    p.append(t(58, cy + 24, "DENTEX", size=11.5, weight="bold", anchor="start"))
    p.append(t(308, cy + 24, "2,332 images", size=9.2, fill=C_BLUE_D, anchor="end"))
    p.append(t(34, cy + 46, "quadrant · enumeration · diagnosis labels", size=8.8, anchor="start", maxw=276))
    p.append(t(34, cy + 59, "4 pathology classes + box coordinates", size=8.8, anchor="start", maxw=276))
    p.append(t(34, cy + 75, "the only panoramic source pairing tooth position", size=8.6, fill=C_SUB, anchor="start", maxw=276))
    p.append(t(34, cy + 87, "with all four reward-relevant classes", size=8.6, fill=C_SUB, anchor="start", maxw=276))
    p.append(R(34, cy + 98, 130, 20, rx=10, fill="#F0F7F0", stroke=C_GREEN, sw=1.0))
    p.append(t(99, cy + 112, "loader: complete", size=8.8, weight="bold", fill=C_GREEN_D))
    p.append(t(34, cy + 140, "contributes 705 corpus traces", size=9.4, weight="bold", fill=C_BLUE_D, anchor="start", maxw=276))
    p.append(t(34, cy + 153, "(678 pathology + 27 healthy)", size=8.6, fill=C_SUB, anchor="start", maxw=276))
    p.append(t(34, cy + 172, "grounding pool: 1,339 images · 21,624 tooth boxes", size=8.6, fill=C_SUB, anchor="start", maxw=276))

    # Tufts card
    tx = 330
    p.append(R(tx, cy, 290, 184))
    p.append(icon_db(tx + 12, cy + 10))
    p.append(t(tx + 36, cy + 24, "Tufts Dental Database", size=11.5, weight="bold", anchor="start", maxw=150))
    p.append(t(tx + 278, cy + 24, "1,000 images", size=9.2, fill=C_BLUE_D, anchor="end"))
    p.append(t(tx + 12, cy + 46, "per-tooth boxes (Universal Numbering)", size=8.8, anchor="start", maxw=268))
    p.append(t(tx + 12, cy + 59, "expert pathology layer · 25,184 boxes", size=8.8, anchor="start", maxw=268))
    p.append(t(tx + 12, cy + 75, "periapical maps onto the DENTEX taxonomy;", size=8.6, fill=C_SUB, anchor="start", maxw=268))
    p.append(t(tx + 12, cy + 87, "no caries findings in this layer", size=8.6, fill=C_SUB, anchor="start", maxw=268))
    p.append(R(tx + 12, cy + 98, 190, 20, rx=10, fill=C_ORANGE_L, stroke="#F0B27A", sw=1.0))
    p.append(t(tx + 107, cy + 112, "loader: partial - 1 open question", size=8.8, weight="bold", fill=C_ORANGE_D, maxw=180))
    p.append(t(tx + 12, cy + 140, "contributes 1,142 corpus traces", size=9.4, weight="bold", fill=C_BLUE_D, anchor="start", maxw=268))
    p.append(t(tx + 12, cy + 153, "(202 overlap + 660 healthy + 280 all-diseases)", size=8.6, fill=C_SUB, anchor="start", maxw=268))
    p.append(t(tx + 12, cy + 172, "grounding pool: 1,000 images · 25,184 boxes", size=8.6, fill=C_SUB, anchor="start", maxw=268))

    # ---- candidates ----
    c0 = y0 + 232 + 12
    ch = 38 + 4 * 66 + 44
    p.append(R(10, c0, 620, ch, fill="#FBFBFC", stroke="#CCCCCC", sw=1.2))
    p.append(R(22, c0 + 10, 124, 18, rx=9, fill=PILL_GREY, stroke="none", sw=0))
    p.append(t(84, c0 + 23, "CANDIDATE POOL", size=8.8, weight="bold", fill="#FFFFFF"))
    p.append(t(618, c0 + 23, "pending - not used in this work", size=8.8, fill=C_SUB, anchor="end"))

    candidates = [
        ("Dental Radiography", "1,272", "no confirmed diagnosis taxonomy", "systematic review"),
        ("DNS (UFBA-UESC)", "543", "FDI boxes in COCO - access on request", "targeted follow-up"),
        ("Segmented Mandibles", "232", "mandible-only masks, not per-tooth", "systematic review"),
        ("TL-pano", "197", "FDI + quadrant labels - NC license", "targeted follow-up"),
        ("TK Tooth Number Code", "188", "numbering only, no diagnosis", "systematic review"),
        ("Tunisia Panoramic", "180", "in progress - 1 open FDI question", "systematic review"),
        ("Panoramic-Paraguay", "135", "thin documentation in the review", "systematic review"),
        ("Panoramic-Caries-Seg", "75", "binary caries masks (pixel level)", "systematic review"),
    ]
    _check_dataset_catalog({"DENTEX": 2332, "Tufts Dental Database": 1000}, [(c[0], c[1]) for c in candidates])
    for i, (name, n, reason, source) in enumerate(candidates):
        col, row = i % 2, i // 2
        cx = 22 + col * 304
        cyy = c0 + 38 + row * 66
        p.append(R(cx, cyy, 292, 58))
        p.append(t(cx + 10, cyy + 17, name, size=9.4, weight="bold", anchor="start", maxw=200))
        p.append(t(cx + 282, cyy + 17, n, size=9.0, fill=C_BLUE_D, anchor="end"))
        p.append(t(cx + 10, cyy + 33, reason, size=8.6, fill=C_SUB, anchor="start", maxw=272))
        p.append(t(cx + 10, cyy + 48, f"({source})", size=8.4, fill=C_SUB, anchor="start", maxw=272))
    ny = c0 + 38 + 4 * 66 + 8
    p.append(t(22, ny + 6, "diagnosis-labeled: 1 of 8 (binary caries masks) · grounding-only: 7 of 8", size=8.8, anchor="start", maxw=596))
    p.append(t(22, ny + 20, "promotion rule: verified end-to-end annotation semantics, per dataset", size=8.8, anchor="start", maxw=596))

    # ---- footer ----
    f0 = c0 + ch + 12
    p.append(R(10, f0, 620, 76, fill="#F7F8FA", stroke="#E3E3E3", sw=1.0))
    foot = [
        "catalog transcribed from a systematic review of 16 public dental imaging datasets",
        "(Uribe et al. 2024) plus targeted follow-up search",
        "the diagnosis-label vs grounding-only flag drives purpose: grounding candidates generalize",
        "locate_tooth, diagnosis candidates feed trace generation",
    ]
    for i, line in enumerate(foot):
        p.append(t(22, f0 + 16 + i * 13 + (6 if i >= 2 else 0), line, size=8.6, fill=C_SUB, anchor="start", maxw=596))
    if warn:
        print("  dataset_catalog text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, f0 + 88, "".join(p))


# ---------------------------------------------------------------------------
# Figure 6: grounding model CV + held-out verification
# ---------------------------------------------------------------------------


def build_grounding_cv(a):
    """Grounding CV on a 620-unit canvas: fold chart on top, two text columns below."""
    W = 620
    warn: list[str] = []
    t = _checked(warn)
    p = []
    p.append(TC(W / 2, 22, "Grounding model: 5-fold CV and held-out verification", size=13, weight="bold"))
    p.append(t(W / 2, 38, "YOLOv8m backend of locate_tooth (one class per FDI quadrant-position pair) - target-filtered protocol", size=9.0, fill=C_SUB, maxw=600))

    # ---- panel A: fold bars ----
    p.append(R(10, 50, 600, 284, fill="#FCFCFD", stroke="#DDDDDD", sw=1.2))
    p.append(t(26, 74, "in-fold CV: target mAP50 per fold", size=10.4, weight="bold", anchor="start"))
    p.append(L(26, 90, 50, 90, color=C_BLUE, sw=3))
    p.append(t(56, 93, "DENTEX-only (mean 0.9508)", size=8.8, anchor="start", maxw=180))
    p.append(L(262, 90, 286, 90, color=C_GREEN, sw=3))
    p.append(t(292, 93, "DENTEX + Tufts (mean 0.9376)", size=8.8, anchor="start", maxw=200))

    top, bot = 112.0, 290.0

    def yv(v):
        return bot - (v - 0.80) * (bot - top) / 0.20

    x0, x1 = 62, 596
    for gv in (0.85, 0.90, 0.95, 1.00):
        gy = yv(gv)
        p.append(L(x0, gy, x1, gy, color="#E6E6E6", sw=0.8))
        p.append(t(x0 - 6, gy + 3, f"{gv:.2f}", size=8.4, fill=C_SUB, anchor="end"))
    donly = [0.9534, 0.9405, 0.9614, 0.9564, 0.9423]
    dtfts = [0.9676, 0.9502, 0.9384, 0.9504, 0.8814]
    gw = (x1 - x0) / 5.0
    p.append(L(x0, yv(0.9508), x1, yv(0.9508), color=C_BLUE, sw=1.0, dash="5,3"))
    p.append(L(x0, yv(0.9376), x1, yv(0.9376), color=C_GREEN, sw=1.0, dash="5,3"))
    for i, (vd, vt) in enumerate(zip(donly, dtfts)):
        gx = x0 + i * gw
        p.append(R(gx + 6, yv(vd), 40, bot - yv(vd), rx=2, fill=C_BLUE, stroke="none", sw=0))
        p.append(t(gx + 26, yv(vd) + 12, f"{vd:.3f}", size=8.4, fill="#FFFFFF", weight="bold", maxw=42))
        p.append(R(gx + 50, yv(vt), 40, bot - yv(vt), rx=2, fill=C_GREEN, stroke="none", sw=0))
        p.append(t(gx + 70, yv(vt) + 12, f"{vt:.3f}", size=8.4, fill="#FFFFFF", weight="bold", maxw=42))
        p.append(t(gx + 48, bot + 15, f"fold {i}", size=8.8, fill=C_SUB))
    p.append(L(x0, bot, x1, bot, color="#999999", sw=1.0))

    # ---- panel B left: pool + co-training effect ----
    b0, bh = 348, 238
    p.append(R(10, b0, 296, bh, fill="#FCFCFD", stroke="#DDDDDD", sw=1.2))
    lx = 24
    p.append(t(lx, b0 + 22, "training pool", size=10.4, weight="bold", anchor="start"))
    bw1 = 268 * 1339 / 2339
    bw2 = 268 * 1000 / 2339
    p.append(R(lx, b0 + 32, bw1, 16, rx=2, fill=C_BLUE, stroke="none", sw=0))
    p.append(R(lx + bw1, b0 + 32, bw2, 16, rx=2, fill=C_GREEN, stroke="none", sw=0))
    p.append(t(lx, b0 + 68, "DENTEX  1,339 images · 21,624 boxes", size=8.8, anchor="start", maxw=270))
    p.append(t(lx, b0 + 82, "Tufts    1,000 images · 25,184 boxes", size=8.8, anchor="start", maxw=270))
    p.append(t(lx, b0 + 100, "combined 2,339 images · 46,808 boxes", size=9.0, weight="bold", anchor="start", maxw=270))
    p.append(t(lx, b0 + 140, "effect of co-training with Tufts", size=10.4, weight="bold", anchor="start", maxw=270))
    p.append(t(lx, b0 + 160, "precision 0.9637 -> 0.9680", size=8.8, anchor="start"))
    p.append(t(lx, b0 + 175, "raw val mAP50 0.5820 -> 0.8695", size=8.8, anchor="start"))
    p.append(t(lx, b0 + 192, "(different protocol from the target-filtered", size=8.4, fill=C_SUB, anchor="start", maxw=270))
    p.append(t(lx, b0 + 204, "CV above - both measured)", size=8.4, fill=C_SUB, anchor="start", maxw=270))

    # ---- panel B right: held-out + guarantee ----
    rx0 = 318
    p.append(R(rx0, b0, 292, bh, fill="#FCFCFD", stroke="#DDDDDD", sw=1.2))
    rx = rx0 + 14
    p.append(t(rx, b0 + 22, "held-out official DENTEX validation", size=10.4, weight="bold", anchor="start", maxw=270))
    p.append(t(rx, b0 + 38, "50 images · 46 pathological · 182 target boxes", size=8.8, anchor="start", maxw=270))
    p.append(R(rx, b0 + 48, 264, 58, rx=6, fill="#F2F8F2", stroke=C_GREEN, sw=1.3))
    p.append(t(rx + 12, b0 + 66, "best: DENTEX + Tufts, fold 1", size=9.4, weight="bold", fill=C_GREEN_D, anchor="start", maxw=244))
    p.append(t(rx + 12, b0 + 82, "target mAP50 0.9593 · precision 0.9864", size=8.8, anchor="start", maxw=244))
    p.append(t(rx + 12, b0 + 97, "deployed as the locate_tooth backend", size=8.4, fill=C_SUB, anchor="start", maxw=244))
    p.append(R(rx, b0 + 118, 264, 108, rx=6, fill=C_BLUE_FILL, stroke="#C7D3E8", sw=1.1))
    p.append(t(rx + 12, b0 + 136, "trace-generation guarantee", size=9.4, weight="bold", fill=C_BLUE_D, anchor="start", maxw=244))
    guar = [
        "when the requested tooth exists in the ground",
        "truth, locate_tooth returns its exact box - the",
        "detector's measured accuracy governs only the",
        "unannotated-tooth fallback; what the model is",
        "shown is then perturbed independently (tiered),",
        "keeping accept-vs-nudge live",
    ]
    for i, line in enumerate(guar):
        p.append(t(rx + 12, b0 + 153 + i * 12, line, size=8.6, anchor="start", maxw=244))
    if warn:
        print("  grounding_cv text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, b0 + bh + 12, "".join(p))


# ---------------------------------------------------------------------------
# Figure 7: image-input geometry (TPU canonical vs GPU native, decision open)
# ---------------------------------------------------------------------------


def build_input_geometry(a):
    """Image-input geometry on a 640-unit canvas (labels print at ~6.3-7.5 pt).

    TPU views sit in one row (FULL | CROP -> COMPARE) so the panel stays
    short; the GPU branch is a dashed route down the left margin.
    """
    W = 640
    warn: list[str] = []
    t = _checked(warn)
    p = []
    p.append(TC(W / 2, 22, "Image-input geometry: two execution targets, one invariant", size=13, weight="bold"))
    p.append(t(W / 2, 38, "canonical views rendered by the project's own geometry module from DENTEX image 624; the target choice is deliberately open", size=8.8, fill=C_SUB, maxw=620))

    # native input
    nx, ny, nw, nh = 16, 54, 270, 127
    p.append(IMG(nx, ny, nw, nh, a["opg"]))
    p.append(R(nx, ny, nw, nh, rx=3, stroke="#666666", sw=1.2, fill="none"))
    sx, sy = nw / a["native_w"], nh / a["native_h"]
    mx = nx + CROP_FINAL_REGION[0] * sx
    my = ny + CROP_FINAL_REGION[1] * sy
    mw = (CROP_FINAL_REGION[2] - CROP_FINAL_REGION[0]) * sx
    mh = (CROP_FINAL_REGION[3] - CROP_FINAL_REGION[1]) * sy
    p.append(R(mx, my, mw, mh, rx=0, fill="none", stroke=C_ORANGE, sw=1.6))
    p.append(t(302, 90, "native · 2,808 × 1,316 px", size=9.8, weight="bold", anchor="start", maxw=320))
    p.append(t(302, 105, "~17k visual tokens if fed as-is", size=8.8, fill=C_SUB, anchor="start", maxw=320))

    # TPU panel (canonical + static slots)
    ty0, th = 210, 296
    p.append(arrow([(nx + nw / 2, ny + nh), (nx + nw / 2, ty0 - 2)], color="#999999", sw=1.5))
    p.append(R(16, ty0, 614, th, fill=C_BLUE_FILL, stroke="#C7D3E8", sw=1.2))
    p.append(R(28, ty0 + 10, 92, 20, rx=10, fill=C_BLUE, stroke="none", sw=0))
    p.append(t(74, ty0 + 24, "TPU v5e-8", size=9.0, weight="bold", fill="#FFFFFF"))
    p.append(t(130, ty0 + 24, "canonical resizing + static slot padding (required)", size=8.6, fill=C_BLUE_D, anchor="start", maxw=240))

    iy = ty0 + 62
    p.append(t(28, iy - 6, "FULL · 1536 × 768 · 1,152 tokens", size=8.6, weight="bold", fill=C_BLUE_D, anchor="start", maxw=290))
    p.append(IMG(28, iy, 280, 140, a["canon_full"]))
    p.append(R(28, iy, 280, 140, rx=2, stroke="#666666", sw=1.0, fill="none"))
    p.append(t(324, iy + 188, "CROP · 256 × 384 · 96 tokens", size=8.6, weight="bold", fill=C_BLUE_D, anchor="start", maxw=134))
    p.append(IMG(324, iy, 116, 174, a["canon_crop"]))
    p.append(R(324, iy, 116, 174, rx=2, stroke="#666666", sw=1.0, fill="none"))
    p.append(t(462, iy - 6, "COMPARE · 512 × 384 · 192 tokens", size=8.6, weight="bold", fill=C_BLUE_D, anchor="start", maxw=160))
    p.append(IMG(462, iy, 160, 120, a["canon_compare"]))
    p.append(R(462, iy, 160, 120, rx=2, stroke="#666666", sw=1.0, fill="none"))
    # the CROP view feeds the COMPARE composite (real tool output)
    p.append(arrow([(440, iy + 60), (460, iy + 60)], color="#999999", sw=1.3))
    # leader: the marked crop on the native image feeds the CROP view
    lx = 382
    p.append(arrow([(mx + mw, my + mh / 2), (lx, my + mh / 2), (lx, iy - 1)], color=C_ORANGE, sw=1.3))

    p.append(t(28, iy + 206, "static slots: 5 FULL + 10 CROP + 4 COMPARE = 19 slots = 7,488 vision tokens -", size=8.8, fill=C_BLUE_D, anchor="start", maxw=590))
    p.append(t(28, iy + 219, "one XLA graph, no recompilation", size=8.8, fill=C_BLUE_D, anchor="start", maxw=590))

    # GPU panel
    gy0, gh = ty0 + th + 12, 108
    p.append(R(16, gy0, 614, gh, fill="#F7F8FA", stroke="#BBBBBB", sw=1.2))
    p.append(R(28, gy0 + 10, 56, 20, rx=10, fill=PILL_GREY, stroke="none", sw=0))
    p.append(t(56, gy0 + 24, "GPU", size=9.0, weight="bold", fill="#FFFFFF"))
    p.append(t(618, gy0 + 24, "runs so far: original images, dynamic shapes", size=8.6, fill="#555555", anchor="end", maxw=470))
    p.append(t(28, gy0 + 48, "GPU runs so far feed native images with dynamic padding; a --canonical-resize flag", size=8.8, anchor="start", maxw=590))
    p.append(t(28, gy0 + 61, "exists for the same views - the final choice is pending the first GPU run", size=8.8, anchor="start", maxw=590))
    cx = 28
    for cw, label in ((126, "RTX 4090 · QLoRA"), (118, "RTX Pro 6000"), (100, "A100 · bf16")):
        p.append(R(cx, gy0 + 74, cw, 20, rx=10, fill="#FFFFFF", stroke="#BBBBBB", sw=1.0))
        p.append(t(cx + cw / 2, gy0 + 88, label, size=8.4, fill="#555555", maxw=cw - 8))
        cx += cw + 8
    # branch to the GPU target runs down the left margin (dashed)
    p.append(arrow([(nx, ny + nh / 2), (6, ny + nh / 2), (6, gy0 + gh / 2), (14, gy0 + gh / 2)], color="#999999", sw=1.5))

    # decision band
    dy0 = gy0 + gh + 12
    p.append(R(10, dy0, 620, 104, rx=8, fill=C_ORANGE_L, stroke="#F0B27A", sw=1.2))
    dx = 22
    p.append(t(dx, dy0 + 18, "OPEN DECISION - execution target: GPU (RTX 4090 / RTX Pro 6000 / A100) vs TPU v5e-8;", size=9.0, weight="bold", fill=C_ORANGE_D, anchor="start", maxw=596))
    p.append(t(dx, dy0 + 30, "to be settled by the first GPU run", size=9.0, weight="bold", fill=C_ORANGE_D, anchor="start", maxw=596))
    p.append(t(dx, dy0 + 50, "shared invariant: tools always execute on the native image · evaluation runs at original size ·", size=8.6, fill=C_ORANGE_D, anchor="start", maxw=596))
    p.append(t(dx, dy0 + 62, "SFT, GRPO and evaluation use the same setting", size=8.6, fill=C_ORANGE_D, anchor="start", maxw=596))
    p.append(t(dx, dy0 + 82, "TPU padding changes no gradients: the padded tail carries labels = -100 and attention_mask = 0", size=8.4, fill=C_ORANGE_D, anchor="start", maxw=596))
    p.append(t(dx, dy0 + 94, "(verified against the reference implementation)", size=8.4, fill=C_ORANGE_D, anchor="start", maxw=596))

    if warn:
        print("  input_geometry text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, dy0 + 116, "".join(p))


def build_pipeline(a):
    """Pipeline overview, rebuilt with real radiograph pixels (image 624 assets).

    Narrow 640-unit canvas laid out as a snake (datasets -> loop, then
    verification -> corpus right-to-left, then SFT -> GRPO -> policy -> eval)
    so that, at \\linewidth (468 pt), body text lands at ~7 pt on the page
    instead of ~4.7 pt on the old 960-unit canvas. Content is unchanged.
    """
    W = 640
    G = "#666666"
    p = []
    p.append(T(W / 2, 22, "VLM-DENTAL: synthetic trace synthesis and two-stage training pipeline", size=13, weight="bold"))

    # ---- row 1: public datasets -> trace generation -----------------------
    y1, h1 = 38, 232
    p.append(R(10, y1, 160, h1, fill=C_BLUE_FILL, stroke=C_BLUE, sw=1.5))
    p.append(T(90, y1 + 36, "Public datasets", size=11.5, weight="bold"))
    bd_ = _showcase()["breadth"]
    bw_, bh_ = _fit(*bd_["meta"]["image_size"], 140, 66)
    p.append(IMG(20, y1 + 62, bw_, bh_, bd_["image"]))
    _gt_overlay(p, bd_["meta"], 20, y1 + 62, bw_, bh_, sw=0.8)
    p.append(R(20, y1 + 62, bw_, bh_, rx=0, stroke="#666666", sw=1.0, fill="none"))
    p.append(T(90, y1 + 154, "DENTEX + Tufts", size=10))
    p.append(T(90, y1 + 168, "expert labels (FDI)", size=10))

    ym = y1 + h1 / 2
    p.append(arrow([(170, ym), (196, ym)], color=G, sw=1.5))

    # teacher loop (real tool-output pixels: full OPG, canonical crop, real
    # contralateral composite returned by tool_contralateral_compare)
    p.append(R(200, y1, 430, h1, rx=8, fill="#FBF6F0", stroke="#DD8452", sw=1.5))
    cx = 415
    p.append(T(cx, y1 + 22, "Trace generation: LangGraph agent loop", size=11.5, weight="bold"))
    p.append(T(cx, y1 + 39, "frontier teacher via rate-limited provider pool", size=9.8))
    p.append(T(cx, y1 + 52, "ground-truth-directed agent loop", size=9.8))
    iy, ih = y1 + 62, 80
    p.append(IMG(242, iy, 170, ih, a["opg"]))
    p.append(R(242, iy, 170, ih, rx=0, stroke="#888888", sw=0.9, fill="none"))
    p.append(IMG(420, iy, 54, ih, a["canon_crop"]))
    p.append(R(420, iy, 54, ih, rx=0, stroke="#888888", sw=0.9, fill="none"))
    p.append(IMG(482, iy, 106, ih, a["canon_compare"]))
    p.append(R(482, iy, 106, ih, rx=0, stroke="#888888", sw=0.9, fill="none"))
    p.append(T(cx, y1 + 160, "reasoning turns interleaved with real tool output", size=9.4, fill=C_SUB))
    p.append(T(cx, y1 + 176, "8 tools: zoom_crop - window_level - denoise - enhance_contrast -", size=9.4))
    p.append(T(cx, y1 + 189, "contralateral_compare - locate_tooth - fdi_label - nudge_crop", size=9.4))
    p.append(R(226, y1 + 198, 378, 24, rx=4, fill="#FDEBE6", stroke=C_RED, sw=1.2))
    p.append(T(cx, y1 + 214, "tiered perturbation of the box shown to the teacher", size=9.8, weight="bold", fill=C_RED))

    # ---- row 2: verification -> verified corpus (right to left) -----------
    y2, h2 = y1 + h1 + 32, 128
    p.append(arrow([(480, y1 + h1), (480, y2 - 2)], color=G, sw=1.5))

    p.append(R(330, y2, 300, h2, fill=C_GREEN_FILL, stroke=C_GREEN, sw=1.5))
    p.append(T(480, y2 + 24, "Verification", size=11.5, weight="bold"))
    p.append(T(480, y2 + 46, "independent, cross-family", size=10))
    p.append(T(480, y2 + 60, "verifier vs expert labels;", size=10))
    p.append(T(480, y2 + 74, "repair-and-re-verify loop", size=10))
    p.append(T(480, y2 + 88, "dual-gate directive audit", size=10))
    p.append(T(480, y2 + 108, "92 first-pass rejections,", size=9.4, fill=C_SUB))
    p.append(T(480, y2 + 120, "3 measured failure modes", size=9.4, fill=C_SUB))

    p.append(arrow([(330, y2 + h2 / 2), (312, y2 + h2 / 2)], color=G, sw=1.5))

    p.append(R(10, y2, 300, h2, fill=C_BLUE_FILL, stroke=C_BLUE, sw=1.5))
    p.append(T(160, y2 + 24, "Verified corpus", size=11.5, weight="bold"))
    p.append(T(160, y2 + 46, "3,694 traces = 1,847 + 1,847", size=10))
    p.append(T(160, y2 + 60, "two arms, exact 1:1 parity", size=10))
    p.append(T(160, y2 + 74, "5 cohorts, 2 institutions", size=10))
    p.append(T(160, y2 + 88, "44,350 real tool calls", size=10))
    p.append(T(160, y2 + 112, "0 directive leaks", size=10, weight="bold", fill=C_GREEN_D))

    # ---- row 3: two-stage training -> policy -> eval ----------------------
    y3, h3 = y2 + h2 + 32, 148
    p.append(arrow([(92, y2 + h2), (92, y3 - 2)], color=G, sw=1.5))
    ym3 = y3 + h3 / 2

    p.append(R(12, y3, 160, h3, fill=C_BLUE_FILL, stroke=C_BLUE, sw=1.5))
    p.append(T(92, y3 + 24, "Stage 1 -- SFT", size=11.5, weight="bold"))
    p.append(T(92, y3 + 44, "QLoRA on Qwen3.5-9B", size=10))
    p.append(T(92, y3 + 66, "multi-turn trajectories", size=9.8))
    p.append(T(92, y3 + 80, "interleave the real crops", size=9.8))
    p.append(T(92, y3 + 94, "the tools produced;", size=9.8))
    p.append(T(92, y3 + 108, "loss masked to", size=9.8))
    p.append(T(92, y3 + 122, "assistant tokens", size=9.8))

    p.append(arrow([(172, ym3), (192, ym3)], color=G, sw=1.5))

    p.append(R(194, y3, 214, h3, rx=8, fill="#FBF6F0", stroke="#DD8452", sw=1.5))
    p.append(T(301, y3 + 24, "Stage 2 -- GRPO", size=11.5, weight="bold"))
    p.append(T(301, y3 + 44, "ground-truth-anchored", size=9.8))
    p.append(T(301, y3 + 58, "composite reward:", size=9.8))
    p.append(T(301, y3 + 72, "multi-finding F1 accuracy,", size=9.8))
    p.append(T(301, y3 + 86, "format, tool validity,", size=9.8))
    p.append(T(301, y3 + 100, "complexity-scaled efficiency", size=9.8))
    p.append(T(301, y3 + 122, "locate_tooth and nudge_crop", size=9.5, weight="bold", fill=C_RED))
    p.append(T(301, y3 + 135, "exempt from call cost", size=9.5, weight="bold", fill=C_RED))

    p.append(arrow([(408, ym3), (428, ym3)], color=G, sw=1.5))

    p.append(R(430, y3, 90, h3, fill=C_GREEN_FILL, stroke=C_GREEN, sw=1.5))
    p.append(T(475, y3 + 24, "Policy", size=11.5, weight="bold"))
    p.append(T(475, y3 + 48, "self-correcting", size=9.6))
    p.append(T(475, y3 + 62, "tool-use", size=9.6))
    p.append(T(475, y3 + 76, "agent", size=9.6))

    p.append(arrow([(520, ym3), (540, ym3)], color=G, sw=1.5))

    p.append(R(542, y3, 88, h3, fill=C_BLUE_FILL, stroke=C_BLUE, sw=1.5))
    p.append(T(586, y3 + 24, "Eval", size=11.5, weight="bold"))
    p.append(T(586, y3 + 48, "DENTEX-native", size=9.6))
    p.append(T(586, y3 + 62, "metrics,", size=9.6))
    p.append(T(586, y3 + 76, "ablations", size=9.6))

    # ---- footnotes ----------------------------------------------------------
    yf = y3 + h3 + 26
    p.append(T(W / 2, yf, "One agent loop, one tool dispatcher, one image-input geometry (native, or canonical views when training on TPU)", size=9.4, fill=C_TXT))
    p.append(T(W / 2, yf + 13, "across trace generation, RL rollouts, and evaluation; tool calls execute for real against the source image at every stage.", size=9.4, fill=C_TXT))
    p.append(T(W / 2, yf + 33, "During trace generation only: locate_tooth returns the expert-verified box for annotated teeth,", size=9.2, fill=C_SUB))
    p.append(T(W / 2, yf + 46, "and the perturbation hook independently perturbs the box shown.", size=9.2, fill=C_SUB))

    return svg(W, yf + 62, "".join(p))


# ---------------------------------------------------------------------------


_SHOWCASE: dict = {}


def _showcase() -> dict:
    if not _SHOWCASE:
        _SHOWCASE.update(load_showcase())
    return _SHOWCASE


def build_concept(a):
    """Teaser: grounding taken as an answer vs grounding as a hypothesis.

    Real episode from the verified corpus (DENTEX image 127, FDI 48, impacted):
    locate_tooth returned a box 0.92 box-sizes from the expert box (IoU 0.07);
    the policy saw a crop "dominated by a maxillary molar", judged the box
    "about one arch too high", and one nudge_crop(dx -0.57, dy +0.69) landed on
    the expert box (IoU 0.94). All pixels, boxes and numbers come from
    paper/figures/assets/showcase (see prepare_showcase.py).
    """
    sc = _showcase()["flagship"]
    m, f = sc["meta"], sc["meta"]["focus"]
    W = 640
    warn: list[str] = []
    t = _checked(warn)
    p = []
    G = "#777777"
    region = tuple(f["context_region"])
    cw, ch = f["context_size"]
    cr_w, cr_h = f["crop_before_size"]

    def context(x, y, w, with_final):
        h = w * ch / cw
        p.append(IMG(x, y, w, h, sc["context"]))
        p.append(R(x, y, w, h, rx=2, stroke="#666666", sw=1.1, fill="none"))
        layers = [(f["gt_box"], "#2ca02c", 1.6), (f["returned_box"], C_BLUE, 1.9)]
        if with_final:
            layers.append((f["nudges"][-1], C_ORANGE, 1.9))
        for box, color, sw in layers:
            bx, by, bw, bh = native_box(box, m, x, y, w, h, region)
            p.append(R(bx, by, bw, bh, rx=0, fill="none", stroke=color, sw=sw))
        return h

    def crop(x, y, h, which):
        w = h * cr_w / cr_h
        p.append(IMG(x, y, w, h, sc[which]))
        p.append(R(x, y, w, h, rx=2, stroke="#666666", sw=1.1, fill="none"))
        return w

    # ---------------- (a) grounding taken as an answer ----------------
    ya, ha = 10, 196
    p.append(R(10, ya, 620, ha, rx=8, fill="#F8F8F8", stroke="#BBBBBB", sw=1.2))
    p.append(R(10, ya, 232, 22, rx=6, fill=PILL_GREY, stroke="none", sw=0))
    p.append(t(126, ya + 15, "(a) Grounding taken as an answer", size=9.0, weight="bold", fill="#FFFFFF", maxw=220))
    iy = ya + 36
    cw_a = 128
    hh = context(22, iy, cw_a, False)
    p.append(t(22, iy + hh + 14, "returned box (blue), expert box (green)", size=8.2, fill=C_SUB, anchor="start", maxw=190))
    p.append(arrow([(cw_a + 28, iy + hh / 2), (cw_a + 52, iy + hh / 2)], color=G, sw=1.5))
    x_c = cw_a + 58
    ww = crop(x_c, iy + 6, 120, "crop_before")
    p.append(t(x_c, iy + 148, "crop of the returned box", size=8.2, fill=C_SUB, anchor="start", maxw=150))
    p.append(arrow([(x_c + ww + 8, iy + 66), (x_c + ww + 30, iy + 66)], color=G, sw=1.5))
    bx0 = x_c + ww + 36
    p.append(R(bx0, iy + 4, 618 - bx0, 128, rx=6, fill="#FFFFFF", stroke="#CCCCCC", sw=1.1))
    tw = 618 - bx0 - 20
    p.append(t(bx0 + 12, iy + 26, "The crop looks authoritative.", size=9.6, weight="bold", anchor="start", maxw=tw))
    p.append(t(bx0 + 12, iy + 44, "Every later observation is built on", size=8.8, anchor="start", maxw=tw))
    p.append(t(bx0 + 12, iy + 57, "a box that is wrong - here it sits", size=8.8, anchor="start", maxw=tw))
    p.append(t(bx0 + 12, iy + 70, f"{f['offset_before']:.2f} box-sizes from the expert box", size=8.8, anchor="start", maxw=tw))
    p.append(t(bx0 + 12, iy + 83, f"(IoU {f['iou_before']:.2f}).", size=8.8, anchor="start", maxw=tw))
    p.append(t(bx0 + 12, iy + 108, "Nothing in the loop questions the box.", size=8.6, fill=C_RED, weight="bold", anchor="start", maxw=tw))

    # ---------------- (b) grounding as a hypothesis (ours) ----------------
    yb, hb = ya + ha + 14, 214
    p.append(R(10, yb, 620, hb, rx=8, fill=C_BLUE_FILL, stroke=C_BLUE, sw=1.4))
    p.append(R(10, yb, 262, 22, rx=6, fill=C_BLUE, stroke="none", sw=0))
    p.append(t(141, yb + 15, "(b) Grounding as a hypothesis (ours)", size=9.0, weight="bold", fill="#FFFFFF", maxw=250))
    iy = yb + 36
    hh = context(22, iy, cw_a, True)
    p.append(t(22, iy + hh + 14, "same episode; box corrected (orange)", size=8.2, fill=C_SUB, anchor="start", maxw=190))
    p.append(arrow([(cw_a + 28, iy + hh / 2), (cw_a + 52, iy + hh / 2)], color=G, sw=1.5))
    ch_ = 104
    w1 = crop(x_c, iy + 6, ch_, "crop_before")
    p.append(t(x_c, iy + 126, "inspect: “... about one arch", size=8.2, fill=C_SUB, anchor="start", maxw=130))
    p.append(t(x_c, iy + 138, "too high”", size=8.2, fill=C_SUB, anchor="start", maxw=130))
    ax0 = x_c + w1 + 8
    x_f = ax0 + 104
    p.append(arrow([(ax0, iy + 58), (x_f - 8, iy + 58)], color=C_ORANGE, sw=2.0))
    p.append(t((ax0 + x_f - 8) / 2, iy + 40, "nudge_crop × 1", size=9.2, weight="bold", fill=C_ORANGE_D, maxw=x_f - 8 - ax0))
    a0 = f["nudge_args"][0]
    p.append(t((ax0 + x_f - 8) / 2, iy + 76, f"dx {a0['dx_frac']:+.2f} · dy {a0['dy_frac']:+.2f}".replace("-", "−"), size=8.4, fill=C_ORANGE_D, maxw=x_f - 8 - ax0))
    p.append(t((ax0 + x_f - 8) / 2, iy + 94, f"IoU {f['iou_before']:.2f} → {f['iou_after']:.2f}", size=8.8, weight="bold", fill=C_GREEN_D, maxw=x_f - 8 - ax0))
    w2 = crop(x_f, iy + 6, ch_, "crop_after")
    p.append(t(x_f, iy + 126, "re-inspect: after nudge_crop", size=8.2, fill=C_SUB, anchor="start", maxw=140))
    p.append(arrow([(x_f + w2 + 8, iy + 58), (x_f + w2 + 26, iy + 58)], color=G, sw=1.5))
    qx = x_f + w2 + 32
    p.append(R(qx, iy + 14, 618 - qx, 92, rx=6, fill="#F2F8F2", stroke=C_GREEN, sw=1.3))
    qw = 618 - qx - 16
    fa = next(x for x in [{"q": 4, "pos": 8, "dx": "Impacted tooth", "conf": 0.92}])
    p.append(t(qx + 10, iy + 34, "final answer", size=9.4, weight="bold", fill=C_GREEN_D, anchor="start", maxw=qw))
    p.append(t(qx + 10, iy + 50, f"Q{fa['q']}T{fa['pos']} · {fa['dx']}", size=8.8, anchor="start", maxw=qw))
    p.append(t(qx + 10, iy + 63, f"confidence {fa['conf']:.2f}", size=8.6, anchor="start", maxw=qw))
    p.append(t(qx + 10, iy + 90, "matches expert label", size=8.6, weight="bold", fill=C_GREEN_D, anchor="start", maxw=qw))

    # legend
    ly = yb + hb + 20
    p.append(L(12, ly - 3, 36, ly - 3, color=C_BLUE, sw=2.4))
    p.append(t(42, ly, "box returned by the grounding tool", size=8.6, anchor="start", maxw=190))
    p.append(L(250, ly - 3, 274, ly - 3, color="#2ca02c", sw=2.4))
    p.append(t(280, ly, "expert box", size=8.6, anchor="start", maxw=90))
    p.append(L(380, ly - 3, 404, ly - 3, color=C_ORANGE, sw=2.4))
    p.append(t(410, ly, "box after nudge_crop", size=8.6, anchor="start", maxw=160))
    if warn:
        print("  concept text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, ly + 12, "".join(p))


def _cover_b64(b64: str, aspect: float) -> str:
    """Centre-crop a base64 image to width/height == aspect (no stretching)."""
    import base64
    import io

    from PIL import Image

    im = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    w, h = im.size
    if w / h > aspect:
        nw = round(h * aspect)
        im = im.crop(((w - nw) // 2, 0, (w - nw) // 2 + nw, h))
    else:
        nh = round(w / aspect)
        im = im.crop((0, (h - nh) // 2, w, (h - nh) // 2 + nh))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _gt_overlay(p, item, x, y, w, h, color="#2ca02c", sw=0.9, alpha_boxes=None):
    """Thin vector boxes for every ground-truth finding on a full radiograph drawn at (x, y, w, h)."""
    for g in item["ground_truth"]:
        bx, by, bw, bh = native_box(g["bbox"], item, x, y, w, h)
        p.append(R(bx, by, bw, bh, rx=0, fill="none", stroke=color, sw=sw))


def _tier(dx: float, dy: float) -> str:
    """Perturbation tier recovered from the box offset (as in the miner)."""
    m = max(abs(dx), abs(dy))
    return "none" if m < 0.05 else ("small" if m <= 0.32 else "big")


def _fit(w: float, h: float, box_w: float, box_h: float) -> tuple[float, float]:
    """Largest (w', h') with the aspect of (w, h) that fits inside box."""
    s_ = min(box_w / w, box_h / h)
    return w * s_, h * s_


def build_gallery(a):
    """Qualitative gallery: six real episodes from the verified corpus.

    Per cell: the episode context (expert box green, returned box blue, final
    nudge orange), the crop of the returned box and the crop after the nudge.
    All numbers come from paper/figures/assets/showcase (prepare_showcase.py).
    """
    sc = _showcase()
    cells = [("three_step", "DENTEX"), ("breadth", "DENTEX"), ("deep_caries", "DENTEX"), ("tufts", "Tufts"), ("small_tier", "DENTEX"), ("healthy", "DENTEX")]
    W = 640
    warn: list[str] = []
    t = _checked(warn)
    p = []
    cw_, chh = 302, 190
    TILE_H = 110
    for i, (pid, dsname) in enumerate(cells):
        col, row = i % 2, i // 2
        x0, y0 = 10 + col * 318, 10 + row * (chh + 10)
        m = sc[pid]["meta"]
        letter = "abcdef"[i]
        counts = f"{m['n_findings']} findings · {m['n_turns']} turns · {m['n_calls']} tool calls"
        p.append(R(x0, y0, cw_, chh, rx=8, fill="#FBFBFC", stroke="#C9CED6", sw=1.2))
        f = m.get("focus")
        ty = y0 + 40  # top of the tile area
        if f:
            p.append(t(x0 + 10, y0 + 17, f"({letter}) {dsname} image {m['image_id']} · FDI {f['fdi']} {f['diagnosis'].lower()}", size=9.2, weight="bold", anchor="start", maxw=cw_ - 20))
            tier = _tier(f["dx_frac"], f["dy_frac"])
            n = len(f["nudges"])
            p.append(t(x0 + 10, y0 + 30, f"{tier} offset · {n} nudge{'s' if n != 1 else ''} · IoU {f['iou_before']:.2f} → {f['iou_after']:.2f}", size=8.6, fill=C_SUB, anchor="start", maxw=cw_ - 20))
            region = tuple(f["context_region"])
            cw, chx = f["context_size"]
            iw, ih = _fit(cw, chx, 142, TILE_H)
            ix, iy = x0 + 10, ty + (TILE_H - ih) / 2
            p.append(IMG(ix, iy, iw, ih, sc[pid]["context"]))
            p.append(R(ix, iy, iw, ih, rx=2, stroke="#666666", sw=1.0, fill="none"))
            for box, color, sw in ((f["gt_box"], "#2ca02c", 1.5), (f["returned_box"], C_BLUE, 1.7), (f["nudges"][-1], C_ORANGE, 1.7)):
                bx, by, bw, bh = native_box(box, m, ix, iy, iw, ih, region)
                p.append(R(bx, by, bw, bh, rx=0, fill="none", stroke=color, sw=sw))
            p.append(t(ix + iw / 2, ty + TILE_H + 13, "context", size=8.2, fill=C_SUB, maxw=iw + 10))
            xx = x0 + 10 + 142 + 12
            for nm, lab in (("crop_before", "returned box"), ("crop_after", "after nudge")):
                cwid, chid = f[nm + "_size"]
                w_, h_ = _fit(cwid, chid, 62, TILE_H)
                p.append(IMG(xx, ty + (TILE_H - h_) / 2, w_, h_, sc[pid][nm]))
                p.append(R(xx, ty + (TILE_H - h_) / 2, w_, h_, rx=1, stroke="#666666", sw=0.9, fill="none"))
                p.append(t(xx + 31, ty + TILE_H + 13, lab, size=8.2, fill=C_SUB, maxw=70))
                xx += 62 + 10
        else:
            p.append(t(x0 + 10, y0 + 17, f"({letter}) {dsname} healthy image {m['image_id']} · negative control", size=9.2, weight="bold", anchor="start", maxw=cw_ - 20))
            p.append(t(x0 + 10, y0 + 30, "no findings returned - cleared, no hallucination", size=8.6, fill=C_SUB, anchor="start", maxw=cw_ - 20))
            iw, ih = _fit(*m["image_size"], cw_ - 20, 104)
            ix, iy = x0 + 10 + (cw_ - 20 - iw) / 2, ty
            p.append(IMG(ix, iy, iw, ih, sc[pid]["image"]))
            p.append(R(ix, iy, iw, ih, rx=2, stroke="#666666", sw=1.0, fill="none"))
            p.append(t(x0 + 10, ty + TILE_H + 13, "a partially edentulous scan with restorations - still an empty list", size=8.2, fill=C_SUB, anchor="start", maxw=cw_ - 20))
        p.append(t(x0 + 10, y0 + chh - 9, counts, size=8.2, fill=C_SUB, anchor="start", maxw=cw_ - 20))
    ly = 10 + 3 * (chh + 10) + 8
    p.append(L(12, ly - 3, 36, ly - 3, color="#2ca02c", sw=2.4))
    p.append(t(42, ly, "expert box", size=8.6, anchor="start", maxw=90))
    p.append(L(150, ly - 3, 174, ly - 3, color=C_BLUE, sw=2.4))
    p.append(t(180, ly, "box returned by locate_tooth", size=8.6, anchor="start", maxw=170))
    p.append(L(370, ly - 3, 394, ly - 3, color=C_ORANGE, sw=2.4))
    p.append(t(400, ly, "box after the nudge_crop chain", size=8.6, anchor="start", maxw=200))
    if warn:
        print("  gallery text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, ly + 12, "".join(p))


def build_design(a):
    """Experimental design: the 2x2 of conditions, hypotheses H1-H3, frozen protocol, probes.

    Every statement is taken from the Experiments section (conditions,
    hypotheses, protocol, probes) and from the zero-shot table; result cells
    stay "pending" exactly like the main-results table.
    """
    W = 640
    warn: list[str] = []
    t = _checked(warn)
    p = []
    G = "#777777"
    PEND = C_RED

    def cell(x, y, w, h, title, sub, line3, hero=False):
        p.append(R(x, y, w, h, rx=8, fill=C_GREEN_FILL if hero else C_BLUE_FILL, stroke=C_GREEN if hero else C_BLUE, sw=1.8 if hero else 1.3))
        p.append(t(x + w / 2, y + 22, title, size=10.4, weight="bold", fill=C_GREEN_D if hero else C_BLUE_D, maxw=w - 14))
        p.append(t(x + w / 2, y + 38, sub, size=8.8, fill=C_SUB, maxw=w - 14))
        p.append(t(x + w / 2, y + 52, line3, size=8.6, fill=C_SUB, maxw=w - 14))
        p.append(t(x + w / 2, y + 72, "F1 / invented: pending", size=8.8, weight="bold", fill=PEND, maxw=w - 14))

    # column + row headers
    c1x, c2x, cw, ch = 96, 306, 140, 86
    r1y, r2y = 42, 174
    p.append(t(c1x + cw / 2, 28, "SFT", size=10.4, weight="bold", maxw=cw))
    p.append(t(c2x + cw / 2, 28, "SFT + GRPO", size=10.4, weight="bold", maxw=cw))
    p.append(t(12, r1y + 44, "no tools", size=10.0, weight="bold", anchor="start", maxw=76))
    p.append(t(12, r2y + 44, "with tools", size=10.0, weight="bold", anchor="start", maxw=76))
    cell(c1x, r1y, cw, ch, "SFT only", "no tools", "1,847 single-turn traces")
    cell(c2x, r1y, cw, ch, "SFT + GRPO", "no tools", "same recipe, single-turn")
    cell(c1x, r2y, cw, ch, "SFT + tools", "1,847 with-tools traces", "real tool execution")
    cell(c2x, r2y, cw, ch, "VLM-DENTAL", "SFT + GRPO + tools", "live tool rollouts", hero=True)
    # H2 (RL) horizontal arrows, H1 (tools) vertical arrows
    for ry in (r1y, r2y):
        ymid = ry + ch / 2
        p.append(arrow([(c1x + cw, ymid), (c2x - 2, ymid)], color=C_ORANGE, sw=1.8))
        p.append(t((c1x + cw + c2x) / 2, ymid - 7, "H2: RL", size=8.8, weight="bold", fill=C_ORANGE_D, maxw=62))
    for cx in (c1x, c2x):
        xm = cx + cw / 2
        p.append(arrow([(xm, r1y + ch), (xm, r2y - 2)], color=C_BLUE, sw=1.8))
        p.append(t(xm + 8, (r1y + ch + r2y) / 2 + 4, "H1: tools", size=8.8, weight="bold", fill=C_BLUE_D, anchor="start", maxw=70))

    # references (right)
    rx_, rw_ = 478, 152
    p.append(R(rx_, r1y, rw_, ch, rx=8, fill="#F7F8FA", stroke="#BBBBBB", sw=1.2))
    p.append(t(rx_ + rw_ / 2, r1y + 20, "zero-shot references", size=9.6, weight="bold", maxw=rw_ - 12))
    p.append(t(rx_ + 12, r1y + 38, "best frontier VLM", size=8.6, fill=C_SUB, anchor="start", maxw=100))
    p.append(t(rx_ + rw_ - 12, r1y + 38, "0.222", size=8.8, weight="bold", anchor="end"))
    p.append(t(rx_ + 12, r1y + 52, "Qwen3.5-9B", size=8.6, fill=C_SUB, anchor="start", maxw=100))
    p.append(t(rx_ + rw_ - 12, r1y + 52, "0.112", size=8.8, weight="bold", anchor="end"))
    p.append(t(rx_ + 12, r1y + 66, "majority floor", size=8.6, fill=C_SUB, anchor="start", maxw=100))
    p.append(t(rx_ + rw_ - 12, r1y + 66, "0.038", size=8.8, weight="bold", anchor="end"))
    p.append(t(rx_ + rw_ / 2, r1y + ch + 12, "exact F1 (zero-shot table)", size=8.2, fill=C_SUB, maxw=rw_))
    p.append(R(rx_, r2y, rw_, ch, rx=8, fill="#FDF3EA", stroke="#DD8452", sw=1.2))
    p.append(t(rx_ + rw_ / 2, r2y + 20, "supervised specialist", size=9.6, weight="bold", maxw=rw_ - 12))
    p.append(t(rx_ + rw_ / 2, r2y + 36, "HierarchicalDet on DENTEX", size=8.6, fill=C_SUB, maxw=rw_ - 12))
    p.append(t(rx_ + rw_ / 2, r2y + 50, "AP 0.372 · AP50 0.601", size=8.8, maxw=rw_ - 12))
    p.append(t(rx_ + rw_ / 2, r2y + 63, "AR 0.653", size=8.8, maxw=rw_ - 12))
    p.append(t(rx_ + rw_ / 2, r2y + 78, "a fair test, not a sure win", size=8.2, fill=C_ORANGE_D, weight="bold", maxw=rw_ - 12))
    p.append(arrow([(c2x + cw, r2y + ch / 2), (rx_ - 2, r2y + ch / 2)], color=G, sw=1.5))
    p.append(t((c2x + cw + rx_) / 2, r2y + ch / 2 - 7, "H3", size=8.8, weight="bold", fill=C_SUB, maxw=34))

    # protocol band
    py0 = r2y + ch + 24
    ph = 120
    p.append(R(10, py0, 620, ph, rx=8, fill="#F7F8FA", stroke="#CCCCCC", sw=1.2))
    p.append(R(10, py0, 196, 22, rx=6, fill=PILL_GREY, stroke="none", sw=0))
    p.append(t(108, py0 + 15, "evaluation protocol (frozen)", size=8.8, weight="bold", fill="#FFFFFF", maxw=184))
    items = [
        ("test", "DENTEX official validation: 50 images (46 pathological, 182 target teeth; 4 healthy), used once"),
        ("dev", "all choices on a fixed dev split of ~70 training images, drawn once with a fixed seed"),
        ("metrics", "set-matched pooled F1 (exact, tooth-only) · invented findings on healthy images · no-answer rate · ECE"),
        ("uncertainty", "bootstrap over images; paired comparisons on the images conditions share"),
    ]
    yy = py0 + 40
    for k, v in items:
        p.append(t(22, yy, k, size=8.8, weight="bold", fill=C_BLUE_D, anchor="start", maxw=70))
        p.append(t(96, yy, v, size=8.6, anchor="start", maxw=526))
        yy += 15
    p.append(t(22, yy + 2, "probes on the trained checkpoint (nothing retrained):", size=8.8, weight="bold", fill=C_ORANGE_D, anchor="start", maxw=300))
    p.append(t(22, yy + 15, "tool removal · grounding perturbation · group-size sweep K ∈ {1, 2, 4, 8, 16} on dev", size=8.6, anchor="start", maxw=596))
    if warn:
        print("  design text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, py0 + ph + 10, "".join(p))


def build_perturbation_mechanism(a):
    """Tiered-perturbation mechanism + nudge_crop loop (was a hand-authored SVG).

    Moved into the builder framework on a 640-unit canvas (labels print at
    ~6.5-8.4 pt instead of ~5 pt). Wording, rates and offsets are copied
    from the previous hand-authored figure; geometry is new.
    """
    W = 640
    warn: list[str] = []
    t = _checked(warn)
    G = "#666666"
    BLUE_F, BLUE_S = "#F0F4FA", "#4C72B0"
    ORA_F, ORA_S = "#FBF6F0", "#DD8452"
    GRN_F, GRN_S = "#F2F8F2", "#55A868"
    NOTE_F, NOTE_S = "#FFFBF0", "#BCBD22"
    TXT, SUB = "#222222", "#444444"
    p = []
    p.append(t(W / 2, 22, "Self-correcting grounding: tiered perturbation and the nudge_crop correction loop", size=12.4, weight="bold", fill=TXT, maxw=620))

    def box(x, y, w, h, fill, stroke, head, lines, head_size=11.2, size=9.0, pitch=13.5):
        p.append(R(x, y, w, h, rx=8, fill=fill, stroke=stroke, sw=1.5))
        cx = x + w / 2
        p.append(t(cx, y + 21, head, size=head_size, weight="bold", fill=TXT, maxw=w - 16))
        yy = y + 38
        for ln in lines:
            p.append(t(cx, yy, ln, size=size, fill=SUB, maxw=w - 16))
            yy += pitch

    # ---- row 1: locate_tooth -> perturbation roll ----
    y1, h1 = 36, 62
    box(10, y1, 285, h1, BLUE_F, BLUE_S, "locate_tooth(FDI tooth)",
        ["returns a reference box (expert-verified box for", "annotated teeth; detector box otherwise)"])
    p.append(arrow([(295, y1 + h1 / 2), (321, y1 + h1 / 2)], color=G, sw=1.5))
    box(323, y1, 307, h1, ORA_F, ORA_S, "Perturbation roll (fixed rates)",
        ["applied to the box SHOWN to the policy,", "never to the logged reference box"])

    # ---- row 2: three tiers ----
    xs = [109, 319, 529]
    bus = y1 + h1 + 18
    p.append(L(476.5, y1 + h1, 476.5, bus, color=G, sw=1.5))
    p.append(L(xs[0], bus, xs[2], bus, color=G, sw=1.5))
    y2, h2 = bus + 20, 100
    for cx in xs:
        p.append(arrow([(cx, bus), (cx, y2 - 1)], color=G, sw=1.5))
    box(10, y2, 198, h2, "#F4F4F4", "#999999", "none - 45%",
        ["box unchanged", "clean accept-and-proceed", "demonstration"])
    box(220, y2, 198, h2, "#FDF3EA", ORA_S, "small - 25%",
        ["offset 12-28% of box size, random sign", "genuine judgment call: often still fine",
         "given zoom padding - never resolved", "by the prompt"])
    box(430, y2, 198, h2, "#FDEBE6", "#C44E52", "big - 30%",
        ["offset 45-75% of box size, random sign", "visually self-evident without any", "verbal cue"])

    # ---- merge -> teacher -> nudge -> zoom ----
    mb = y2 + h2 + 16
    for cx in (xs[0], xs[2]):
        p.append(L(cx, y2 + h2, cx, mb, color=G, sw=1.5))
    p.append(L(xs[0], mb, xs[2], mb, color=G, sw=1.5))
    y3, h3 = mb + 18, 50
    p.append(L(319, mb, 319, y3 - 10, color=G, sw=1.5))
    p.append(arrow([(319, y3 - 12), (319, y3 - 1)], color=G, sw=1.5))
    box(119, y3, 400, h3, BLUE_F, BLUE_S, "Teacher inspects the crop (zoom_crop)",
        ["and decides from the pixels alone whether the box is acceptable"])
    y4, h4 = y3 + h3 + 18, 44
    p.append(arrow([(319, y3 + h3), (319, y4 - 1)], color=G, sw=1.5))
    box(169, y4, 300, h4, GRN_F, GRN_S, "nudge_crop", ["dx_frac, dy_frac, scale"])
    y5, h5 = y4 + h4 + 18, 58
    p.append(arrow([(319, y4 + h4), (319, y5 - 1)], color=G, sw=1.5))
    box(109, y5, 420, h5, GRN_F, GRN_S, "zoom_crop on the corrected box - verify - commit finding",
        ["the same judgment the deployed policy faces at inference,", "with no ground truth available"],
        head_size=10.6)

    # ---- row: side notes + schematic ----
    y6, h6 = y5 + h5 + 20, 98
    for x0, head, lines in (
        (10, "Decoupled from the detector",
         ["rates are configured constants,", "not the detector's real error rate,", "so demonstrations stay valid as", "the grounding tool improves."]),
        (220, "No verbal cue",
         ["an error is never announced in text;", "magnitude alone must carry the", "signal, so the learned behavior", "survives to GRPO and inference."]),
    ):
        p.append(R(x0, y6, 198, h6, rx=8, fill=NOTE_F, stroke=NOTE_S, sw=1.2))
        p.append(t(x0 + 99, y6 + 20, head, size=9.8, weight="bold", fill=TXT, maxw=184))
        for i, ln in enumerate(lines):
            p.append(t(x0 + 99, y6 + 38 + i * 13.5, ln, size=9.0, fill=SUB, maxw=184))
    sx = 430
    p.append(R(sx, y6, 198, h6, rx=8, fill="#FAFAFA", stroke="#CCCCCC", sw=1.2))
    p.append(t(sx + 99, y6 + 17, "reference vs. shown box", size=9.0, fill=SUB, maxw=184))
    p.append(R(sx + 44, y6 + 28, 66, 40, rx=0, fill="none", stroke="#2ca02c", sw=2))
    p.append(R(sx + 70, y6 + 46, 66, 40 - 6, rx=0, fill="none", stroke="#d62728", sw=2, dash="6,3"))
    p.append(arrow([(sx + 82, y6 + 50), (sx + 104, y6 + 66)], color="#444444", sw=1.2))
    p.append(t(sx + 99, y6 + 94, "shift given in box-relative fractions", size=8.6, fill=SUB, maxw=190))

    if warn:
        print("  perturbation_mechanism text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, y6 + h6 + 12, "".join(p))


BUILDERS = {
    "fig_traces_evidence.svg": build_traces_evidence,
    "fig_traces_comparison.svg": build_traces_comparison,
    "fig_trace_generation.svg": build_trace_generation,
    "fig_trace_verification.svg": build_trace_verification,
    "fig_architecture.svg": build_architecture,
    "fig_architecture_training.svg": build_architecture_training,
    "fig_corpus_composition.svg": build_corpus_composition,
    "fig_datasets.svg": build_dataset_catalog,
    "fig_grounding_cv.svg": build_grounding_cv,
    "fig_input_geometry.svg": build_input_geometry,
    "fig_concept.svg": build_concept,
    "fig_gallery.svg": build_gallery,
    "fig_design.svg": build_design,
    "fig_pipeline.svg": build_pipeline,
    "fig_perturbation_mechanism.svg": build_perturbation_mechanism,
}


# y (in the builder's own coordinates) where each figure's real content starts.
# Everything above it is the in-figure title/subtitle: journals expect the
# caption to carry that, so it is removed and the content shifted up.
CONTENT_TOP = {
    "fig_architecture.svg": 12,
    "fig_architecture_training.svg": 2,
    "fig_corpus_composition.svg": 50,
    "fig_datasets.svg": 58,
    "fig_grounding_cv.svg": 50,
    "fig_input_geometry.svg": 50,
    "fig_pipeline.svg": 36,
    "fig_perturbation_mechanism.svg": 34,
}


def strip_title(svg_text: str, content_top: float) -> str:
    """Drop <text> above content_top and shift the figure up (margin 8 units)."""
    import re

    shift = content_top - 8
    out = re.sub(
        r'<text x="[-\d.]+" y="([-\d.]+)"[^>]*>[^<]*</text>',
        lambda m: "" if float(m.group(1)) < content_top - 2 else m.group(0),
        svg_text,
    )
    m = re.match(r'(<svg [^>]*?)width="([\d.]+)" height="([\d.]+)" viewBox="0 0 ([\d.]+) ([\d.]+)">(<rect [^>]*/>)', out)
    w, h = float(m.group(2)), float(m.group(3))
    head = f'{m.group(1)}width="{w:g}" height="{h - shift:g}" viewBox="0 0 {w:g} {h - shift:g}">'
    bg = f'<rect width="{w:g}" height="{h - shift:g}" fill="#FFFFFF"/>'
    rest = out[m.end():]
    assert rest.endswith("</svg>")
    return head + bg + f'<g transform="translate(0,{-shift:g})">' + rest[: -len("</svg>")] + "</g></svg>"


def main() -> None:
    force = "--force" in sys.argv
    a = load_assets(force=force)
    print(f"assets: source {a['source_file']} ({a['native_w']}x{a['native_h']})")
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in BUILDERS.items():
        out = SVG_DIR / name
        text = builder(a)
        if name in CONTENT_TOP:
            text = strip_title(text, CONTENT_TOP[name])
        out.write_text(text, encoding="utf-8")
        print(f"  wrote {out.name} ({out.stat().st_size / 1024:.0f} KB)")

    from generate_paper_assets import convert_diagrams

    convert_diagrams(PAPER_DIR)

    if "--preview" in sys.argv:
        import pypdfium2 as pdfium

        prev = PAPER_DIR.parent / "scratch" / "preview"
        prev.mkdir(parents=True, exist_ok=True)
        for name in BUILDERS:
            pdf = PAPER_DIR / "figures" / "pdf" / name.replace(".svg", ".pdf")
            doc = pdfium.PdfDocument(str(pdf))
            im = doc[0].render(scale=1.5).to_pil()
            im.save(prev / f"diagram_{name.replace('.svg', '')}.png")
            print(f"  preview scratch/preview/diagram_{name.replace('.svg', '')}.png")


if __name__ == "__main__":
    main()
