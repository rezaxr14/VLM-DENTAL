"""Build the three redesigned paper diagrams with embedded real radiograph assets.

Generates (into paper/figures/svg/, then converts to PDF):
- fig_traces_comparison.svg : paired with-tools vs no-tools traces for DENTEX
  image 624, with a real radiograph panel showing the actual box trajectory
  parsed from the released trace observations.
- fig_trace_multistage.svg  : the generate -> verify -> decontaminate/release
  pipeline with icons, mini-visuals and data-driven bars.
- fig_architecture.svg      : system architecture with real image assets,
  numbered flow badges and orthogonal connectors.

All numbers are measured corpus statistics; all radiograph pixels come from
the local DENTEX copy (image 624 = train_318.png). No downloads (Rule 16).
"""

from __future__ import annotations

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
    scaled_box,
    svg,
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


def build_traces_comparison(a):
    """Traces comparison on a 760-unit canvas.

    Top: the real radiograph + evidence trail (zoom inset, final crop).
    Bottom: no-tools trace (left) and with-tools trace (right). The old
    1060x850 three-panel layout printed at ~3.3 pt; this one prints at
    ~5.3-7 pt (the densest figure in the paper: ~110 text lines).
    """
    from reportlab.pdfbase.pdfmetrics import stringWidth

    W0, H0 = a["native_w"], a["native_h"]
    gt, loc = BOXES_624["gt"], BOXES_624["loc"]
    n1, n2, n3 = BOXES_624["n1"], BOXES_624["n2"], BOXES_624["n3"]
    W = 760
    warn: list[str] = []
    t = _checked(warn)
    p = []
    p.append(TC(W / 2, 22, "Same image, same ground truth: two training traces", size=13, weight="bold"))
    p.append(t(W / 2, 38, "DENTEX image 624 (train_318.png) · FDI 34 · Caries · quotes abridged from the released trace", size=9.0, fill=C_SUB, maxw=700))

    # ---------------- top panel: image + evidence trail ----------------
    ay, ah = 50, 290
    p.append(R(10, ay, 740, ah, fill="#F7F9FC", stroke=C_BLUE, sw=1.4))
    p.append(t(24, ay + 20, "The image and the evidence trail", size=11.5, weight="bold", fill=C_BLUE_D, anchor="start"))
    p.append(t(236, ay + 20, "real radiograph + real returned boxes", size=8.8, fill=C_SUB, anchor="start"))

    ox, oy, ow, oh = 24, ay + 34, 380, 178
    p.append(IMG(ox, oy, ow, oh, a["opg"]))
    p.append(R(ox, oy, ow, oh, rx=2, stroke="#666666", sw=1.1, fill="none"))
    for box, color, sw, dash in (
        (gt, "#2ca02c", 1.6, "4,2.5"),
        (loc, C_BLUE, 1.6, None),
        (n1, "#FFC287", 1.0, None),
        (n2, "#FF9A4D", 1.1, None),
        (n3, C_ORANGE, 1.6, None),
    ):
        bx, by, bw, bh = scaled_box(box, (0, 0, W0, H0), ox, oy, ow, oh)
        p.append(R(bx, by, bw, bh, rx=0, fill="none", stroke=color, sw=sw, dash=dash))

    # magnifier inset + final crop (region aspect-matched)
    ih = 230
    iw, cw = 98, 68
    ix, iy = 470, ay + 38
    cx = 640
    ssx, ssy = ow / W0, oh / H0  # INSET_REGION is (x0, y0, x1, y1) in native pixels
    rx_ = ox + INSET_REGION[0] * ssx
    ry_ = oy + INSET_REGION[1] * ssy
    rw_ = (INSET_REGION[2] - INSET_REGION[0]) * ssx
    rh_ = (INSET_REGION[3] - INSET_REGION[1]) * ssy
    p.append(R(rx_, ry_, rw_, rh_, rx=0, fill="none", stroke="#9A9A9A", sw=0.8, dash="2.5,2"))
    p.append(L(rx_ + rw_, ry_, ix, iy, color="#C9C9C9", sw=0.9, dash="2.5,2.5"))
    p.append(L(rx_ + rw_, ry_ + rh_, ix, iy + ih, color="#C9C9C9", sw=0.9, dash="2.5,2.5"))
    p.append(t(ix, iy - 5, "4.5× zoom", size=8.8, fill="#555555", anchor="start"))
    p.append(IMG(ix, iy, iw, ih, a["inset"]))
    p.append(R(ix, iy, iw, ih, rx=4, stroke="#666666", sw=1.2, fill="none"))
    for box, color, sw, dash in (
        (gt, "#2ca02c", 1.8, "5,3"),
        (loc, C_BLUE, 1.8, None),
        (n1, "#FFC287", 1.1, None),
        (n2, "#FF9A4D", 1.3, None),
        (n3, C_ORANGE, 2.0, None),
    ):
        bx, by, bw, bh = scaled_box(box, INSET_REGION, ix, iy, iw, ih)
        p.append(R(bx, by, bw, bh, rx=0, fill="none", stroke=color, sw=sw, dash=dash))
    # walk arrow (loc, n1, n2, n3 share centre x ~ 0.538 of the inset width)
    wx = ix + 0.538 * iw
    p.append(arrow([(wx, iy + 0.941 * ih), (wx, iy + 0.163 * ih)], color=C_ORANGE, sw=1.5))
    p.append(t(cx, iy - 5, "final crop · turn 10", size=8.8, fill="#555555", anchor="start", maxw=110))
    p.append(IMG(cx, iy, cw, ih, a["crop_final"]))
    p.append(R(cx, iy, cw, ih, rx=4, stroke="#666666", sw=1.2, fill="none"))

    # legend + real coordinates under the radiograph
    ly = oy + oh + 18
    p.append(L(24, ly - 3, 48, ly - 3, color=C_BLUE, sw=2.4))
    p.append(t(54, ly, "locate_tooth(34) box", size=8.8, anchor="start", maxw=150))
    p.append(L(220, ly - 3, 228, ly - 3, color="#FFC287", sw=2.4))
    p.append(L(228, ly - 3, 236, ly - 3, color="#FF9A4D", sw=2.4))
    p.append(L(236, ly - 3, 244, ly - 3, color=C_ORANGE, sw=2.4))
    p.append(t(250, ly, "nudges 1-3, light to dark", size=8.8, anchor="start", maxw=150))
    p.append(L(24, ly + 13, 48, ly + 13, color="#2ca02c", sw=2.4, dash="4,2.5"))
    p.append(t(54, ly + 16, "ground-truth box", size=8.8, anchor="start", maxw=150))
    p.append(t(250, ly + 16, "coords as returned", size=8.8, fill=C_SUB, anchor="start", maxw=150))
    p.append(t(24, ly + 34, "[1668,917,84,322] - [1685,885,50,193]", size=8.2, fill=C_SUB, anchor="start", font=MONO, maxw=380))
    p.append(t(24, ly + 46, "- [1693,818,35,135] - [1696,750,28,108]", size=8.2, fill=C_SUB, anchor="start", font=MONO, maxw=380))

    # ---------------- bottom-left: no-tools trace ----------------
    by0 = ay + ah + 10
    bh = 424
    bx0, bw = 10, 300
    bcx = bx0 + bw / 2
    p.append(R(bx0, by0, bw, bh, fill="#FBFBFB", stroke="#BBBBBB", sw=1.2))
    p.append(t(bcx, by0 + 22, "No-tools trace (baseline)", size=11.5, weight="bold", fill="#555555"))
    p.append(t(bcx, by0 + 36, "1 turn · 0 tool calls · hindsight description", size=8.8, fill=C_SUB, maxw=280))
    p.append(R(bx0 + 14, by0 + 46, bw - 28, 40, fill="#E8EEF7", stroke=C_BLUE, sw=1.1))
    p.append(t(bcx, by0 + 62, "user: panoramic X-ray, tool-free prompt", size=9.2, weight="bold", fill=C_BLUE_D, maxw=260))
    p.append(t(bcx, by0 + 77, "“identify any abnormal teeth and diagnose”", size=8.6, fill="#555555", maxw=260))
    p.append(arrow([(bcx, by0 + 86), (bcx, by0 + 98)], color="#999999", sw=1.3))
    tb_y, tb_h = by0 + 100, 182
    p.append(R(bx0 + 14, tb_y, bw - 28, tb_h, fill="#FFFFFF", stroke="#CCCCCC", sw=1.1))
    p.append(t(bcx, tb_y + 19, "thought (single turn)", size=9.8, weight="bold"))
    paras = [
        "“On systematic review of this panoramic radiograph, I begin by orienting to the patient’s right and left sides as marked. The image shows a mixed dentition with multiple developing permanent teeth ...",
        "Focusing on the lower-left quadrant (Q3), I examine the posterior region ... I can see a radiolucent area involving the occlusal/cervical portion of the tooth, with loss of normal enamel contour and a darker zone consistent with a carious lesion ...”",
    ]
    yy = tb_y + 36
    for i, para in enumerate(paras):
        for ln in _wrap(para, 8.6, bw - 52):
            p.append(t(bx0 + 26, yy, ln, size=8.6, fill="#444444", anchor="start", maxw=bw - 50))
            yy += 11.8
        yy += 5
    yy += 4
    for ln in ("One-shot description from the full image:", "no crop, no second look, nothing checkable", "against the pixels it actually saw."):
        p.append(t(bx0 + 26, yy, ln, size=8.4, fill="#777777", anchor="start", style="italic", maxw=bw - 50))
        yy += 11.8
    fy = tb_y + tb_h + 12
    p.append(arrow([(bcx, tb_y + tb_h), (bcx, fy)], color="#999999", sw=1.3))
    p.append(R(bx0 + 14, fy + 2, bw - 28, 42, fill="#F2F8F2", stroke=C_GREEN, sw=1.3))
    p.append(t(bcx, fy + 19, "final answer", size=9.8, weight="bold", fill=C_GREEN_D))
    p.append(t(bcx, fy + 35, "Q3T4 · Caries · confidence 0.82", size=9.0, fill="#444444"))
    ty_ = fy + 54
    p.append(R(bx0 + 14, ty_, bw - 28, bx0 + by0 + bh - ty_ - 12 - bx0, fill="#FBF6F0", stroke="#DD8452", sw=1.1))
    p.append(t(bcx, ty_ + 16, "What this arm can teach", size=9.8, weight="bold", fill="#8A5A2E"))
    p.append(t(bx0 + 26, ty_ + 31, "- output format + clinical vocabulary", size=8.8, fill="#555555", anchor="start", maxw=bw - 50))
    p.append(t(bx0 + 26, ty_ + 43, "- quadrant-survey habit", size=8.8, fill="#555555", anchor="start", maxw=bw - 50))
    p.append(t(bcx, ty_ + 60, "No evidence trail to verify or correct.", size=9.0, weight="bold", fill=C_RED, maxw=bw - 50))

    # ---------------- bottom-right: with-tools trace ----------------
    cx0, cw0 = 320, 430
    ccx = cx0 + cw0 / 2
    p.append(R(cx0, by0, cw0, bh, fill="#FCFCFD", stroke=C_BLUE, sw=1.4))
    p.append(t(ccx, by0 + 22, "With-tools trace (main policy)", size=11.5, weight="bold", fill=C_BLUE_D))
    p.append(t(ccx, by0 + 36, "23 turns · 22 real tool calls · 3 corrective nudges", size=8.8, fill=C_SUB, maxw=410))
    rows = [
        ("T0", "window_level", "survey the full field at high contrast", 0),
        ("T1", "fdi_label", "orient to FDI notation first", 0),
        ("T2", "locate_tooth(34)", "box [1668, 917, 84, 322] (GT-grounded detector)", 0),
        ("T3", "zoom_crop", "“shows only bone and root area ... too low”", 0),
        ("T4", "nudge_crop #1", "dy -0.3 · scale 0.6 -> [1685, 885, 50, 193]", 1),
        ("T6", "nudge_crop #2", "dy -0.5 · scale 0.7 -> [1693, 818, 35, 135]", 1),
        ("T8", "nudge_crop #3", "dy -0.6 · scale 0.8 -> [1696, 750, 28, 108]", 1),
        ("T9-10", "zoom + denoise", "“Excellent! Now I can clearly see tooth 34”", 0),
        ("T14+", "compare + enhance", "contralateral check, contrast pass, re-verify", 0),
    ]
    ry = by0 + 46
    for turn, name, sub, hot in rows:
        fill = C_ORANGE_L if hot else "#FFFFFF"
        stroke = "#F0B27A" if hot else "#DDDDDD"
        p.append(R(cx0 + 12, ry, cw0 - 24, 25, rx=5, fill=fill, stroke=stroke, sw=1.0))
        p.append(R(cx0 + 12, ry, 4, 25, rx=2, fill=C_ORANGE if hot else C_BLUE, stroke="none", sw=0))
        tcol = "#B45A04" if hot else C_BLUE_D
        p.append(t(cx0 + 24, ry + 17, turn, size=8.8, weight="bold", fill=tcol, anchor="start"))
        p.append(t(cx0 + 64, ry + 17, name, size=9.4, weight="bold", fill=C_TXT, anchor="start"))
        nx = cx0 + 64 + stringWidth(name, "Helvetica-Bold", 9.4) + 10
        p.append(t(nx, ry + 17, sub, size=8.6, fill=C_SUB, anchor="start", maxw=cx0 + cw0 - 20 - nx))
        ry += 30
    fa = ry + 2
    p.append(R(cx0 + 12, fa, cw0 - 24, 44, rx=6, fill="#F2F8F2", stroke=C_GREEN, sw=1.4))
    p.append(t(ccx, fa + 19, "final answer (turn 22) · Q3T4 · Caries · conf 0.82", size=9.6, weight="bold", fill=C_GREEN_D, maxw=400))
    p.append(t(ccx, fa + 35, "matches ground truth - passed strict verification", size=8.8, fill=C_SUB, maxw=400))
    sm = fa + 52
    p.append(R(cx0 + 12, sm, cw0 - 24, by0 + bh - sm - 12, rx=6, fill="#F7F9FC", stroke="#C9D4E4", sw=1.0))
    p.append(t(ccx, sm + 20, "22 real tool calls · 3 corrective · every observation", size=8.8, fill=C_BLUE_D, maxw=400))
    p.append(t(ccx, sm + 33, "executed against the actual radiograph", size=8.8, fill=C_BLUE_D, maxw=400))

    # ---------------- bottom band ----------------
    dy = by0 + bh + 10
    p.append(R(10, dy, 740, 96, fill="#F7F8FA", stroke="#DDDDDD", sw=1.1))
    p.append(t(24, dy + 22, "What the exact 1:1 pairing isolates", size=12, weight="bold", anchor="start"))
    p.append(R(380, dy + 8, 358, 22, rx=11, fill="#EEF2FA", stroke=C_BLUE, sw=1.1))
    p.append(t(559, dy + 23, "5 cohorts × 2 arms · 1,847 + 1,847 · 3,694 verified traces", size=8.8, weight="bold", fill=C_BLUE_D, maxw=346))
    cols = [
        (24, dy + 48, ["same image · same prompt template · same ground truth -", "only the tool channel differs"]),
        (390, dy + 48, ["any SFT delta is attributable to demonstrated tool use,", "not to data volume or prompt wording"]),
        (24, dy + 76, ["with-tools traces carry 44,350 executed tool calls;", "the no-tools arm carries zero"]),
        (390, dy + 76, ["per-cohort counts match exactly (678/27/660/202/280", "on each side) - parity by construction"]),
    ]
    for bx, by_, lines in cols:
        for i, ln in enumerate(lines):
            p.append(t(bx, by_ + i * 12 - (6 if i == 0 and False else 0), ln, size=9.0, fill="#444444", anchor="start", maxw=340))
    if warn:
        print("  traces_comparison text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, dy + 108, "".join(p))


# ---------------------------------------------------------------------------
# Figure 2: multi-stage trace synthesis (generate -> verify -> release)
# ---------------------------------------------------------------------------


def build_multistage(a):
    """Trace synthesis (generate -> verify -> decontaminate) on a 780-unit canvas.

    Three full-width bands of three cards each. The old 1080x870 canvas
    printed ~3 pt labels; here labels print at ~5-6.5 pt. Every label
    carries a width limit derived from its card and is checked with font
    metrics at build time (overflow is printed).
    """
    W = 780
    warn: list[str] = []
    t = _checked(warn)
    p = []
    G = "#999999"
    p.append(TC(W / 2, 22, "Trace synthesis: generate, verify, decontaminate", size=13, weight="bold"))
    p.append(t(W / 2, 38, "all counts measured on the released corpus - nothing projected", size=9.2, fill=C_SUB, maxw=700))

    def band(y, h, fill, stroke, pill_fill, pill_text, pill_w, right_text, right_col):
        p.append(R(10, y, 760, h, fill=fill, stroke=stroke, sw=1.2))
        p.append(R(10, y, pill_w, 22, rx=6, fill=pill_fill, stroke="none", sw=0))
        p.append(t(10 + pill_w / 2, y + 15, pill_text, size=8.8, weight="bold", fill="#FFFFFF", maxw=pill_w - 12))
        p.append(t(758, y + 15, right_text, size=8.6, fill=right_col, anchor="end", maxw=520))

    def title(x, y, icon, text, w):
        p.append(icon)
        p.append(t(x + 38, y + 24, text, size=10.4, weight="bold", anchor="start", maxw=w - 50))

    # ===================== band A: generation =====================
    ya = 50
    ca = ya + 34
    cha = 244
    ha = 34 + cha + 12
    band(ya, ha, C_BLUE_FILL, "#C7D3E8", C_BLUE, "STAGE 1 · GENERATION", 176, "provider family A: generators", C_BLUE_D)

    # inputs card
    ix, iw = 22, 160
    p.append(R(ix, ca, iw, cha))
    title(ix, ca, icon_db(ix + 12, ca + 10), "Expert labels", iw)
    p.append(IMG(ix + 14, ca + 38, 132, 62, a["opg"]))
    p.append(R(ix + 14, ca + 38, 132, 62, rx=2, stroke="#666666", sw=1.0, fill="none"))
    p.append(t(ix + 14, ca + 116, "DENTEX · Tufts loaders", size=8.8, fill=C_SUB, anchor="start", maxw=iw - 24))
    p.append(t(ix + 14, ca + 129, "FDI conversion at ingest", size=8.8, fill=C_SUB, anchor="start", maxw=iw - 24))
    p.append(R(ix + 14, ca + 142, 132, 46, fill="#F0F7F0", stroke=C_GREEN, sw=1.1))
    p.append(t(ix + 80, ca + 160, "ground truth per image", size=8.6, weight="bold", fill=C_GREEN_D, maxw=122))
    p.append(t(ix + 80, ca + 175, "FDI 34 · Caries · bbox", size=8.4, fill=C_GREEN_D, maxw=122))
    p.append(arrow([(ix + iw, ca + 110), (ix + iw + 18, ca + 110)], color=G, sw=1.5))

    # loop card
    lx, lw = ix + iw + 20, 312
    p.append(R(lx, ca, lw, cha))
    p.append(t(lx + 12, ca + 22, "Ground-truth-directed LangGraph loop", size=10.2, weight="bold", anchor="start", maxw=lw - 24))
    py = ca + 32
    p.append(R(lx + 12, py, 58, 22, rx=11, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(t(lx + 41, py + 15, "thought", size=8.6, fill=C_BLUE_D, maxw=50))
    p.append(arrow([(lx + 71, py + 11), (lx + 81, py + 11)], color=G, sw=1.1))
    p.append(R(lx + 82, py, 68, 22, rx=11, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(t(lx + 116, py + 15, "tool call", size=8.6, fill=C_BLUE_D, maxw=60))
    p.append(arrow([(lx + 151, py + 11), (lx + 161, py + 11)], color=G, sw=1.1))
    p.append(R(lx + 162, py, 82, 22, rx=11, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(t(lx + 203, py + 15, "observation", size=8.6, fill=C_BLUE_D, maxw=74))
    p.append(arrow([(lx + 203, py + 22), (lx + 203, py + 40), (lx + 41, py + 40), (lx + 41, py + 23)], color="#B9C6DC", sw=1.2))
    p.append(t(lx + 122, py + 35, "repeat every turn", size=8.4, fill="#6F86AD", maxw=140))
    # generation pool (stacked chips) | zoom monitor
    gy = py + 56
    p.append(t(lx + 12, gy, "generation pool:", size=8.6, fill=C_SUB, anchor="start", maxw=140))
    cy_ = gy + 6
    for cwid, label in ((104, "Gemini 3.5 Flash Lite"), (66, "MiniMax M3"), (66, "NVIDIA NIM"), (66, "local vLLM")):
        p.append(R(lx + 12, cy_, cwid, 17, rx=8.5, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
        p.append(t(lx + 12 + cwid / 2, cy_ + 12, label, size=8.2, fill=C_BLUE_D, maxw=cwid - 8))
        cy_ += 21
    mx, my, mw, mh = lx + 168, gy - 8, 132, 84
    p.append(R(mx, my, mw, mh, rx=4, fill="#FAFBFD", stroke="#DDDDDD", sw=1.0))
    p.append(IMG(mx + 10, my + 8, 24, 56, a["crop_located"]))
    p.append(R(mx + 10, my + 8, 24, 56, rx=1, stroke="#888888", sw=0.8, fill="none"))
    p.append(t(mx + 66, my + 30, "nudge", size=8.2, fill=C_ORANGE_D, maxw=40))
    p.append(arrow([(mx + 40, my + 42), (mx + 92, my + 42)], color=C_ORANGE, sw=1.6))
    p.append(IMG(mx + 98, my + 8, 16, 56, a["crop_final"]))
    p.append(R(mx + 98, my + 8, 16, 56, rx=1, stroke="#888888", sw=0.8, fill="none"))
    p.append(t(mx + mw / 2, my + 78, "before / after", size=8.2, fill="#777777", maxw=mw - 8))
    # perturbation callout
    qy = ca + 188
    p.append(R(lx + 12, qy, lw - 24, 46, rx=6, fill=C_ORANGE_L, stroke="#F0B27A", sw=1.1))
    p.append(t(lx + 22, qy + 14, "tiered display perturbation, independent of the true box:", size=8.6, fill=C_ORANGE_D, anchor="start", maxw=lw - 44))
    p.append(t(lx + 22, qy + 27, "45% small · 25% big · 30% none -", size=8.4, fill=C_ORANGE_D, anchor="start", maxw=lw - 44))
    p.append(t(lx + 22, qy + 39, "keeps accept-vs-nudge demonstrable", size=8.4, fill=C_ORANGE_D, anchor="start", maxw=lw - 44))
    p.append(arrow([(lx + lw, ca + 110), (lx + lw + 18, ca + 110)], color=G, sw=1.5))

    # unverified pool card
    ux, uw = lx + lw + 20, 232
    p.append(R(ux, ca, uw, cha))
    title(ux, ca, icon_doc(ux + 12, ca + 10), "Unverified trace pool", uw)
    p.append(R(ux + 12, ca + 36, uw - 24, 64, rx=4, fill="#F7F8FA", stroke="#E3E3E3", sw=1.0))
    snip = [
        'T4 thought "crop shows only bone and',
        '         root area ... too low"',
        "T4 call    nudge_crop(dy=-0.30, s=0.60)",
        "T5 obs     bbox -> [1685, 885, 50, 193]",
    ]
    yy = ca + 52
    for ln in snip:
        p.append(t(ux + 20, yy, ln, size=7.8, fill="#444444", anchor="start", font=MONO, maxw=uw - 40))
        yy += 13
    p.append(R(ux + 12, ca + 112, uw - 24, 22, rx=11, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(t(ux + uw / 2, ca + 127, "44,350 executed tool calls", size=8.8, weight="bold", fill=C_BLUE_D, maxw=uw - 36))
    p.append(R(ux + 12, ca + 142, uw - 24, 22, rx=11, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(t(ux + uw / 2, ca + 157, "2,292 corrective nudges", size=8.8, weight="bold", fill=C_BLUE_D, maxw=uw - 36))
    p.append(t(ux + uw / 2, ca + 184, "no training value until verified", size=8.6, fill=C_SUB, maxw=uw - 24))

    # ===================== band B: verification =====================
    yb = ya + ha + 12
    cb = yb + 46
    chb = 196
    hb = 46 + chb + 12
    band(yb, hb, C_GREEN_FILL, "#BFDCC6", C_GREEN, "STAGE 2 · VERIFICATION", 176, "provider family B - never the generator pool", C_GREEN_D)

    vx, vw = 22, 220
    p.append(R(vx, cb, vw, chb))
    title(vx, cb, icon_shield(vx + 14, cb + 12), "Strict verifier", vw)
    p.append(t(vx + 14, cb + 44, "staged cross-family check per trace:", size=8.4, fill=C_SUB, anchor="start", maxw=vw - 24))
    items = [
        (cb + 66, "diagnosis set-matches the full", "multi-finding ground truth"),
        (cb + 96, "evidence trail consistent - no", "claims on healthy tissue"),
        (cb + 126, "no directive leakage in CoT", None),
        (cb + 144, "format + FDI notation valid", None),
    ]
    for iy_, l1, l2 in items:
        p.append(icon_check(vx + 14, iy_ - 5.5))
        p.append(t(vx + 28, iy_, l1, size=8.6, anchor="start", maxw=vw - 38))
        if l2:
            p.append(t(vx + 28, iy_ + 12, l2, size=8.6, anchor="start", maxw=vw - 38))
    p.append(t(vx + 14, cb + 176, "strict rate pacing via the provider pool", size=8.2, fill=C_SUB, anchor="start", maxw=vw - 24))

    rx_, rw_ = vx + vw + 45, 190
    p.append(arrow([(vx + vw, cb + 96), (rx_ - 2, cb + 96)], color=C_RED, sw=1.6))
    p.append(t((vx + vw + rx_) / 2, cb + 80, "REJECT", size=8.2, weight="bold", fill=C_RED, maxw=42))
    p.append(t((vx + vw + rx_) / 2, cb + 90, "· 92", size=8.2, weight="bold", fill=C_RED, maxw=42))
    p.append(R(rx_, cb, rw_, chb))
    title(rx_, cb, icon_gear(rx_ + 14, cb + 12), "Repair loop", rw_)
    p.append(t(rx_ + 14, cb + 44, "editor rewrites the trace,", size=8.4, fill=C_SUB, anchor="start", maxw=rw_ - 24))
    p.append(t(rx_ + 14, cb + 57, "keeping the valid CoT prefix:", size=8.4, fill=C_SUB, anchor="start", maxw=rw_ - 24))
    for ry_, txt in ((cb + 82, "fix tool-arg / FDI slips"), (cb + 102, "drop phantom claims"), (cb + 122, "excise directive leakage")):
        p.append(R(rx_ + 14, ry_ - 5, 5, 5, rx=1, fill=C_ORANGE, stroke="none", sw=0))
        p.append(t(rx_ + 26, ry_, txt, size=8.6, anchor="start", maxw=rw_ - 38))
    p.append(t(rx_ + 14, cb + 160, "re-verified after every repair", size=8.2, fill=C_SUB, anchor="start", maxw=rw_ - 24))
    # re-verify loop above the cards
    lane = yb + 38
    p.append(arrow([(rx_ + rw_ / 2, cb), (rx_ + rw_ / 2, lane), (vx + vw / 2, lane), (vx + vw / 2, cb - 1)], color="#7FA98B", sw=1.3))
    p.append(t((rx_ + rw_ / 2 + vx + vw / 2) / 2, lane - 4, "re-verify until pass", size=8.4, fill=C_GREEN_D, maxw=200))

    ax, aw = rx_ + rw_ + 45, 236
    p.append(arrow([(rx_ + rw_, cb + 96), (ax - 2, cb + 96)], color="#BBBBBB", sw=1.2, dash="3,2.5"))
    p.append(t((rx_ + rw_ + ax) / 2, cb + 88, "logged", size=8.2, fill=C_SUB, maxw=42))
    p.append(R(ax, cb, aw, chb))
    title(ax, cb, icon_magnifier(ax + 18, cb + 18, r=4.5), "First-pass audit", aw)
    p.append(t(ax + 14, cb + 46, "92 rejections on 3,694 traces", size=8.8, weight="bold", anchor="start", maxw=aw - 24))
    p.append(t(ax + 14, cb + 58, "(97.5% pass)", size=8.8, weight="bold", anchor="start", maxw=aw - 24))
    bars = [
        (cb + 82, "hallucinated findings on healthy tissue", 130, C_RED, "66", C_RED),
        (cb + 114, "contradicted the ground truth", 30, "#DD8452", "15", C_RED),
        (cb + 146, "drift left uncorrected", 22, C_ORANGE, "11", C_ORANGE_D),
    ]
    for by_, label, bw_, bc, cnt, cc in bars:
        p.append(t(ax + 14, by_, label, size=8.2, fill=C_SUB, anchor="start", maxw=aw - 24))
        p.append(R(ax + 14, by_ + 6, bw_, 9, rx=2, fill=bc, stroke="none", sw=0))
        p.append(t(ax + 14 + bw_ + 6, by_ + 14, cnt, size=8.4, weight="bold", fill=cc, anchor="start"))
    p.append(t(ax + 14, cb + 180, "category 3 exists because perturbation pressure", size=8.0, fill=C_ORANGE_D, anchor="start", maxw=aw - 24))

    # ===================== band C: decontamination + release =====================
    yc = yb + hb + 24
    cc_ = yc + 34
    chc = 168
    hc = 34 + chc + 12
    band(yc, hc, C_SAND_FILL, "#E5D9B8", "#DD8452", "STAGE 3 · DECONTAMINATION + RELEASE", 214, "only clean traces enter the corpus", C_SAND_D)

    gx, gw = 22, 244
    # PASS arrows from the verification band
    p.append(arrow([(238, cb + chb), (238, cc_ - 1)], color=C_GREEN, sw=1.8))
    p.append(t(248, yc + 15, "PASS · 97.5% first-pass", size=8.4, fill=C_GREEN_D, anchor="start", maxw=140))
    cx_, cw_ = gx + gw + 26, 244
    px_ = cx_ + 150
    p.append(arrow([(px_, cb + chb), (px_, cc_ - 1)], color="#7FA98B", sw=1.3))
    p.append(t(px_ + 8, yc + 15, "PASS", size=8.4, fill=C_GREEN_D, anchor="start", maxw=60))

    p.append(R(gx, cc_, gw, chc))
    title(gx, cc_, icon_shield(gx + 14, cc_ + 12, color=C_SAND_D), "Dual-gate decontamination", gw)
    p.append(t(gx + 14, cc_ + 46, "Gate 1 - regex sweep for directive", size=8.6, anchor="start", maxw=gw - 24))
    p.append(t(gx + 14, cc_ + 58, "phrasings leaked into the CoT", size=8.6, anchor="start", maxw=gw - 24))
    p.append(t(gx + 14, cc_ + 78, "Gate 2 - flagged traces re-verified", size=8.6, anchor="start", maxw=gw - 24))
    p.append(t(gx + 14, cc_ + 90, "by family B before any action", size=8.6, anchor="start", maxw=gw - 24))
    p.append(R(gx + 14, cc_ + 104, 150, 22, rx=11, fill="#FDEEEE", stroke=C_RED, sw=1.0))
    p.append(t(gx + 89, cc_ + 119, "105 traces excised", size=8.6, weight="bold", fill="#8A2E33", maxw=140))
    p.append(t(gx + 14, cc_ + 146, '0 false positives on clinical "hint of"', size=8.2, fill=C_GREEN_D, anchor="start", maxw=gw - 24))
    p.append(arrow([(gx + gw, cc_ + 90), (cx_ - 2, cc_ + 90)], color=G, sw=1.5))
    p.append(t((gx + gw + cx_) / 2, cc_ + 82, "clean", size=8.0, fill=C_SUB, maxw=24))

    p.append(R(cx_, cc_, cw_, chc))
    title(cx_, cc_, icon_db(cx_ + 12, cc_ + 10, color=C_GREEN_D), "Released corpus", cw_)
    p.append(R(cx_ + 14, cc_ + 40, 100, 14, rx=2, fill=C_BLUE, stroke="none", sw=0))
    p.append(t(cx_ + 64, cc_ + 68, "1,847 with-tools", size=8.4, fill=C_BLUE_D, maxw=100))
    p.append(R(cx_ + 124, cc_ + 40, 100, 14, rx=2, fill=C_GREEN, stroke="none", sw=0))
    p.append(t(cx_ + 174, cc_ + 68, "1,847 no-tools", size=8.4, fill=C_GREEN_D, maxw=100))
    p.append(t(cx_ + 14, cc_ + 88, "exact per-cohort 1:1 parity (5 cohorts)", size=8.2, fill=C_SUB, anchor="start", maxw=cw_ - 24))
    p.append(t(cx_ + 14, cc_ + 114, "3,694 verified traces · 5,144 findings", size=8.8, anchor="start", maxw=cw_ - 24))
    p.append(t(cx_ + 14, cc_ + 128, "687 healthy images · 44,350 tool calls", size=8.8, anchor="start", maxw=cw_ - 24))
    p.append(arrow([(cx_ + cw_, cc_ + 90), (cx_ + cw_ + 24, cc_ + 90)], color=G, sw=1.5))
    p.append(t(cx_ + cw_ + 12, cc_ + 82, "clean", size=8.0, fill=C_SUB, maxw=24))

    rlx, rlw = cx_ + cw_ + 26, 192
    p.append(R(rlx, cc_, rlw, chc))
    title(rlx, cc_, icon_doc(rlx + 12, cc_ + 10), "Release", rlw)
    p.append(t(rlx + 14, cc_ + 46, "public on the Hugging Face Hub", size=8.6, anchor="start", maxw=rlw - 24))
    p.append(t(rlx + 14, cc_ + 62, "per-call audit: perturbation tier,", size=8.6, anchor="start", maxw=rlw - 24))
    p.append(t(rlx + 14, cc_ + 74, "shown vs true bbox, offsets", size=8.6, anchor="start", maxw=rlw - 24))
    p.append(R(rlx + 14, cc_ + 90, rlw - 28, 22, rx=11, fill="#EEF2FA", stroke=C_BLUE, sw=1.0))
    p.append(t(rlx + rlw / 2, cc_ + 105, "every number regenerable", size=8.6, weight="bold", fill=C_BLUE_D, maxw=rlw - 36))
    p.append(t(rlx + 14, cc_ + 134, "scripts/paper rebuilds all figures", size=8.2, fill=C_SUB, anchor="start", maxw=rlw - 24))
    p.append(t(rlx + 14, cc_ + 146, "and tables in this paper", size=8.2, fill=C_SUB, anchor="start", maxw=rlw - 24))

    if warn:
        print("  multistage text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, yc + hc + 12, "".join(p))


# ---------------------------------------------------------------------------
# Figure 3: architecture
# ---------------------------------------------------------------------------


def build_architecture(a):
    """System architecture on a 764-unit, two-column canvas.

    The previous 1200x760 three-column canvas printed its 7-8 unit labels at
    ~3 pt. Same content, two columns (offline data/training on the left, the
    agent loop + evaluation on the right), numbered flow 1-6 unchanged.
    Every label is checked against its box width with reportlab metrics.
    """
    from reportlab.pdfbase.pdfmetrics import stringWidth

    W = 764
    warn: list[str] = []

    def t(x, y, s_, size=9.0, fill=C_TXT, anchor="middle", weight=None, maxw=None, font=None):
        face = "Courier" if font else ("Helvetica-Bold" if weight == "bold" else "Helvetica")
        if maxw is not None and stringWidth(s_, face, size) > maxw:
            warn.append(f"overflow {stringWidth(s_, face, size):.0f}>{maxw}: {s_!r}")
        return TC(x, y, s_, size=size, fill=fill, anchor=anchor, weight=weight, font=font)

    p = []
    p.append(TC(W / 2, 22, "VLM-DENTAL system architecture", size=13, weight="bold"))
    p.append(t(W / 2, 38, "an agentic tool loop over native-resolution radiographs, trained on verified self-correcting traces", size=9.4, fill=C_SUB, maxw=700))

    LX, LW = 10, 350           # left column
    RX, RW = 410, 340          # right column
    LC_ = LX + LW / 2          # left column centre x
    G = "#999999"

    # =================== LEFT COLUMN ===================
    # ---- Data layer ----
    y1, h1 = 54, 222
    p.append(R(LX, y1, LW, h1))
    p.append(icon_db(LX + 14, y1 + 12))
    p.append(t(LX + 38, y1 + 24, "Data layer", size=11.5, weight="bold", anchor="start"))
    tx = LX + 14
    p.append(t(tx, y1 + 46, "2 datasets in use: DENTEX · Tufts", size=9.2, anchor="start", maxw=320))
    p.append(t(tx, y1 + 59, "(8 more catalogued as candidates - pending)", size=8.8, fill=C_SUB, anchor="start", maxw=320))
    p.append(t(tx, y1 + 76, "loaders: DENTEX full · Tufts partial (1 open question)", size=8.8, fill=C_SUB, anchor="start", maxw=320))
    p.append(t(tx, y1 + 89, "FDI notation conversion at ingest", size=8.8, fill=C_SUB, anchor="start", maxw=320))
    p.append(IMG(tx, y1 + 100, 150, 70, a["opg"]))
    p.append(R(tx, y1 + 100, 150, 70, rx=2, stroke="#666666", sw=1.0, fill="none"))
    p.append(t(tx + 75, y1 + 183, "panoramic radiograph + per-tooth", size=8.6, fill=C_SUB, maxw=156))
    p.append(t(tx + 75, y1 + 194, "boxes + diagnoses", size=8.6, fill=C_SUB, maxw=156))
    p.append(R(LX + 186, y1 + 100, 150, 54, fill="#F0F7F0", stroke=C_GREEN, sw=1.1))
    p.append(t(LX + 261, y1 + 118, "multi-finding ground truth", size=8.8, weight="bold", fill=C_GREEN_D, maxw=140))
    p.append(t(LX + 261, y1 + 132, "1-7 findings per image", size=8.8, fill=C_GREEN_D, maxw=140))
    p.append(t(LX + 261, y1 + 146, "full set - never truncated", size=8.8, fill=C_GREEN_D, maxw=140))
    p.append(R(tx, y1 + 196, LW - 28, 20, rx=10, fill="#EEF2FA", stroke=C_BLUE, sw=1.0))
    p.append(t(LC_, y1 + 210, "grounding model: YOLOv8m · 5-fold CV · mAP50 ~ 0.87", size=8.8, fill=C_BLUE_D, maxw=318))
    ya = y1 + h1
    p.append(arrow([(LC_, ya), (LC_, ya + 26)], color=G, sw=1.5))
    p.append(badge(1, LC_, ya + 13))

    # ---- Trace synthesis ----
    y2, h2 = ya + 26, 262
    p.append(R(LX, y2, LW, h2, rx=8, stroke=C_BLUE, sw=1.4))
    p.append(icon_doc(LX + 14, y2 + 10))
    p.append(t(LX + 38, y2 + 22, "Trace synthesis (offline)", size=11.5, weight="bold", fill=C_BLUE_D, anchor="start"))
    flow = [
        (C_BLUE_FILL, "#C7D3E8", C_BLUE_D, "family A generates with-tools traces", "+ paired no-tools traces"),
        (C_ORANGE_L, "#F0B27A", C_ORANGE_D, "tiered perturbation of displayed boxes", "(nudge behavior stays demonstrable)"),
        (C_GREEN_FILL, "#BFDCC6", C_GREEN_D, "family B verifies + repairs every trace", "staged · cross-provider · multi-finding"),
        (C_SAND_FILL, "#E5D9B8", C_SAND_D, "dual-gate decontamination", "regex sweep + re-verification"),
    ]
    fy = y2 + 34
    for fill, stroke, col, l1, l2 in flow:
        p.append(R(LX + 14, fy, LW - 28, 32, rx=6, fill=fill, stroke=stroke, sw=1.0))
        p.append(t(LC_, fy + 14, l1, size=9.2, fill=col, maxw=310))
        p.append(t(LC_, fy + 26, l2, size=8.6, fill=col, maxw=310))
        p.append(arrow([(LC_, fy + 32), (LC_, fy + 42)], color=G, sw=1.2))
        fy += 42
    p.append(R(LX + 14, fy, LW - 28, 36, rx=6, fill="#EEF2FA", stroke=C_BLUE, sw=1.2))
    p.append(t(LC_, fy + 15, "corpus: 1,847 + 1,847 · 3,694 traces", size=9.2, weight="bold", fill=C_BLUE_D, maxw=310))
    p.append(t(LC_, fy + 28, "exact per-cohort parity", size=8.6, fill=C_BLUE_D, maxw=310))
    p.append(t(LX + 14, fy + 54, "details in the trace-synthesis figure", size=8.8, fill=C_SUB, anchor="start", maxw=320))
    p.append(t(LX + 14, fy + 66, "(all quotes, boxes and stats are real)", size=8.8, fill=C_SUB, anchor="start", maxw=320))
    yb = y2 + h2
    p.append(arrow([(LC_, yb), (LC_, yb + 26)], color="#7FA98B", sw=1.6))
    p.append(badge(2, LC_, yb + 13))

    # ---- Training ----
    y3 = yb + 26
    h3 = 306
    p.append(R(LX, y3, LW, h3, rx=8, stroke=C_BLUE, sw=1.4))
    p.append(icon_gear(LX + 22, y3 + 20))
    p.append(t(LX + 38, y3 + 24, "Training", size=11.5, weight="bold", fill=C_BLUE_D, anchor="start"))
    ix, iw = LX + 14, LW - 28
    p.append(R(ix, y3 + 34, iw, 52, rx=6, fill=C_BLUE_FILL, stroke="#C7D3E8", sw=1.0))
    p.append(t(ix + 12, y3 + 51, "1 · SFT on verified traces", size=9.6, weight="bold", fill=C_BLUE_D, anchor="start", maxw=300))
    p.append(t(ix + 12, y3 + 65, "behavior cloning of the tool loop", size=8.8, fill=C_SUB, anchor="start", maxw=300))
    p.append(t(ix + 12, y3 + 78, "(1,847 with-tools traces)", size=8.6, fill=C_SUB, anchor="start", maxw=300))
    p.append(arrow([(LC_, y3 + 86), (LC_, y3 + 104)], color=G, sw=1.2))
    p.append(t(LC_ + 12, y3 + 99, "then", size=8.6, fill=C_SUB, anchor="start"))
    p.append(badge(3, LC_ + 42, y3 + 95, r=7.5))
    p.append(R(ix, y3 + 104, iw, 92, rx=6, fill=C_BLUE_FILL, stroke="#C7D3E8", sw=1.0))
    p.append(t(ix + 12, y3 + 121, "2 · GRPO on fresh rollouts", size=9.6, weight="bold", fill=C_BLUE_D, anchor="start", maxw=300))
    p.append(t(ix + 12, y3 + 135, "same loop, live tool execution", size=8.8, fill=C_SUB, anchor="start", maxw=300))
    p.append(t(ix + 12, y3 + 147, "group-relative advantage", size=8.8, fill=C_SUB, anchor="start", maxw=300))
    px = ix + 12
    for cw, label in ((60, "accuracy"), (54, "format"), (58, "tool use"), (66, "efficiency")):
        p.append(R(px, y3 + 154, cw, 17, rx=8.5, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
        p.append(t(px + cw / 2, y3 + 166, label, size=8.4, fill=C_BLUE_D, maxw=cw - 6))
        px += cw + 8
    p.append(t(ix + 12, y3 + 186, "efficiency term exempts locate_tooth + nudge_crop", size=8.6, fill=C_ORANGE_D, anchor="start", maxw=300))
    oy = y3 + 206
    p.append(R(ix, oy, iw, 40, rx=6, fill="#F0F7F0", stroke=C_GREEN, sw=1.1))
    p.append(t(ix + 12, oy + 16, "output: LoRA adapters", size=9.4, weight="bold", fill=C_GREEN_D, anchor="start", maxw=300))
    p.append(t(ix + 12, oy + 31, "loaded by the policy at inference", size=8.8, fill=C_SUB, anchor="start", maxw=300))
    p.append(t(ix, y3 + 264, "rollouts execute the real dispatcher - same code path as trace generation", size=8.6, fill=C_SUB, anchor="start", maxw=iw))
    p.append(t(ix, y3 + 277, "checkpoint selection on the dev split", size=8.6, fill=C_SUB, anchor="start", maxw=iw))
    p.append(t(ix, y3 + 290, "(test set evaluated once, at the end)", size=8.6, fill=C_SUB, anchor="start", maxw=iw))
    bottom = y3 + h3

    # =================== RIGHT COLUMN ===================
    # ---- Evaluation (top) ----
    e0, eh = 54, 214
    p.append(R(RX, e0, RW, eh))
    p.append(icon_magnifier(RX + 20, e0 + 20, r=4.5))
    p.append(t(RX + 38, e0 + 24, "Evaluation", size=11.5, weight="bold", anchor="start"))
    p.append(t(RX + 14, e0 + 46, "set-matched precision / recall / F1", size=9.2, anchor="start", maxw=316))
    p.append(t(RX + 14, e0 + 59, "over multi-finding predictions", size=8.8, fill=C_SUB, anchor="start", maxw=316))
    p.append(t(RX + 14, e0 + 76, "95% bootstrap confidence intervals", size=9.2, anchor="start", maxw=316))
    p.append(R(RX + 14, e0 + 88, RW - 28, 84, rx=6, fill=C_BLUE_FILL, stroke="#C7D3E8", sw=1.0))
    p.append(t(RX + 26, e0 + 104, "probes (mechanism evidence):", size=9.0, weight="bold", fill=C_BLUE_D, anchor="start", maxw=290))
    p.append(t(RX + 26, e0 + 120, "- tool removal: with vs without", size=8.8, fill=C_BLUE_D, anchor="start", maxw=290))
    p.append(t(RX + 26, e0 + 133, "- perturbation: shown-box vs true-box", size=8.8, fill=C_BLUE_D, anchor="start", maxw=290))
    p.append(t(RX + 26, e0 + 146, "- nudge budget K sweep", size=8.8, fill=C_BLUE_D, anchor="start", maxw=290))
    p.append(t(RX + 26, e0 + 163, "(results pending, reported as open cells)", size=8.6, fill=C_SUB, anchor="start", maxw=290))
    p.append(t(RX + 14, e0 + 190, "heavy runs on GPU or TPU", size=8.8, fill=C_SUB, anchor="start", maxw=316))
    p.append(t(RX + 14, e0 + 203, "(target decision pending - see input geometry)", size=8.8, fill=C_SUB, anchor="start", maxw=316))

    # ---- Final answer ----
    a0 = e0 + eh + 34
    p.append(arrow([(RX + RW / 2, a0), (RX + RW / 2, e0 + eh + 2)], color=G, sw=1.5))
    p.append(badge(5, RX + RW / 2, a0 - 17))
    p.append(t(RX + RW / 2 + 16, a0 - 14, "per-image findings", size=8.6, fill=C_SUB, anchor="start"))
    p.append(R(RX, a0, RW, 70, rx=8, fill="#F2F8F2", stroke=C_GREEN, sw=1.4))
    p.append(t(RX + RW / 2, a0 + 19, "final answer - findings JSON", size=9.8, weight="bold", fill=C_GREEN_D, maxw=316))
    p.append(t(RX + 14, a0 + 39, '{"findings":[{"quadrant":3,"tooth":34,', size=8.4, fill=C_BLUE_D, anchor="start", font=MONO, maxw=316))
    p.append(t(RX + 14, a0 + 53, ' "diagnosis":"Caries","confidence":0.82}]}', size=8.4, fill=C_BLUE_D, anchor="start", font=MONO, maxw=316))

    # ---- Policy <-> Tool node ----
    c0 = a0 + 70 + 36
    ch = 170
    px0, pw = RX, 140
    tx0, tw = RX + 170, 170
    p.append(arrow([(px0 + 60, c0), (px0 + 60, a0 + 72)], color=G, sw=1.4))
    p.append(t(px0 + 70, a0 + 70 + 20, "on answer / budget", size=8.6, fill=C_SUB, anchor="start", maxw=120))
    p.append(R(px0, c0, pw, ch, rx=8, stroke=C_BLUE, sw=1.5))
    p.append(t(px0 + pw / 2, c0 + 24, "Policy", size=11.5, weight="bold", fill=C_BLUE_D))
    p.append(t(px0 + pw / 2, c0 + 44, "Qwen3.5-9B", size=10.2, weight="bold", maxw=pw - 12))
    p.append(t(px0 + pw / 2, c0 + 64, "reasoning node", size=8.8, fill=C_SUB, maxw=pw - 12))
    p.append(t(px0 + pw / 2, c0 + 77, "thought + JSON tool calls", size=8.8, fill=C_SUB, maxw=pw - 10))
    p.append(t(px0 + pw / 2, c0 + 102, "LoRA adapters:", size=8.8, fill=C_SUB, maxw=pw - 12))
    p.append(t(px0 + pw / 2, c0 + 115, "SFT · GRPO (pluggable)", size=8.8, fill=C_SUB, maxw=pw - 10))
    p.append(R(tx0, c0, tw, ch, rx=8, stroke=C_BLUE, sw=1.5))
    p.append(t(tx0 + tw / 2, c0 + 24, "Tool node", size=11.5, weight="bold", fill=C_BLUE_D))
    p.append(IMG(tx0 + 10, c0 + 34, 34, 110, a["crop_final"]))
    p.append(R(tx0 + 10, c0 + 34, 34, 110, rx=1, stroke="#888888", sw=0.9, fill="none"))
    p.append(t(tx0 + 27, c0 + 158, "live crop", size=8.4, fill=C_SUB))
    tt = tx0 + 52
    p.append(t(tt, c0 + 46, "8 registered tools", size=9.2, weight="bold", anchor="start", maxw=tw - 58))
    for i, line in enumerate(("zoom · window · denoise", "enhance · contrast", "locate · nudge", "bilateral compare · fdi")):
        p.append(t(tt, c0 + 62 + 12 * i, line, size=8.8, fill=C_SUB, anchor="start", maxw=tw - 58))
    p.append(t(tt, c0 + 124, "continuous parameters:", size=8.8, fill=C_SUB, anchor="start", maxw=tw - 58))
    p.append(t(tt, c0 + 136, "strength · center · width", size=8.8, fill=C_SUB, anchor="start", maxw=tw - 58))
    # cycle arrows in the 30-unit corridor
    gx0, gx1 = px0 + pw, tx0
    p.append(arrow([(gx0, c0 + 56), (gx1, c0 + 56)], color=C_BLUE, sw=1.6))
    p.append(t((gx0 + gx1) / 2, c0 + 48, "call", size=8.6, fill=C_BLUE_D))
    p.append(arrow([(gx1, c0 + 108), (gx0, c0 + 108)], color="#8DA2C4", sw=1.6))
    p.append(t((gx0 + gx1) / 2, c0 + 100, "obs", size=8.6, fill="#8DA2C4"))

    # ---- OPG input (feeds policy solid, tools dashed base_image) ----
    o0 = c0 + ch + 34
    p.append(IMG(RX + 10, o0, 224, 105, a["opg"]))
    p.append(R(RX + 10, o0, 224, 105, rx=3, stroke="#666666", sw=1.2, fill="none"))
    p.append(arrow([(px0 + 60, o0), (px0 + 60, c0 + ch + 2)], color=G, sw=1.4))
    p.append(t(RX + 122, o0 + 119, "input · native OPG (2,808 × 1,316 px)", size=8.8, fill=C_SUB, maxw=330))
    p.append(t(RX + 122, o0 + 131, "every tool executes at full resolution", size=8.6, fill=C_SUB, maxw=330))
    bx = tx0 + tw - 24
    p.append(arrow([(RX + 234, o0 + 52), (bx, o0 + 52), (bx, c0 + ch + 2)], color=G, sw=1.2, dash="4,3"))
    p.append(t(bx - 6, o0 + 44, "base_image", size=8.6, fill=C_SUB, anchor="end"))

    # ---- dispatcher / input-geometry note ----
    d0 = o0 + 150
    p.append(R(RX, d0, RW, 80, rx=8, fill=C_BLUE_FILL, stroke="#AAB8CE", sw=1.2, dash="5,3"))
    p.append(t(RX + 14, d0 + 18, "shared dispatcher - every image-consuming tool runs on base_image;", size=8.8, fill=C_BLUE_D, anchor="start", maxw=316))
    p.append(t(RX + 14, d0 + 31, "the crop chain never compounds", size=8.8, fill=C_BLUE_D, anchor="start", maxw=316))
    p.append(t(RX + 14, d0 + 50, "input geometry: TPU = canonical views + static slots (no recompile);", size=8.8, fill=C_BLUE_D, anchor="start", maxw=316))
    p.append(t(RX + 14, d0 + 63, "GPU = native so far - decision pending; tools always run native", size=8.8, fill=C_BLUE_D, anchor="start", maxw=316))
    right_bottom = d0 + 80

    # ---- cross-column connectors ----
    # 4: LoRA adapters -> policy (short horizontal hop across the gutter)
    ay = oy + 20
    p.append(arrow([(ix + iw, ay), (LX + LW + 18, ay), (LX + LW + 18, c0 + 60), (px0, c0 + 60)], color="#7FA98B", sw=1.6))
    p.append(badge(4, LX + LW + 18, ay - 30))
    # 6: evaluation -> training (dev-split signal, right margin then bottom lane, dashed)
    lane = max(bottom, right_bottom) + 22
    mx = RX + RW + 8
    p.append(arrow([(RX + RW, e0 + 100), (mx, e0 + 100), (mx, lane), (LC_ + 70, lane), (LC_ + 70, bottom + 2)], color=G, sw=1.3, dash="5,3"))
    p.append(badge(6, mx, e0 + eh + 17))
    p.append(t(mx - 12, e0 + eh + 20, "dev-split signal", size=8.6, fill=C_SUB, anchor="end"))
    p.append(
        t(
            LX,
            lane + 24,
            "numbered path: 1 data -> traces · 2 corpus -> SFT · 3 SFT -> GRPO · 4 adapters -> policy · 5 findings -> eval · 6 dev-split signal",
            size=8.6,
            fill=C_SUB,
            anchor="start",
            maxw=W - 2 * LX,
        )
    )
    if warn:
        print("  architecture text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W + 8, lane + 40, "".join(p))


# ---------------------------------------------------------------------------
# Figure 4: corpus composition (per-cohort, measured)
# ---------------------------------------------------------------------------


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

    scale = 205.0 / 678
    rows = [
        ("dentex · pathology", 678, "3,529", "0", 25.2, C_BLUE),
        ("dentex · healthy", 27, "-", "27", 29.9, C_GREEN),
        ("tufts · periapical overlap", 202, "631", "0", 18.0, C_BLUE),
        ("tufts · healthy", 660, "-", "660", 27.0, C_GREEN),
        ("tufts · all diseases", 280, "984", "0", 17.8, C_BLUE),
    ]
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

    ty = 246
    p.append(R(22, ty - 8, 596, 52, rx=6, fill="#F7F8FA", stroke="#E3E3E3", sw=1.0))
    p.append(t(34, ty + 8, "totals", size=9.6, weight="bold", anchor="start"))
    p.append(t(84, ty + 8, "1,847 with-tools + 1,847 no-tools · exact parity · 44,350 executed tool calls", size=9.0, anchor="start", maxw=526))
    p.append(t(84, ty + 21, "5,144 findings (4,513 unique) · 687 healthy images", size=9.0, anchor="start", maxw=526))
    p.append(t(84, ty + 34, "with-tools arm: 12-13 turns and 18-30 real tool calls per trace", size=8.6, fill=C_SUB, anchor="start", maxw=526))

    ly = 304
    p.append(L(24, ly - 3, 46, ly - 3, color=C_BLUE, sw=3))
    p.append(t(52, ly, "pathology cohorts (findings-bearing)", size=8.8, anchor="start", maxw=190))
    p.append(L(262, ly - 3, 284, ly - 3, color=C_GREEN, sw=3))
    p.append(t(290, ly, "healthy cohorts (negative controls)", size=8.8, anchor="start", maxw=190))

    # unique findings by class
    py0 = 326
    p.append(R(10, py0, 620, 154, fill="#FCFCFD", stroke="#DDDDDD", sw=1.2))
    p.append(t(24, py0 + 22, "unique findings by class", size=10, weight="bold", anchor="start"))
    classes = [
        ("caries", 2189),
        ("deep caries", 578),
        ("impacted teeth", 604),
        ("periapical lesions", 789),
        ("further Tufts classes", 353),
    ]
    cy = py0 + 44
    for name, n in classes:
        bw = 300.0 * n / 2189
        p.append(t(24, cy, name, size=8.8, anchor="start", maxw=104))
        p.append(R(134, cy - 8, bw, 10, rx=2, fill=C_BLUE, stroke="none", sw=0))
        p.append(t(134 + bw + 6, cy, f"{n:,}", size=8.8, weight="bold", anchor="start"))
        cy += 17
    p.append(t(24, py0 + 134, "4,513 unique findings; 5,144 annotations total (the 202-image Tufts overlap cohort", size=8.6, fill=C_SUB, anchor="start", maxw=596))
    p.append(t(24, py0 + 146, "is contained in the 280-image all-diseases cohort)", size=8.6, fill=C_SUB, anchor="start", maxw=596))
    if warn:
        print("  corpus_composition text-fit warnings:\n   " + "\n   ".join(warn))
    return svg(W, py0 + 162, "".join(p))


# ---------------------------------------------------------------------------
# Figure 5: dataset catalog (two in use, eight candidates)
# ---------------------------------------------------------------------------


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
    p.append(R(22, y0 + 10, 62, 18, rx=9, fill=C_GREEN, stroke="none", sw=0))
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
    p.append(R(22, c0 + 10, 124, 18, rx=9, fill="#9AA5B1", stroke="none", sw=0))
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
    p.append(arrow([(mx + mw, my + mh / 2), (lx, my + mh / 2), (lx, iy - 1)], color=C_ORANGE, sw=1.3, dash="4,3"))

    p.append(t(28, iy + 206, "static slots: 5 FULL + 10 CROP + 4 COMPARE = 19 slots = 7,488 vision tokens -", size=8.8, fill=C_BLUE_D, anchor="start", maxw=590))
    p.append(t(28, iy + 219, "one XLA graph, no recompilation", size=8.8, fill=C_BLUE_D, anchor="start", maxw=590))

    # GPU panel
    gy0, gh = ty0 + th + 12, 108
    p.append(R(16, gy0, 614, gh, fill="#F7F8FA", stroke="#BBBBBB", sw=1.2))
    p.append(R(28, gy0 + 10, 56, 20, rx=10, fill="#777777", stroke="none", sw=0))
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
    p.append(arrow([(nx, ny + nh / 2), (6, ny + nh / 2), (6, gy0 + gh / 2), (14, gy0 + gh / 2)], color="#999999", sw=1.3, dash="4,3"))

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
    p.append(IMG(20, y1 + 62, 140, 66, a["opg"]))
    p.append(R(20, y1 + 62, 140, 66, rx=0, stroke="#666666", sw=1.0, fill="none"))
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
    "fig_traces_comparison.svg": build_traces_comparison,
    "fig_trace_multistage.svg": build_multistage,
    "fig_architecture.svg": build_architecture,
    "fig_corpus_composition.svg": build_corpus_composition,
    "fig_datasets.svg": build_dataset_catalog,
    "fig_grounding_cv.svg": build_grounding_cv,
    "fig_input_geometry.svg": build_input_geometry,
    "fig_pipeline.svg": build_pipeline,
    "fig_perturbation_mechanism.svg": build_perturbation_mechanism,
}


def main() -> None:
    force = "--force" in sys.argv
    a = load_assets(force=force)
    print(f"assets: source {a['source_file']} ({a['native_w']}x{a['native_h']})")
    SVG_DIR.mkdir(parents=True, exist_ok=True)
    for name, builder in BUILDERS.items():
        out = SVG_DIR / name
        out.write_text(builder(a), encoding="utf-8")
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
