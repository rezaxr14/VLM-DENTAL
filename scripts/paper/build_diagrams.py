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
    p = []
    W, H = a["native_w"], a["native_h"]
    gt, loc = BOXES_624["gt"], BOXES_624["loc"]
    n1, n2, n3 = BOXES_624["n1"], BOXES_624["n2"], BOXES_624["n3"]

    # header
    p.append(T(530, 26, "Same image, same ground truth: two training traces", size=14.5, weight="bold"))
    p.append(
        T(
            530,
            44,
            "DENTEX image 624 (train_318.png) · FDI 34 · Caries · quotes abridged from the released trace",
            size=9.5,
            fill=C_SUB,
        )
    )

    # ---------------- left panel: image + evidence trail ----------------
    p.append(R(16, 58, 320, 598, fill="#F7F9FC", stroke=C_BLUE, sw=1.4))
    p.append(T(176, 80, "The image and the evidence trail", size=11.5, weight="bold", fill=C_BLUE_D))
    p.append(T(176, 95, "real radiograph + real returned boxes", size=8.2, fill=C_SUB))

    # full OPG with trajectory boxes (small) at (32,104) 288x135
    p.append(IMG(32, 104, 288, 135, a["opg"]))
    p.append(R(32, 104, 288, 135, rx=2, stroke="#666666", sw=1.1, fill="none"))
    for box, color, sw, dash in (
        (gt, "#2ca02c", 1.6, "4,2.5"),
        (loc, C_BLUE, 1.6, None),
        (n1, "#FFC287", 1.0, None),
        (n2, "#FF9A4D", 1.1, None),
        (n3, C_ORANGE, 1.6, None),
    ):
        bx, by, bw, bh = scaled_box(box, (0, 0, W, H), 32, 104, 288, 135)
        p.append(R(bx, by, bw, bh, rx=0, fill="none", stroke=color, sw=sw, dash=dash))

    # leader lines to the magnifier inset
    p.append(L(200, 246, 72, 266, color="#C9C9C9", sw=0.9, dash="2.5,2.5"))
    p.append(L(210, 246, 164, 266, color="#C9C9C9", sw=0.9, dash="2.5,2.5"))

    # magnifier inset at (40,268) 130x306 (region aspect-matched)
    p.append(T(40, 262, "4.5× zoom", size=8.4, fill="#555555", anchor="start"))
    p.append(IMG(40, 268, 130, 306, a["inset"]))
    p.append(R(40, 268, 130, 306, rx=4, stroke="#666666", sw=1.2, fill="none"))
    for box, color, sw, dash in (
        (gt, "#2ca02c", 1.8, "5,3"),
        (loc, C_BLUE, 1.8, None),
        (n1, "#FFC287", 1.1, None),
        (n2, "#FF9A4D", 1.3, None),
        (n3, C_ORANGE, 2.0, None),
    ):
        bx, by, bw, bh = scaled_box(box, INSET_REGION, 40, 268, 130, 306)
        p.append(R(bx, by, bw, bh, rx=0, fill="none", stroke=color, sw=sw, dash=dash))
    # the walk arrow: loc, n1, n2 and n3 all share center x ~= 110 in inset
    # coordinates, so one vertical arrow threads through the nested stack and
    # shows the upward correction walk without any stray segments
    p.append(arrow([(110, 556), (110, 318)], color=C_ORANGE, sw=1.5))

    # final crop panel at (212,268) 90x304
    p.append(T(257, 260, "final crop · turn 10", size=8.4, fill="#555555"))
    p.append(IMG(212, 268, 90, 304, a["crop_final"]))
    p.append(R(212, 268, 90, 304, rx=4, stroke="#666666", sw=1.2, fill="none"))

    # legend + real coords
    p.append(L(36, 592, 60, 592, color=C_BLUE, sw=2.4))
    p.append(T(66, 595, "locate_tooth(34) box", size=8.2, anchor="start"))
    p.append(L(186, 592, 194, 592, color="#FFC287", sw=2.4))
    p.append(L(194, 592, 202, 592, color="#FF9A4D", sw=2.4))
    p.append(L(202, 592, 210, 592, color=C_ORANGE, sw=2.4))
    p.append(T(216, 595, "nudges 1-3, light to dark", size=8.2, anchor="start"))
    p.append(L(36, 608, 60, 608, color="#2ca02c", sw=2.4, dash="4,2.5"))
    p.append(T(66, 611, "ground-truth box", size=8.2, anchor="start"))
    p.append(T(216, 611, "coords as returned", size=8.2, fill=C_MUT, anchor="start"))
    p.append(T(36, 630, "[1668,917,84,322] - [1685,885,50,193]", size=7.2, fill=C_SUB, anchor="start", font=MONO))
    p.append(T(36, 642, "- [1693,818,35,135] - [1696,750,28,108]", size=7.2, fill=C_SUB, anchor="start", font=MONO))

    # ---------------- middle panel: no-tools trace ----------------
    p.append(R(352, 58, 272, 598, fill="#FBFBFB", stroke="#BBBBBB", sw=1.2))
    p.append(T(488, 80, "No-tools trace (baseline)", size=11.5, weight="bold", fill="#555555"))
    p.append(T(488, 95, "1 turn · 0 tool calls · hindsight description", size=8.2, fill=C_SUB))

    p.append(R(368, 108, 240, 42, fill="#E8EEF7", stroke=C_BLUE, sw=1.1))
    p.append(T(488, 124, "user: panoramic X-ray, tool-free prompt", size=8.8, weight="bold", fill=C_BLUE_D))
    p.append(T(488, 139, "“identify any abnormal teeth and diagnose”", size=8.0, fill="#555555"))
    p.append(arrow([(488, 150), (488, 164)], color="#999999", sw=1.3))

    p.append(R(368, 166, 240, 300, fill="#FFFFFF", stroke="#CCCCCC", sw=1.1))
    p.append(T(488, 186, "thought (single turn)", size=9.5, weight="bold"))
    no_tools_lines = [
        "“On systematic review of this panoramic",
        "radiograph, I begin by orienting to the",
        "patient’s right and left sides as marked.",
        "The image shows a mixed dentition with",
        "multiple developing permanent teeth ...",
        "",
        "Focusing on the lower-left quadrant (Q3),",
        "I examine the posterior region ... I can",
        "see a radiolucent area involving the",
        "occlusal/cervical portion of the tooth,",
        "with loss of normal enamel contour and",
        "a darker zone consistent with a carious",
        "lesion ...”",
    ]
    y = 206
    for ln in no_tools_lines:
        if ln:
            p.append(T(380, y, ln, size=8.4, fill="#444444", anchor="start"))
        y += 13.5
    p.append(T(380, y + 12, "One-shot description from the full image:", size=8.0, fill="#777777", anchor="start", style="italic"))
    p.append(T(380, y + 25, "no crop, no second look, nothing checkable", size=8.0, fill="#777777", anchor="start", style="italic"))
    p.append(T(380, y + 38, "against the pixels it actually saw.", size=8.0, fill="#777777", anchor="start", style="italic"))
    p.append(arrow([(488, 466), (488, 480)], color="#999999", sw=1.3))

    p.append(R(368, 482, 240, 46, fill="#F2F8F2", stroke=C_GREEN, sw=1.3))
    p.append(T(488, 502, "final answer", size=9.5, weight="bold", fill=C_GREEN_D))
    p.append(T(488, 518, "Q3T4 · Caries · confidence 0.82", size=8.8, fill="#444444"))
    p.append(R(368, 540, 240, 100, fill="#FBF6F0", stroke="#DD8452", sw=1.1))
    p.append(T(488, 560, "What this arm can teach", size=9.5, weight="bold", fill="#8A5A2E"))
    p.append(T(488, 578, "- output format + clinical vocabulary", size=8.4, fill="#555555"))
    p.append(T(488, 592, "- quadrant-survey habit", size=8.4, fill="#555555"))
    p.append(T(488, 622, "No evidence trail to verify or correct.", size=8.8, weight="bold", fill=C_RED))

    # ---------------- right panel: with-tools trace ----------------
    p.append(R(640, 58, 404, 598, fill="#FCFCFD", stroke=C_BLUE, sw=1.4))
    p.append(T(842, 80, "With-tools trace (main policy)", size=11.5, weight="bold", fill=C_BLUE_D))
    p.append(T(842, 95, "23 turns · 22 real tool calls · 3 corrective nudges", size=8.2, fill=C_SUB))

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
    y = 108
    for turn, name, sub, hot in rows:
        fill = C_ORANGE_L if hot else "#FFFFFF"
        stroke = "#F0B27A" if hot else "#DDDDDD"
        p.append(R(656, y, 372, 42, rx=5, fill=fill, stroke=stroke, sw=1.0))
        p.append(R(656, y, 4, 42, rx=2, fill=C_ORANGE if hot else C_BLUE, stroke="none", sw=0))
        tcol = "#B45A04" if hot else C_BLUE_D
        p.append(T(670, y + 17, turn, size=8.6, weight="bold", fill=tcol, anchor="start"))
        p.append(T(706, y + 17, name, size=9.4, weight="bold", fill=C_TXT, anchor="start"))
        p.append(T(706, y + 33, sub, size=8.0, fill=C_SUB, anchor="start"))
        y += 46

    p.append(R(656, 530, 372, 52, rx=6, fill="#F2F8F2", stroke=C_GREEN, sw=1.4))
    p.append(T(842, 550, "final answer (turn 22) · Q3T4 · Caries · conf 0.82", size=9.4, weight="bold", fill=C_GREEN_D))
    p.append(T(842, 568, "matches ground truth - passed strict verification", size=8.2, fill=C_SUB))
    p.append(R(656, 592, 372, 48, rx=6, fill="#F7F9FC", stroke="#C9D4E4", sw=1.0))
    p.append(T(842, 612, "22 real tool calls · 3 corrective · every observation", size=8.2, fill=C_BLUE_D))
    p.append(T(842, 626, "executed against the actual radiograph", size=8.2, fill=C_BLUE_D))

    # ---------------- bottom band ----------------
    p.append(R(16, 672, 1028, 162, fill="#F7F8FA", stroke="#DDDDDD", sw=1.1))
    p.append(T(530, 696, "What the exact 1:1 pairing isolates", size=12, weight="bold"))
    bullets = [
        (40, 724, "same image · same prompt template · same ground truth -"),
        (40, 738, "only the tool channel differs"),
        (530, 724, "any SFT delta is attributable to demonstrated tool use,"),
        (530, 738, "not to data volume or prompt wording"),
        (40, 766, "with-tools traces carry 44,350 executed tool calls;"),
        (40, 780, "the no-tools arm carries zero"),
        (530, 766, "per-cohort counts match exactly (678/27/660/202/280"),
        (530, 780, "on each side) - parity by construction"),
    ]
    for bx, by, txt in bullets:
        p.append(T(bx, by, txt, size=9.0, fill="#444444", anchor="start"))
    p.append(R(330, 796, 400, 26, rx=13, fill="#EEF2FA", stroke=C_BLUE, sw=1.1))
    p.append(T(530, 813, "5 cohorts × 2 arms · 1,847 + 1,847 · 3,694 verified traces", size=8.6, weight="bold", fill=C_BLUE_D))

    return svg(1060, 850, "".join(p))


# ---------------------------------------------------------------------------
# Figure 2: multi-stage trace synthesis (generate -> verify -> release)
# ---------------------------------------------------------------------------


def build_multistage(a):
    p = []
    p.append(T(540, 26, "Trace synthesis: generate, verify, decontaminate", size=14.5, weight="bold"))
    p.append(T(540, 44, "all counts measured on the released corpus - nothing projected", size=9.5, fill=C_SUB))

    # ================= band A: generation =================
    p.append(R(16, 64, 1048, 252, fill=C_BLUE_FILL, stroke="#C7D3E8", sw=1.2))
    p.append(R(16, 64, 196, 24, rx=6, fill=C_BLUE, stroke="none", sw=0))
    p.append(T(114, 80, "STAGE 1 · GENERATION", size=9.2, weight="bold", fill="#FFFFFF"))
    p.append(T(1048, 80, "provider family A: generators", size=8.4, fill=C_BLUE_D, anchor="end"))

    # inputs card
    p.append(R(32, 100, 190, 200))
    p.append(icon_db(46, 112))
    p.append(T(98, 124, "Expert labels", size=10, weight="bold", anchor="start"))
    p.append(IMG(44, 136, 150, 70, a["opg"]))
    p.append(R(44, 136, 150, 70, rx=2, stroke="#666666", sw=1.0, fill="none"))
    p.append(T(44, 222, "DENTEX · Tufts loaders", size=8.0, fill=C_SUB, anchor="start"))
    p.append(T(44, 234, "FDI conversion at ingest", size=8.0, fill=C_SUB, anchor="start"))
    p.append(R(44, 244, 150, 44, fill="#F0F7F0", stroke=C_GREEN, sw=1.1))
    p.append(T(119, 260, "ground truth per image", size=7.8, weight="bold", fill=C_GREEN_D))
    p.append(T(119, 274, "FDI 34 · Caries · bbox", size=7.6, fill=C_GREEN_D))
    p.append(arrow([(222, 200), (234, 200)], color="#999999", sw=1.5))

    # loop card
    p.append(R(234, 100, 414, 200))
    p.append(T(250, 120, "Ground-truth-directed LangGraph loop", size=10, weight="bold", anchor="start"))
    p.append(R(250, 134, 64, 24, rx=12, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(T(282, 149, "thought", size=8, fill=C_BLUE_D))
    p.append(arrow([(316, 146), (326, 146)], color="#999999", sw=1.1))
    p.append(R(326, 134, 72, 24, rx=12, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(T(362, 149, "tool call", size=8, fill=C_BLUE_D))
    p.append(arrow([(400, 146), (410, 146)], color="#999999", sw=1.1))
    p.append(R(410, 134, 84, 24, rx=12, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(T(452, 149, "observation", size=8, fill=C_BLUE_D))
    p.append(arrow([(496, 146), (506, 146)], color="#999999", sw=1.1))

    # zoom monitor: before / after a nudge (real crops)
    p.append(R(506, 128, 130, 84, rx=4, fill="#FAFBFD", stroke="#DDDDDD", sw=1.0))
    p.append(IMG(514, 136, 20, 46, a["crop_located"]))
    p.append(R(514, 136, 20, 46, rx=1, stroke="#888888", sw=0.8, fill="none"))
    p.append(T(549, 150, "nudge", size=6.6, fill=C_ORANGE_D))
    p.append(arrow([(540, 159), (558, 159)], color=C_ORANGE, sw=1.6))
    p.append(IMG(562, 136, 13, 46, a["crop_final"]))
    p.append(R(562, 136, 13, 46, rx=1, stroke="#888888", sw=0.8, fill="none"))
    p.append(T(571, 196, "before / after", size=6.8, fill="#888888"))
    p.append(arrow([(452, 158), (452, 186), (282, 186), (282, 158)], color="#B9C6DC", sw=1.2))
    p.append(T(367, 181, "repeat every turn", size=7.2, fill="#8DA2C4"))

    # provider chips
    p.append(T(250, 213, "generation pool:", size=8.0, fill=C_SUB, anchor="start"))
    p.append(R(250, 218, 104, 18, rx=9, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(T(302, 231, "Gemini 3.5 Flash Lite", size=7.4, fill=C_BLUE_D))
    p.append(R(360, 218, 72, 18, rx=9, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(T(396, 231, "MiniMax M3", size=7.4, fill=C_BLUE_D))
    p.append(R(438, 218, 72, 18, rx=9, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(T(474, 231, "NVIDIA NIM", size=7.4, fill=C_BLUE_D))
    p.append(R(516, 218, 62, 18, rx=9, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(T(547, 231, "local vLLM", size=7.4, fill=C_BLUE_D))

    # perturbation callout
    p.append(R(250, 246, 380, 42, rx=6, fill=C_ORANGE_L, stroke="#F0B27A", sw=1.1))
    p.append(T(262, 262, "tiered display perturbation, independent of the true box:", size=7.8, fill=C_ORANGE_D, anchor="start"))
    p.append(T(262, 277, "45% small · 25% big · 30% none - keeps accept-vs-nudge demonstrable", size=7.4, fill=C_ORANGE_D, anchor="start"))
    p.append(arrow([(648, 200), (660, 200)], color="#999999", sw=1.5))

    # unverified pool card
    p.append(R(660, 100, 388, 200))
    p.append(icon_doc(674, 112))
    p.append(T(700, 124, "Unverified trace pool", size=10, weight="bold", anchor="start"))
    p.append(R(672, 136, 364, 92, rx=4, fill="#F7F8FA", stroke="#E3E3E3", sw=1.0))
    snip = [
        "T4 thought \"crop shows only bone and",
        "         root area ... too low\"",
        "T4 call    nudge_crop(dy=-0.30, s=0.60)",
        "T5 obs     bbox -> [1685, 885, 50, 193]",
    ]
    y = 152
    for ln in snip:
        p.append(T(682, y, ln, size=7.4, fill="#444444", anchor="start", font=MONO))
        y += 13
    p.append(R(672, 238, 178, 24, rx=12, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(T(761, 254, "44,350 executed tool calls", size=7.8, weight="bold", fill=C_BLUE_D))
    p.append(R(858, 238, 178, 24, rx=12, fill="#EEF2FA", stroke="#C7D3E8", sw=1.0))
    p.append(T(947, 254, "2,292 corrective nudges", size=7.8, weight="bold", fill=C_BLUE_D))
    p.append(T(854, 286, "no training value until verified", size=8.2, fill=C_SUB))

    # ================= band B: verification =================
    p.append(R(16, 332, 1048, 252, fill=C_GREEN_FILL, stroke="#BFDCC6", sw=1.2))
    p.append(R(16, 332, 196, 24, rx=6, fill=C_GREEN, stroke="none", sw=0))
    p.append(T(114, 348, "STAGE 2 · VERIFICATION", size=9.2, weight="bold", fill="#FFFFFF"))
    p.append(T(1048, 348, "provider family B - never the generator pool", size=8.4, fill=C_GREEN_D, anchor="end"))

    # strict verifier card
    p.append(R(32, 368, 300, 190))
    p.append(icon_shield(48, 382))
    p.append(T(72, 394, "Strict verifier", size=10, weight="bold", anchor="start"))
    p.append(T(48, 414, "staged cross-family check per trace:", size=8.2, fill=C_SUB, anchor="start"))
    items = [
        (432, "diagnosis set-matches the full", "multi-finding ground truth"),
        (462, "evidence trail consistent - no", "claims on healthy tissue"),
        (492, "no directive leakage in CoT", None),
        (510, "format + FDI notation valid", None),
    ]
    for iy, l1, l2 in items:
        p.append(icon_check(48, iy - 5.5))
        p.append(T(60, iy, l1, size=8.2, anchor="start"))
        if l2:
            p.append(T(60, iy + 12, l2, size=8.2, anchor="start"))
    p.append(T(48, 538, "strict rate pacing via the provider pool", size=7.6, fill=C_MUT, anchor="start"))

    p.append(arrow([(332, 462), (396, 462)], color=C_RED, sw=1.6))
    p.append(T(364, 454, "REJECT · 92", size=8, fill=C_RED))

    # repair card
    p.append(R(400, 368, 260, 190))
    p.append(icon_gear(416, 384))
    p.append(T(440, 394, "Repair loop", size=10, weight="bold", anchor="start"))
    p.append(T(416, 414, "editor rewrites the trace,", size=8.2, fill=C_SUB, anchor="start"))
    p.append(T(416, 426, "keeping the valid CoT prefix:", size=8.2, fill=C_SUB, anchor="start"))
    for ry, txt in ((448, "fix tool-arg / FDI slips"), (466, "drop phantom claims"), (484, "excise directive leakage")):
        p.append(R(416, ry - 5, 5, 5, rx=1, fill=C_ORANGE, stroke="none", sw=0))
        p.append(T(428, ry, txt, size=8.2, anchor="start"))
    p.append(T(416, 514, "re-verified after every repair", size=7.8, fill=C_MUT, anchor="start"))
    p.append(arrow([(530, 368), (530, 350), (182, 350), (182, 368)], color="#7FA98B", sw=1.3))
    p.append(T(356, 344, "re-verify until pass", size=7.8, fill=C_GREEN_D))

    p.append(arrow([(660, 462), (732, 462)], color="#BBBBBB", sw=1.2, dash="3,2.5"))
    p.append(T(696, 454, "logged", size=7.4, fill=C_MUT))

    # audit card with data-driven bars
    p.append(R(736, 368, 312, 190))
    p.append(icon_magnifier(752, 386, r=4.5))
    p.append(T(770, 394, "First-pass audit", size=10, weight="bold", anchor="start"))
    p.append(T(752, 416, "92 rejections on 3,694 traces (97.5% pass)", size=8.4, weight="bold", anchor="start"))
    bars = [
        (438, "hallucinated findings on healthy tissue", 150, C_RED, "66", C_RED),
        (472, "contradicted the ground truth", 34, "#DD8452", "15", C_RED),
        (506, "drift left uncorrected", 25, C_ORANGE, "11", C_ORANGE_D),
    ]
    for by, label, bw, bc, cnt, cc in bars:
        p.append(T(752, by, label, size=7.6, fill=C_SUB, anchor="start"))
        p.append(R(752, by + 6, bw, 9, rx=2, fill=bc, stroke="none", sw=0))
        p.append(T(752 + bw + 6, by + 14, cnt, size=8, weight="bold", fill=cc, anchor="start"))
    p.append(T(752, 544, "category 3 exists because perturbation pressure", size=7.4, fill=C_ORANGE_D, anchor="start"))

    # PASS arrow down to band C
    p.append(arrow([(182, 558), (182, 672)], color=C_GREEN, sw=1.8))
    p.append(T(190, 590, "PASS · 97.5% first-pass", size=8.2, fill=C_GREEN_D, anchor="start"))
    p.append(arrow([(530, 558), (530, 672)], color="#7FA98B", sw=1.3))
    p.append(T(538, 600, "PASS", size=7.6, fill=C_GREEN_D, anchor="start"))

    # ================= band C: decontamination + release =================
    p.append(R(16, 644, 1048, 210, fill=C_SAND_FILL, stroke="#E5D9B8", sw=1.2))
    p.append(R(16, 644, 240, 24, rx=6, fill="#DD8452", stroke="none", sw=0))
    p.append(T(136, 660, "STAGE 3 · DECONTAMINATION + RELEASE", size=9.2, weight="bold", fill="#FFFFFF"))
    p.append(T(1048, 660, "only clean traces enter the corpus", size=8.4, fill=C_SAND_D, anchor="end"))

    # dual-gate card
    p.append(R(32, 672, 300, 160))
    p.append(icon_shield(48, 686, color=C_SAND_D))
    p.append(T(72, 698, "Dual-gate decontamination", size=10, weight="bold", anchor="start"))
    p.append(T(48, 720, "Gate 1 - regex sweep for directive", size=8.2, anchor="start"))
    p.append(T(48, 732, "phrasings leaked into the CoT", size=8.2, anchor="start"))
    p.append(T(48, 752, "Gate 2 - flagged traces re-verified", size=8.2, anchor="start"))
    p.append(T(48, 764, "by family B before any action", size=8.2, anchor="start"))
    p.append(R(48, 776, 170, 22, rx=11, fill="#FDEEEE", stroke=C_RED, sw=1.0))
    p.append(T(133, 791, "105 traces excised", size=7.8, weight="bold", fill="#8A2E33"))
    p.append(T(48, 814, "0 false positives on clinical \"hint of\"", size=7.6, fill=C_GREEN_D, anchor="start"))
    p.append(arrow([(332, 752), (396, 752)], color="#999999", sw=1.5))
    p.append(T(364, 744, "clean", size=7.4, fill=C_MUT))

    # corpus card with parity bars
    p.append(R(400, 672, 320, 160))
    p.append(icon_db(416, 686, color=C_GREEN_D))
    p.append(T(440, 698, "Released corpus", size=10, weight="bold", anchor="start"))
    p.append(R(416, 714, 120, 14, rx=2, fill=C_BLUE, stroke="none", sw=0))
    p.append(T(476, 742, "1,847 with-tools", size=7.8, fill=C_BLUE_D))
    p.append(R(544, 714, 120, 14, rx=2, fill=C_GREEN, stroke="none", sw=0))
    p.append(T(604, 742, "1,847 no-tools", size=7.8, fill=C_GREEN_D))
    p.append(T(416, 758, "exact per-cohort 1:1 parity (5 cohorts)", size=7.6, fill=C_MUT, anchor="start"))
    p.append(T(416, 780, "3,694 verified traces · 5,144 findings", size=8.2, anchor="start"))
    p.append(T(416, 796, "687 healthy images · 44,350 tool calls", size=8.2, anchor="start"))
    p.append(arrow([(720, 752), (784, 752)], color="#999999", sw=1.5))
    p.append(T(752, 744, "clean", size=7.4, fill=C_MUT))

    # release card
    p.append(R(788, 672, 260, 160))
    p.append(icon_doc(804, 686))
    p.append(T(828, 698, "Release", size=10, weight="bold", anchor="start"))
    p.append(T(804, 720, "public on the Hugging Face Hub", size=8.2, anchor="start"))
    p.append(T(804, 736, "per-call audit: perturbation tier,", size=8.2, anchor="start"))
    p.append(T(804, 748, "shown vs true bbox, offsets", size=8.2, anchor="start"))
    p.append(R(804, 762, 220, 22, rx=11, fill="#EEF2FA", stroke=C_BLUE, sw=1.0))
    p.append(T(914, 777, "every number regenerable", size=7.8, weight="bold", fill=C_BLUE_D))
    p.append(T(804, 804, "scripts/paper rebuilds all figures", size=7.6, fill=C_MUT, anchor="start"))
    p.append(T(804, 816, "and tables in this paper", size=7.6, fill=C_MUT, anchor="start"))

    return svg(1080, 870, "".join(p))


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


def build_corpus_composition(a):
    p = []
    p.append(T(530, 24, "Corpus composition: five cohorts, two exact arms", size=14.5, weight="bold"))
    p.append(T(530, 42, "all counts measured from the released trace files (per-cohort 1:1 parity by construction)", size=9.5, fill=C_SUB))

    p.append(R(16, 58, 1028, 400, fill="#FCFCFD", stroke="#DDDDDD", sw=1.2))
    p.append(T(40, 88, "cohort", size=8, fill=C_MUT, anchor="start"))
    p.append(T(300, 88, "with-tools traces (bar length = cohort size)", size=8, fill=C_MUT, anchor="start"))
    p.append(T(690, 88, "traces", size=8, fill=C_MUT, anchor="start"))
    p.append(T(760, 88, "findings", size=8, fill=C_MUT, anchor="start"))
    p.append(T(850, 88, "healthy", size=8, fill=C_MUT, anchor="start"))
    p.append(T(930, 88, "avg calls", size=8, fill=C_MUT, anchor="start"))

    scale = 380.0 / 678
    rows = [
        ("dentex · pathology", 678, "3,529", "0", 25.2, C_BLUE),
        ("dentex · healthy", 27, "-", "27", 29.9, C_GREEN),
        ("tufts · periapical overlap", 202, "631", "0", 18.0, C_BLUE),
        ("tufts · healthy", 660, "-", "660", 27.0, C_GREEN),
        ("tufts · all diseases", 280, "984", "0", 17.8, C_BLUE),
    ]
    y = 104
    for name, n, findings, healthy, calls, color in rows:
        bw = n * scale
        p.append(T(40, y + 20, name, size=9.2, weight="bold", anchor="start"))
        p.append(R(300, y + 6, bw, 20, rx=3, fill=color, stroke="none", sw=0))
        p.append(T(690, y + 20, f"{n}", size=8.6, weight="bold", anchor="start"))
        p.append(T(760, y + 20, findings, size=8.6, anchor="start"))
        p.append(T(850, y + 20, healthy, size=8.6, anchor="start"))
        p.append(T(930, y + 20, f"{calls:.1f}", size=8.6, anchor="start"))
        y += 52

    p.append(R(32, 352, 996, 56, rx=6, fill="#F7F8FA", stroke="#E3E3E3", sw=1.0))
    p.append(T(48, 378, "totals", size=9.5, weight="bold", anchor="start"))
    p.append(T(120, 378, "1,847 with-tools + 1,847 no-tools · exact parity", size=9.0, anchor="start"))
    p.append(T(430, 378, "5,144 findings (4,513 unique)", size=9.0, anchor="start"))
    p.append(T(640, 378, "687 healthy images", size=9.0, anchor="start"))
    p.append(T(790, 378, "44,350 executed tool calls", size=9.0, anchor="start"))
    p.append(T(120, 398, "with-tools arm: 12-13 turns and 18-30 real tool calls per trace", size=8.0, fill=C_SUB, anchor="start"))

    p.append(L(300, 442, 324, 442, color=C_BLUE, sw=3))
    p.append(T(330, 445, "pathology cohorts (findings-bearing)", size=8.4, anchor="start"))
    p.append(L(560, 442, 584, 442, color=C_GREEN, sw=3))
    p.append(T(590, 445, "healthy cohorts (negative controls)", size=8.4, anchor="start"))

    # unique findings by class
    p.append(R(16, 474, 1028, 112, fill="#FCFCFD", stroke="#DDDDDD", sw=1.2))
    p.append(T(40, 498, "unique findings by class", size=9.5, weight="bold", anchor="start"))
    classes = [
        ("caries", 2189, 500),
        ("deep caries", 578, 132),
        ("impacted teeth", 604, 138),
        ("periapical lesions", 789, 180),
        ("further Tufts classes", 353, 81),
    ]
    cy = 522
    for name, n, bw in classes:
        p.append(T(40, cy, name, size=8.2, anchor="start"))
        p.append(R(190, cy - 8, bw, 9, rx=2, fill=C_BLUE, stroke="none", sw=0))
        p.append(T(190 + bw + 6, cy, f"{n:,}", size=8.0, weight="bold", anchor="start"))
        cy += 15
    p.append(T(830, 528, "4,513 unique findings;", size=8.4, anchor="start"))
    p.append(T(830, 544, "5,144 annotations total", size=8.4, anchor="start"))
    p.append(T(830, 560, "(the 202-image Tufts overlap cohort", size=7.6, fill=C_SUB, anchor="start"))
    p.append(T(830, 572, "is contained in the 280-image all-diseases cohort)", size=7.6, fill=C_SUB, anchor="start"))

    return svg(1060, 606, "".join(p))


# ---------------------------------------------------------------------------
# Figure 5: dataset catalog (two in use, eight candidates)
# ---------------------------------------------------------------------------


def build_dataset_catalog(a):
    p = []
    p.append(T(540, 24, "The data layer: two sources in use, eight candidates on file", size=14.5, weight="bold"))
    p.append(T(540, 42, "a candidate is promoted only after its annotation semantics are verified - two loaders deliberately stop at one open question each", size=9.5, fill=C_SUB))

    # ---- in use ----
    p.append(R(16, 58, 384, 522, fill=C_GREEN_FILL, stroke=C_GREEN, sw=1.4))
    p.append(R(32, 74, 70, 20, rx=10, fill=C_GREEN, stroke="none", sw=0))
    p.append(T(67, 88, "IN USE", size=8.6, weight="bold", fill="#FFFFFF"))
    p.append(T(384, 88, "feeds the released corpus", size=8.2, fill=C_GREEN_D, anchor="end"))

    p.append(R(32, 106, 352, 210))
    p.append(icon_db(48, 122))
    p.append(T(76, 134, "DENTEX", size=12, weight="bold", anchor="start"))
    p.append(T(368, 134, "2,332 images", size=9.2, fill=C_BLUE_D, anchor="end"))
    p.append(T(48, 158, "quadrant · enumeration · diagnosis labels", size=8.2, anchor="start"))
    p.append(T(48, 172, "4 pathology classes + box coordinates", size=8.2, anchor="start"))
    p.append(T(48, 188, "the only panoramic source pairing tooth position", size=7.8, fill=C_SUB, anchor="start"))
    p.append(T(48, 200, "with all four reward-relevant classes", size=7.8, fill=C_SUB, anchor="start"))
    p.append(R(48, 214, 160, 22, rx=11, fill="#F0F7F0", stroke=C_GREEN, sw=1.0))
    p.append(T(128, 229, "loader: complete", size=8.2, weight="bold", fill=C_GREEN_D))
    p.append(T(48, 260, "contributes 705 corpus traces", size=9.0, weight="bold", fill=C_BLUE_D, anchor="start"))
    p.append(T(48, 276, "(678 pathology + 27 healthy)", size=8.0, fill=C_SUB, anchor="start"))
    p.append(T(48, 296, "grounding pool: 1,339 images · 21,624 tooth boxes", size=7.8, fill=C_SUB, anchor="start"))

    p.append(R(32, 328, 352, 210))
    p.append(icon_db(48, 344))
    p.append(T(76, 356, "Tufts Dental Database", size=12, weight="bold", anchor="start"))
    p.append(T(368, 356, "1,000 images", size=9.2, fill=C_BLUE_D, anchor="end"))
    p.append(T(48, 380, "per-tooth boxes (Universal Numbering)", size=8.2, anchor="start"))
    p.append(T(48, 394, "expert pathology layer · 25,184 boxes", size=8.2, anchor="start"))
    p.append(T(48, 410, "periapical maps onto the DENTEX taxonomy;", size=7.8, fill=C_SUB, anchor="start"))
    p.append(T(48, 422, "no caries findings in this layer", size=7.8, fill=C_SUB, anchor="start"))
    p.append(R(48, 436, 230, 22, rx=11, fill=C_ORANGE_L, stroke="#F0B27A", sw=1.0))
    p.append(T(163, 451, "loader: partial - 1 open question", size=8.2, weight="bold", fill=C_ORANGE_D))
    p.append(T(48, 482, "contributes 1,142 corpus traces", size=9.0, weight="bold", fill=C_BLUE_D, anchor="start"))
    p.append(T(48, 498, "(202 overlap + 660 healthy + 280 all-diseases)", size=8.0, fill=C_SUB, anchor="start"))
    p.append(T(48, 518, "grounding pool: 1,000 images · 25,184 boxes", size=7.8, fill=C_SUB, anchor="start"))

    # ---- candidates ----
    p.append(R(416, 58, 648, 522, fill="#FBFBFC", stroke="#CCCCCC", sw=1.2))
    p.append(R(432, 74, 140, 20, rx=10, fill="#9AA5B1", stroke="none", sw=0))
    p.append(T(502, 88, "CANDIDATE POOL", size=8.6, weight="bold", fill="#FFFFFF"))
    p.append(T(1048, 88, "pending - not used in this work", size=8.2, fill=C_SUB, anchor="end"))

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
        cx = 432 + col * 316
        cy = 106 + row * 88
        p.append(R(cx, cy, 300, 80))
        p.append(T(cx + 12, cy + 20, name, size=9.2, weight="bold", anchor="start"))
        p.append(T(cx + 288, cy + 20, n, size=8.6, fill=C_BLUE_D, anchor="end"))
        p.append(T(cx + 12, cy + 38, reason, size=7.6, fill=C_SUB, anchor="start"))
        p.append(T(cx + 12, cy + 62, f"({source})", size=7.0, fill=C_MUT, anchor="start"))
    p.append(T(432, 496, "diagnosis-labeled: 1 of 8 (binary caries masks) · grounding-only: 7 of 8", size=7.8, anchor="start"))
    p.append(T(432, 512, "promotion rule: verified end-to-end annotation semantics, per dataset", size=7.8, anchor="start"))

    # ---- footer ----
    p.append(R(16, 596, 1048, 52, fill="#F7F8FA", stroke="#E3E3E3", sw=1.0))
    p.append(T(32, 618, "catalog transcribed from a systematic review of 16 public dental imaging datasets (Uribe et al. 2024) plus targeted follow-up search", size=7.8, fill=C_SUB, anchor="start"))
    p.append(T(32, 636, "the diagnosis-label vs grounding-only flag drives purpose: grounding candidates generalize locate_tooth, diagnosis candidates feed trace generation", size=7.8, fill=C_SUB, anchor="start"))

    return svg(1080, 664, "".join(p))


# ---------------------------------------------------------------------------
# Figure 6: grounding model CV + held-out verification
# ---------------------------------------------------------------------------


def build_grounding_cv(a):
    p = []
    p.append(T(540, 24, "Grounding model: 5-fold CV and held-out verification", size=14.5, weight="bold"))
    p.append(T(540, 42, "YOLOv8m backend of locate_tooth (one class per FDI quadrant-position pair) - target-filtered protocol", size=9.5, fill=C_SUB))

    # left: fold bars
    p.append(R(16, 58, 620, 500, fill="#FCFCFD", stroke="#DDDDDD", sw=1.2))
    p.append(T(36, 82, "in-fold CV: target mAP50 per fold", size=10.5, weight="bold", anchor="start"))
    p.append(L(36, 98, 60, 98, color=C_BLUE, sw=3))
    p.append(T(66, 101, "DENTEX-only (mean 0.9508)", size=8.4, anchor="start"))
    p.append(L(250, 98, 274, 98, color=C_GREEN, sw=3))
    p.append(T(280, 101, "DENTEX + Tufts (mean 0.9376)", size=8.4, anchor="start"))

    top, bot = 130.0, 500.0

    def yv(v):
        return bot - (v - 0.80) * (bot - top) / 0.20

    for gv in (0.85, 0.90, 0.95, 1.00):
        gy = yv(gv)
        p.append(L(70, gy, 600, gy, color="#E6E6E6", sw=0.8))
        p.append(T(64, gy + 3, f"{gv:.2f}", size=7.4, fill=C_MUT, anchor="end"))
    donly = [0.9534, 0.9405, 0.9614, 0.9564, 0.9423]
    dtfts = [0.9676, 0.9502, 0.9384, 0.9504, 0.8814]
    for i, (vd, vt) in enumerate(zip(donly, dtfts)):
        gx = 70 + i * 106
        p.append(R(gx + 8, yv(vd), 34, bot - yv(vd), rx=2, fill=C_BLUE, stroke="none", sw=0))
        p.append(T(gx + 25, yv(vd) - 5, f"{vd:.3f}", size=6.6, fill=C_BLUE_D))
        p.append(R(gx + 46, yv(vt), 34, bot - yv(vt), rx=2, fill=C_GREEN, stroke="none", sw=0))
        p.append(T(gx + 63, yv(vt) - 5, f"{vt:.3f}", size=6.6, fill=C_GREEN_D))
        p.append(T(gx + 44, bot + 16, f"fold {i}", size=7.8, fill=C_SUB))
    p.append(L(70, bot, 600, bot, color="#999999", sw=1.0))
    p.append(L(70, yv(0.9508), 600, yv(0.9508), color=C_BLUE, sw=1.0, dash="5,3"))
    p.append(L(70, yv(0.9376), 600, yv(0.9376), color=C_GREEN, sw=1.0, dash="5,3"))

    # right: pool + held-out
    p.append(R(652, 58, 412, 500, fill="#FCFCFD", stroke="#DDDDDD", sw=1.2))
    p.append(T(668, 82, "training pool", size=10.5, weight="bold", anchor="start"))
    bw1 = 380 * 1339 / 2339
    bw2 = 380 * 1000 / 2339
    p.append(R(668, 96, bw1, 18, rx=2, fill=C_BLUE, stroke="none", sw=0))
    p.append(R(668 + bw1, 96, bw2, 18, rx=2, fill=C_GREEN, stroke="none", sw=0))
    p.append(T(668, 134, "DENTEX  1,339 images · 21,624 boxes", size=8.2, anchor="start"))
    p.append(T(668, 148, "Tufts    1,000 images · 25,184 boxes", size=8.2, anchor="start"))
    p.append(T(668, 166, "combined 2,339 images · 46,808 boxes", size=8.4, weight="bold", anchor="start"))

    p.append(T(668, 198, "held-out official DENTEX validation", size=10.5, weight="bold", anchor="start"))
    p.append(T(668, 216, "50 images · 46 pathological · 182 target boxes", size=8.2, anchor="start"))
    p.append(R(668, 228, 380, 64, rx=6, fill="#F2F8F2", stroke=C_GREEN, sw=1.3))
    p.append(T(682, 250, "best: DENTEX + Tufts, fold 1", size=9.0, weight="bold", fill=C_GREEN_D, anchor="start"))
    p.append(T(682, 268, "target mAP50 0.9593 · precision 0.9864", size=8.6, anchor="start"))
    p.append(T(682, 284, "deployed as the locate_tooth backend", size=7.8, fill=C_SUB, anchor="start"))

    p.append(T(668, 320, "effect of co-training with Tufts", size=10.5, weight="bold", anchor="start"))
    p.append(T(668, 340, "precision 0.9637 -> 0.9680", size=8.4, anchor="start"))
    p.append(T(668, 356, "raw val mAP50 0.5820 -> 0.8695", size=8.4, anchor="start"))
    p.append(T(668, 372, "(different protocol from the target-filtered", size=7.6, fill=C_SUB, anchor="start"))
    p.append(T(668, 384, "CV above - both measured)", size=7.6, fill=C_SUB, anchor="start"))

    p.append(R(668, 404, 380, 130, rx=6, fill=C_BLUE_FILL, stroke="#C7D3E8", sw=1.1))
    p.append(T(682, 426, "trace-generation guarantee", size=9.0, weight="bold", fill=C_BLUE_D, anchor="start"))
    p.append(T(682, 444, "when the requested tooth exists in the ground truth,", size=7.8, anchor="start"))
    p.append(T(682, 456, "locate_tooth returns its exact box - the detector's", size=7.8, anchor="start"))
    p.append(T(682, 468, "measured accuracy governs only the unannotated-tooth", size=7.8, anchor="start"))
    p.append(T(682, 480, "fallback; what the model is shown is then perturbed", size=7.8, anchor="start"))
    p.append(T(682, 492, "independently (tiered), keeping accept-vs-nudge live", size=7.8, anchor="start"))

    return svg(1080, 584, "".join(p))


# ---------------------------------------------------------------------------
# Figure 7: image-input geometry (TPU canonical vs GPU native, decision open)
# ---------------------------------------------------------------------------


def build_input_geometry(a):
    p = []
    p.append(T(560, 24, "Image-input geometry: two execution targets, one invariant", size=14.5, weight="bold"))
    p.append(T(560, 42, "canonical views rendered by the project's own geometry module from DENTEX image 624; the target choice is deliberately open", size=9.5, fill=C_SUB))

    # native input
    p.append(IMG(32, 140, 320, 150, a["opg"]))
    p.append(R(32, 140, 320, 150, rx=3, stroke="#666666", sw=1.2, fill="none"))
    sx, sy = 320.0 / a["native_w"], 150.0 / a["native_h"]
    mx = 32 + CROP_FINAL_REGION[0] * sx
    my = 140 + CROP_FINAL_REGION[1] * sy
    mw = (CROP_FINAL_REGION[2] - CROP_FINAL_REGION[0]) * sx
    mh = (CROP_FINAL_REGION[3] - CROP_FINAL_REGION[1]) * sy
    p.append(R(mx, my, mw, mh, rx=0, fill="none", stroke=C_ORANGE, sw=1.6))
    p.append(T(192, 310, "native · 2,808 × 1,316 px", size=8.8, weight="bold"))
    p.append(T(192, 324, "~17k visual tokens if fed as-is", size=7.8, fill=C_SUB))

    # branch stem
    p.append(L(352, 215, 376, 215, color="#999999", sw=1.5))
    p.append(arrow([(376, 215), (398, 215)], color="#999999", sw=1.5))
    p.append(arrow([(376, 215), (376, 554), (398, 554)], color="#999999", sw=1.3, dash="4,3"))

    # TPU panel (canonical + static slots)
    p.append(R(400, 64, 704, 416, fill=C_BLUE_FILL, stroke="#C7D3E8", sw=1.2))
    p.append(R(416, 78, 96, 22, rx=11, fill=C_BLUE, stroke="none", sw=0))
    p.append(T(464, 93, "TPU v5e-8", size=8.8, weight="bold", fill="#FFFFFF"))
    p.append(T(1088, 93, "canonical resizing + static slot padding (required)", size=8.2, fill=C_BLUE_D, anchor="end"))

    p.append(IMG(430, 120, 290, 145, a["canon_full"]))
    p.append(R(430, 120, 290, 145, rx=2, stroke="#666666", sw=1.0, fill="none"))
    p.append(T(575, 116, "FULL · 1536 × 768 · 1,152 tokens", size=8.4, weight="bold", fill=C_BLUE_D))
    p.append(IMG(790, 112, 120, 180, a["canon_crop"]))
    p.append(R(790, 112, 120, 180, rx=2, stroke="#666666", sw=1.0, fill="none"))
    p.append(T(840, 312, "CROP · 256 × 384 · 96 tokens", size=8.4, weight="bold", fill=C_BLUE_D, anchor="end"))
    p.append(IMG(430, 300, 200, 150, a["canon_compare"]))
    p.append(R(430, 300, 200, 150, rx=2, stroke="#666666", sw=1.0, fill="none"))
    p.append(T(530, 294, "COMPARE · 512 × 384 · 192 tokens", size=8.4, weight="bold", fill=C_BLUE_D))

    # leader: the marked crop feeds the CROP view
    p.append(arrow([(226, 199), (226, 104), (850, 104), (850, 112)], color=C_ORANGE, sw=1.3, dash="4,3"))
    # the CROP view feeds the COMPARE composite (real tool output)
    p.append(arrow([(850, 292), (850, 375), (632, 375)], color="#999999", sw=1.3))

    p.append(T(430, 468, "static slots: 5 FULL + 10 CROP + 4 COMPARE = 19 slots = 7,488 vision tokens - one XLA graph, no recompilation", size=8.0, fill=C_BLUE_D, anchor="start"))

    # GPU panel
    p.append(R(400, 496, 704, 116, fill="#F7F8FA", stroke="#BBBBBB", sw=1.2))
    p.append(R(416, 510, 60, 22, rx=11, fill="#777777", stroke="none", sw=0))
    p.append(T(446, 525, "GPU", size=8.8, weight="bold", fill="#FFFFFF"))
    p.append(T(1088, 525, "runs so far: original images, dynamic shapes", size=8.2, fill="#555555", anchor="end"))
    p.append(T(416, 552, "GPU runs so far feed native images with dynamic padding; a --canonical-resize flag", size=8.0, anchor="start"))
    p.append(T(416, 566, "exists for the same views - the final choice is pending the first GPU run", size=8.0, anchor="start"))
    p.append(R(416, 580, 124, 22, rx=11, fill="#FFFFFF", stroke="#BBBBBB", sw=1.0))
    p.append(T(478, 594, "RTX 4090 · QLoRA", size=7.6, fill="#555555"))
    p.append(R(548, 580, 122, 22, rx=11, fill="#FFFFFF", stroke="#BBBBBB", sw=1.0))
    p.append(T(609, 594, "RTX Pro 6000", size=7.6, fill="#555555"))
    p.append(R(678, 580, 100, 22, rx=11, fill="#FFFFFF", stroke="#BBBBBB", sw=1.0))
    p.append(T(728, 594, "A100 · bf16", size=7.6, fill="#555555"))

    # decision band
    p.append(R(16, 628, 1088, 76, rx=8, fill=C_ORANGE_L, stroke="#F0B27A", sw=1.2))
    p.append(T(32, 650, "OPEN DECISION - execution target: GPU (RTX 4090 / RTX Pro 6000 / A100) vs TPU v5e-8; to be settled by the first GPU run", size=8.8, weight="bold", fill=C_ORANGE_D, anchor="start"))
    p.append(T(32, 668, "shared invariant: tools always execute on the native image · evaluation runs at original size · SFT, GRPO and evaluation use the same setting", size=8.2, fill=C_ORANGE_D, anchor="start"))
    p.append(T(32, 688, "TPU padding changes no gradients: the padded tail carries labels = -100 and attention_mask = 0 (verified against the reference implementation)", size=7.8, fill=C_ORANGE_D, anchor="start"))

    return svg(1120, 720, "".join(p))


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


BUILDERS = {
    "fig_traces_comparison.svg": build_traces_comparison,
    "fig_trace_multistage.svg": build_multistage,
    "fig_architecture.svg": build_architecture,
    "fig_corpus_composition.svg": build_corpus_composition,
    "fig_datasets.svg": build_dataset_catalog,
    "fig_grounding_cv.svg": build_grounding_cv,
    "fig_input_geometry.svg": build_input_geometry,
    "fig_pipeline.svg": build_pipeline,
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
