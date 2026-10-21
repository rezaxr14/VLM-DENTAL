"""Audit every connector (arrow) in the generated diagram SVGs.

Run after ``build_diagrams.py``::

    python scripts/paper/audit_diagram_arrows.py            # all figures
    python scripts/paper/audit_diagram_arrows.py fig_architecture   # one figure
    python scripts/paper/audit_diagram_arrows.py --strict   # warnings fail too

An arrow is the ``<polyline>`` + ``<polygon>`` pair written by
``diagram_common.arrow()``. Rules (ERROR unless noted):

  tip-float      arrow tip is not within TOL of any box/image edge
  tip-inside     arrow tip lies inside a card by more than DEPTH (overshoot)
  tail-float     arrow tail is not on a box edge (allowed only when the arrow was
                 emitted with ``free="tail"`` -- an input that enters from outside)
  diagonal       a segment is neither horizontal nor vertical (also bare <line>s)
  crosses-text   a segment passes through a text label
  crosses-card   a segment crosses the border of a card it does not start/end on
  badge-on-arrow a numbered badge touches an arrow shaft
  label-on-arrow (WARN) a text label sits within 2 units of a shaft
  tiny-segment   an elbow leg is < 12 units, or < MIN_SHAFT of shaft shows behind the head
  overlap        two connectors share a collinear stretch
  along-border   a segment runs along a box border for more than a few units

Boxes are every <rect>/<image> except the full-canvas background. Full-width
stage bands count as containers: crossing them is allowed.
"""

from __future__ import annotations

import html
import math
import re
import sys
from pathlib import Path

from reportlab.pdfbase.pdfmetrics import stringWidth

SVG_DIR = Path(__file__).resolve().parent.parent.parent / "paper" / "figures" / "svg"

TOL = 1.8  # tip/tail must be this close to an edge
DEPTH = 3.0  # tip may not be deeper than this inside a card
HEAD_K = 5.6  # arrow(): head length = HEAD_K * stroke width
MIN_SHAFT = 4.0  # visible shaft required behind the head (WARN below 8)
ATTR = re.compile(r'([\w:-]+)="([^"]*)"')


def attrs(tag: str) -> dict:
    return {k: html.unescape(v) for k, v in ATTR.findall(tag)}


def f(d: dict, k: str, default: float = 0.0) -> float:
    try:
        return float(d.get(k, default))
    except ValueError:
        return default


class Fig:
    def __init__(self, path: Path):
        self.name = path.stem
        s = path.read_text(encoding="utf-8")
        m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', s)
        self.W, self.H = float(m.group(1)), float(m.group(2))
        self.rects: list[tuple[float, float, float, float]] = []
        self.text: list[tuple[float, float, float, float, str]] = []
        self.badges: list[tuple[float, float, float]] = []
        self.lines: list[tuple[float, float, float, float]] = []
        self.arrows: list[dict] = []
        self._parse(s)
        self._mark_containers()

    # ---- parsing -----------------------------------------------------
    def _parse(self, s: str) -> None:
        # arrows: optional <!--free:tail--> comment, polyline, polygon
        arrow_re = re.compile(r'(?:<!--free:(\w+)-->)?<polyline ([^>]*)/><polygon ([^>]*)/>')
        for m in arrow_re.finditer(s):
            free = m.group(1) or ""
            pl, pg = attrs(m.group(2)), attrs(m.group(3))
            pts = [tuple(map(float, p.split(","))) for p in pl["points"].split()]
            tri = [tuple(map(float, p.split(","))) for p in pg["points"].split()]
            tip = tri[0]
            self.arrows.append(
                {"pts": pts + [tip], "sw": f(pl, "stroke-width", 1.5), "free": free, "tip": tip}
            )
        arrow_spans = [m.span() for m in arrow_re.finditer(s)]

        def in_arrow(pos: int) -> bool:
            return any(a <= pos < b for a, b in arrow_spans)

        for m in re.finditer(r"<rect ([^>]*)/>", s):
            a = attrs(m.group(1))
            w, h = f(a, "width"), f(a, "height")
            if w >= self.W - 0.5 and h >= self.H - 0.5:
                continue  # canvas background
            if w < 3 or h < 3:
                continue  # bars/ticks are not connector targets
            self.rects.append((f(a, "x"), f(a, "y"), w, h))
        for m in re.finditer(r"<image ([^>]*)/>", s):
            a = attrs(m.group(1))
            self.rects.append((f(a, "x"), f(a, "y"), f(a, "width"), f(a, "height")))
        for m in re.finditer(r"<line ([^>]*)/>", s):
            if in_arrow(m.start()):
                continue
            a = attrs(m.group(1))
            self.lines.append((f(a, "x1"), f(a, "y1"), f(a, "x2"), f(a, "y2")))
        for m in re.finditer(r"<circle ([^>]*)/>", s):
            a = attrs(m.group(1))
            if f(a, "r") >= 6.5:
                self.badges.append((f(a, "cx"), f(a, "cy"), f(a, "r")))
        for m in re.finditer(r"<text ([^>]*)>([^<]*)</text>", s):
            a = attrs(m.group(1))
            txt = html.unescape(m.group(2))
            if not txt.strip():
                continue
            size = f(a, "font-size", 9)
            face = "Courier" if "Courier" in a.get("font-family", "") else (
                "Helvetica-Bold" if a.get("font-weight") == "bold" else "Helvetica"
            )
            w = stringWidth(txt, face, size)
            x, y = f(a, "x"), f(a, "y")
            anchor = a.get("text-anchor", "start")
            x0 = x - w / 2 if anchor == "middle" else (x - w if anchor == "end" else x)
            self.text.append((x0, y - 0.74 * size, x0 + w, y + 0.2 * size, txt))

    def _mark_containers(self) -> None:
        self.container = []
        for i, (x, y, w, h) in enumerate(self.rects):
            inside = sum(
                1
                for j, (a, b, c, d) in enumerate(self.rects)
                if j != i and a >= x - 0.5 and b >= y - 0.5 and a + c <= x + w + 0.5 and b + d <= y + h + 0.5
                and c * d < w * h * 0.9
            )
            self.container.append(inside >= 3 or w >= 560)

    # ---- geometry ----------------------------------------------------
    @staticmethod
    def edge_dist(p, r) -> float:
        """Distance from p to the *outline* of rect r (0 when p is on it)."""
        x, y, w, h = r
        px, py = p
        dx = max(x - px, 0, px - (x + w))
        dy = max(y - py, 0, py - (y + h))
        if dx > 0 or dy > 0:  # outside
            return math.hypot(dx, dy)
        return min(px - x, x + w - px, py - y, y + h - py)

    @staticmethod
    def inside(p, r, depth=0.0) -> bool:
        x, y, w, h = r
        return x + depth < p[0] < x + w - depth and y + depth < p[1] < y + h - depth

    @staticmethod
    def seg_border_crossings(a, b, r):
        """Parameter t (0..1) where segment a-b crosses rect r's outline."""
        x, y, w, h = r
        out = []
        for (x1, y1), (x2, y2) in (
            ((x, y), (x + w, y)), ((x, y + h), (x + w, y + h)),
            ((x, y), (x, y + h)), ((x + w, y), (x + w, y + h)),
        ):
            d1x, d1y, d2x, d2y = b[0] - a[0], b[1] - a[1], x2 - x1, y2 - y1
            den = d1x * d2y - d1y * d2x
            if abs(den) < 1e-9:
                continue
            t = ((x1 - a[0]) * d2y - (y1 - a[1]) * d2x) / den
            u = ((x1 - a[0]) * d1y - (y1 - a[1]) * d1x) / den
            if 0 <= t <= 1 and 0 <= u <= 1:
                out.append(t)
        return out

    @staticmethod
    def seg_rect_hit(a, b, box, pad=0.0) -> bool:
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
        lo, hi = 0.0, 1.0
        for p, q, mn, mx in ((a[0], b[0], x0, x1), (a[1], b[1], y0, y1)):
            d = q - p
            if abs(d) < 1e-9:
                if p < mn or p > mx:
                    return False
            else:
                t1, t2 = (mn - p) / d, (mx - p) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                lo, hi = max(lo, t1), min(hi, t2)
                if lo > hi:
                    return False
        return True

    @staticmethod
    def seg_point_dist(a, b, p) -> float:
        dx, dy = b[0] - a[0], b[1] - a[1]
        n2 = dx * dx + dy * dy
        t = 0 if n2 == 0 else max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / n2))
        return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))

    # ---- audit -------------------------------------------------------
    def audit(self) -> list[tuple[str, str, str]]:
        issues: list[tuple[str, str, str]] = []

        def add(sev, rule, msg):
            issues.append((sev, rule, msg))

        segs_all = []
        for ai, ar in enumerate(self.arrows):
            pts = ar["pts"]
            sw = ar["sw"]
            head = HEAD_K * sw
            tag = f"arrow#{ai} ({pts[0][0]:.0f},{pts[0][1]:.0f})->({pts[-1][0]:.0f},{pts[-1][1]:.0f})"
            tip, tail = pts[-1], pts[0]

            # tip
            dists = [(self.edge_dist(tip, r), r) for r in self.rects]
            dmin, rmin = min(dists, key=lambda t: t[0]) if dists else (99, None)
            if ar["free"] in ("tip", "both"):
                pass
            elif dmin > TOL:
                add("ERROR", "tip-float", f"{tag}: tip is {dmin:.1f} from the nearest edge")
            else:
                for i, r in enumerate(self.rects):
                    if self.container[i]:
                        continue
                    if self.inside(tip, r, DEPTH):
                        add("ERROR", "tip-inside", f"{tag}: tip enters a card by >{DEPTH:.0f}")
                        break
            # tail
            if ar["free"] not in ("tail", "both"):
                if min((self.edge_dist(tail, r) for r in self.rects), default=99) > TOL:
                    add("ERROR", "tail-float", f"{tag}: tail is not on a box edge")

            touched = [i for i, r in enumerate(self.rects)
                       if self.edge_dist(tip, r) <= TOL or self.edge_dist(tail, r) <= TOL]

            for si in range(len(pts) - 1):
                a, b = pts[si], pts[si + 1]
                L = math.hypot(b[0] - a[0], b[1] - a[1])
                segs_all.append((ai, a, b))
                if abs(b[0] - a[0]) > 0.6 and abs(b[1] - a[1]) > 0.6:
                    add("ERROR", "diagonal", f"{tag}: segment {si} is diagonal")
                # pts = polyline points (ending at the head BASE) + tip, so the last
                # entry is the head itself and the visible shaft is the segment before it
                if si == len(pts) - 2:
                    continue  # the arrowhead (length HEAD_K*sw by construction)
                final_leg = si == len(pts) - 3
                if final_leg and L < MIN_SHAFT:
                    add("ERROR", "tiny-segment", f"{tag}: only {L:.1f} of shaft behind the head")
                elif final_leg and L < 8 and len(pts) == 3:
                    add("WARN", "tiny-segment", f"{tag}: short shaft ({L:.1f}) behind the head")
                elif not final_leg and L < 12:
                    add("ERROR", "tiny-segment", f"{tag}: elbow leg {si} only {L:.1f} long")
                for t in self.text:
                    if any(math.hypot((t[0] + t[2]) / 2 - bx, (t[1] + t[3]) / 2 - by) < br for bx, by, br in self.badges):
                        continue  # the digit inside a badge
                    box = (t[0] + 0.3, t[1] + 0.3, t[2] - 0.3, t[3] - 0.3)
                    if self.seg_rect_hit(a, b, box):
                        add("ERROR", "crosses-text", f"{tag}: crosses text {t[4]!r}")
                    elif self.seg_rect_hit(a, b, (t[0], t[1], t[2], t[3]), pad=2.0) and L > 8:
                        add("WARN", "label-on-arrow", f"{tag}: text {t[4]!r} within 2 units of the shaft")
                for bx, by, br in self.badges:
                    if self.seg_point_dist(a, b, (bx, by)) < br + 1.2:
                        add("ERROR", "badge-on-arrow", f"{tag}: badge at ({bx:.0f},{by:.0f}) touches the shaft")
                for i, r in enumerate(self.rects):
                    if i in touched or self.container[i]:
                        continue
                    ts = [t for t in self.seg_border_crossings(a, b, r) if 0.02 < t < 0.98]
                    if ts and L * min(ts) > 2 and L * (1 - max(ts)) > 2:
                        add("ERROR", "crosses-card", f"{tag}: crosses the border of rect {tuple(round(v) for v in r)}")
                    # running along a border
                    x, y, w, h = r
                    if abs(b[1] - a[1]) < 0.3 and (abs(a[1] - y) < 1.2 or abs(a[1] - (y + h)) < 1.2):
                        ov = min(max(a[0], b[0]), x + w) - max(min(a[0], b[0]), x)
                        if ov > 6:
                            add("ERROR", "along-border", f"{tag}: runs along a horizontal border for {ov:.0f}")
                    if abs(b[0] - a[0]) < 0.3 and (abs(a[0] - x) < 1.2 or abs(a[0] - (x + w)) < 1.2):
                        ov = min(max(a[1], b[1]), y + h) - max(min(a[1], b[1]), y)
                        if ov > 6:
                            add("ERROR", "along-border", f"{tag}: runs along a vertical border for {ov:.0f}")

        # overlapping connectors
        for i, (ai, a, b) in enumerate(segs_all):
            for aj, c, d in segs_all[i + 1:]:
                if ai == aj:
                    continue
                if abs(a[0] - b[0]) < 0.3 and abs(c[0] - d[0]) < 0.3 and abs(a[0] - c[0]) < 1.5:
                    ov = min(max(a[1], b[1]), max(c[1], d[1])) - max(min(a[1], b[1]), min(c[1], d[1]))
                elif abs(a[1] - b[1]) < 0.3 and abs(c[1] - d[1]) < 0.3 and abs(a[1] - c[1]) < 1.5:
                    ov = min(max(a[0], b[0]), max(c[0], d[0])) - max(min(a[0], b[0]), min(c[0], d[0]))
                else:
                    continue
                if ov > 3:
                    add("ERROR", "overlap", f"arrows #{ai} and #{aj} share a {ov:.0f}-unit stretch")
        # bare <line> leaders (zoom leaders, rules): diagonals are not allowed
        for x1, y1, x2, y2 in self.lines:
            if abs(x2 - x1) > 0.6 and abs(y2 - y1) > 0.6 and math.hypot(x2 - x1, y2 - y1) > 10:  # shorter strokes are icon glyphs
                issues.append(("ERROR", "diagonal", f"leader line ({x1:.0f},{y1:.0f})->({x2:.0f},{y2:.0f}) is diagonal"))
        return issues


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv
    files = sorted(SVG_DIR.glob("fig_*.svg"))
    if args:
        files = [p for p in files if any(a in p.stem for a in args)]
    n_err = n_warn = 0
    for p in files:
        fig = Fig(p)
        if not fig.arrows and not fig.lines:
            continue
        issues = fig.audit()
        errs = [i for i in issues if i[0] == "ERROR"]
        warns = [i for i in issues if i[0] == "WARN"]
        n_err += len(errs)
        n_warn += len(warns)
        status = "ok" if not issues else f"{len(errs)} error(s), {len(warns)} warning(s)"
        print(f"{fig.name}: {len(fig.arrows)} arrows - {status}")
        for sev, rule, msg in issues:
            print(f"   {sev:5s} [{rule}] {msg}")
    print(f"\n{n_err} error(s), {n_warn} warning(s)")
    return 1 if n_err or (strict and n_warn) else 0


if __name__ == "__main__":
    sys.exit(main())
