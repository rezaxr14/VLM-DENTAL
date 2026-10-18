"""Audit builder-generated diagrams for text that overflows its box.

For every <text> in paper/figures/svg/fig_*.svg (the builder-generated
diagrams) the label's width is measured with reportlab's Helvetica /
Courier metrics and checked against the smallest <rect> that contains its
anchor point (and against the canvas). Exit status 1 if anything overflows.

    python scripts/paper/audit_diagram_text.py
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

from reportlab.pdfbase.pdfmetrics import stringWidth

SVG_DIR = Path(__file__).resolve().parents[2] / "paper" / "figures" / "svg"
NAMES = sorted(f.stem[len("fig_"):] for f in SVG_DIR.glob("fig_*.svg"))  # matplotlib figures have no <text>
RECT = re.compile(r'<rect x="([-\d.]+)" y="([-\d.]+)" width="([\d.]+)" height="([\d.]+)"[^>]*?/>')
TEXT = re.compile(r'<text x="([-\d.]+)" y="([-\d.]+)" font-family="([^"]*)" font-size="([\d.]+)"([^>]*)>([^<]*)</text>')


def audit(name: str) -> list[str]:
    svg = (SVG_DIR / f"fig_{name}.svg").read_text(encoding="utf-8")
    width = float(re.search(r'viewBox="0 0 ([\d.]+)', svg).group(1))
    rects = [tuple(float(m.group(i)) for i in range(1, 5)) for m in RECT.finditer(svg)]
    rects = [r for r in rects if r[2] < width - 5]  # drop the page background
    problems = []
    for m in TEXT.finditer(svg):
        x, y, fam, size = float(m.group(1)), float(m.group(2)), m.group(3), float(m.group(4))
        rest, txt = m.group(5), html.unescape(m.group(6))
        if not txt.strip():
            continue
        face = "Courier" if ("Mono" in fam or "Courier" in fam) else ("Helvetica-Bold" if "bold" in rest else "Helvetica")
        w = stringWidth(txt, face, size)
        anchor = "end" if 'text-anchor="end"' in rest else ("middle" if 'text-anchor="middle"' in rest else "start")
        x0 = x - w / 2 if anchor == "middle" else (x - w if anchor == "end" else x)
        x1 = x0 + w
        inside = [r for r in rects if r[0] <= x <= r[0] + r[2] and r[1] <= y <= r[1] + r[3]]
        if not inside:
            if x0 < 0 or x1 > width:
                problems.append(f"off-canvas {x0:.0f}..{x1:.0f}: {txt[:60]!r}")
            continue
        r = min(inside, key=lambda r: r[2] * r[3])
        if x0 < r[0] + 2 or x1 > r[0] + r[2] - 2 or y > r[1] + r[3] - 1:
            problems.append(f"{txt[:60]!r}: text {x0:.0f}..{x1:.0f} vs box {r[0]:.0f}..{r[0] + r[2]:.0f} (y {y:.0f}/{r[1] + r[3]:.0f})")
    return problems


def main() -> int:
    total = 0
    for name in NAMES:
        problems = audit(name)
        total += len(problems)
        print(f"{name:24s} {'ok' if not problems else str(len(problems)) + ' issue(s)'}")
        for pr in problems:
            print("   ", pr)
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
