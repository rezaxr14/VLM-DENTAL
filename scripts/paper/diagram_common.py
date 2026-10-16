"""Shared helpers for the generated paper diagrams (build_diagrams.py).

Everything here is plain-SVG string assembly with two hard constraints from
the toolchain (svglib -> reportlab -> PDF):

- Only Helvetica-family fonts survive conversion; no text wrapping happens
  automatically, so every label is hand-broken into lines.
- reportlab base-14 fonts use the WinAnsi charset: use "-" not U+2192,
  draw check/cross marks as paths, never as glyphs.

Raster assets are embedded as base64 PNG data URIs (verified to survive
svg2rlg). All radiograph content is extracted from the local DENTEX copy and
the released trace for DENTEX image 624 -- no downloads (Rule 16).
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_DIR = REPO_ROOT / "paper"
ASSET_DIR = PAPER_DIR / "figures" / "assets"
SVG_DIR = PAPER_DIR / "figures" / "svg"

FONT = "Helvetica"
MONO = "Courier"

# Muted Q1-style palette (matplotlib "deep"-adjacent, print-safe).
C_BLUE = "#4C72B0"
C_BLUE_D = "#34548A"
C_BLUE_FILL = "#F4F8FD"
C_GREEN = "#55A868"
C_GREEN_D = "#2F6B3C"
C_GREEN_FILL = "#F4FBF6"
C_ORANGE = "#ff7f0e"
C_ORANGE_L = "#FFF4E8"
C_ORANGE_D = "#8A4B06"
C_RED = "#C44E52"
C_SAND_FILL = "#FDFBF2"
C_SAND_D = "#8A6D1F"
C_TXT = "#444444"
C_SUB = "#777777"
C_MUT = "#999999"


_AMP = chr(38)
_LT = chr(60)
_GT = chr(62)


def esc(s: str) -> str:
    return (
        s.replace(_AMP, _AMP + "amp;")
        .replace(_LT, _AMP + "lt;")
        .replace(_GT, _AMP + "gt;")
    )


def T(x, y, s, size=9, fill=C_TXT, anchor="middle", weight=None, style=None, font=None):
    a = f'<text x="{x}" y="{y}" font-family="{font or FONT}" font-size="{size}"'
    if weight:
        a += f' font-weight="{weight}"'
    if style:
        a += f' font-style="{style}"'
    if anchor != "start":
        a += f' text-anchor="{anchor}"'
    return a + f' fill="{fill}">{esc(s)}</text>'


def R(x, y, w, h, rx=6, fill="#FFFFFF", stroke=C_MUT, sw=1.1, dash=None, opacity=None):
    a = f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"'
    if dash:
        a += f' stroke-dasharray="{dash}"'
    if opacity is not None:
        a += f' opacity="{opacity}"'
    return a + "/>"


def L(x1, y1, x2, y2, color=C_MUT, sw=1.0, dash=None):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{d}/>'


def IMG(x, y, w, h, b64):
    return f'<image x="{x}" y="{y}" width="{w}" height="{h}" xlink:href="data:image/png;base64,{b64}"/>'


_ARROW_COLORS = {}


def arrow(points, color="#666666", sw=1.5, dash=None):
    """Orthogonal polyline with an arrowhead. Registers its marker color."""
    key = color.strip("#")
    _ARROW_COLORS.setdefault(key, color)
    pts = " ".join(f"{p[0]:.1f},{p[1]:.1f}" for p in points)
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{sw}"'
        f'{d} marker-end="url(#arr{key})"/>'
    )


def marker_defs() -> str:
    parts = ["<defs>"]
    for key, color in sorted(_ARROW_COLORS.items()):
        parts.append(
            f'<marker id="arr{key}" viewBox="0 0 10 10" refX="8.5" refY="5" '
            f'markerWidth="6.5" markerHeight="6.5" orient="auto">'
            f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{color}"/></marker>'
        )
    parts.append("</defs>")
    return "".join(parts)


def svg(w, h, body):
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{w}" height="{h}" viewBox="0 0 {w} {h}">'
        f'<rect width="{w}" height="{h}" fill="#FFFFFF"/>'
        + body
        + marker_defs()
        + "</svg>"
    )


def badge(n, x, y, r=8.0, fill=C_BLUE):
    return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}"/>' + T(
        x, y + 3.2, str(n), size=9.5, fill="#FFFFFF", weight="bold"
    )


# ---------------------------------------------------------------------------
# Icons (vector paths -- glyphs like U+2713 do not survive WinAnsi)
# ---------------------------------------------------------------------------


def icon_check(x, y, s=6.0, color=C_GREEN_D):
    return (
        f'<path d="M {x} {y + s * 0.55} L {x + s * 0.36} {y + s} L {x + s} {y - s * 0.25}" '
        f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>'
    )


def icon_cross(x, y, s=5.0, color=C_RED):
    return (
        f'<path d="M {x} {y} L {x + s} {y + s} M {x + s} {y} L {x} {y + s}" '
        f'fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round"/>'
    )


def icon_db(x, y, w=14.0, h=16.0, color=C_BLUE):
    rx = w / 2
    return (
        f'<ellipse cx="{x + rx}" cy="{y + 3}" rx="{rx}" ry="2.8" fill="none" stroke="{color}" stroke-width="1.4"/>'
        f'<path d="M {x} {y + 3} L {x} {y + h - 3} A {rx} 2.8 0 0 0 {x + w} {y + h - 3} L {x + w} {y + 3}" '
        f'fill="none" stroke="{color}" stroke-width="1.4"/>'
        f'<path d="M {x} {y + 8.5} A {rx} 2.8 0 0 0 {x + w} {y + 8.5}" fill="none" stroke="{color}" stroke-width="1.1"/>'
    )


def icon_img(x, y, w=16.0, h=12.0, color="#888888"):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="1.5" fill="none" stroke="{color}" stroke-width="1.3"/>'
        f'<circle cx="{x + w * 0.3}" cy="{y + h * 0.35}" r="1.2" fill="{color}"/>'
        f'<path d="M {x + 1.5} {y + h - 2} L {x + w * 0.45} {y + h * 0.45} '
        f'L {x + w * 0.65} {y + h * 0.7} L {x + w - 2} {y + h * 0.4}" '
        f'fill="none" stroke="{color}" stroke-width="1.2"/>'
    )


def icon_magnifier(x, y, r=5.0, color="#555555"):
    return (
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="{color}" stroke-width="1.6"/>'
        f'<line x1="{x + r * 0.75}" y1="{y + r * 0.75}" x2="{x + r * 1.9}" y2="{y + r * 1.9}" '
        f'stroke="{color}" stroke-width="2.2" stroke-linecap="round"/>'
    )


def icon_shield(x, y, w=13.0, h=15.0, color=C_GREEN_D):
    return (
        f'<path d="M {x} {y} L {x + w} {y} L {x + w} {y + h * 0.55} '
        f'Q {x + w} {y + h * 0.85} {x + w / 2} {y + h} '
        f'Q {x} {y + h * 0.85} {x} {y + h * 0.55} Z" '
        f'fill="none" stroke="{color}" stroke-width="1.4"/>'
        + icon_check(x + w * 0.28, y + h * 0.3, s=4.6, color=color)
    )


def icon_doc(x, y, w=11.0, h=14.0, color="#888888"):
    return (
        f'<path d="M {x} {y} L {x + w * 0.65} {y} L {x + w} {y + h * 0.25} L {x + w} {y + h} L {x} {y + h} Z" '
        f'fill="none" stroke="{color}" stroke-width="1.3"/>'
        f'<path d="M {x + w * 0.65} {y} L {x + w * 0.65} {y + h * 0.25} L {x + w} {y + h * 0.25}" '
        f'fill="none" stroke="{color}" stroke-width="1.1"/>'
        f'<line x1="{x + 2}" y1="{y + h * 0.5}" x2="{x + w - 2}" y2="{y + h * 0.5}" stroke="{color}" stroke-width="1"/>'
        f'<line x1="{x + 2}" y1="{y + h * 0.68}" x2="{x + w - 2}" y2="{y + h * 0.68}" stroke="{color}" stroke-width="1"/>'
    )


def icon_gear(x, y, r=6.0, color="#666666"):
    import math

    parts = [f'<circle cx="{x}" cy="{y}" r="{r * 0.55}" fill="none" stroke="{color}" stroke-width="1.5"/>']
    for i in range(8):
        a = math.pi / 4 * i
        x1, y1 = x + r * 0.72 * math.cos(a), y + r * 0.72 * math.sin(a)
        x2, y2 = x + r * 1.08 * math.cos(a), y + r * 1.08 * math.sin(a)
        parts.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{color}" stroke-width="1.7"/>'
        )
    return "".join(parts)


# ---------------------------------------------------------------------------
# Real radiograph assets: DENTEX image 624 (train_318.png) + its released trace
# ---------------------------------------------------------------------------

# Real returned/ground-truth boxes for image 624, parsed from the trace
# observations (locate_tooth + the three nudge_crop results) and the
# quadrant-enumeration-disease ground truth. Native pixel coords.
BOXES_624 = {
    "gt": [1621.0, 696.0, 84.0, 322.0],
    "loc": [1668.4, 917.3, 84.0, 322.0],
    "n1": [1685.2, 885.1, 50.4, 193.2],
    "n2": [1692.8, 817.5, 35.3, 135.2],
    "n3": [1696.3, 749.9, 28.2, 108.2],
}

# Inset crop region for the magnifier view [x0, y0, x1, y1] (native px).
# Must fully contain the located box (y extends to 1239) and the ground-truth box.
INSET_REGION = (1560, 630, 1840, 1290)
# Final zoom-crop region shown as "the crop the teacher finally saw".
CROP_FINAL_REGION = (1655, 700, 1760, 1055)
# Pre-correction crop region (what the first (mis)located box actually framed).
CROP_LOCATED_REGION = (1630, 880, 1800, 1270)


def b64_of(path: Path) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


def _save(img, name: str) -> tuple[str, int, int]:
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    p = ASSET_DIR / name
    img.save(p)
    return b64_of(p), img.width, img.height


def load_assets(force: bool = False) -> dict:
    """Extract and cache base64 PNG assets from DENTEX image 624 (train_318)."""
    cache = ASSET_DIR / "_assets_b64.json"
    if cache.exists() and not force:
        try:
            data = json.loads(cache.read_text(encoding="utf-8"))
            if len(data.get("opg", "")) > 1000:
                return data
        except (json.JSONDecodeError, OSError):
            pass

    from PIL import Image

    xray_dir = REPO_ROOT / "data/dentex/DENTEX/training_data/quadrant-enumeration-disease/xrays"
    ann_json = REPO_ROOT / "data/dentex/DENTEX/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"
    ann = json.loads(ann_json.read_text(encoding="utf-8"))
    id2file = {im["id"]: im["file_name"] for im in ann["images"]}
    src = xray_dir / id2file[624]
    im = Image.open(src).convert("RGB")
    W, H = im.size

    assets: dict = {"source_file": id2file[624], "native_w": W, "native_h": H}

    opg = im.resize((608, round(608 * H / W)), Image.LANCZOS)
    assets["opg"], assets["opg_w"], assets["opg_h"] = _save(opg, "opg_624.png")

    c = im.crop(CROP_FINAL_REGION)
    ch = 520
    assets["crop_final"], assets["crop_final_w"], assets["crop_final_h"] = _save(
        c.resize((round(ch * c.width / c.height), ch), Image.LANCZOS), "crop_final_624.png"
    )
    assets["crop_final_aspect"] = c.height / c.width

    c2 = im.crop(CROP_LOCATED_REGION)
    ch2 = 460
    assets["crop_located"], assets["crop_located_w"], assets["crop_located_h"] = _save(
        c2.resize((round(ch2 * c2.width / c2.height), ch2), Image.LANCZOS), "crop_located_624.png"
    )
    assets["crop_located_aspect"] = c2.height / c2.width

    x0, y0, x1, y1 = INSET_REGION
    ins = im.crop((x0, y0, x1, y1))
    assets["inset"], assets["inset_w"], assets["inset_h"] = _save(
        ins.resize((260, 612), Image.LANCZOS), "inset_624.png"
    )

    # Canonical views: REAL outputs of the project's own geometry module
    # (dental_agent.utils.canonical.to_canonical) and the real
    # contralateral_compare tool -- the exact views the policy sees.
    crop_box = [
        CROP_FINAL_REGION[0],
        CROP_FINAL_REGION[1],
        CROP_FINAL_REGION[2] - CROP_FINAL_REGION[0],
        CROP_FINAL_REGION[3] - CROP_FINAL_REGION[1],
    ]
    try:
        import sys as _sys

        if str(REPO_ROOT) not in _sys.path:
            _sys.path.insert(0, str(REPO_ROOT))
        from dental_agent.utils.canonical import to_canonical
        from dental_agent.tools.contralateral import tool_contralateral_compare

        full_view = to_canonical(im, "FULL")
        crop_view = to_canonical(im.crop(tuple(CROP_FINAL_REGION)), "CROP")
        composite = tool_contralateral_compare(im, crop_box, 3)
        comp_view = to_canonical(composite, "COMPARE")
    except Exception as exc:  # noqa: BLE001
        print(f"warn: could not import canonical module ({exc}); replicating geometry inline")
        full_view = _letterbox(im, 1536, 768)
        crop_view = _letterbox(im.crop(tuple(CROP_FINAL_REGION)), 256, 384)
        comp_view = _letterbox(_mirror_composite(im, crop_box), 512, 384)

    assets["canon_full"], *_ = _save(full_view.resize((768, 384), Image.LANCZOS), "canon_full_624.png")
    assets["canon_crop"], *_ = _save(crop_view, "canon_crop_624.png")
    assets["canon_compare"], *_ = _save(comp_view, "canon_compare_624.png")

    cache.write_text(json.dumps(assets), encoding="utf-8")
    return assets


def _letterbox(img, cw: int, ch: int):
    """Inline replica of canonical.to_canonical (fallback only)."""
    from PIL import Image

    scale = min(cw / img.width, ch / img.height)
    nw, nh = max(1, round(img.width * scale)), max(1, round(img.height * scale))
    resized = img.resize((nw, nh), Image.LANCZOS)
    canvas = Image.new("RGB", (cw, ch), (0, 0, 0))
    canvas.paste(resized, ((cw - nw) // 2, (ch - nh) // 2))
    return canvas


def _mirror_composite(im, crop_box):
    """Inline replica of tools.contralateral.tool_contralateral_compare (fallback only)."""
    from PIL import Image

    x, y, w, h = [int(v) for v in crop_box]
    W, H = im.size
    target = im.crop((max(0, x), max(0, y), min(W, x + w), min(H, y + h)))
    mx1, mx2 = max(0, int(W - x - w)), min(W, int(W - x))
    my1, my2 = max(H // 2, y), min(H, y + h)  # quadrant 3 -> lower jaw half
    mirror = im.crop((mx1, my1, mx2, my2)) if mx2 > mx1 and my2 > my1 else target
    comp = Image.new("RGB", (target.width + mirror.width + 10, max(target.height, mirror.height)), (0, 0, 0))
    comp.paste(target, (0, 0))
    comp.paste(mirror, (target.width + 10, 0))
    return comp


def scaled_box(box, region, disp_x, disp_y, disp_w, disp_h):
    """Map a native box into display coords given the native region it sits in."""
    rx0, ry0, rx1, ry1 = region
    sx = disp_w / (rx1 - rx0)
    sy = disp_h / (ry1 - ry0)
    return (
        disp_x + (box[0] - rx0) * sx,
        disp_y + (box[1] - ry0) * sy,
        box[2] * sx,
        box[3] * sy,
    )
