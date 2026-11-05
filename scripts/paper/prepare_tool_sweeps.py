#!/usr/bin/env python3
"""Parameter sweeps of the project's own tools on one real DENTEX region (image 334).

Writes three images next to the params_dentex showcase assets:

  sweep_window.jpg    x = window center 0..255, y = window width 20..250 (one tooth region, windowed pointwise;
                      checked against dental_agent/tools/windowing.py)
  sweep_denoise.jpg   bilateral strength 0 -> 1 left to right (tool_denoise run per slice)
  sweep_contrast.jpg  contrast factor 1 -> 3 left to right (tool_enhance_contrast run per slice)

Each slice of a sweep is the real tool output at that parameter, so the picture is a continuous parameter axis.
Used by scripts/paper/q1_figures/fig_tools_output.py.

    python scripts/paper/prepare_tool_sweeps.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_showcase import REPO, _load_tools  # noqa: E402

D = REPO / "paper" / "figures" / "assets" / "showcase" / "params_dentex"
NATIVE = (2953, 1316)
REGION_MAP = (760, 480, 760, 320)      # native px: x, y, w, h  (teeth 14-17, upper right)
REGION_STRIP = (760, 520, 760, 200)
C_RANGE, W_RANGE = (0, 255), (20, 250)
K_SLICES = 96
STRENGTH = (0.0, 1.0)
FACTOR = (1.0, 3.0)


def crop(im: Image.Image, r):
    x, y, w, h = r
    return im.crop((x, y, x + w, y + h))


def window_map(region: Image.Image, tools) -> Image.Image:
    g = np.array(region.convert("L"), dtype=np.float32)
    h, w = g.shape
    c = np.linspace(*C_RANGE, w, dtype=np.float32)[None, :]
    wd = np.linspace(*W_RANGE, h, dtype=np.float32)[:, None]
    lo, hi = c - wd / 2.0, c + wd / 2.0
    out = ((np.clip(g, lo, hi) - lo) / (hi - lo)) * 255.0
    out = out.astype(np.uint8)
    # check the pointwise formula against the real tool at the four values the agent used
    for preset, (cc, ww) in {"bone": (128, 100), "enamel": (200, 100), "soft_tissue": (80, 150)}.items():
        ref = np.array(tools["windowing"].tool_window_level(region, preset=preset).convert("L"), dtype=np.int16)
        lo2, hi2 = cc - ww / 2.0, cc + ww / 2.0
        mine = (((np.clip(g, lo2, hi2) - lo2) / (hi2 - lo2)) * 255.0).astype(np.uint8).astype(np.int16)
        assert np.abs(ref - mine).max() <= 1, preset
    return Image.fromarray(out).convert("RGB")


def sweep(region: Image.Image, fn, lo: float, hi: float) -> Image.Image:
    w, h = region.size
    out = Image.new("RGB", (w, h))
    edges = np.linspace(0, w, K_SLICES + 1).round().astype(int)
    for i in range(K_SLICES):
        p = lo + (hi - lo) * (i + 0.5) / K_SLICES
        im = fn(region, p).convert("RGB")
        out.paste(im.crop((edges[i], 0, edges[i + 1], h)), (edges[i], 0))
    return out


def main() -> None:
    tools = _load_tools()
    full = Image.open(D / "image.jpg").convert("RGB")
    native = full.resize(NATIVE, Image.LANCZOS)
    wm = window_map(crop(native, REGION_MAP), tools)
    wm.save(D / "sweep_window.jpg", quality=90, optimize=True)
    rs = crop(native, REGION_STRIP)
    sweep(rs, lambda im, p: tools["denoise"].tool_denoise(im, method="bilateral", strength=p), *STRENGTH) \
        .save(D / "sweep_denoise.jpg", quality=90, optimize=True)
    sweep(rs, lambda im, p: tools["contrast"].tool_enhance_contrast(im, factor=p), *FACTOR) \
        .save(D / "sweep_contrast.jpg", quality=90, optimize=True)
    (D / "sweeps.json").write_text(json.dumps({
        "region_map": REGION_MAP, "region_strip": REGION_STRIP, "center_range": C_RANGE, "width_range": W_RANGE,
        "strength_range": STRENGTH, "factor_range": FACTOR, "slices": K_SLICES}, indent=1))
    print("wrote", [p.name for p in sorted(D.glob("sweep_*.jpg"))])


if __name__ == "__main__":
    main()
