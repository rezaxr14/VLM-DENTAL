"""Visual-preview generator: re-runs the paper figure functions and saves PNG previews.

    python scripts/paper/preview_figures.py

Writes PNG previews of every generated figure to ``scratch/preview`` so they can be
visually verified (per the repo's visual-verification discipline) without a PDF renderer.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO_ROOT))

import generate_paper_assets as g  # noqa: E402

out = REPO_ROOT / 'scratch' / 'preview'
(out / 'figures' / 'svg').mkdir(parents=True, exist_ok=True)
(out / 'figures' / 'pdf').mkdir(parents=True, exist_ok=True)

# Monkeypatch Figure.savefig so every .svg save also writes a .png preview.
_orig = g.plt.Figure.savefig


def savefig_with_png(self, path, *a, **k):
    r = _orig(self, path, *a, **k)
    p = str(path)
    if p.endswith('.svg'):
        _orig(self, p[:-4] + '.png', dpi=150)
    return r


g.plt.Figure.savefig = savefig_with_png

s = json.load(open(REPO_ROOT / 'paper' / 'assets_summary.json', encoding='utf-8'))
g.plot_tool_usage(out, s['tool_totals'])
g.plot_nudge_tier(out, s['nudge_by_tier'])

args = argparse.Namespace(
    dentex_disease_json=REPO_ROOT / 'data/dentex/DENTEX/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json',
    dentex_xrays=REPO_ROOT / 'data/dentex/DENTEX/training_data/quadrant-enumeration-disease/xrays',
    tufts_radiographs=REPO_ROOT / 'data/tufts/Radiographs',
)
from generate_paper_assets import build_size_lookups, collect  # noqa: E402

per_cohort, _, _, _ = collect(REPO_ROOT / 'data/traces', build_size_lookups(args))
meta = g.render_perturbation_example(out, per_cohort, args)
print('example:', meta)
print('previews:', sorted(p.name for p in (out / 'figures' / 'svg').glob('*.png')))
