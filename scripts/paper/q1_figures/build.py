#!/usr/bin/env python3
"""Regenerate the Q1-style paper figures from source.

    python scripts/paper/q1_figures/build.py            # all figures
    python scripts/paper/q1_figures/build.py overview   # one figure

Writes paper/figures/svg/<name>.svg and, when svglib is installed,
paper/figures/pdf/<name>.pdf. Inputs are the real radiograph assets in
paper/figures/assets (see scripts/paper/prepare_showcase.py); no model runs.
Requires: Pillow, svglib, reportlab.
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import ROOT  # noqa: E402

FIGURES = {
    "overview": ("fig_overview", "fig_overview"),
    "mechanism": ("fig_correction_mechanism", "fig_correction_mechanism"),
    "trace": ("fig_trace_pipeline", "fig_trace_pipeline"),
    "training": ("fig_training_reward", "fig_training_reward"),
    "model": ("fig_model_internals", "fig_model_internals"),
    "data": ("fig_data_detector", "fig_data_detector"),
    "tools": ("fig_tools_output", "fig_tools_output"),
    "evidence": ("fig_evidence_trail", "fig_evidence_trail"),
    "design": ("fig_design_2x2", "fig_design_2x2"),
    "behaviour": ("fig_correction_behaviour", "fig_correction_behaviour"),
    "gallery": ("fig_gallery_episodes", "fig_gallery_episodes"),
}


def to_pdf(svg: Path, pdf: Path):
    try:
        from reportlab.graphics import renderPDF
        from svglib.svglib import svg2rlg
    except ImportError:
        print("  warn: svglib/reportlab missing; PDF not written")
        return
    drawing = svg2rlg(str(svg))
    pdf.parent.mkdir(parents=True, exist_ok=True)
    renderPDF.drawToFile(drawing, str(pdf))


def main(argv):
    wanted = argv or list(FIGURES)
    svg_dir = ROOT / "paper" / "figures" / "svg"
    pdf_dir = ROOT / "paper" / "figures" / "pdf"
    svg_dir.mkdir(parents=True, exist_ok=True)
    for key in wanted:
        module_name, stem = FIGURES[key]
        print(f"{stem}:")
        module = __import__(module_name)
        svg = svg_dir / f"{stem}.svg"
        module.build(str(svg))
        to_pdf(svg, pdf_dir / f"{stem}.pdf")


if __name__ == "__main__":
    main(sys.argv[1:])
