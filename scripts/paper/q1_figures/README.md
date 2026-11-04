# Q1-style paper figures

Figures that show model architecture, data flow and real examples, with short
labels. Each figure is generated as editable SVG (and converted to PDF) from
plain Python, so every figure can be rebuilt from source:

    python scripts/paper/q1_figures/build.py            # all
    python scripts/paper/q1_figures/build.py trace      # one (overview | mechanism | trace)

| Figure | Module | Output |
|---|---|---|
| 1 overview | `fig_overview.py` | `paper/figures/svg/fig_overview.svg` |
| 2 nudge_crop mechanism | `fig_correction_mechanism.py` | `paper/figures/svg/fig_correction_mechanism.svg` |
| 3 trace pipeline | `fig_trace_pipeline.py` | `paper/figures/svg/fig_trace_pipeline.svg` |

`common.py` holds the palette, text-fit checks and drawing helpers. Radiographs
and crops come from `paper/figures/assets/showcase` (real episodes; see
`scripts/paper/prepare_showcase.py`). Requires Pillow, svglib and reportlab.
