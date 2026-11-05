# Q1-style paper figures

The figures used in `paper/main.tex`. They show model architecture, data flow and
real examples (radiographs, crops, expert boxes, real tool outputs) with short
labels. Each figure is generated as editable SVG (and converted to PDF) from plain
Python, so every figure can be rebuilt from source:

    python scripts/paper/q1_figures/build.py            # all
    python scripts/paper/q1_figures/build.py tools      # one (key in the table below)

| Key | Figure in the paper | Module | Label |
|---|---|---|---|
| `mechanism` | 1, grounding is a hypothesis (`nudge_crop` on a real episode) | `fig_correction_mechanism.py` | `fig:mechanism` |
| `overview` | 2, system overview, harness and speculative decoding | `fig_overview.py` | `fig:overview` |
| `model` | 3, Qwen3.5-9B opened, frozen vs trained | `fig_model_internals.py` | `fig:internals` |
| `data` | 4, data sources, grounding pool, YOLOv8m detector, cohorts | `fig_data_detector.py` | `fig:data` |
| `tools` | 5, what the tools return (parameter sweeps) | `fig_tools_output.py` | `fig:tools` |
| `trace` | 6, trace synthesis around the image-624 trace | `fig_trace_pipeline.py` | `fig:tracepipeline` |
| `evidence` | 7, one image in both arms, tool-call distribution, 1:1 pairing | `fig_evidence_trail.py` | `fig:evidence` |
| `training` | 8, SFT, GRPO and the reward | `fig_training_reward.py` | `fig:training` |
| `design` | 9, experimental design (2x2) | `fig_design_2x2.py` | `fig:design` |
| `behaviour` | 10, correction behaviour by perturbation tier | `fig_correction_behaviour.py` | `fig:nudgetier` |
| `gallery` | 11, six verified episodes | `fig_gallery_episodes.py` | `fig:gallery` |

Outputs: `paper/figures/svg/<module>.svg` and `paper/figures/pdf/<module>.pdf`.

`common.py` holds the palette, text-fit checks and drawing helpers. Radiographs and
crops come from `paper/figures/assets/showcase` (real episodes; see
`scripts/paper/prepare_showcase.py`); the parameter sweeps in the tools figure are
made once by `scripts/paper/prepare_tool_sweeps.py`, which runs the project's own tool
functions across their parameter range. Numbers in the figures are read from
`paper/assets_summary.json` or are the paper's own constants. Requires Pillow, svglib
and reportlab (the sweep script also needs OpenCV).

Rules: arrowheads are explicit polygons (no `<marker>`), no filters or gradients,
images embedded as base64, every label checked for fit (`fit ok`), no glyphs the font
lacks. The superseded thesis-style figures are kept in `paper/figures/archive/svg`
and `scripts/paper/archive/thesis_figures`; they are no longer included in the paper.
