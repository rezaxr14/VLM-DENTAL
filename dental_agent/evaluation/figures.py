"""Paper figures of the evaluation matrix, drawn from the tables ``scripts/aggregate_eval.py`` writes (no model needed).

Inputs are the files in ``<output-dir>/matrix_<dataset>_<split>/``: ``matrix_pooled.csv`` (every condition x image size,
exact and FDI F1 with image-resampling intervals), ``matrix_comparisons.csv`` (paired differences), ``matrix_ablations.csv``
(tool-ablation effects), ``matrix_localization.csv`` (box hit rate before and after ``nudge_crop``) and
``matrix_summary.json`` (only for facts the CSVs do not carry: resample count, cells that used another prompt, image
coverage). The plotted numbers are the CSV numbers, so a figure and its table can never disagree.

Figures (one file per level where it applies; ``exact`` = right tooth and diagnosis, ``fdi`` = right tooth):

* ``matrix_forest_<level>``      pooled F1 with 95% interval of every condition, base / SFT / GRPO, with and without
                                 tools, one panel per image size (canonical first, it is the primary result).
* ``paired_comparisons_<level>`` paired F1 differences (tools vs no tools, SFT vs base, GRPO vs SFT, canonical vs
                                 original) with paired-bootstrap intervals; filled marker = interval excludes zero.
* ``ablation_effects_<level>``   ablated minus unablated F1 for every tool-ablation run, same paired intervals.
* ``localization_before_after``  share of located findings whose working box hits the annotated tooth right after
                                 ``locate_tooth`` (hollow) and after the agent's ``nudge_crop`` calls (filled).

Worked example for the localization figure: three located findings with box IoU (initial, final) of (0.33, 1.0),
(0.8, 0.8) and (0.2, 0.3) at threshold 0.5 give hit rate 1/3 before and 2/3 after nudging; the dumbbell runs from 0.33 to
0.67 and is annotated ``recovered 1/2`` (one of the two initial misses was brought back onto the tooth).

Colours: the validated categorical slots 1 (blue) and 2 (orange) of the data-viz palette, validated against the chart
surface; text always wears the ink tokens, never a series colour. Every figure has a table twin (the CSV it came from).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Tokens (light surface; see the data-viz palette). One place, no colour anywhere else in this file.
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SERIES_TOOLS = "#2a78d6"      # categorical slot 1: with tools (also the single series of the effect / localization plots)
SERIES_NO_TOOLS = "#eb6834"   # categorical slot 2: without tools

MODELS = ("base", "sft", "grpo")
MODEL_LABEL = {"base": "Base", "sft": "SFT", "grpo": "GRPO"}
SIZES = ("canonical", "original")
LEVELS = ("exact", "fdi")
LEVEL_LABEL = {"exact": "exact match (tooth and diagnosis)", "fdi": "tooth identified (FDI)"}
LEVEL_SHORT = {"exact": "exact-match", "fdi": "tooth-identified"}

TEXT_COLUMNS = {"cell", "ablation", "label", "a", "b", "level", "diagnosis"}

POOLED_CSV = "matrix_pooled.csv"
COMPARISONS_CSV = "matrix_comparisons.csv"
ABLATIONS_CSV = "matrix_ablations.csv"
LOCALIZATION_CSV = "matrix_localization.csv"
SUMMARY_JSON = "matrix_summary.json"


# ---------------------------------------------------------------------------
# Reading the aggregate tables
# ---------------------------------------------------------------------------

def _num(text: str) -> float | str | None:
    if text == "" or text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def read_table(path: str | Path) -> list[dict[str, Any]]:
    """Rows of one aggregate CSV. Empty cells become ``None``, numeric cells floats, label columns stay text."""
    with open(path, "r", newline="", encoding="utf-8") as f:
        return [
            {k: (v if k in TEXT_COLUMNS else _num(v)) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def load_tables(matrix_dir: str | Path) -> dict[str, Any]:
    """Everything the figures need from one ``matrix_<dataset>_<split>/`` folder. A table that is missing (for example no
    ablation was run) comes back as an empty list, and the figure that needs it is skipped by ``make_all``."""
    d = Path(matrix_dir)
    out: dict[str, Any] = {"summary": {}}
    for key, name in (("pooled", POOLED_CSV), ("comparisons", COMPARISONS_CSV), ("ablations", ABLATIONS_CSV),
                      ("localization", LOCALIZATION_CSV)):
        out[key] = read_table(d / name) if (d / name).is_file() else []
    if (d / SUMMARY_JSON).is_file():
        out["summary"] = json.loads((d / SUMMARY_JSON).read_text(encoding="utf-8"))
    return out


# ---------------------------------------------------------------------------
# Naming helpers
# ---------------------------------------------------------------------------

def split_cell(cell: str) -> tuple[str, str, str]:
    """``'sft_with_tools@canonical'`` -> (``'sft'``, ``'with_tools'``, ``'canonical'``)."""
    cond, size = cell.split("@", 1)
    model, tools = cond.split("_", 1)
    return model, tools, size


def describe_tag(tag: str) -> str:
    """Reader-facing name of an ablation tag (``AblationSpec.tag()``). ``''`` is the unablated run.

    ``__no-nudge_crop`` -> ``without nudge_crop``;  ``__loc-shift0.3-scale1-seed0`` -> ``locate_tooth box shifted by 30% of
    its size (seed 0)``."""
    if not tag:
        return "Unablated"
    parts = []
    for part in tag.lstrip("_").split("+"):
        m = re.fullmatch(r"loc-shift(?P<shift>[^-]+)-scale(?P<scale>[^-]+)-seed(?P<seed>.+)", part)
        if m:
            bits = []
            if float(m["shift"]) > 0:
                bits.append(f"shifted by {float(m['shift']) * 100:g}% of its size")
            if float(m["scale"]) != 1.0:
                bits.append(f"scaled x{float(m['scale']):g}")
            parts.append("locate_tooth box " + " and ".join(bits) + f" (seed {m['seed']})")
        elif part.startswith("no-"):
            parts.append(f"without {part[3:]}")
        else:
            parts.append(part)
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Plot data (pure functions of the tables; these are what the tests check)
# ---------------------------------------------------------------------------

def forest_cells(pooled: Iterable[dict[str, Any]], level: str = "exact") -> dict[str, list[dict[str, Any]]]:
    """Rows of the matrix forest per image size, ordered base / SFT / GRPO, no tools before tools inside a model."""
    rows = [r for r in pooled if r["level"] == level]
    out: dict[str, list[dict[str, Any]]] = {}
    for size in SIZES:
        picked = []
        for model in MODELS:
            for tools in ("no_tools", "with_tools"):
                for r in rows:
                    m, t, s = split_cell(r["cell"])
                    if (m, t, s) == (model, tools, size):
                        picked.append({"cell": r["cell"], "model": model, "tools": tools, "size": size,
                                       "value": r["f1"], "low": r["f1_ci_low"], "high": r["f1_ci_high"],
                                       "images": int(r["images"])})
        if picked:
            out[size] = picked
    return out


def _block(label: str) -> str:
    return "canonical" if label.endswith(", canonical)") else "original" if label.endswith(", original)") else "size"


def comparison_rows(comparisons: Iterable[dict[str, Any]], level: str = "exact") -> list[dict[str, Any]]:
    """Paired differences in the order the table lists them. ``significant`` = the interval excludes zero."""
    out = []
    for r in comparisons:
        if r["level"] != level:
            continue
        out.append({"label": r["label"], "block": _block(r["label"]), "diff": r["diff"], "low": r["ci_low"],
                    "high": r["ci_high"], "p": r["p_value"], "n_paired": int(r["n_paired"]),
                    "significant": r["ci_low"] > 0 or r["ci_high"] < 0})
    return out


def ablation_groups(ablations: Iterable[dict[str, Any]], level: str = "exact") -> list[tuple[str, list[dict[str, Any]]]]:
    """(tag, effect rows) per ablation run, tags in file order. Effect = ablated minus unablated F1."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in ablations:
        if r["level"] != level:
            continue
        m, _, s = split_cell(r["cell"])
        groups.setdefault(r["ablation"], []).append({
            "label": f"{MODEL_LABEL.get(m, m)}, {s}", "diff": r["effect"], "low": r["ci_low"], "high": r["ci_high"],
            "p": r["p_value"], "n_paired": int(r["n_paired"]), "significant": r["ci_low"] > 0 or r["ci_high"] < 0,
            "ablated": r["ablated_f1"], "unablated": r["unablated_f1"]})
    return list(groups.items())


def localization_groups(localization: Iterable[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    """(tag, rows) with the unablated run first. ``hit_*`` are shares of the located findings."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for r in localization:
        m, _, s = split_cell(r["cell"])
        n_found = int(round((r["located_rate"] or 0.0) * (r["n_findings_with_box"] or 0)))
        groups.setdefault(r["ablation"] or "", []).append({
            "label": f"{MODEL_LABEL.get(m, m)}, {s}", "initial": r["hit_initial"], "final": r["hit_final"],
            "recovered": None if r["recovery_rate"] is None else int(r["n_recovered"]),
            "recoverable": int(r["n_recoverable"]), "n_located": n_found, "iou_threshold": r["iou_threshold"]})
    ordered = sorted(groups.items(), key=lambda kv: (kv[0] != "", kv[0]))
    return ordered


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _style() -> dict[str, Any]:
    return {"font.size": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK_2,
            "xtick.color": INK_2, "ytick.color": INK_2, "text.color": INK, "pdf.fonttype": 42, "svg.fonttype": "none"}


def _new_figure(height_in: float, ncols: int = 1, width_in: float = 7.2, sharex: bool = True) -> tuple[Figure, list[Any]]:
    fig = Figure(figsize=(width_in, height_in), facecolor=SURFACE, layout="constrained")
    axes = fig.subplots(1, ncols, sharex=sharex, sharey=True) if ncols > 1 else [fig.subplots()]
    axes = list(axes) if ncols > 1 else axes
    for ax in axes:
        ax.set_facecolor(SURFACE)
        ax.grid(axis="x", color=GRID, linewidth=0.7, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.tick_params(axis="x", length=3, color=AXIS)
    return fig, axes


def _footnote(fig: Figure, text: str) -> None:
    fig.get_layout_engine().set(rect=(0.0, 0.07, 1.0, 0.93))
    fig.text(0.012, 0.012, text, fontsize=6.5, color=MUTED, ha="left", va="bottom", wrap=True)


def _dot(ax: Any, x: float, y: float, color: str, filled: bool = True, size: float = 7.5) -> None:
    ax.plot([x], [y], linestyle="none", marker="o", markersize=size, markerfacecolor=color if filled else SURFACE,
            markeredgecolor=color if not filled else SURFACE, markeredgewidth=1.6 if not filled else 1.4, zorder=4)


def _interval(ax: Any, lo: float, hi: float, y: float, color: str) -> None:
    ax.hlines(y, lo, hi, color=color, linewidth=1.6, zorder=3, capstyle="round")


def _layout_rows(groups: Sequence[tuple[str | None, int]]) -> tuple[list[list[float]], list[tuple[float, str]], float]:
    """y positions for groups of (header, n_rows): a header takes a row of its own, groups are separated by a gap."""
    ys: list[list[float]] = []
    headers: list[tuple[float, str]] = []
    y = 0.0
    for header, n in groups:
        if header is not None:
            headers.append((y, header))
            y += 1.0
        ys.append([y + i for i in range(n)])
        y += n + 0.6
    return ys, headers, y - 0.6


def _set_rows(ax: Any, labels: list[tuple[float, str]], headers: list[tuple[float, str]], bottom: float) -> None:
    ticks = sorted(labels + [(y, h) for y, h in headers])
    ax.set_yticks([y for y, _ in ticks])
    ax.set_yticklabels([t for _, t in ticks], fontsize=8)
    header_y = {y for y, _ in headers}
    for y, lbl in zip([y for y, _ in ticks], ax.get_yticklabels()):
        if y in header_y:
            lbl.set_fontweight("bold")
            lbl.set_color(INK)
    ax.set_ylim(bottom + 0.6, -0.6)


def _image_note(summary: dict[str, Any], pooled: Sequence[dict[str, Any]]) -> list[str]:
    notes = []
    n_boot = summary.get("n_boot")
    counts = sorted({int(r["images"]) for r in pooled}) if pooled else []
    n_txt = (f"{counts[0]} images per cell" if len(counts) == 1 else f"{counts[0]}-{counts[-1]} images per cell") if counts else ""
    notes.append("Pooled over all findings; intervals resample whole images" + (f" ({n_boot} resamples, 95%)" if n_boot else " (95%)")
                 + (f"; {n_txt}" if n_txt else "") + ".")
    cached = sorted(n for n, c in (summary.get("cells") or {}).items() if any("zero_shot_cached" in p for p in c.get("prompt", [])))
    if cached:
        notes.append("(*) cached zero-shot run with a different prompt, not comparable with the other conditions: " + ", ".join(cached) + ".")
    missing = (summary.get("coverage") or {}).get("cells_missing_images") or {}
    if missing:
        notes.append("Cells cover different image sets: " + "; ".join(f"{k} lacks {len(v)}" for k, v in missing.items()) + ".")
    return notes


def plot_matrix_forest(pooled: Sequence[dict[str, Any]], summary: dict[str, Any] | None = None, level: str = "exact") -> Figure | None:
    """Pooled F1 with 95% interval of every condition, one panel per image size."""
    by_size = forest_cells(pooled, level)
    if not by_size:
        return None
    summary = summary or {}
    cached = {n for n, c in (summary.get("cells") or {}).items() if any("zero_shot_cached" in p for p in c.get("prompt", []))}
    hi = max(r["high"] for rows in by_size.values() for r in rows)
    with mpl.rc_context(_style()):
        tallest = max(len({r["model"] for r in rows}) for rows in by_size.values())
        fig, axes = _new_figure(1.0 + 0.62 * 2 * tallest, ncols=len(by_size))
        for ax, (size, rows) in zip(axes, by_size.items()):
            models = [m for m in MODELS if any(r["model"] == m for r in rows)]
            ys, _, bottom = _layout_rows([(None, 2) for _ in models])
            centres = []
            for m, yy in zip(models, ys):
                for r in (r for r in rows if r["model"] == m):
                    y = yy[0 if r["tools"] == "no_tools" else 1]
                    color = SERIES_TOOLS if r["tools"] == "with_tools" else SERIES_NO_TOOLS
                    _interval(ax, r["low"], r["high"], y, color)
                    _dot(ax, r["value"], y, color)
                centres.append((sum(yy) / 2, MODEL_LABEL[m] + (" *" if any(r["cell"] in cached for r in rows if r["model"] == m) else "")))
            _set_rows(ax, centres, [], bottom)
            ax.set_xlim(0, min(1.0, hi * 1.08 + 0.02))
            ax.set_title(size.capitalize() + (" size (primary)" if size == "canonical" else " size"), fontsize=8.5, color=INK, loc="left")
            ax.set_xlabel(f"F1, {LEVEL_LABEL[level]}", fontsize=8)
        fig.legend(handles=[Line2D([], [], marker="o", linestyle="-", color=SERIES_TOOLS, markersize=7, label="with tools"),
                            Line2D([], [], marker="o", linestyle="-", color=SERIES_NO_TOOLS, markersize=7, label="no tools")],
                   loc="outside upper right", ncol=2, frameon=False, fontsize=7.5)
        _footnote(fig, "\n".join(_image_note(summary, [r for r in pooled if r["level"] == level])))
    return fig


def _p_text(p: float) -> str:
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _effect_text(r: dict[str, Any]) -> str:
    return f"{r['diff']:+.3f} [{r['low']:+.3f}, {r['high']:+.3f}]"


def _forest_of_differences(groups: Sequence[tuple[str | None, list[dict[str, Any]]]], xlabel: str, notes: list[str],
                           with_p: bool = False) -> Figure:
    spec = [(h, len(rows)) for h, rows in groups]
    ys, headers, bottom = _layout_rows(spec)
    lo = min(r["low"] for _, rows in groups for r in rows)
    hi = max(r["high"] for _, rows in groups for r in rows)
    pad = 0.08 * max(hi - lo, 0.05)
    with mpl.rc_context(_style()):
        fig, (ax,) = _new_figure(0.9 + 0.2 * (bottom + 1), ncols=1, width_in=7.2)
        labels = []
        for (_, rows), yy in zip(groups, ys):
            for r, y in zip(rows, yy):
                _interval(ax, r["low"], r["high"], y, SERIES_TOOLS)
                _dot(ax, r["diff"], y, SERIES_TOOLS, filled=r["significant"])
                labels.append((y, r["label"]))
                ax.text(1.02, y, _effect_text(r) + (f"   {_p_text(r['p'])}" if with_p else ""), transform=ax.get_yaxis_transform(),
                        fontsize=7, color=INK_2, va="center", ha="left", family="monospace")
        _set_rows(ax, labels, headers, bottom)
        ax.axvline(0, color=AXIS, linewidth=1.0, zorder=2)
        ax.set_xlim(min(lo, 0) - pad, max(hi, 0) + pad)
        ax.set_xlabel(xlabel, fontsize=8)
        fig.legend(handles=[Line2D([], [], marker="o", linestyle="-", color=SERIES_TOOLS, markersize=7, label="interval excludes 0"),
                            Line2D([], [], marker="o", linestyle="-", color=SERIES_TOOLS, markerfacecolor=SURFACE, markeredgewidth=1.6,
                                   markersize=7, label="interval includes 0")], loc="outside upper left", ncol=2, frameon=False, fontsize=7.5)
        fig.get_layout_engine().set(rect=(0.0, 0.07, 0.86, 0.93))
        fig.text(0.012, 0.012, "\n".join(notes), fontsize=6.5, color=MUTED, ha="left", va="bottom", wrap=True)
    return fig


def plot_paired_comparisons(comparisons: Sequence[dict[str, Any]], summary: dict[str, Any] | None = None,
                            level: str = "exact") -> Figure | None:
    """Paired F1 differences A minus B, one row per comparison, blocks separated by a gap."""
    rows = comparison_rows(comparisons, level)
    if not rows:
        return None
    blocks: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        blocks.setdefault(r["block"], []).append(r)
    groups = [(None, blocks[b]) for b in ("canonical", "original", "size") if b in blocks]
    n_boot = (summary or {}).get("n_boot")
    return _forest_of_differences(
        groups, f"Difference in {LEVEL_SHORT[level]} F1 (A minus B, same images)",
        ["Each row is A minus B on the images both conditions cover; intervals are paired bootstrap over images"
         + (f" ({n_boot} resamples, 95%)." if n_boot else " (95%)."),
         "Intervals are not corrected for the number of comparisons shown; read a filled marker as a screening result."], with_p=True)


def plot_ablation_effects(ablations: Sequence[dict[str, Any]], summary: dict[str, Any] | None = None,
                          level: str = "exact") -> Figure | None:
    """Ablated minus unablated F1 for every tool-ablation run."""
    groups = [(describe_tag(tag), rows) for tag, rows in ablation_groups(ablations, level)]
    if not groups:
        return None
    n_boot = (summary or {}).get("n_boot")
    return _forest_of_differences(
        groups, f"Change in {LEVEL_SHORT[level]} F1 (ablated minus unablated)",
        ["Evaluation-only ablations of the trained agent (dev split); not models retrained without the tool. Paired bootstrap over images"
         + (f" ({n_boot} resamples, 95%)." if n_boot else " (95%).")], with_p=True)


def plot_localization(localization: Sequence[dict[str, Any]], summary: dict[str, Any] | None = None) -> Figure | None:
    """Box hit rate right after ``locate_tooth`` (hollow) against after the agent's ``nudge_crop`` calls (filled)."""
    groups = [(describe_tag(tag), rows) for tag, rows in localization_groups(localization)]
    if not groups:
        return None
    thr = groups[0][1][0]["iou_threshold"]
    ys, headers, bottom = _layout_rows([(h, len(rows)) for h, rows in groups])
    with mpl.rc_context(_style()):
        fig, (ax,) = _new_figure(0.9 + 0.2 * (bottom + 1), ncols=1, width_in=7.2)
        labels = []
        for (_, rows), yy in zip(groups, ys):
            for r, y in zip(rows, yy):
                ax.hlines(y, r["initial"], r["final"], color=SERIES_TOOLS, linewidth=1.6, zorder=3)
                _dot(ax, r["initial"], y, SERIES_TOOLS, filled=False)
                _dot(ax, r["final"], y, SERIES_TOOLS, filled=True)
                labels.append((y, r["label"]))
                rec = "no initial misses" if r["recovered"] is None else f"recovered {r['recovered']}/{r['recoverable']}"
                ax.text(1.02, y, f"n={r['n_located']}  {rec}", transform=ax.get_yaxis_transform(), fontsize=7, color=INK_2,
                        va="center", ha="left", family="monospace")
        _set_rows(ax, labels, headers, bottom)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel(f"Share of located findings whose working box hits the annotated tooth (IoU >= {thr:g})", fontsize=8)
        fig.legend(handles=[Line2D([], [], marker="o", linestyle="none", markerfacecolor=SURFACE, markeredgecolor=SERIES_TOOLS,
                                   markeredgewidth=1.6, markersize=7, label="after locate_tooth"),
                            Line2D([], [], marker="o", linestyle="none", color=SERIES_TOOLS, markersize=7, label="after nudge_crop")],
                   loc="outside upper left", ncol=2, frameon=False, fontsize=7.5)
        fig.get_layout_engine().set(rect=(0.0, 0.07, 0.86, 0.93))
        fig.text(0.012, 0.012, "n = located findings (annotated tooth box present); recovered = initial misses brought onto the tooth.\n"
                 "Under a shift or scale ablation the initial box is the perturbed one; without nudge_crop the two markers coincide.",
                 fontsize=6.5, color=MUTED, ha="left", va="bottom", wrap=True)
    return fig


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------

def save_figure(fig: Figure, stem: str | Path, formats: Sequence[str] = ("png", "pdf"), dpi: int = 300) -> list[Path]:
    """Write ``<stem>.<format>`` for every format and return the paths."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    written = []
    for fmt in formats:
        path = stem.with_suffix("." + fmt)
        fig.savefig(path, dpi=dpi, facecolor=SURFACE, format=fmt)
        written.append(path)
    return written


def figure_names(levels: Sequence[str] = LEVELS) -> list[str]:
    """Every figure ``make_all`` can write for these levels (a figure without data is simply not written)."""
    return [f"{k}_{lv}" for lv in levels for k in ("matrix_forest", "paired_comparisons", "ablation_effects")] + ["localization_before_after"]


def make_all(matrix_dir: str | Path, out_dir: str | Path | None = None, formats: Sequence[str] = ("png", "pdf"),
             levels: Sequence[str] = LEVELS, dpi: int = 300) -> dict[str, list[Path]]:
    """Draw every figure whose table exists in ``matrix_dir`` and write it to ``out_dir`` (default ``<matrix_dir>/figures``).
    Returns ``{figure name: files}``; a figure with no data (no ablation was run, no tool cells) is left out."""
    t = load_tables(matrix_dir)
    out = Path(out_dir) if out_dir else Path(matrix_dir) / "figures"
    written: dict[str, list[Path]] = {}
    jobs: list[tuple[str, Figure | None]] = []
    for level in levels:
        jobs += [(f"matrix_forest_{level}", plot_matrix_forest(t["pooled"], t["summary"], level)),
                 (f"paired_comparisons_{level}", plot_paired_comparisons(t["comparisons"], t["summary"], level)),
                 (f"ablation_effects_{level}", plot_ablation_effects(t["ablations"], t["summary"], level))]
    jobs.append(("localization_before_after", plot_localization(t["localization"], t["summary"])))
    for name, fig in jobs:
        if fig is not None:
            written[name] = save_figure(fig, out / name, formats, dpi)
    return written


# ---------------------------------------------------------------------------
# GRPO group-size (K) sweep: figures from the tables ``scripts/aggregate_k_sweep.py`` writes
# ---------------------------------------------------------------------------
#
# * ``k_sweep_<level>``   top: pooled F1 with 95% interval per K, the SFT reference as a dashed line; bottom: paired
#                         difference to SFT per K (hollow marker = the interval includes 0).
# * ``k_sweep_training``  per K: training hours, share of groups whose K rewards tied (no gradient), and mean reward of the
#                         first against the last quarter of the steps.
#
# Worked example: K = 1, 2, 4 pool to exact F1 0.40, 0.55, 0.57 against an SFT reference of 0.45; the paired differences
# -0.05 [-0.15, +0.05], +0.10 [+0.02, +0.18], +0.12 [+0.04, +0.20] draw K = 1 hollow (no difference established) and K = 2
# and 4 filled (above SFT).

K_TABLE_CSV = "k_sweep_table.csv"
K_TRAINING_CSV = "k_sweep_training.csv"
K_SUMMARY_JSON = "k_sweep_summary.json"


def load_k_sweep_tables(k_dir: str | Path) -> dict[str, Any]:
    """The tables of one ``ksweep_<dataset>_<split>/`` folder; a missing table is an empty list."""
    d = Path(k_dir)
    out: dict[str, Any] = {"table": [], "training": [], "summary": {}}
    if (d / K_TABLE_CSV).is_file():
        out["table"] = read_table(d / K_TABLE_CSV)
    if (d / K_TRAINING_CSV).is_file():
        out["training"] = read_table(d / K_TRAINING_CSV)
    if (d / K_SUMMARY_JSON).is_file():
        out["summary"] = json.loads((d / K_SUMMARY_JSON).read_text(encoding="utf-8"))
    return out


def k_sweep_rows(table: Iterable[dict[str, Any]], level: str = "exact") -> list[dict[str, Any]]:
    """Plot rows of one level, ascending in K: F1 with interval, the SFT reference F1 and the paired difference to it."""
    rows = []
    for r in sorted((r for r in table if r["level"] == level), key=lambda r: r["k"]):
        has_ref = r.get("ref_diff") is not None
        rows.append({
            "k": int(r["k"]), "value": r["f1"], "low": r["f1_ci_low"], "high": r["f1_ci_high"], "images": int(r["images"]),
            "reference": r.get("ref_f1"), "diff": r["ref_diff"] if has_ref else None,
            "diff_low": r["ref_ci_low"] if has_ref else None, "diff_high": r["ref_ci_high"] if has_ref else None,
            "significant": has_ref and (r["ref_ci_low"] > 0 or r["ref_ci_high"] < 0),
            "is_best": bool(r.get("is_best")), "within_noise": bool(r.get("within_noise_of_best")),
        })
    return rows


def _k_axes(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.7, linestyle="-")
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="x", length=3, color=AXIS)
    ax.tick_params(axis="y", length=0)


def plot_k_sweep(table: Sequence[dict[str, Any]], summary: dict[str, Any] | None = None, level: str = "exact") -> Figure | None:
    """Pooled F1 per K with the SFT reference, and the paired difference to SFT (when a reference was given)."""
    rows = k_sweep_rows(table, level)
    if not rows:
        return None
    summary = summary or {}
    has_ref = rows[0]["diff"] is not None
    xs = list(range(len(rows)))
    with mpl.rc_context(_style()):
        fig = Figure(figsize=(7.2, 4.8 if has_ref else 3.0), facecolor=SURFACE, layout="constrained")
        axes = list(fig.subplots(2 if has_ref else 1, 1, sharex=True))if has_ref else [fig.subplots()]
        top = axes[0]
        _k_axes(top)
        for x, r in zip(xs, rows):
            top.vlines(x, r["low"], r["high"], color=SERIES_TOOLS, linewidth=1.6, zorder=3, capstyle="round")
            _dot(top, x, r["value"], SERIES_TOOLS)
            top.text(x + 0.08, r["value"], f"{r['value']:.2f}", fontsize=7, color=INK_2, va="center", ha="left")
        if has_ref and rows[0]["reference"] is not None:
            ref = rows[0]["reference"]
            top.axhline(ref, color=MUTED, linewidth=1.0, linestyle="--", zorder=2)
            top.text(1.0, ref, f"SFT reference {ref:.2f} ", transform=top.get_yaxis_transform(), fontsize=7, color=INK_2,
                     va="bottom", ha="right")
        top.set_ylabel(f"Pooled {LEVEL_SHORT[level]} F1", fontsize=8)
        top.set_ylim(max(0.0, min(r["low"] for r in rows) - 0.05), min(1.0, max(r["high"] for r in rows) + 0.06))
        if has_ref:
            bot = axes[1]
            _k_axes(bot)
            bot.axhline(0.0, color=AXIS, linewidth=1.0, zorder=2)
            for x, r in zip(xs, rows):
                bot.vlines(x, r["diff_low"], r["diff_high"], color=SERIES_TOOLS, linewidth=1.6, zorder=3, capstyle="round")
                _dot(bot, x, r["diff"], SERIES_TOOLS, filled=r["significant"])
                bot.text(x + 0.08, r["diff"], f"{r['diff']:+.2f}", fontsize=7, color=INK_2, va="center", ha="left")
            bot.set_ylabel("GRPO minus SFT (paired)", fontsize=8)
        axes[-1].set_xticks(xs)
        axes[-1].set_xticklabels([f"K={r['k']}" for r in rows], fontsize=8)
        axes[-1].set_xlim(-0.5, len(rows) - 0.4)
        n_boot, imgs = summary.get("n_boot"), sorted({r["images"] for r in rows})
        _footnote(fig, "Pooled over all findings on the dev images; intervals resample whole images"
                  + (f" ({n_boot} resamples, 95%)" if n_boot else " (95%)") + f"; {imgs[0]}" + (f"-{imgs[-1]}" if len(imgs) > 1 else "")
                  + " images per K.\nOne training run per K: the intervals do not include variation between training runs."
                  + (" Hollow marker: the paired interval includes 0." if has_ref else ""))
    return fig


def plot_k_sweep_training(training: Sequence[dict[str, Any]], summary: dict[str, Any] | None = None) -> Figure | None:
    """Training hours, tie rate and first-versus-last-quarter reward per K, from the GRPO step logs."""
    rows = [{**r, "k": int(r["k"])} for r in sorted(training, key=lambda r: r["k"])]
    panels = []
    if any(r.get("hours") is not None for r in rows):
        panels.append("hours")
    if any(r.get("tie_rate") is not None for r in rows):
        panels.append("tie")
    if any(r.get("reward_first_quarter") is not None for r in rows):
        panels.append("reward")
    if not panels:
        return None
    xs = list(range(len(rows)))
    with mpl.rc_context(_style()):
        fig = Figure(figsize=(7.2, 2.9), facecolor=SURFACE, layout="constrained")
        axes = fig.subplots(1, len(panels), sharex=True)
        axes = [axes] if len(panels) == 1 else list(axes)
        for ax, kind in zip(axes, panels):
            _k_axes(ax)
            if kind == "hours":
                for x, r in zip(xs, rows):
                    if r.get("hours") is not None:
                        ax.bar(x, r["hours"], width=0.55, color=SERIES_TOOLS, zorder=3)
                        ax.text(x, r["hours"], f"{r['hours']:.1f}", fontsize=7, color=INK_2, ha="center", va="bottom")
                ax.set_ylabel("Training hours", fontsize=8)
            elif kind == "tie":
                for x, r in zip(xs, rows):
                    if r.get("tie_rate") is not None:
                        ax.bar(x, r["tie_rate"], width=0.55, color=SERIES_TOOLS, zorder=3)
                        ax.text(x, r["tie_rate"], f"{r['tie_rate']:.2f}", fontsize=7, color=INK_2, ha="center", va="bottom")
                ax.set_ylim(0, 1.05)
                ax.set_ylabel("Share of groups with all rewards equal", fontsize=8)
            else:
                for x, r in zip(xs, rows):
                    if r.get("reward_first_quarter") is not None:
                        ax.vlines(x, r["reward_first_quarter"], r["reward_last_quarter"], color=SERIES_TOOLS, linewidth=1.6, zorder=3)
                        _dot(ax, x, r["reward_first_quarter"], SERIES_TOOLS, filled=False)
                        _dot(ax, x, r["reward_last_quarter"], SERIES_TOOLS, filled=True)
                ax.set_ylabel("Mean rollout reward", fontsize=8)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"K={r['k']}" for r in rows], fontsize=7.5)
            ax.set_xlim(-0.6, len(rows) - 0.4)
        _footnote(fig, "From the GRPO step logs, one run per K. Ties need K >= 2 (no bar at K=1). Reward: hollow = first quarter of the steps, "
                       "filled = last quarter.")
    return fig


def k_sweep_figure_names(levels: Sequence[str] = LEVELS) -> list[str]:
    return [f"k_sweep_{lv}" for lv in levels] + ["k_sweep_training"]


def make_k_sweep_figures(k_dir: str | Path, out_dir: str | Path | None = None, formats: Sequence[str] = ("png", "pdf"),
                         levels: Sequence[str] = LEVELS, dpi: int = 300) -> dict[str, list[Path]]:
    """Draw the K-sweep figures whose tables exist in ``k_dir`` (default output ``<k_dir>/figures``); returns ``{name: files}``."""
    t = load_k_sweep_tables(k_dir)
    out = Path(out_dir) if out_dir else Path(k_dir) / "figures"
    jobs = [(f"k_sweep_{lv}", plot_k_sweep(t["table"], t["summary"], lv)) for lv in levels]
    jobs.append(("k_sweep_training", plot_k_sweep_training(t["training"], t["summary"])))
    return {name: save_figure(fig, out / name, formats, dpi) for name, fig in jobs if fig is not None}
