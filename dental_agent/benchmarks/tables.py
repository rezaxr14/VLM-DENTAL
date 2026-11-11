"""Per-question score vectors, cluster-bootstrap intervals, paired comparisons and the paper table for MMOral-OPG-Bench.

Questions of one radiograph are not independent, so every interval resamples IMAGES (cluster bootstrap), not questions.
Published numbers of other papers are never computed here: they come from a file you fill in, each with its judge and source, and
are printed in separate rows below the numbers measured here.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from dental_agent.benchmarks import mmoral_opg as mm


def closed_vectors(rows: Sequence[dict[str, Any]], seed: int = 0, mode: str = "official") -> dict[int, tuple[int, float, str]]:
    """{question index: (image_id, correct 0/1, category)} under the official (random fallback) or strict extraction."""
    out = {}
    for r in rows:
        if r.get("generation_refused"):                      # no answer was produced (TPU static generation): wrong, never a random option
            out[int(r["index"])] = (int(r["image_id"]), 0.0, str(r["category"]))
            continue
        idx2 = {L: r[f"option{i}"] for L, i in zip(mm.LETTERS, (1, 2, 3, 4))}
        ch = mm.extract_choice(r["prediction"], mm.LETTERS, idx2, random.Random(f"{seed}:{r['index']}"), strict=mode == "strict")
        out[int(r["index"])] = (int(r["image_id"]), float(ch == str(r["answer"])), str(r["category"]))
    return out


def open_vectors(answers: Sequence[dict[str, Any]], judged: Sequence[dict[str, Any]]) -> dict[int, tuple[int, float, str]]:
    by = {int(j["index"]): j for j in judged}
    out = {}
    for a in answers:
        j = by.get(int(a["index"]))
        if j is None:
            raise ValueError(f"open-ended question {a['index']} has no judge score")
        out[int(a["index"])] = (int(a["image_id"]), float(j["score"]), str(a["category"]))
    return out


def _select(vec: dict[int, tuple[int, float, str]], group: str | None):
    return {i: v for i, v in vec.items() if group is None or group in v[2]}


def cluster_mean_ci(vec: dict[int, tuple[int, float, str]], group: str | None = None, n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean x 100 and 95% percentile interval, resampling images."""
    sel = _select(vec, group)
    if not sel:
        return dict(n=0, mean=float("nan"), lo=float("nan"), hi=float("nan"))
    by_img: dict[int, list[float]] = {}
    for img, val, _ in sel.values():
        by_img.setdefault(img, []).append(val)
    sums = np.array([sum(v) for v in by_img.values()]); cnt = np.array([len(v) for v in by_img.values()], dtype=float)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    boots = sums[picks].sum(axis=1) / cnt[picks].sum(axis=1)
    return dict(n=len(sel), mean=100 * sums.sum() / cnt.sum(), lo=100 * float(np.percentile(boots, 2.5)), hi=100 * float(np.percentile(boots, 97.5)))


def paired_delta(a: dict[int, tuple[int, float, str]], b: dict[int, tuple[int, float, str]], group: str | None = None,
                 n_boot: int = 2000, seed: int = 0) -> dict[str, float]:
    """Mean (a - b) x 100 over the same questions, with a cluster-bootstrap interval. a and b must cover the same questions."""
    if set(a) != set(b):
        raise ValueError("The two systems answered different question sets; a paired comparison needs identical ones.")
    diff = {i: (a[i][0], a[i][1] - b[i][1], a[i][2]) for i in a}
    r = cluster_mean_ci(diff, group, n_boot, seed)
    r["significant_95"] = bool(r["n"] and (r["lo"] > 0 or r["hi"] < 0))
    return r


def summary_rows(closed: dict, open_: dict, n_boot: int = 2000, seed: int = 0) -> list[dict[str, Any]]:
    rows = []
    for part, vec, cats in (("closed", closed, mm.CLOSED_CATEGORIES), ("open", open_, mm.OPEN_CATEGORIES)):
        if not vec:
            continue
        for g in (*cats, None):
            ci = cluster_mean_ci(vec, g, n_boot, seed)
            rows.append(dict(part=part, category=g or "Overall", **ci))
    return rows


def render_markdown(measured: list[dict[str, Any]], published: Sequence[dict[str, Any]] = ()) -> str:
    """measured: [{'system', 'judge', 'part', 'category', 'mean', 'lo', 'hi', 'n'}]; published: [{'name','part','judge','source', <cat>: value}]."""
    out = []
    for part, cats in (("closed", mm.CLOSED_CATEGORIES), ("open", mm.OPEN_CATEGORIES)):
        systems = sorted({m["system"] for m in measured if m["part"] == part})
        pub = [p for p in published if p.get("part") == part]
        if not systems and not pub:
            continue
        title = "Closed-ended accuracy (%)" if part == "closed" else "Open-ended judged score (x100)"
        out += [f"### MMOral-OPG-Bench, {title}", "", "| System | Judge | " + " | ".join([*cats, "Overall"]) + " |", "|---|---|" + "---|" * (len(cats) + 1)]
        for s in systems:
            rows = {m["category"]: m for m in measured if m["system"] == s and m["part"] == part}
            judge = next(m["judge"] for m in measured if m["system"] == s and m["part"] == part)
            cells = []
            for c in (*cats, "Overall"):
                m = rows.get(c)
                cells.append("-" if m is None or m["n"] == 0 else f"{m['mean']:.1f} [{m['lo']:.1f}, {m['hi']:.1f}]")
            out.append(f"| {s} | {judge if part == 'open' else 'none (exact match)'} | " + " | ".join(cells) + " |")
        for p in pub:
            cells = [("-" if p.get(c) is None else f"{float(p[c]):.1f}") for c in (*cats, "Overall")]
            out.append(f"| *{p['name']}* (published; {p.get('source', 'source missing')}) | {p.get('judge', 'judge not stated')} | " + " | ".join(cells) + " |")
        out.append("")
    out += ["Measured rows: mean and 95% interval from resampling radiographs. Published rows are copied from the cited paper and were "
            "scored with the judge named in the row; they are not comparable to measured rows unless the judge and protocol match.", ""]
    return "\n".join(out)


def load_published(path: str | Path | None) -> list[dict[str, Any]]:
    if not path or not Path(path).is_file():
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for p in data:
        if not p.get("source"):
            raise ValueError(f"Published row {p.get('name')!r} has no 'source': every copied number needs its source")
    return data
