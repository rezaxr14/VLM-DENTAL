"""
Dataset-aware FDI (quadrant/tooth_position) conversion.

dentex_row_to_fdi() in dentex.py exists because DENTEX's raw JSON encodes
category_id_1/category_id_2 as 0-indexed, while every prompt/tool/reward
in this codebase is written against 1-indexed FDI -- and that +1 was once
reimplemented, incorrectly, by hand in seven other files (see
dentex_row_to_fdi's docstring for the full history).

Tufts' loader (tufts.py) hands back already-correct 1-indexed values, by
design (see tufts.py's module docstring point 1) -- so a second dataset
means there are now two valid conventions, not one. row_to_fdi() is the
single place that knows which convention applies to which row, so the
call sites that need an FDI value (grpo.py, judge.py, detector.py,
trace_generation.py, run_zero_shot.py) ask this function instead of
importing dentex_row_to_fdi directly and assuming it's the only dataset
in play -- exactly the assumption that produced the original bug.

Dispatch key is the "source_dataset" column every loader now tags its
annots_df rows with ("dentex" or "tufts"). Rows without that column
(e.g. an existing DENTEX parquet cache written before this column
existed) default to "dentex" -- safe, since DENTEX was the only dataset
in the codebase before Tufts was added, so anything untagged predates
Tufts' existence.
"""

from __future__ import annotations

from typing import Any

from dental_agent.data.dentex import dentex_row_to_fdi


def row_to_fdi(row: Any, default: int = 0) -> tuple[int, int]:
    """One annots_df row -> (quadrant, tooth_position) in 1-indexed FDI,
    regardless of which dataset the row came from.

    Adding a third dataset? Add its convention here explicitly -- do NOT
    let it fall through to the "dentex" default just because that's what
    happens to run without an error. The ValueError below is intentional:
    a silently-wrong assumption about indexing convention is exactly the
    failure mode this module exists to prevent, so an unrecognized
    source_dataset should stop the run, not guess.
    """
    import pandas as pd

    source = (row.get("source_dataset") if hasattr(row, "get") else None) or "dentex"
    if source == "dentex":
        return dentex_row_to_fdi(row, default=default)
    if source == "tufts":
        v1 = row.get("category_id_1") if hasattr(row, "get") else None
        v2 = row.get("category_id_2") if hasattr(row, "get") else None
        q = int(v1) if v1 is not None and not (isinstance(v1, float) and pd.isna(v1)) else default
        tp = int(v2) if v2 is not None and not (isinstance(v2, float) and pd.isna(v2)) else default
        return q, tp
    raise ValueError(
        f"row_to_fdi: unrecognized source_dataset {source!r}. Add an explicit "
        "conversion for it in this function rather than assuming it matches "
        "either DENTEX's 0-indexed convention or Tufts' already-1-indexed "
        "one -- that assumption is exactly the bug class dentex_row_to_fdi "
        "was created to stop."
    )
