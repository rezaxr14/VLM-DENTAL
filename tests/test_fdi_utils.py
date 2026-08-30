import pytest
import pandas as pd
from dental_agent.data.fdi_utils import row_to_fdi


def test_row_to_fdi_dentex():
    # DENTEX is 0-indexed: Quadrant 0..3 -> 1..4, Position 0..7 -> 1..8
    row = {"source_dataset": "dentex", "category_id_1": 0, "category_id_2": 0}
    assert row_to_fdi(row) == (1, 1)

    row = {"source_dataset": "dentex", "category_id_1": 3, "category_id_2": 7}
    assert row_to_fdi(row) == (4, 8)


def test_row_to_fdi_tufts():
    # Tufts is already 1-indexed: Quadrant 1..4, Position 1..8
    row = {"source_dataset": "tufts", "category_id_1": 1, "category_id_2": 1}
    assert row_to_fdi(row) == (1, 1)

    row = {"source_dataset": "tufts", "category_id_1": 4, "category_id_2": 8}
    assert row_to_fdi(row) == (4, 8)


def test_row_to_fdi_untagged_legacy_defaults_to_dentex():
    # Rows without source_dataset (e.g. legacy cache) default to dentex (0-indexed)
    row = {"category_id_1": 2, "category_id_2": 5}
    assert row_to_fdi(row) == (3, 6)


def test_row_to_fdi_pandas_series():
    s_dentex = pd.Series({"source_dataset": "dentex", "category_id_1": 1, "category_id_2": 2})
    assert row_to_fdi(s_dentex) == (2, 3)

    s_tufts = pd.Series({"source_dataset": "tufts", "category_id_1": 2, "category_id_2": 3})
    assert row_to_fdi(s_tufts) == (2, 3)


def test_row_to_fdi_null_and_none_safety():
    # Missing source_dataset with explicit None
    row = {"source_dataset": None, "category_id_1": 0, "category_id_2": 1}
    assert row_to_fdi(row) == (1, 2)

    # Tufts with None / missing values
    row_tufts_missing = {"source_dataset": "tufts", "category_id_1": None, "category_id_2": 5}
    assert row_to_fdi(row_tufts_missing, default=0) == (0, 5)


def test_row_to_fdi_unrecognized_dataset_raises():
    row = {"source_dataset": "unrecognized_source", "category_id_1": 1, "category_id_2": 1}
    with pytest.raises(ValueError, match="unrecognized source_dataset"):
        row_to_fdi(row)
