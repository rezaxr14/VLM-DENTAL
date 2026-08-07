"""
Unit tests for FDI numbering and anatomical mapping utilities.
"""

import pytest

from dental_agent.tools.fdi import (
    fdi_encode,
    fdi_decode,
    tool_fdi_label,
    get_anatomical_name,
    flip_quadrant,
)


def test_fdi_encode_decode() -> None:
    assert fdi_encode(3, 6) == 36
    assert fdi_encode(1, 1) == 11
    assert fdi_encode(4, 8) == 48

    assert fdi_decode(36) == (3, 6)
    assert fdi_decode("21") == (2, 1)


def test_fdi_invalid() -> None:
    with pytest.raises(ValueError):
        fdi_encode(5, 1)
    with pytest.raises(ValueError):
        fdi_encode(1, 9)
    with pytest.raises(ValueError):
        fdi_decode(59)


def test_tool_fdi_label() -> None:
    assert tool_fdi_label(3, 6) == "36"
    assert tool_fdi_label(9, 1) is None
    assert tool_fdi_label(2, 0) is None


def test_get_anatomical_name() -> None:
    name = get_anatomical_name(1, 1)
    assert "Maxillary Right" in name
    assert "Central Incisor" in name
    assert "FDI #11" in name


def test_flip_quadrant() -> None:
    assert flip_quadrant(1) == 2
    assert flip_quadrant(2) == 1
    assert flip_quadrant(3) == 4
    assert flip_quadrant(4) == 3

    with pytest.raises(ValueError):
        flip_quadrant(5)
