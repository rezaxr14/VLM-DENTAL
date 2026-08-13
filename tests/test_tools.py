"""
Unit tests for deterministic diagnostic tools and grounding interfaces.
"""

import pytest
from PIL import Image

from dental_agent.tools.zoom_crop import tool_zoom_crop
from dental_agent.tools.windowing import tool_window_level
from dental_agent.tools.grounding import ToothGrounder
from dental_agent.training.detector import compute_iou
from dental_agent.tools.registry import ToolRegistry


def test_zoom_crop(synthetic_image: Image.Image) -> None:
    bbox = [100, 100, 50, 50]
    crop = tool_zoom_crop(synthetic_image, bbox, padding_frac=0.2)

    assert isinstance(crop, Image.Image)
    # pad_x = max(50*0.2, 50.0) = 50, so width = 50 + 2*50 = 150
    assert crop.width == 150
    assert crop.height == 150


def test_zoom_crop_clamping(synthetic_image: Image.Image) -> None:
    # Test boundary clamping at top-left
    bbox = [0, 0, 50, 50]
    crop = tool_zoom_crop(synthetic_image, bbox, padding_frac=0.5)
    assert crop.width <= synthetic_image.width
    assert crop.height <= synthetic_image.height


def test_window_level(synthetic_image: Image.Image) -> None:
    windowed = tool_window_level(synthetic_image, preset="enamel")
    assert isinstance(windowed, Image.Image)
    assert windowed.size == synthetic_image.size


def test_compute_iou() -> None:
    box_a = [0, 0, 10, 10]
    box_b = [0, 0, 10, 10]
    assert abs(compute_iou(box_a, box_b) - 1.0) < 1e-6

    box_c = [20, 20, 30, 30]
    assert compute_iou(box_a, box_c) == 0.0

    # 50% overlap in 1D => area intersection = 5*10=50, union = 100+100-50=150 => 1/3
    box_d = [5, 0, 15, 10]
    assert abs(compute_iou(box_a, box_d) - (50.0 / 150.0)) < 1e-6


def test_tool_registry() -> None:
    registry = ToolRegistry.create_default()
    assert registry.get("zoom_crop") is not None
    assert registry.get("window_level") is not None
    assert registry.get("denoise") is not None
    assert registry.get("contralateral_compare") is not None
    assert registry.get("fdi_label") is not None

    desc = registry.format_tool_descriptions()
    assert "zoom_crop" in desc
    assert "window_level" in desc
    assert "denoise" in desc
