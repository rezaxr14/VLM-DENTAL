"""
Unit tests for deterministic diagnostic tools and grounding interfaces.
"""

import pytest
from PIL import Image

from dental_agent.tools.zoom_crop import tool_zoom_crop
from dental_agent.tools.contrast import tool_enhance_contrast
from dental_agent.tools.grounding import compute_iou, OracleGroundingTool
from dental_agent.tools.registry import ToolRegistry


def test_zoom_crop(synthetic_image: Image.Image) -> None:
    bbox = [100, 100, 50, 50]
    crop = tool_zoom_crop(synthetic_image, bbox, padding_frac=0.2)

    assert isinstance(crop, Image.Image)
    # 50 + 2*10 = 70
    assert crop.width == 70
    assert crop.height == 70


def test_zoom_crop_clamping(synthetic_image: Image.Image) -> None:
    # Test boundary clamping at top-left
    bbox = [0, 0, 50, 50]
    crop = tool_zoom_crop(synthetic_image, bbox, padding_frac=0.5)
    assert crop.width <= synthetic_image.width
    assert crop.height <= synthetic_image.height


def test_contrast_enhancement(synthetic_image: Image.Image) -> None:
    enhanced = tool_enhance_contrast(synthetic_image, factor=2.0)
    assert isinstance(enhanced, Image.Image)
    assert enhanced.size == synthetic_image.size


def test_compute_iou() -> None:
    box_a = [0, 0, 10, 10]
    box_b = [0, 0, 10, 10]
    assert abs(compute_iou(box_a, box_b) - 1.0) < 1e-6

    box_c = [20, 20, 30, 30]
    assert compute_iou(box_a, box_c) == 0.0

    # 50% overlap in 1D => area intersection = 5*10=50, union = 100+100-50=150 => 1/3
    box_d = [5, 0, 15, 10]
    assert abs(compute_iou(box_a, box_d) - (50.0 / 150.0)) < 1e-6


def test_oracle_grounding_tool(sample_annotations_df, sample_categories_df) -> None:
    oracle = OracleGroundingTool(
        annots_df=sample_annotations_df,
        categories_df=sample_categories_df,
        diag_col="category_id_3",
    )

    results = oracle(image_id=100)
    assert len(results) == 2
    assert results[0]["quadrant"] == 1
    assert results[0]["tooth_position"] == 6
    assert results[0]["fdi_label"] == "16"
    assert results[0]["diagnosis"] == "Caries"

    assert results[1]["quadrant"] == 3
    assert results[1]["tooth_position"] == 8
    assert results[1]["fdi_label"] == "38"
    assert results[1]["diagnosis"] == "Impacted Tooth"


def test_tool_registry() -> None:
    registry = ToolRegistry.create_default()
    assert registry.get("zoom_crop") is not None
    assert registry.get("enhance_contrast") is not None
    assert registry.get("fdi_label") is not None

    desc = registry.format_tool_descriptions()
    assert "zoom_crop" in desc
    assert "enhance_contrast" in desc
