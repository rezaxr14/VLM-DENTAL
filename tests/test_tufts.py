import json
from pathlib import Path
import pytest
from dental_agent.data.tufts import (
    _permanent_universal_to_fdi,
    _title_to_fdi,
    _stem_id,
    _bbox_yxyx_to_xywh,
    _is_degenerate_bbox,
    _ring_bboxes,
    _bbox_overlap_area,
    _map_tufts_pathology_to_dentex_category,
    _load_tooth_index,
    _load_findings,
    _PRIMARY_LETTER_TO_FDI,
)


def test_permanent_universal_to_fdi():
    # Quadrant 1 (UR 18..11 -> Univ 1..8)
    assert _permanent_universal_to_fdi(1) == (1, 8)
    assert _permanent_universal_to_fdi(8) == (1, 1)

    # Quadrant 2 (UL 21..28 -> Univ 9..16)
    assert _permanent_universal_to_fdi(9) == (2, 1)
    assert _permanent_universal_to_fdi(16) == (2, 8)

    # Quadrant 3 (LL 38..31 -> Univ 17..24)
    assert _permanent_universal_to_fdi(17) == (3, 8)
    assert _permanent_universal_to_fdi(24) == (3, 1)

    # Quadrant 4 (LR 41..48 -> Univ 25..32)
    assert _permanent_universal_to_fdi(25) == (4, 1)
    assert _permanent_universal_to_fdi(32) == (4, 8)

    with pytest.raises(ValueError):
        _permanent_universal_to_fdi(0)
    with pytest.raises(ValueError):
        _permanent_universal_to_fdi(33)


def test_title_to_fdi():
    # Permanent
    assert _title_to_fdi("1", include_primary_teeth=False) == (1, 8, False)
    assert _title_to_fdi("14", include_primary_teeth=False) == (2, 6, False)

    # Primary excluded by default
    assert _title_to_fdi("A", include_primary_teeth=False) is None
    # Primary included
    assert _title_to_fdi("A", include_primary_teeth=True) == (5, 5, True)
    assert _title_to_fdi("T", include_primary_teeth=True) == (8, 5, True)

    # Invalid title
    assert _title_to_fdi("invalid", include_primary_teeth=True) is None
    assert _title_to_fdi("99", include_primary_teeth=True) is None


def test_stem_id_cross_platform():
    assert _stem_id("797.jpg") == 797
    assert _stem_id("245.JPG") == 245
    assert _stem_id("Radiographs/100.jpg") == 100
    assert _stem_id("Radiographs\\1051.JPG") == 1051
    assert _stem_id("  folder/sub/5.png  ") == 5


def test_bbox_conversions_and_degeneracy():
    # ymin, xmin, ymax, xmax -> x, y, w, h
    yxyx = [100.0, 50.0, 300.0, 150.0]
    xywh = _bbox_yxyx_to_xywh(yxyx)
    assert xywh == [50.0, 100.0, 100.0, 200.0]

    # Degenerate boxes (span < 3px)
    assert _is_degenerate_bbox([100.0, 50.0, 101.0, 150.0]) is True
    assert _is_degenerate_bbox([100.0, 50.0, 200.0, 51.0]) is True
    assert _is_degenerate_bbox([100.0, 50.0, 200.0, 150.0]) is False


def test_ring_bboxes_and_overlap():
    # Multi-ring extraction
    polygons = [
        [[10.0, 20.0], [30.0, 20.0], [30.0, 40.0]],  # valid ring (3 pts) -> (10, 20, 30, 40)
        [[100.0, 200.0], [105.0, 205.0]],             # degenerate ring (<3 pts) -> ignored
        [[50.0, 60.0], [80.0, 60.0], [80.0, 90.0], [50.0, 90.0]], # valid ring -> (50, 60, 80, 90)
    ]
    boxes = _ring_bboxes(polygons)
    assert len(boxes) == 2
    assert boxes[0] == (10.0, 20.0, 30.0, 40.0)
    assert boxes[1] == (50.0, 60.0, 80.0, 90.0)

    # Overlap area
    box1 = (0.0, 0.0, 10.0, 10.0)
    box2 = (5.0, 5.0, 15.0, 15.0)
    box3 = (20.0, 20.0, 30.0, 30.0)
    assert _bbox_overlap_area(box1, box2) == 25.0
    assert _bbox_overlap_area(box1, box3) == 0.0


def test_taxonomy_mapping():
    assert _map_tufts_pathology_to_dentex_category("Periapical") == 2
    assert _map_tufts_pathology_to_dentex_category("None") is None
    assert _map_tufts_pathology_to_dentex_category("Non-Odontogenic") is None
    assert _map_tufts_pathology_to_dentex_category("Pericoronal") is None
    assert _map_tufts_pathology_to_dentex_category("Inter-Radicular") is None


def test_load_tooth_index_and_findings_mock(tmp_path: Path):
    # Create mock teeth_bbox.json
    teeth_bbox_data = [
        {
            "External ID": "1.jpg",
            "Label": {
                "objects": [
                    {"title": "1", "bounding box": [100, 50, 200, 150]},
                    {"title": "A", "bounding box": [300, 100, 400, 200]},
                    {"title": "2", "bounding box": [100, 50, 101, 150]},  # degenerate
                ]
            }
        }
    ]
    teeth_file = tmp_path / "teeth_bbox.json"
    teeth_file.write_text(json.dumps(teeth_bbox_data))

    index_no_primary, stats = _load_tooth_index(teeth_file, include_primary_teeth=False)
    assert 1 in index_no_primary
    assert len(index_no_primary[1]) == 1
    assert index_no_primary[1][0]["quadrant"] == 1
    assert index_no_primary[1][0]["tooth_position"] == 8
    assert stats["n_degenerate_excluded"] == 1
    assert stats["n_primary_excluded"] == 1

    index_with_primary, _ = _load_tooth_index(teeth_file, include_primary_teeth=True)
    assert len(index_with_primary[1]) == 2

    # Create mock expert.json
    expert_data = [
        {
            "External ID": "1.JPG",
            "Description": "Periapical lesion on tooth 1",
            "Label": {
                "objects": [
                    {
                        "title": "Periapical",
                        "polygons": [[[50, 100], [150, 100], [150, 200], [50, 200]]],
                        "classifications": [
                            {"title": "Level one", "answer": {"value": "Periapical Radiolucency"}},
                            {"title": "Level two", "answers": [{"value": "Ill-defined"}]},
                        ]
                    },
                    {
                        "title": "None",
                        "polygons": []
                    }
                ]
            }
        }
    ]
    expert_file = tmp_path / "expert.json"
    expert_file.write_text(json.dumps(expert_data))

    findings = _load_findings(expert_file)
    assert len(findings) == 1
    assert findings[0]["image_id"] == 1
    assert len(findings[0]["findings"]) == 1
    f0 = findings[0]["findings"][0]
    assert f0["dentex_category_id"] == 2
    assert f0["level_1"] == "Periapical Radiolucency"
    assert f0["level_2"] == "Ill-defined"
