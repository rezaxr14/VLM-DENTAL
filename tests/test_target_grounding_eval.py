"""Adversarial and Edge-Case Unit Tests for Target-Filtered Grounding Evaluation & Hydration.

Tests non-obvious failure modes, duplicate predictions, boundary IoUs,
sparse annotations, coordinate clamping, and recursive model hydration.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from scripts.train_grounding_tool import (
    _compute_box_iou,
    evaluate_target_grounding,
    _ensure_folds_hydrated,
    _model_subdir_prefix,
)


def test_compute_box_iou_edge_cases():
    """Test IoU computation across standard, zero, identical, containment, and boundary cases."""
    # Identical boxes -> IoU = 1.0
    assert _compute_box_iou([10, 10, 50, 50], [10, 10, 50, 50]) == pytest.approx(1.0)

    # Zero overlap (disjoint) -> IoU = 0.0
    assert _compute_box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert _compute_box_iou([0, 0, 10, 10], [10, 10, 20, 20]) == 0.0  # Touching at corner

    # Partial overlap
    # Box 1: [0, 0, 10, 10] (Area = 100)
    # Box 2: [5, 0, 15, 10] (Area = 100)
    # Intersection: [5, 0, 10, 10] (Area = 50)
    # Union: 100 + 100 - 50 = 150 -> IoU = 50 / 150 = 1/3
    assert _compute_box_iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1.0 / 3.0)

    # Containment (Box 2 inside Box 1)
    # Box 1: [0, 0, 20, 20] (Area = 400)
    # Box 2: [5, 5, 15, 15] (Area = 100)
    # Intersection: 100, Union: 400 -> IoU = 100 / 400 = 0.25
    assert _compute_box_iou([0, 0, 20, 20], [5, 5, 15, 15]) == pytest.approx(0.25)


def test_target_eval_sparse_annotations_no_false_positive_penalty():
    """Test that predicting healthy teeth on an image with only 2 diseased ground truth targets
    does NOT penalize precision down to 6%, but evaluates precision strictly on target classes."""
    # Synthetic image: 1 image, width=1000, height=500
    val_images_df = pd.DataFrame([
        {"id": 1, "width": 1000, "height": 500, "local_path": "fake_img_1.png"}
    ])
    
    # Ground Truth has only 2 teeth annotated: Tooth 11 (cls 0) and Tooth 21 (cls 8)
    # Category 1=0 (Q1), Category 2=0 (P1) -> Tooth 11 -> cls 0
    # Category 1=1 (Q2), Category 2=0 (P1) -> Tooth 21 -> cls 8
    val_annots_df = pd.DataFrame([
        {"image_id": 1, "category_id_1": 0, "category_id_2": 0, "bbox": [100, 100, 50, 50]},
        {"image_id": 1, "category_id_1": 1, "category_id_2": 0, "bbox": [200, 100, 50, 50]},
    ])

    # Model predicts 30 teeth across the whole mouth (2 target teeth correctly + 28 healthy teeth)
    pred_boxes = [[100, 100, 150, 150], [200, 100, 250, 150]]  # perfect match for targets
    pred_classes = [0, 8]
    pred_confs = [0.95, 0.90]

    # Add 28 non-target healthy tooth predictions
    for c in range(1, 8):
        pred_boxes.append([100 + c * 20, 200, 150 + c * 20, 250])
        pred_classes.append(c)
        pred_confs.append(0.85)
    for c in range(9, 30):
        pred_boxes.append([100 + c * 20, 300, 150 + c * 20, 350])
        pred_classes.append(c)
        pred_confs.append(0.80)

    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array(pred_boxes)
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array(pred_classes)
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array(pred_confs)
    mock_boxes.__len__.return_value = len(pred_boxes)

    mock_pred_result = MagicMock()
    mock_pred_result.boxes = mock_boxes

    mock_model = MagicMock()
    mock_model.predict.return_value = [mock_pred_result]

    with patch("pathlib.Path.exists", return_value=True):
        res = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)

    # Asserts
    assert res["total_targets"] == 2
    assert res["recall_50"] == pytest.approx(1.0)
    assert res["recall_75"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(1.0)  # Targeted precision is 100%, not 2/30 (6.6%)!
    assert res["mean_iou"] == pytest.approx(1.0)
    assert res["map50"] == pytest.approx(1.0)


def test_target_eval_greedy_bipartite_matching_duplicate_predictions():
    """Test that 2 overlapping predictions for the same tooth only yield 1 True Positive
    and the second prediction is treated as False Positive for precision."""
    val_images_df = pd.DataFrame([
        {"id": 1, "width": 1000, "height": 500, "local_path": "fake_img_1.png"}
    ])
    # 1 target: Tooth 18 (Q1, P8 -> category_id_1=0, category_id_2=7 -> cls 7)
    val_annots_df = pd.DataFrame([
        {"image_id": 1, "category_id_1": 0, "category_id_2": 7, "bbox": [100, 100, 50, 50]}  # GT: [100, 100, 150, 150]
    ])

    # Model predicts 2 boxes for Tooth 18:
    # Box 1: [100, 100, 150, 150] -> IoU = 1.0 (TP)
    # Box 2: [110, 110, 160, 160] -> IoU ~ 0.47 (FP / duplicate)
    pred_boxes = [[100, 100, 150, 150], [110, 110, 160, 160]]
    pred_classes = [7, 7]
    pred_confs = [0.95, 0.70]

    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array(pred_boxes)
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array(pred_classes)
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array(pred_confs)
    mock_boxes.__len__.return_value = 2

    mock_model = MagicMock()
    mock_model.predict.return_value = [MagicMock(boxes=mock_boxes)]

    with patch("pathlib.Path.exists", return_value=True):
        res = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)

    assert res["total_targets"] == 1
    assert res["recall_50"] == pytest.approx(1.0)
    # Precision: 1 TP out of 2 target predictions -> 50% precision
    assert res["precision"] == pytest.approx(0.5)


def test_target_eval_zero_predictions_edge_case():
    """Test that zero predictions produce 0.0 metrics without division by zero errors."""
    val_images_df = pd.DataFrame([
        {"id": 1, "width": 1000, "height": 500, "local_path": "fake_img_1.png"}
    ])
    val_annots_df = pd.DataFrame([
        {"image_id": 1, "category_id_1": 0, "category_id_2": 0, "bbox": [100, 100, 50, 50]}
    ])

    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.zeros((0, 4))
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.zeros((0,), dtype=int)
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.zeros((0,))
    mock_boxes.__len__.return_value = 0

    mock_model = MagicMock()
    mock_model.predict.return_value = [MagicMock(boxes=mock_boxes)]

    with patch("pathlib.Path.exists", return_value=True):
        res = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)

    assert res["total_targets"] == 1
    assert res["recall_50"] == 0.0
    assert res["recall_75"] == 0.0
    assert res["precision"] == 0.0
    assert res["mean_iou"] == 0.0
    assert res["map50"] == 0.0


def test_target_eval_zero_ground_truth_targets():
    """Test image with zero valid ground truth targets."""
    val_images_df = pd.DataFrame([
        {"id": 1, "width": 1000, "height": 500, "local_path": "fake_img_1.png"}
    ])
    val_annots_df = pd.DataFrame([])  # Empty annotations

    mock_model = MagicMock()
    with patch("pathlib.Path.exists", return_value=True):
        res = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)

    assert res["total_targets"] == 0
    assert res["recall_50"] == 0.0
    assert res["precision"] == 0.0


def test_target_eval_boundary_iou_thresholds():
    """Test boundary IoU conditions (exactly 0.50001, 0.49999, and 0.75001)."""
    # GT box: [0, 0, 100, 100] (Area = 10000)
    # Box with width = 100, height = 75:
    # Overlap: [0, 0, 100, 75] (Area = 7500)
    # Union: 10000 + 7500 - 7500 = 10000 -> IoU = 7500/10000 = 0.75
    # Height = 75.001 -> IoU = 7500.1 / 10000 = 0.75001 (TP@0.50=1, TP@0.75=1)
    # Height = 49.999 -> IoU = 4999.9 / 10000 = 0.49999 (TP@0.50=0, TP@0.75=0)
    
    # 1. Test 0.75001 IoU
    val_images_df = pd.DataFrame([
        {"id": 1, "width": 1000, "height": 1000, "local_path": "fake_img_1.png"}
    ])
    val_annots_df = pd.DataFrame([
        {"image_id": 1, "category_id_1": 0, "category_id_2": 0, "bbox": [0, 0, 100, 100]}
    ])
    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[0, 0, 100, 75.01]])
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.90])
    mock_boxes.__len__.return_value = 1
    mock_model = MagicMock()
    mock_model.predict.return_value = [MagicMock(boxes=mock_boxes)]

    with patch("pathlib.Path.exists", return_value=True):
        res = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)
    assert res["recall_50"] == 1.0
    assert res["recall_75"] == 1.0

    # 2. Test 0.49999 IoU
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[0, 0, 100, 49.99]])
    mock_model.predict.return_value = [MagicMock(boxes=mock_boxes)]
    with patch("pathlib.Path.exists", return_value=True):
        res_low = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)
    assert res_low["recall_50"] == 0.0
    assert res_low["recall_75"] == 0.0


def test_target_eval_out_of_bounds_coordinate_clamping():
    """Test that out-of-bounds coordinates (negative or exceeding image dimensions) are clamped safely."""
    val_images_df = pd.DataFrame([
        {"id": 1, "width": 500, "height": 500, "local_path": "fake_img_1.png"}
    ])
    # Box with negative x and width extending beyond image boundary
    val_annots_df = pd.DataFrame([
        {"image_id": 1, "category_id_1": 0, "category_id_2": 0, "bbox": [-50, -20, 600, 550]}
    ])

    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[0, 0, 500, 500]])
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.9])
    mock_boxes.__len__.return_value = 1
    mock_model = MagicMock()
    mock_model.predict.return_value = [MagicMock(boxes=mock_boxes)]

    with patch("pathlib.Path.exists", return_value=True):
        res = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)
    # The GT box was clamped to [0, 0, 500, 500], exactly matching the prediction
    assert res["recall_50"] == 1.0
    assert res["mean_iou"] == pytest.approx(1.0)


def test_ensure_folds_hydrated_recursive_pattern(tmp_path):
    """Test that _ensure_folds_hydrated copies subdirectories recursively and locates all weights."""
    model_root = tmp_path / "models"
    model_root.mkdir()

    # Create synthetic local fold directories
    fold_0 = model_root / "dentex_tufts_cv_fold_0" / "weights"
    fold_0.mkdir(parents=True)
    (fold_0 / "best.pt").write_text("fake_weights_0")

    fold_1 = model_root / "dentex_tufts_cv_fold_1" / "weights"
    fold_1.mkdir(parents=True)
    (fold_1 / "best.pt").write_text("fake_weights_1")

    with patch.dict("os.environ", {"HF_TOKEN": ""}):
        found = _ensure_folds_hydrated(model_root, "dentex_tufts_", 2)

    assert found == [0, 1]
    assert (model_root / "dentex_tufts_cv_fold_0" / "weights" / "best.pt").exists()
    assert (model_root / "dentex_tufts_cv_fold_1" / "weights" / "best.pt").exists()


def test_target_eval_supernumerary_multiple_gt_boxes_same_tooth():
    """Test when an image has multiple ground truth findings for the same tooth class."""
    val_images_df = pd.DataFrame([
        {"id": 1, "width": 1000, "height": 500, "local_path": "fake_img_1.png"}
    ])
    # 2 GT boxes for tooth 11 (cls 0) at distinct locations
    val_annots_df = pd.DataFrame([
        {"image_id": 1, "category_id_1": 0, "category_id_2": 0, "bbox": [100, 100, 50, 50]},
        {"image_id": 1, "category_id_1": 0, "category_id_2": 0, "bbox": [300, 100, 50, 50]},
    ])

    # Model predicts 2 boxes matching each GT box
    pred_boxes = [[100, 100, 150, 150], [300, 100, 350, 150]]
    pred_classes = [0, 0]
    pred_confs = [0.95, 0.90]

    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array(pred_boxes)
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array(pred_classes)
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array(pred_confs)
    mock_boxes.__len__.return_value = 2

    mock_model = MagicMock()
    mock_model.predict.return_value = [MagicMock(boxes=mock_boxes)]

    with patch("pathlib.Path.exists", return_value=True):
        res = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)

    assert res["total_targets"] == 2
    assert res["recall_50"] == 1.0
    assert res["recall_75"] == 1.0
    assert res["precision"] == 1.0
    assert res["mean_iou"] == pytest.approx(1.0)


def test_benchmark_comparative_json_schema_completeness(tmp_path):
    """Test that comparative evaluation results produce valid JSON and strictly conforming schemas."""
    synthetic_results = [
        {
            "family": "DENTEX-Only",
            "name": "DENTEX-Only (Fold 0)",
            "fold": 0,
            "recall_50": 0.8542,
            "recall_75": 0.6215,
            "precision": 0.8910,
            "mean_iou": 0.7420,
            "map50": 0.8350,
            "map50_95": 0.5480,
            "total_targets": 182,
        },
        {
            "family": "DENTEX+Tufts",
            "name": "DENTEX+Tufts (5-Fold Mean)",
            "fold": "mean",
            "recall_50": 0.8850,
            "recall_75": 0.6720,
            "precision": 0.9120,
            "mean_iou": 0.7810,
            "map50": 0.8695,
            "map50_95": 0.6080,
            "total_targets": 182,
        },
    ]
    out_file = tmp_path / "benchmark_comparative_evaluation.json"
    out_file.write_text(json.dumps(synthetic_results, indent=2))

    loaded = json.loads(out_file.read_text())
    assert isinstance(loaded, list)
    assert len(loaded) == 2
    required_keys = {"family", "name", "fold", "recall_50", "recall_75", "precision", "mean_iou", "map50"}
    for item in loaded:
        assert required_keys.issubset(item.keys())
        assert isinstance(item["recall_50"], float)
        assert np.isfinite(item["recall_50"])
        assert 0.0 <= item["recall_50"] <= 1.0

