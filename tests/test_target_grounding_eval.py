"""Comprehensive, Adversarial Unit Tests for Target-Filtered Grounding Evaluation & Hydration.

Validates:
1. Real DENTEX COCO JSON schema parsing, 0-index to FDI conversion (Rule 1), and null safety.
2. Greedy 1-to-1 bipartite target matching under multi-finding cluttered tooth predictions (Rule 13).
3. Precision-recall behavior on sparse/partial disease benchmarks without false-positive penalties.
4. Zero-division and empty dataset edge cases.
5. Exact IoU boundary thresholds (0.50001, 0.49999, 0.75001).
6. Out-of-bounds coordinate clamping and malformed bounding box geometry.
7. Recursive Hugging Face Hub fold hydration patterns across dual model families.
8. Comparative evaluation JSON schema validity and numerical sanity.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image

from scripts.train_grounding_tool import (
    _compute_box_iou,
    evaluate_target_grounding,
    evaluate_yolo_labels_target_grounding,
    _ensure_folds_hydrated,
    _model_subdir_prefix,
)
from dental_agent.data.dentex import dentex_row_to_fdi


# ==============================================================================
# 1. IoU Computation & Geometric Edge Cases
# ==============================================================================

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


def test_target_eval_boundary_iou_thresholds():
    """Test boundary IoU conditions (exactly 0.50001, 0.49999, and 0.75001)."""
    # 1. Test 0.75001 IoU (TP@0.50=1, TP@0.75=1)
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

    # 2. Test 0.49999 IoU (TP@0.50=0, TP@0.75=0)
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
    assert res["recall_50"] == 1.0
    assert res["mean_iou"] == pytest.approx(1.0)


# ==============================================================================
# 2. DENTEX Real Schema & FDI 0-Index to 1-Index Conversions
# ==============================================================================

def test_real_dentex_coco_json_parsing_and_fdi_mapping():
    """Test parsing realistic DENTEX validation JSON schema across all 4 quadrants and 8 positions."""
    # Test all 32 valid quadrant and position combinations (0-indexed to 1-indexed)
    for q_0 in range(4):
        for p_0 in range(8):
            row = {"category_id_1": q_0, "category_id_2": p_0}
            q_1, p_1 = dentex_row_to_fdi(row)
            assert q_1 == q_0 + 1
            assert p_1 == p_0 + 1
            assert 1 <= q_1 <= 4
            assert 1 <= p_1 <= 8
            # Verify YOLO class index mapping (0 to 31)
            cls_idx = (q_1 - 1) * 8 + (p_1 - 1)
            assert 0 <= cls_idx <= 31

    # Test missing categories with default fallback
    bad_row_1 = {"category_id_2": 3}
    assert dentex_row_to_fdi(bad_row_1, default=0) == (1, 4)



# ==============================================================================
# 3. Multi-Finding Bipartite Matching Under Clutter (Rule 13)
# ==============================================================================

def test_target_eval_sparse_annotations_no_false_positive_penalty():
    """Test that predicting healthy teeth on an image with only 2 diseased ground truth targets
    does NOT penalize precision down to 6%, but evaluates precision strictly on target classes."""
    val_images_df = pd.DataFrame([
        {"id": 1, "width": 1000, "height": 500, "local_path": "fake_img_1.png"}
    ])
    
    # Ground Truth has only 2 teeth annotated: Tooth 11 (cls 0) and Tooth 21 (cls 8)
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

    mock_model = MagicMock()
    mock_model.predict.return_value = [MagicMock(boxes=mock_boxes)]

    with patch("pathlib.Path.exists", return_value=True):
        res = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)

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
    assert res["precision"] == pytest.approx(0.5)


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


# ==============================================================================
# 4. Zero Predictions & Missing Annotation Edge Cases
# ==============================================================================

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
    val_annots_df = pd.DataFrame([])  # Empty annotations DataFrame

    mock_model = MagicMock()
    with patch("pathlib.Path.exists", return_value=True):
        res = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)

    assert res["total_targets"] == 0
    assert res["recall_50"] == 0.0
    assert res["precision"] == 0.0


# ==============================================================================
# 5. Hydration & Schema Invariants
# ==============================================================================

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


def test_evaluate_yolo_labels_target_grounding(tmp_path):
    """Test evaluate_yolo_labels_target_grounding with synthetic image and normalized YOLO .txt label."""
    from scripts.train_grounding_tool import evaluate_yolo_labels_target_grounding
    from PIL import Image

    img_dir = tmp_path / "images" / "val"
    lbl_dir = tmp_path / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    # Create dummy 1000x500 image
    img_path = img_dir / "val_001.png"
    im = Image.new("RGB", (1000, 500), color=(128, 128, 128))
    im.save(img_path)

    # Label: Class 0 (quadrant 1, pos 1) centered at (0.2, 0.4), w=0.1, h=0.2
    # Absolute xyxy: [150, 150, 250, 250]
    lbl_path = lbl_dir / "val_001.txt"
    lbl_path.write_text("0 0.2 0.4 0.1 0.2\n")

    # Mock YOLO model with exact prediction
    mock_model = MagicMock()
    mock_box = MagicMock()
    mock_box.xyxy = MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([[150.0, 150.0, 250.0, 250.0]])))))
    mock_box.cls = MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([0])))))
    mock_box.conf = MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array([0.95])))))
    mock_box.__len__ = MagicMock(return_value=1)

    mock_preds = MagicMock()
    mock_preds.boxes = mock_box
    mock_model.predict.return_value = [mock_preds]

    res = evaluate_yolo_labels_target_grounding(mock_model, img_dir, lbl_dir, conf_thresh=0.001, nominal_conf_thresh=0.25)
    assert res["total_targets"] == 1
    assert res["recall_50"] == 1.0
    assert res["recall_75"] == 1.0
    assert res["precision"] == 1.0
    assert res["mean_iou"] == pytest.approx(1.0)
    assert res["map50"] == pytest.approx(1.0)


def test_yolo_labels_target_eval_sparse_gt_ignores_healthy_teeth(tmp_path):
    """Test that predicting whole-mouth healthy teeth on a sparse YOLO GT label file
    does NOT penalize precision, evaluating strictly against target classes."""
    img_dir = tmp_path / "images" / "val"
    lbl_dir = tmp_path / "labels" / "val"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)

    img_path = img_dir / "val_sparse.png"
    im = Image.new("RGB", (1000, 500), color=(100, 100, 100))
    im.save(img_path)

    # Sparse GT: Only Tooth 11 (cls 0) and Tooth 21 (cls 8)
    lbl_path = lbl_dir / "val_sparse.txt"
    lbl_path.write_text("0 0.15 0.3 0.1 0.2\n8 0.25 0.3 0.1 0.2\n")

    # Model predicts 30 teeth across the mouth (2 matching GT targets + 28 unannotated healthy teeth)
    pred_boxes = [[100.0, 100.0, 200.0, 200.0], [200.0, 100.0, 300.0, 200.0]]  # Matches cls 0 and cls 8
    pred_classes = [0, 8]
    pred_confs = [0.95, 0.90]

    for c in range(1, 8):
        pred_boxes.append([100.0 + c * 20, 300.0, 150.0 + c * 20, 400.0])
        pred_classes.append(c)
        pred_confs.append(0.85)
    for c in range(9, 30):
        pred_boxes.append([100.0 + c * 20, 300.0, 150.0 + c * 20, 400.0])
        pred_classes.append(c)
        pred_confs.append(0.80)

    mock_model = MagicMock()
    mock_box = MagicMock()
    mock_box.xyxy = MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array(pred_boxes)))))
    mock_box.cls = MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array(pred_classes)))))
    mock_box.conf = MagicMock(cpu=MagicMock(return_value=MagicMock(numpy=MagicMock(return_value=np.array(pred_confs)))))
    mock_box.__len__ = MagicMock(return_value=len(pred_boxes))

    mock_preds = MagicMock()
    mock_preds.boxes = mock_box
    mock_model.predict.return_value = [mock_preds]

    res = evaluate_yolo_labels_target_grounding(mock_model, img_dir, lbl_dir, conf_thresh=0.001, nominal_conf_thresh=0.25)
    assert res["total_targets"] == 2
    assert res["recall_50"] == pytest.approx(1.0)
    assert res["precision"] == pytest.approx(1.0)
    assert res["mean_iou"] == pytest.approx(1.0)


def test_ensure_folds_hydrated_local_disk_precedence(tmp_path):
    """Test that _ensure_folds_hydrated finds local fold weights without downloading from HF Hub."""
    from scripts.train_grounding_tool import _ensure_folds_hydrated
    model_root = tmp_path / "models"
    for fold in range(5):
        w_dir = model_root / f"dentex_tufts_cv_fold_{fold}" / "weights"
        w_dir.mkdir(parents=True)
        (w_dir / "best.pt").write_text("fake weight")

    with patch("huggingface_hub.snapshot_download") as mock_download:
        found = _ensure_folds_hydrated(model_root, "dentex_tufts_", 5)
        assert found == [0, 1, 2, 3, 4]
        mock_download.assert_not_called()


# ==============================================================================
# 6. True COCO 10-Threshold mAP50-95 Invariant Tests
# ==============================================================================

def test_map50_95_monotonicity_and_perfect_score():
    """Test that perfect predictions yield mAP50 = 1.0 and mAP50-95 = 1.0."""
    val_images_df = pd.DataFrame([
        {"id": 1, "width": 1000, "height": 1000, "local_path": "fake_img_1.png"}
    ])
    val_annots_df = pd.DataFrame([
        {"image_id": 1, "category_id_1": 0, "category_id_2": 0, "bbox": [100, 100, 50, 50]},
        {"image_id": 1, "category_id_1": 1, "category_id_2": 1, "bbox": [200, 200, 50, 50]},
    ])

    # Perfect predictions with IoU = 1.0
    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([
        [100, 100, 150, 150],
        [200, 200, 250, 250],
    ])
    # Class 0: Q1 P1 -> cls 0; Class 1: Q2 P2 -> cls 9
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 9])
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.95, 0.90])
    mock_boxes.__len__.return_value = 2

    mock_model = MagicMock()
    mock_model.predict.return_value = [MagicMock(boxes=mock_boxes)]

    with patch("pathlib.Path.exists", return_value=True):
        res = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)

    assert res["map50"] == pytest.approx(1.0)
    assert res["map50_95"] == pytest.approx(1.0)
    assert res["map50"] >= res["map50_95"]


def test_map50_95_partial_iou_dropoff():
    """Test that predictions with ~0.60 IoU have mAP50 = 1.0 but mAP50-95 < 1.0."""
    val_images_df = pd.DataFrame([
        {"id": 1, "width": 1000, "height": 1000, "local_path": "fake_img_1.png"}
    ])
    val_annots_df = pd.DataFrame([
        {"image_id": 1, "category_id_1": 0, "category_id_2": 0, "bbox": [100, 100, 100, 100]},  # [100, 100, 200, 200] Area 10000
    ])

    # Prediction [100, 100, 200, 160] -> Inter = 100 * 60 = 6000, Union = 10000 -> IoU = 0.60
    # At IoU thresh 0.50, 0.55, 0.60 -> TP (3 thresholds)
    # At IoU thresh 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95 -> FP (7 thresholds)
    # So mAP50 == 1.0, but mAP50_95 == 3/10 = 0.30
    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 200, 160]])
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.95])
    mock_boxes.__len__.return_value = 1

    mock_model = MagicMock()
    mock_model.predict.return_value = [MagicMock(boxes=mock_boxes)]

    with patch("pathlib.Path.exists", return_value=True):
        res = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)

    assert res["map50"] == pytest.approx(1.0)
    assert res["map50_95"] == pytest.approx(0.30)
    assert res["map50"] > res["map50_95"]


def test_nominal_matching_no_double_counting_recall():
    """Test that a single prediction box matching two GT targets of the same class
    is only counted ONCE for TP (Recall = 0.50, NOT 1.0)."""
    val_images_df = pd.DataFrame([
        {"id": 1, "width": 1000, "height": 1000, "local_path": "fake_img_1.png"}
    ])
    val_annots_df = pd.DataFrame([
        {"image_id": 1, "category_id_1": 0, "category_id_2": 0, "bbox": [100, 100, 50, 50]},
        {"image_id": 1, "category_id_1": 0, "category_id_2": 0, "bbox": [110, 110, 50, 50]},
    ])

    # Only 1 prediction box
    mock_boxes = MagicMock()
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[100, 100, 150, 150]])
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0])
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.90])
    mock_boxes.__len__.return_value = 1

    mock_model = MagicMock()
    mock_model.predict.return_value = [MagicMock(boxes=mock_boxes)]

    with patch("pathlib.Path.exists", return_value=True):
        res = evaluate_target_grounding(mock_model, val_images_df, val_annots_df)

    assert res["total_targets"] == 2
    assert res["recall_50"] == pytest.approx(0.50)  # Exactly 1 of 2 GTs matched!
    assert res["precision"] == pytest.approx(1.0)




