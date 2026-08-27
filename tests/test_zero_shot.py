"""
Unit tests for Zero-Shot evaluation parsing, finding matching, and baseline metrics.
Zero API calls are executed during these tests (mock stubs only per Rule 10).
"""

import json
import pytest
from dental_agent.evaluation.baselines import (
    parse_zero_shot_response,
    match_zero_shot_finding,
)
from dental_agent.evaluation.metrics import (
    compute_evaluation_metrics,
    compute_diagnostic_metrics,
    expected_calibration_error,
)


def test_parse_zero_shot_response_findings_schema():
    raw_text = """```json
    {
      "findings": [
        {
          "quadrant": 3,
          "tooth_position": 6,
          "diagnosis": "Periapical Lesion",
          "confidence": 0.95
        },
        {
          "quadrant": 1,
          "tooth_position": 8,
          "diagnosis": "Impacted Tooth",
          "confidence": 0.88
        }
      ]
    }
    ```"""
    parsed = parse_zero_shot_response(raw_text)
    assert isinstance(parsed, dict)
    assert "findings" in parsed
    assert len(parsed["findings"]) == 2
    assert parsed["findings"][0]["quadrant"] == 3
    assert parsed["findings"][0]["tooth_position"] == 6
    assert parsed["findings"][0]["diagnosis"] == "Periapical Lesion"


def test_parse_zero_shot_response_with_think_tags():
    raw_text = """<think>
    Looking at quadrant 2, tooth 4 has a dark radiolucent area.
    </think>
    {"findings": [{"quadrant": 2, "tooth_position": 4, "diagnosis": "Caries", "confidence": 0.85}]}"""
    parsed = parse_zero_shot_response(raw_text)
    assert isinstance(parsed, dict)
    assert "findings" in parsed
    assert parsed["findings"][0]["quadrant"] == 2
    assert parsed["findings"][0]["tooth_position"] == 4
    assert parsed["findings"][0]["diagnosis"] == "Caries"


def test_parse_zero_shot_response_direct_dict():
    raw_text = '{"quadrant": 4, "tooth_position": 7, "diagnosis": "Deep Caries", "confidence": 0.90}'
    parsed = parse_zero_shot_response(raw_text)
    assert isinstance(parsed, dict)
    assert parsed["quadrant"] == 4
    assert parsed["tooth_position"] == 7
    assert parsed["diagnosis"] == "Deep Caries"


def test_parse_zero_shot_response_trailing_commas():
    raw_text = '{"findings": [{"quadrant": 1, "tooth_position": 1, "diagnosis": "Caries",}],}'
    parsed = parse_zero_shot_response(raw_text)
    assert isinstance(parsed, dict)
    assert "findings" in parsed
    assert parsed["findings"][0]["quadrant"] == 1


def test_match_zero_shot_finding_multi_match():
    parsed = {
        "findings": [
            {"quadrant": 1, "tooth_position": 5, "diagnosis": "Caries"},
            {"quadrant": 3, "tooth_position": 6, "diagnosis": "Periapical Lesion"},
            {"quadrant": 4, "tooth_position": 8, "diagnosis": "Impacted Tooth"},
        ]
    }
    # Match tooth 36
    matched = match_zero_shot_finding(parsed, gt_quadrant=3, gt_tooth_position=6)
    assert matched is not None
    assert matched["quadrant"] == 3
    assert matched["tooth_position"] == 6
    assert matched["diagnosis"] == "Periapical Lesion"

    # Match tooth 48
    matched_48 = match_zero_shot_finding(parsed, gt_quadrant=4, gt_tooth_position=8)
    assert matched_48 is not None
    assert matched_48["quadrant"] == 4
    assert matched_48["tooth_position"] == 8

    # Fallback to quadrant match if exact position missing
    matched_q1 = match_zero_shot_finding(parsed, gt_quadrant=1, gt_tooth_position=2)
    assert matched_q1 is not None
    assert matched_q1["quadrant"] == 1

    # Fallback to first if nothing matches
    matched_q2 = match_zero_shot_finding(parsed, gt_quadrant=2, gt_tooth_position=1)
    assert matched_q2 is not None
    assert matched_q2["quadrant"] == 1


def test_zero_shot_metric_evaluation_mock():
    mock_results = [
        {
            "image_id": 1,
            "ground_truth": {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"},
            "final_answer": {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries", "confidence": 0.9},
            "format_ok": True,
            "tool_calls": 0,
        },
        {
            "image_id": 2,
            "ground_truth": {"quadrant": 2, "tooth_position": 5, "diagnosis": "Periapical Lesion"},
            "final_answer": {"quadrant": 2, "tooth_position": 5, "diagnosis": "Deep Caries", "confidence": 0.7},
            "format_ok": True,
            "tool_calls": 0,
        },
        {
            "image_id": 3,
            "ground_truth": {"quadrant": 3, "tooth_position": 1, "diagnosis": "Impacted Tooth"},
            "final_answer": {"quadrant": 4, "tooth_position": 1, "diagnosis": "Impacted Tooth", "confidence": 0.6},
            "format_ok": True,
            "tool_calls": 0,
        },
    ]

    metrics = compute_evaluation_metrics(mock_results)
    assert metrics["n_examples"] == 3
    # FDI: 1/3 (image 1) + 1/3 (image 2) = 2/3
    assert abs(metrics["fdi_accuracy"] - 2.0 / 3.0) < 1e-4
    assert metrics["format_compliance_rate"] == 1.0


def test_multi_finding_matching_and_precision_recall():
    from dental_agent.evaluation.metrics import match_multi_findings, extract_predicted_findings

    # Ground truth: 3 findings on radiograph
    gt = [
        {"quadrant": 4, "tooth_position": 8, "diagnosis": "Impacted"},
        {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"},
        {"quadrant": 3, "tooth_position": 6, "diagnosis": "Periapical Lesion"},
    ]

    # Model predicted 2 findings: one exact match (48 Impacted), one FDI match with wrong diag (16 Deep Caries)
    preds = [
        {"quadrant": 4, "tooth_position": 8, "diagnosis": "Impacted Tooth", "confidence": 0.95},
        {"quadrant": 1, "tooth_position": 6, "diagnosis": "Deep Caries", "confidence": 0.80},
    ]

    clean_preds = extract_predicted_findings(preds)
    res = match_multi_findings(gt, clean_preds)

    assert res["gt_count"] == 3
    assert res["pred_count"] == 2
    # Both 48 and 16 localized
    assert res["fdi_tp"] == 2
    assert res["fdi_precision"] == 1.0  # 2/2
    assert abs(res["fdi_recall"] - 2.0 / 3.0) < 1e-4  # 2/3

    # Only 48 has exact diagnosis match
    assert res["exact_tp"] == 1
    assert res["exact_precision"] == 0.5  # 1/2
    assert abs(res["exact_recall"] - 1.0 / 3.0) < 1e-4  # 1/3

