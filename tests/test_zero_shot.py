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

    # Closeness assertions
    assert "closeness_score" in res
    assert res["closeness_score"] > 0.0


def test_continuous_closeness_scoring():
    from dental_agent.evaluation.metrics import compute_finding_closeness

    # Exact match: 1.0
    gt = {"quadrant": 4, "tooth_position": 8, "diagnosis": "Impacted"}
    pred_exact = {"quadrant": 4, "tooth_position": 8, "diagnosis": "Impacted Tooth"}
    c, s, d = compute_finding_closeness(gt, pred_exact)
    assert c == 1.0
    assert s == 1.0
    assert d == 1.0

    # Adjacent tooth, exact diag: spatial 0.75, diag 1.0 -> closeness 0.875
    pred_adj = {"quadrant": 4, "tooth_position": 7, "diagnosis": "Impacted"}
    c, s, d = compute_finding_closeness(gt, pred_adj)
    assert s == 0.75
    assert d == 1.0
    assert c == 0.875

    # Same tooth, Caries <-> Deep Caries spectrum: spatial 1.0, diag 0.75 -> closeness 0.875
    gt_caries = {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}
    pred_deep = {"quadrant": 1, "tooth_position": 6, "diagnosis": "Deep Caries"}
    c, s, d = compute_finding_closeness(gt_caries, pred_deep)
    assert s == 1.0
    assert d == 0.75
    assert c == 0.875


def test_extract_findings_from_truncated_reasoning_trace():
    """Test realistic truncated Qwen 3.6 reasoning output where the model was cut off
    mid-thought before writing the final JSON."""
    truncated_raw = """
    <think>
    The user wants me to analyze a panoramic dental radiograph (OPG) and identify abnormal teeth.
    1. Examine Quadrant 1:
       - Tooth 18 (Upper Right Third Molar): It appears impacted. Diagnosis: Impacted Tooth.
       - Tooth 16 (Upper Right First Molar): There is a coronal radiolucency. Diagnosis: Caries.
    2. Examine Quadrant 4:
       - Tooth 48 (Lower Right Third Molar): Clearly visible as an impacted tooth.
       - Tooth 46 (Lower Right First Molar): There is periapical radiolucency. Diagnosis: Periapical Lesion.
    3. Examine Quadrant 3:
       - Tooth 38 is impacted horizontally against 37.
    """
    from dental_agent.evaluation.baselines import extract_findings_from_reasoning_text
    from dental_agent.evaluation.metrics import extract_predicted_findings

    extracted = extract_findings_from_reasoning_text(truncated_raw)
    assert len(extracted) >= 4

    clean = extract_predicted_findings(extracted)
    quad_positions = [(p["quadrant"], p["tooth_position"]) for p in clean]
    assert (1, 8) in quad_positions
    assert (1, 6) in quad_positions
    assert (4, 8) in quad_positions
    assert (3, 8) in quad_positions


def test_parse_zero_shot_alternative_key_formats():
    """Test diverse LLM output variations with alternate key names."""
    from dental_agent.evaluation.metrics import extract_predicted_findings

    # Model using 'fdi' string and 'condition'
    sample1 = [{"fdi": "48", "condition": "Impacted", "confidence": 0.9}]
    clean1 = extract_predicted_findings(sample1)
    assert len(clean1) == 1
    assert clean1[0]["quadrant"] == 4
    assert clean1[0]["tooth_position"] == 8
    assert clean1[0]["diagnosis"] == "Impacted"

    # Model using 'tooth' int and 'pathology'
    sample2 = [{"tooth": 36, "pathology": "caries", "confidence": 0.85}]
    clean2 = extract_predicted_findings(sample2)
    assert len(clean2) == 1
    assert clean2[0]["quadrant"] == 3
    assert clean2[0]["tooth_position"] == 6
    assert clean2[0]["diagnosis"] == "Caries"

    # Model using nested 'findings' with 'disease'
    sample3 = {"findings": [{"quadrant": 2, "tooth_position": 5, "disease": "Periapical Lesion"}]}
    clean3 = extract_predicted_findings(sample3)
    assert len(clean3) == 1
    assert clean3[0]["quadrant"] == 2
    assert clean3[0]["tooth_position"] == 5
    assert clean3[0]["diagnosis"] == "Periapical Lesion"


def test_negation_awareness_in_reasoning_extractor():
    """Verify that sentences stating 'no caries' or 'normal' are NOT falsely extracted as findings."""
    from dental_agent.evaluation.baselines import extract_findings_from_reasoning_text

    text_with_negations = """
    <think>
    - Tooth 16 (Upper Right First Molar): There is no caries visible. Tooth is intact and normal.
    - Tooth 17: No evidence of periapical lesion.
    - Tooth 48 (Lower Right Third Molar): Clearly visible as an impacted tooth. Diagnosis: Impacted Tooth.
    - Tooth 36: Ruled out deep caries.
    - Tooth 44: Diagnosis: Periapical Lesion.
    </think>
    """
    findings = extract_findings_from_reasoning_text(text_with_negations)
    positions = [(f["quadrant"], f["tooth_position"]) for f in findings]
    
    # 48 and 44 must be extracted (affirmative findings)
    assert (4, 8) in positions
    assert (4, 4) in positions
    
    # 16, 17, and 36 must NOT be extracted (negated findings)
    assert (1, 6) not in positions
    assert (1, 7) not in positions
    assert (3, 6) not in positions


def test_load_completed_ids_retry_empty(tmp_path):
    """Test that load_completed_ids filters out 0-prediction records when retry_empty=True."""
    from scripts.run_zero_shot import load_completed_ids, save_evaluation_record_atomic

    jsonl_path = tmp_path / "test_eval.jsonl"
    
    # Save record 1 (valid)
    rec1 = {"image_id": 1, "predictions": [{"quadrant": 1, "tooth_position": 8, "diagnosis": "Impacted"}], "format_ok": True}
    save_evaluation_record_atomic(jsonl_path, rec1)

    # Save record 2 (empty/truncated)
    rec2 = {"image_id": 2, "predictions": [], "format_ok": False}
    save_evaluation_record_atomic(jsonl_path, rec2)

    # Without retry_empty: returns both 1 and 2
    all_completed = load_completed_ids(jsonl_path, retry_empty=False)
    assert all_completed == {1, 2}

    # With retry_empty: returns only 1 (image 2 must be re-evaluated)
    valid_completed = load_completed_ids(jsonl_path, retry_empty=True)
    assert valid_completed == {1}

    # Now simulate re-evaluating image 2 with valid predictions
    rec2_fixed = {"image_id": 2, "predictions": [{"quadrant": 4, "tooth_position": 8, "diagnosis": "Impacted"}], "format_ok": True}
    save_evaluation_record_atomic(jsonl_path, rec2_fixed)

    # Now both are completed and file has exactly 2 lines
    final_completed = load_completed_ids(jsonl_path, retry_empty=True)
    assert final_completed == {1, 2}
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    assert len(lines) == 2


def test_parse_zero_shot_thought_unwrapping():
    """Test that JSON containing only a 'thought' key is unwrapped and findings extracted."""
    from dental_agent.evaluation.baselines import parse_zero_shot_response
    from dental_agent.evaluation.metrics import extract_predicted_findings

    sample_thought_json = json.dumps({
        "thought": "Examined all four quadrants. Upper Right (Quadrant 1) shows a possible impacted wisdom tooth (Tooth 8) and subtle caries on Tooth 16. Quadrant 4 shows an impacted tooth 8."
    })
    parsed = parse_zero_shot_response(sample_thought_json)
    assert parsed is not None
    assert "findings" in parsed
    
    clean = extract_predicted_findings(parsed)
    assert len(clean) >= 2
    quad_pos = [(p["quadrant"], p["tooth_position"]) for p in clean]
    assert (1, 8) in quad_pos
    assert (4, 8) in quad_pos


def test_extract_findings_separated_quadrant_tooth():
    """Test diverse separated quadrant and tooth phrasing."""
    from dental_agent.evaluation.baselines import extract_findings_from_reasoning_text

    text = """
    - Quadrant 1, Tooth 8: Diagnosed as Impacted Wisdom Tooth.
    - In the Upper Left, Tooth 6 shows signs of Caries.
    - Position 6 in Quadrant 4 has Deep Caries.
    - Lower Left tooth 7 is intact and normal with no caries.
    """
    findings = extract_findings_from_reasoning_text(text)
    positions = [(f["quadrant"], f["tooth_position"]) for f in findings]
    
    assert (1, 8) in positions
    assert (2, 6) in positions
    assert (4, 6) in positions
    assert (3, 7) not in positions  # Negated: normal with no caries





