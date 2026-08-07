"""
Unit tests for evaluation metrics, ECE, bootstrap CIs, and reporting tables.
"""

import numpy as np
from dental_agent.evaluation.metrics import (
    compute_diagnostic_metrics,
    compute_ece,
    bootstrap_metric_ci,
)
from dental_agent.evaluation.reporting import generate_summary_table


def test_compute_diagnostic_metrics() -> None:
    trajectories = [
        {
            "final_answer": {"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"},
            "format_ok": True,
            "tool_calls": 2,
        },
        {
            "final_answer": {"quadrant": 2, "tooth_position": 4, "diagnosis": "Periapical Lesion"},
            "format_ok": True,
            "tool_calls": 1,
        },
        {
            "final_answer": None,
            "format_ok": False,
            "tool_calls": 0,
        },
    ]

    ground_truths = [
        {"quadrant": 1, "tooth_position": 6, "diagnosis": "caries"},
        {"quadrant": 2, "tooth_position": 5, "diagnosis": "periapical lesion"},
        {"quadrant": 3, "tooth_position": 1, "diagnosis": "deep caries"},
    ]

    metrics = compute_diagnostic_metrics(trajectories, ground_truths)
    assert metrics["format_adherence"] == 2.0 / 3.0
    assert metrics["quadrant_accuracy"] == 2.0 / 3.0
    assert metrics["tooth_position_accuracy"] == 1.0 / 3.0
    assert metrics["exact_match_accuracy"] == 1.0 / 3.0
    assert metrics["mean_tool_calls"] == 1.0


def test_compute_ece() -> None:
    confidences = [0.9, 0.8, 0.7, 0.6, 0.2]
    accuracies = [1, 1, 1, 0, 0]
    ece = compute_ece(confidences, accuracies, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_bootstrap_metric_ci() -> None:
    data = [1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    point, low, high = bootstrap_metric_ci(data, np.mean, n_bootstrap=200, ci=0.95, seed=42)
    assert low <= point <= high
    assert 0.0 <= low <= 1.0


def test_generate_summary_table() -> None:
    results = {
        "Dental-Agent (GRPO)": {
            "format_adherence": 0.98,
            "fdi_localization_accuracy": 0.85,
            "pathology_macro_f1": 0.82,
            "exact_match_accuracy": 0.79,
            "exact_match_ci_95": [0.74, 0.84],
            "ece": 0.035,
            "mean_tool_calls": 2.1,
        },
        "Zero-Shot Baseline": {
            "format_adherence": 0.90,
            "fdi_localization_accuracy": 0.61,
            "pathology_macro_f1": 0.58,
            "exact_match_accuracy": 0.52,
            "exact_match_ci_95": [0.46, 0.58],
            "ece": 0.120,
            "mean_tool_calls": 0.0,
        },
    }
    table_md = generate_summary_table(results, table_format="github")
    assert "Dental-Agent (GRPO)" in table_md
    assert "Zero-Shot Baseline" in table_md
    assert "Exact Match (%)" in table_md
