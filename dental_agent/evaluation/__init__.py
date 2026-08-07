"""
Evaluation harness: metrics (accuracy, F1, calibration, bootstrap CIs),
baselines, ablations (H1, H2), sweeps, failure analysis, and reporting.
"""

from dental_agent.evaluation.metrics import (
    expected_calibration_error,
    compute_evaluation_metrics,
    bootstrap_paired_diff_ci,
    compute_diagnostic_metrics,
    compute_ece,
    bootstrap_metric_ci,
)
from dental_agent.evaluation.baselines import (
    majority_class_baseline_metrics,
    run_zero_shot_baseline,
    run_zeroshot_baseline,
)
from dental_agent.evaluation.diagnosis_baseline import (
    DentexDiagnosisDetectionDataset,
    train_diagnosis_baseline_detector,
    evaluate_diagnosis_baseline_detector,
)
from dental_agent.evaluation.ablations import (
    run_h1_ablation,
    compare_checkpoints,
    run_h2_evaluation,
)
from dental_agent.evaluation.sweep import (
    sweep_reward_weights,
    DEFAULT_WEIGHT_GRID,
    run_hyperparameter_sweep,
)
from dental_agent.evaluation.failure_analysis import (
    categorize_failure,
    failure_mode_breakdown,
    categorize_failures,
    log_failure_cases,
)
from dental_agent.evaluation.reporting import (
    metrics_to_dataframe,
    save_results_report,
    generate_summary_table,
    generate_markdown_report,
)
from dental_agent.evaluation.batch_runner import (
    run_agent_batch,
    run_full_evaluation_suite,
    evaluate_dataset,
)

__all__ = [
    "expected_calibration_error",
    "compute_evaluation_metrics",
    "bootstrap_paired_diff_ci",
    "compute_diagnostic_metrics",
    "compute_ece",
    "bootstrap_metric_ci",
    "majority_class_baseline_metrics",
    "run_zero_shot_baseline",
    "run_zeroshot_baseline",
    "DentexDiagnosisDetectionDataset",
    "train_diagnosis_baseline_detector",
    "evaluate_diagnosis_baseline_detector",
    "run_h1_ablation",
    "compare_checkpoints",
    "run_h2_evaluation",
    "sweep_reward_weights",
    "DEFAULT_WEIGHT_GRID",
    "run_hyperparameter_sweep",
    "categorize_failure",
    "failure_mode_breakdown",
    "categorize_failures",
    "log_failure_cases",
    "metrics_to_dataframe",
    "save_results_report",
    "generate_summary_table",
    "generate_markdown_report",
    "run_agent_batch",
    "run_full_evaluation_suite",
    "evaluate_dataset",
]
