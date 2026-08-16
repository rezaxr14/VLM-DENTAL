"""
Training pipelines: API pool, synthetic trace generation (Aim 1),
Stage 1 SFT, Stage 2 GRPO, and Stage 0 Faster R-CNN detector.
"""

from dental_agent.training.api_pool import (
    call_llm,
    APISessionPool,
    image_to_b64,
)
from dental_agent.training.trace_generation import (
    generate_interactive_trajectory,
    verify_trace,
    build_trace_example,
    run_aim1_batch,
)
from dental_agent.training.sft import (
    build_sft_example,
    TraceSFTDataset,
    DentalSFTDataset,
    load_trace_dataset,
    train_sft,
)
from dental_agent.training.grpo import (
    compute_group_advantages,
    compute_token_log_probs,
    build_full_trajectory_labels,
    validate_span_alignment,
    collect_grpo_group,
    grpo_step,
    log_grpo_step,
    plot_grpo_training_curve,
    train_grpo,
)
from dental_agent.training.detector import (
    flip_quadrant,
    DentexDetectionDataset,
    DentalDetectionDataset,
    detection_collate_fn,
    build_stage0_detector,
    train_stage0_detector,
    tool_locate_abnormal_teeth_learned,
    visualize_detector_predictions,
    compute_iou,
    evaluate_stage0_detector,
)

__all__ = [
    "call_llm",
    "APISessionPool",
    "image_to_b64",
    "generate_interactive_trajectory",
    "verify_trace",
    "build_trace_example",
    "run_aim1_batch",
    "build_sft_example",
    "TraceSFTDataset",
    "DentalSFTDataset",
    "load_trace_dataset",
    "train_sft",
    "compute_group_advantages",
    "compute_token_log_probs",
    "build_full_trajectory_labels",
    "validate_span_alignment",
    "collect_grpo_group",
    "grpo_step",
    "log_grpo_step",
    "plot_grpo_training_curve",
    "train_grpo",
    "flip_quadrant",
    "DentexDetectionDataset",
    "DentalDetectionDataset",
    "detection_collate_fn",
    "build_stage0_detector",
    "train_stage0_detector",
    "tool_locate_abnormal_teeth_learned",
    "visualize_detector_predictions",
    "compute_iou",
    "evaluate_stage0_detector",
]
