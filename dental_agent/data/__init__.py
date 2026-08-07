"""
Data loading, preprocessing, split management, and quality validation for DENTEX.
"""

from dental_agent.data.dentex import (
    load_dentex_dataset,
    load_coco_json,
    score_candidate,
    download_dentex,
    discover_annotation_files,
    build_dataframes,
    resolve_image_paths,
    validate_images,
    check_bbox_bounds,
    list_files,
    hierarchy_summary,
    pick_best_annotation_file,
)
from dental_agent.data.preprocessing import preprocess_image, remap_bbox
from dental_agent.data.splits import get_holdout_ids, get_training_pool, _score_named_split
from dental_agent.data.statistics import (
    plot_image_dimensions,
    plot_diagnosis_distribution,
    plot_bbox_areas,
    plot_annotations_per_image,
    plot_quadrant_diagnosis_matrix,
    plot_cooccurrence_matrix,
    get_diagnosis_column,
)

__all__ = [
    "load_dentex_dataset",
    "load_coco_json",
    "score_candidate",
    "download_dentex",
    "discover_annotation_files",
    "build_dataframes",
    "resolve_image_paths",
    "validate_images",
    "check_bbox_bounds",
    "list_files",
    "hierarchy_summary",
    "pick_best_annotation_file",
    "preprocess_image",
    "remap_bbox",
    "get_holdout_ids",
    "get_training_pool",
    "_score_named_split",
    "plot_image_dimensions",
    "plot_diagnosis_distribution",
    "plot_bbox_areas",
    "plot_annotations_per_image",
    "plot_quadrant_diagnosis_matrix",
    "plot_cooccurrence_matrix",
    "get_diagnosis_column",
]
