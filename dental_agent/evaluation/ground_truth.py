"""Ground-truth findings of one image: the single definition every evaluation path uses.

Every annotation row of an image is a finding (Rule 13: never the first row only, never ``.iloc[0]``), and an image with no
annotation row is healthy and has ground truth ``[]`` (a correct answer for it is an empty one). The result feeds
``match_multi_findings``, ``combine_reward`` and ``compute_diagnostic_metrics`` unchanged.

Worked example, DENTEX image 7 with three annotation rows (raw labels are 0-indexed, ``row_to_fdi`` adds the 1):

    category_id_1  category_id_2  category_id_3          finding
         0              5              1 (Caries)         quadrant 1, tooth 6, Caries
         0              5              3 (Deep Caries)    quadrant 1, tooth 6, Deep Caries
         3              7              0 (Impacted)       quadrant 4, tooth 8, Impacted

gives three findings; first-row code kept one and scored the other two as nothing to find. An image with no row gives ``[]``.

This function used to live in ``scripts/evaluate_models.py`` (which still exposes it under the same name) so that the
matrix runner, the ablation runners, the batch runner and the sweep cannot drift apart.
"""

from __future__ import annotations

from typing import Any

from dental_agent.data.fdi_utils import row_to_fdi
from dental_agent.evaluation.metrics import normalize_dental_diagnosis

DENTEX_DEFAULT_DIAGNOSES = {
    0: "Impacted",
    1: "Caries",
    2: "Periapical Lesion",
    3: "Deep Caries",
}


def extract_ground_truth(image_id: int, annots_df, cats_df, diag_col: str = "category_id_3") -> list[dict[str, Any]]:
    """Extract all ground truth findings for an image using dataset-aware FDI conversion."""
    cat_lookup = (
        dict(zip(cats_df["id"], cats_df["name"]))
        if cats_df is not None and len(cats_df)
        else DENTEX_DEFAULT_DIAGNOSES
    )

    anns = annots_df[annots_df["image_id"] == image_id]
    gt_findings = []
    if not anns.empty:
        for _, ann_row in anns.iterrows():
            q, pos = row_to_fdi(ann_row)
            d_id = ann_row.get(diag_col)
            try:
                d_id_int = int(d_id)
            except (ValueError, TypeError):
                d_id_int = None
            d_raw = cat_lookup.get(d_id, cat_lookup.get(d_id_int, DENTEX_DEFAULT_DIAGNOSES.get(d_id_int, "Caries")))
            finding = {
                "quadrant": q,
                "tooth_position": pos,
                "diagnosis": normalize_dental_diagnosis(d_raw),
                "raw_diagnosis": str(d_raw),
            }
            bbox = ann_row.get("bbox") if hasattr(ann_row, "get") else None
            try:
                if bbox is not None and len(bbox) == 4:
                    finding["bbox"] = [float(v) for v in bbox]   # annotated tooth box [x, y, w, h], used for box-vs-GT IoU
            except (TypeError, ValueError):
                pass
            gt_findings.append(finding)
    return gt_findings
