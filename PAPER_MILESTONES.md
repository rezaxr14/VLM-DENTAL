# PAPER_MILESTONES

This document logs our experimental findings, key metrics, and interesting data for use in future paper writing. 
Future SFT, GRPO, and Trace Generation findings should also be appended here.

## YOLOv8m Object Detection 5-Fold Cross Validation

**Dataset**: Dental Radiographs

**Folds Summary**:
- **Fold 0**: Precision 0.5513, Recall 0.8673, mAP50 0.5854, mAP50-95 0.3355
- **Fold 1 (Best)**: Precision 0.5457, Recall 0.8880, mAP50 0.5901, mAP50-95 0.3464
- **Fold 2**: Precision 0.5304, Recall 0.8393, mAP50 0.5726, mAP50-95 0.3263
- **Fold 3**: Precision 0.5464, Recall 0.8485, mAP50 0.5734, mAP50-95 0.3408
- **Fold 4**: Precision 0.5486, Recall 0.8344, mAP50 0.5887, mAP50-95 0.3405
- **Mean ± Std**: mAP50 = 0.5820 ± 0.0076, mAP50-95 = 0.3379 ± 0.0067

**Best Fold**: Fold 1
- Selected based on overall performance across 5 folds.
- The model weights and evaluation images for this fold are saved in `data/models/grounding_tool_cv_best/`.
