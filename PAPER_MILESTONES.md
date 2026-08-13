# PAPER_MILESTONES

This document logs our experimental findings, key metrics, and interesting data for use in future paper writing. 
Future SFT, GRPO, and Trace Generation findings should also be appended here.

## YOLOv8m Object Detection 5-Fold Cross Validation

**Dataset**: Dental Radiographs

**Folds Summary**:
- **Fold 0**: Precision 0.5948, Recall 0.8493, mAP50 0.6453, mAP50-95 0.3801
- **Fold 1**: Precision 0.5745, Recall 0.8942, mAP50 0.6249, mAP50-95 0.3654
- **Fold 2**: Precision 0.5704, Recall 0.8835, mAP50 0.6430, mAP50-95 0.3730
- **Fold 3 (Best)**: Precision 0.5881, Recall 0.9001, mAP50 0.6474, mAP50-95 0.3860
- **Fold 4**: Precision 0.5866, Recall 0.8563, mAP50 0.6421, mAP50-95 0.3754
- **Mean ± Std**: mAP50 = 0.6406 ± 0.0080, mAP50-95 = 0.3760 ± 0.0069

**Best Fold**: Fold 3
- Selected based on overall performance across 5 folds.
- The model weights and evaluation images for this fold are saved in `data/models/grounding_tool_cv_best/`.
