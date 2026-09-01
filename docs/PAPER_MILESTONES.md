# PAPER_MILESTONES

This document logs our experimental findings, key metrics, and interesting data for use in future paper writing. 
Future SFT, GRPO, and Trace Generation findings should also be appended here.

## 1. YOLOv8m Object Detection 5-Fold Cross Validation (DENTEX Baseline - 634 Images)

**Dataset**: DENTEX `quadrant_enumeration` split

**Folds Summary**:
- **Fold 0**: Precision 0.5513, Recall 0.8673, mAP50 0.5854, mAP50-95 0.3355
- **Fold 1 (Best)**: Precision 0.5457, Recall 0.8880, mAP50 0.5901, mAP50-95 0.3464
- **Fold 2**: Precision 0.5304, Recall 0.8393, mAP50 0.5726, mAP50-95 0.3263
- **Fold 3**: Precision 0.5464, Recall 0.8485, mAP50 0.5734, mAP50-95 0.3408
- **Fold 4**: Precision 0.5486, Recall 0.8344, mAP50 0.5887, mAP50-95 0.3405
- **Mean ± Std**: mAP50 = 0.5820 ± 0.0076, mAP50-95 = 0.3379 ± 0.0067

**Best Fold**: Fold 1
- Saved in `data/models/grounding_tool_cv_best/`.

---

## 2. Multi-Dataset YOLOv8m 5-Fold Cross Validation (DENTEX + Tufts - 1,634 Images)

**Datasets**: Combined DENTEX (634 images) + Tufts Dental Database (1,000 images)

**Folds Summary**:
- **Fold 0**: Precision 0.7309, Recall 0.8242, mAP50 0.8747, mAP50-95 0.5928
- **Fold 1**: Precision 0.6852, Recall 0.8561, mAP50 0.8384, mAP50-95 0.5677
- **Fold 2**: Precision 0.7211, Recall 0.8750, mAP50 0.8676, mAP50-95 0.5783
- **Fold 3**: Precision 0.7262, Recall 0.8666, mAP50 0.8444, mAP50-95 0.5546
- **Fold 4 (Best)**: Precision 0.8780, Recall 0.8190, mAP50 0.9226, mAP50-95 0.6540
- **Mean ± Std**: **mAP50 = 0.8695 ± 0.0298**, **mAP50-95 = 0.5895 ± 0.0346**

**Key Paper Takeaways**:
- Multi-dataset pre-training/co-training with Tufts boosts tooth localization mAP50 by **+28.75% absolute** (0.5820 -> 0.8695) and mAP50-95 by **+25.16% absolute** (0.3379 -> 0.5895).
- Model weights saved in `data/models/dentex_tufts_grounding_tool_cv_best/` and synchronized to Hugging Face Hub `Reza-Nadimi/vlm-dental-models/yolo_cv`.

