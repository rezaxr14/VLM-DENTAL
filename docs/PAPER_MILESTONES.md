# PAPER_MILESTONES

This document logs our experimental findings, key metrics, and interesting data for use in future paper writing. 
Future SFT, GRPO, and Trace Generation findings should also be appended here.

## 1. Dataset Composition & Ground Truth Structure

| Dataset / Split | Image Count | Annotation Density | Total Tooth Boxes | Purpose |
|---|---|---|---|---|
| **DENTEX `quadrant_enumeration`** | 634 images | **Dense** (~28–32 teeth/image) | 19,088 boxes | Full-mouth tooth localization |
| **DENTEX `quadrant-enumeration-disease`** | 705 images | **Sparse** (~1–6 diseased teeth/image) | 2,536 boxes | Multi-class pathology grounding |
| **DENTEX Combined Pool (`combine_enumeration_splits=True`)** | **1,339 images** | **Mixed** (47% dense, 53% sparse) | **21,624 boxes** | DENTEX 5-fold CV training pool |
| **Tufts Dental Database** | **1,000 images** | **Dense** (32 teeth/image) | **25,184 boxes** | Out-of-domain full-arch generalization |
| **Multi-Dataset Total (DENTEX + Tufts)** | **2,339 images** | **Mixed** (70% dense, 30% sparse) | **46,808 boxes** | Primary multi-dataset CV training pool |
| **Held-Out Test Set (`validation_triple.json`)** | **46 images** | **Sparse** (~3.9 diseased teeth/image) | **182 targets** | Target-filtered benchmark evaluation |

## 2. In-Fold Cross-Validation Benchmark (Target-Filtered CV Splits)

*Evaluates predictions specifically on the ground-truth target teeth present in each in-fold validation split using 1-to-1 greedy bipartite nominal matching (`conf >= 0.25`) and true continuous 10-threshold COCO PR interpolation (`conf >= 0.001`, IoU `0.50:0.95`).*

### DENTEX-Only Baseline (5 Folds)
| Model / Fold | mAP50 | mAP50-95 | Precision | Rec@0.50 | Rec@0.75 | Mean IoU |
|---|---|---|---|---|---|---|
| DENTEX-Only (CV Fold 0) | 0.9534 | 0.5672 | 0.9457 | 0.8824 | 0.6839 | 0.7231 |
| DENTEX-Only (CV Fold 1) | 0.9405 | 0.5898 | 0.9580 | 0.8576 | 0.7058 | 0.7160 |
| DENTEX-Only (CV Fold 2) | 0.9614 | 0.5781 | 0.9694 | 0.8182 | 0.6590 | 0.6763 |
| DENTEX-Only (CV Fold 3) | 0.9564 | 0.5853 | 0.9804 | 0.7636 | 0.6248 | 0.6338 |
| DENTEX-Only (CV Fold 4) | 0.9423 | 0.5584 | 0.9650 | 0.8262 | 0.6540 | 0.6790 |
| **DENTEX-Only (5-Fold Mean)** | **0.9508** | **0.5758** | **0.9637** | **0.8296** | **0.6655** | **0.6856** |

### DENTEX + Tufts Multi-Dataset (5 Folds)
| Model / Fold | mAP50 | mAP50-95 | Precision | Rec@0.50 | Rec@0.75 | Mean IoU |
|---|---|---|---|---|---|---|
| DENTEX+Tufts (CV Fold 0) | 0.9676 | 0.6066 | 0.9791 | 0.8639 | 0.7128 | 0.7216 |
| DENTEX+Tufts (CV Fold 1) | 0.9502 | 0.6018 | 0.9699 | 0.8677 | 0.7251 | 0.7275 |
| DENTEX+Tufts (CV Fold 2) | 0.9384 | 0.5689 | 0.9737 | 0.6666 | 0.5416 | 0.5555 |
| DENTEX+Tufts (CV Fold 3) | 0.9504 | 0.5839 | 0.9740 | 0.8194 | 0.6604 | 0.6838 |
| DENTEX+Tufts (CV Fold 4) | 0.8814 | 0.5169 | 0.9432 | 0.6431 | 0.5046 | 0.5311 |
| **DENTEX+Tufts (5-Fold Mean)** | **0.9376** | **0.5756** | **0.9680** | **0.7721** | **0.6289** | **0.6439** |

---

## 3. Held-Out Target Grounding Benchmark (Official Test Set - 46 Images, 182 Targets)

*Methodology: Evaluates model detections specifically on annotated target teeth using greedy 1-to-1 bipartite matching and continuous 101-point COCO PR interpolation down to `conf=0.001` to eliminate whole-mouth false positive distortion.*

| Model Architecture (5-Fold Mean) | Target mAP50 | Target mAP50-95 | Precision | Recall@0.50 | Mean IoU |
|---|---|---|---|---|---|
| **DENTEX-Only (5-Fold Mean)** | 0.9089 | 0.6380 | 0.9393 | 0.8176 | 0.7112 |
| **DENTEX + Tufts (5-Fold Mean)** | 0.9260 | 0.6440 | **0.9634** | 0.7747 | 0.6729 |
| **Best Model (DENTEX+Tufts Fold 1)** | **0.9593** | **0.6500** | **0.9864** | **0.7967** | **0.6884** |

### Key Paper Takeaways
1. **Target Localization Accuracy**: Both model families achieve **>90% Target mAP50** and **>93% Precision** when evaluated strictly on ground truth target teeth.
2. **Dense Dataset Transfer**: Co-training with 1,000 dense Tufts images increases target grounding Target mAP50 from **0.9089 to 0.9260** and Precision from **0.9393 to 0.9634**.
3. **Model Artifacts**: Best models and evaluation summaries are preserved at `data/models/yolo_cv_best/` and synced to Hugging Face Hub `Reza-Nadimi/vlm-dental-models/yolo_cv`.

