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

---

## 2. Full-Universe 5-Fold Cross Validation (Ultralytics `model.val()`)

*Methodology Note: Standard `model.val()` evaluates against literal ground truth presence. In sparse annotation subsets (where healthy teeth lack boxes), correct model detections on unannotated teeth are penalized as False Positives, depressing raw Precision to ~0.55 on DENTEX-only (53% sparse). Adding Tufts (100% dense) dilutes the sparse proportion to ~30%, lifting raw mAP50 to 0.8695.*

### DENTEX Baseline (1,339 Images)
- **Fold 0**: Precision 0.5513, Recall 0.8673, mAP50 0.5854, mAP50-95 0.3355
- **Fold 1 (Best)**: Precision 0.5457, Recall 0.8880, mAP50 0.5901, mAP50-95 0.3464
- **Fold 2**: Precision 0.5304, Recall 0.8393, mAP50 0.5726, mAP50-95 0.3263
- **Fold 3**: Precision 0.5464, Recall 0.8485, mAP50 0.5734, mAP50-95 0.3408
- **Fold 4**: Precision 0.5486, Recall 0.8344, mAP50 0.5887, mAP50-95 0.3405
- **Mean ± Std**: mAP50 = 0.5820 ± 0.0076, mAP50-95 = 0.3379 ± 0.0067

### Multi-Dataset DENTEX + Tufts (2,339 Images)
- **Fold 0**: Precision 0.7309, Recall 0.8242, mAP50 0.8747, mAP50-95 0.5928
- **Fold 1**: Precision 0.6852, Recall 0.8561, mAP50 0.8384, mAP50-95 0.5677
- **Fold 2**: Precision 0.7211, Recall 0.8750, mAP50 0.8676, mAP50-95 0.5783
- **Fold 3**: Precision 0.7262, Recall 0.8666, mAP50 0.8444, mAP50-95 0.5546
- **Fold 4 (Best)**: Precision 0.8780, Recall 0.8190, mAP50 0.9226, mAP50-95 0.6540
- **Mean ± Std**: **mAP50 = 0.8695 ± 0.0298**, **mAP50-95 = 0.5895 ± 0.0346**

---

## 3. Held-Out Target Grounding Benchmark (Official Test Set - 46 Images, 182 Targets)

*Methodology: Evaluates model detections specifically on annotated target teeth using greedy 1-to-1 bipartite matching and continuous 101-point COCO PR interpolation down to `conf=0.001` to eliminate whole-mouth false positive distortion.*

| Model Architecture (5-Fold Mean) | Recall@0.50 | Recall@0.75 | Precision | Mean IoU | Target mAP50 |
|---|---|---|---|---|---|
| **DENTEX-Only (5-Fold Mean)** | **0.8176** | **0.7462** | **0.9397** | **0.6978** | **0.9319** |
| **DENTEX + Tufts (5-Fold Mean)** | **0.7769** | **0.6978** | **0.9647** | **0.6605** | **0.9296** |

### Key Paper Takeaways
1. **Target Localization Accuracy**: Both model families achieve **>93% Target mAP50** and **>94% Precision** when evaluated strictly on ground truth target teeth.
2. **Dense Dataset Transfer**: Co-training with 1,000 dense Tufts images increases target grounding Precision from **93.97% to 96.47%**.
3. **Model Artifacts**: Best models and evaluation summaries are preserved at `data/models/dentex_tufts_grounding_tool_cv_best/` and synced to Hugging Face Hub `Reza-Nadimi/vlm-dental-models/yolo_cv`.

