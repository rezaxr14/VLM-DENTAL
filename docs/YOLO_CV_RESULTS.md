# YOLO Grounding Tool - Cross-Validation & Benchmark Results

## 1. DENTEX-Only Baseline (634 Images)

| Fold | mAP50 | mAP50-95 | Precision | Recall |
|------|-------|----------|-----------|--------|
| 0 | 0.5854 | 0.3355 | 0.5513 | 0.8673 |
| 1 ⭐ (BEST) | 0.5901 | 0.3464 | 0.5457 | 0.8880 |
| 2 | 0.5726 | 0.3263 | 0.5304 | 0.8393 |
| 3 | 0.5734 | 0.3408 | 0.5464 | 0.8485 |
| 4 | 0.5887 | 0.3405 | 0.5486 | 0.8344 |

* **Mean mAP50:** 0.5820 ± 0.0076
* **Mean mAP50-95:** 0.3379 ± 0.0067

---

## 2. Multi-Dataset YOLO: DENTEX + Tufts (1,634 Images)

By expanding the training pool with 1,000 full-mouth tooth annotations from Tufts Dental Database, tooth localization performance increased dramatically:

| Fold | mAP50 | mAP50-95 | Precision | Recall |
|------|-------|----------|-----------|--------|
| 0 | 0.8747 | 0.5928 | 0.7309 | 0.8242 |
| 1 | 0.8384 | 0.5677 | 0.6852 | 0.8561 |
| 2 | 0.8676 | 0.5783 | 0.7211 | 0.8750 |
| 3 | 0.8444 | 0.5546 | 0.7262 | 0.8666 |
| 4 ⭐ (BEST) | 0.9226 | 0.6540 | 0.8780 | 0.8190 |

* **Mean mAP50:** **0.8695 ± 0.0298** (+28.75% absolute gain over DENTEX-only)
* **Mean mAP50-95:** **0.5895 ± 0.0346** (+25.16% absolute gain over DENTEX-only)
* **Average Precision:** **~0.75** (vs 0.54 in DENTEX-only)
* **Average Recall:** **~0.85** (robust tooth localization across quadrants)

---

## 3. Held-Out Official DENTEX Target Grounding Benchmark (`validation_triple.json` - 46 Images, 182 Targets)

Evaluated specifically on the annotated target teeth present in ground truth using greedy 1-to-1 bipartite matching and 101-point continuous COCO PR integration:

### Table 1: DENTEX-Only Baseline (5 Folds)
| Model / Fold | Recall@0.50 | Recall@0.75 | Precision | Mean IoU | Target mAP50 |
|---|---|---|---|---|---|
| DENTEX-Only (Fold 0) | 0.8242 | 0.7582 | 0.9091 | 0.7067 | 0.9773 |
| DENTEX-Only (Fold 1) | 0.7967 | 0.7253 | 0.9355 | 0.6793 | 0.9167 |
| DENTEX-Only (Fold 2) | 0.8022 | 0.7363 | 0.9182 | 0.6825 | 0.9309 |
| DENTEX-Only (Fold 3) | 0.8077 | 0.7418 | 0.9484 | 0.6865 | 0.9432 |
| DENTEX-Only (Fold 4) | 0.8571 | 0.7692 | 0.9873 | 0.7336 | 0.8916 |
| **5-Fold Mean** | **0.8176** | **0.7462** | **0.9397** | **0.6978** | **0.9319** |

### Table 2: DENTEX + Tufts Multi-Dataset (5 Folds)
| Model / Fold | Recall@0.50 | Recall@0.75 | Precision | Mean IoU | Target mAP50 |
|---|---|---|---|---|---|
| DENTEX+Tufts (Fold 0) | 0.8516 | 0.7747 | 0.9810 | 0.7239 | 0.9561 |
| DENTEX+Tufts (Fold 1) | 0.7967 | 0.7088 | 0.9864 | 0.6736 | 0.9617 |
| DENTEX+Tufts (Fold 2) | 0.7308 | 0.6703 | 0.9638 | 0.6307 | 0.9002 |
| DENTEX+Tufts (Fold 3) | 0.8352 | 0.7418 | 0.9682 | 0.7045 | 0.9569 |
| DENTEX+Tufts (Fold 4) | 0.6703 | 0.5934 | 0.9242 | 0.5700 | 0.8732 |
| **5-Fold Mean** | **0.7769** | **0.6978** | **0.9647** | **0.6605** | **0.9296** |

---

## 4. Benchmark Metric Dynamics & Annotation Characteristics

* **Target-Filtered vs Global Full-Universe:** The held-out test split annotates only diseased teeth (~3.9 teeth/image). Standard global evaluation across all 32 classes (`model.val()`) penalizes the ~26 unlabeled healthy teeth as false positives, depressing precision to ~3.5%. Target-Filtered evaluation tests the true tool capability by evaluating precision and recall strictly on the target lesions.
* **Domain Precision vs Domain Recall:** DENTEX+Tufts achieved higher precision (**96.47%** vs 93.97%), while DENTEX-Only had slightly higher recall (**81.76%** vs 77.69%) on this specific DENTEX-only scanner distribution.
* **Full-Mouth Robustness:** DENTEX+Tufts achieves **0.8695 CV mAP50** on full-mouth tooth segmentation across clinical centers, whereas DENTEX-only achieved 0.5820 due to incomplete training labels.

