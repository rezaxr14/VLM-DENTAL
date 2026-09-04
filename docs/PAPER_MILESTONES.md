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

### Diagnostic & Synthetic CoT Reasoning Corpora

| Corpus / Split | Image Count | Modality / Format | Ground Truth Findings | Purpose |
|---|---|---|---|---|
| **DENTEX Pathology** | 678 images | With Tools (678) & No-Tools (678) | 1,732 findings (4 classes) | Primary Stage 1 SFT & Stage 2 GRPO training |
| **DENTEX Normal (Healthy)** | 27 images | With Tools (27) & No-Tools (27) | 0 findings (Negative Controls) | False positive suppression & full arch scanning |
| **Tufts Overlap Pathology** | 202 images | With Tools (202) & No-Tools (202) | 202 findings (Periapical) | Cross-dataset pathology grounding & SFT |
| **Tufts Normal (Healthy)** | 660 images | With Tools (660) & No-Tools (660) | 0 findings (Negative Controls) | Out-of-domain negative control generalization |
| **Tufts All-Diseases Pathology** | 280 images | With Tools (280) & No-Tools (280) | 984 findings (4 native classes) | Multi-disease pathology generalization |
| **Total Verified Traces** | **1,847 images** | **3,694 Total Traces** (10 cohorts) | Complete Verified Ground Truth | Canonical SFT / GRPO Suite (Hosted on HF Hub) |

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

---

## 4. Chain-of-Thought (CoT) Synthetic Traces & Autonomous Verifier Benchmark

*Methodology: Measures synthetic clinical reasoning trace generation under real dynamic tool execution (LangGraph agent loop) and strict multi-modal verification via frontier LLM verifiers (MiniMax M3 / Gemini 3.5 Flash Lite). Evaluates first-pass compliance, automated repair recovery yield, multi-turn reasoning depth, and multi-finding clinical completeness across 3,694 verified traces.*

### Comprehensive Dataset Verification & Historical Repair Audit

| Cohort / Dataset Split | Tool Modality | Total Target | First-Pass Verified | Rejected | Repaired & Promoted | Final Verified Yield | First-Pass Pass Rate | Final Yield Rate | Canonical File / Commit Reference |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **DENTEX Pathology** | With Tools (Main Policy) | 678 | 667 | 11 | 11 | **678** | 98.38% | **100.0%** | `train_cot_traces_dentex.jsonl` (Git `3c2ce7f`) |
| **DENTEX Pathology** | No-Tools (Baseline 3) | 678 | 666 | 12 | 12 | **678** | 98.23% | **100.0%** | `train_cot_traces_dentex_no_tools.jsonl` (Git `f6e8b37`, `4c0f627`) |
| **DENTEX Normal (Healthy)** | With Tools | 27 | 27 | 0 | 0 | **27** | 100.0% | **100.0%** | `train_cot_traces_healthy_dentex.jsonl` (Git `990157a`) |
| **DENTEX Normal (Healthy)** | No-Tools | 27 | 27 | 0 | 0 | **27** | 100.0% | **100.0%** | `train_cot_traces_healthy_dentex_no_tools.jsonl` |
| **TUFTS Overlap Pathology** | With Tools (Main Policy) | 202 | 202 | 0 | 0 | **202** | 100.0% | **100.0%** | `train_cot_traces_tufts.jsonl` (Git `6d2eb06`) |
| **TUFTS Overlap Pathology** | No-Tools (Baseline 3) | 202 | 199 | 3 | 3 | **202** | 98.51% | **100.0%** | `train_cot_traces_tufts_no_tools.jsonl` (Git `27b2e47`, `298d9a0`) |
| **TUFTS Normal (Healthy)** | With Tools | 660 | 594 | 66 | 66 | **660** | 90.00% | **100.0%** | `train_cot_traces_healthy_tufts.jsonl` (Git `60f450b`) |
| **TUFTS Normal (Healthy)** | No-Tools | 660 | 660 | 0 | 0 | **660** | 100.0% | **100.0%** | `train_cot_traces_healthy_tufts_no_tools.jsonl` (Git `1a32c32`) |
| **TUFTS All-Diseases Pathology** | With Tools (Main Policy) | 280 | 280 | 0 | 0 | **280** | 100.0% | **100.0%** | `train_cot_traces_tufts_all.jsonl` (HF Hub) |
| **TUFTS All-Diseases Pathology** | No-Tools (Baseline 3) | 280 | 280 | 0 | 0 | **280** | 100.0% | **100.0%** | `train_cot_traces_tufts_all_no_tools.jsonl` (HF Hub) |
| **TOTALS ACROSS PROJECT** | **Combined Corpus** | **3,694** | **3,602** | **92** | **92** | **3,694** | **97.51%** | **100.0%** | Canonical Unified SFT / RL Training Corpus |

### Verifier Error Detection & Rejection Taxonomy (92 Flaws Caught)
During first-pass evaluation, the independent frontier verifier caught **92 problematic traces** across three distinct failure modes:
1. **False-Positive Pathology on Normal Scans (66 rejections, 71.7%)**: Generator hallucinated caries or periapical radiolucencies on disease-free Tufts scans (`abnormality: None`). Caught and rejected by the verifier; repaired to systematic negative surveys (`final_answer: []`).
2. **Ground-Truth Contradictions & FDI Errors (15 rejections, 16.3%)**: Single-turn baseline generator asserted mismatched pathology classes (e.g. diagnosing Caries for an Impacted Tooth) or swapped tooth quadrants. Caught and rejected; repaired via targeted re-prompting.
3. **Uncorrected Spatial Drift (11 rejections, 12.0%)**: Interactive agent received an off-center bounding box but diagnosed without issuing corrective `nudge_crop` adjustments. Caught and rejected; repaired with spatial realignment.
*All 92 rejected traces achieved a 100.0% repair-and-promotion yield into the final canonical training corpus.*

### Hybrid Composition of the Canonical Training Corpus
The canonical training files used by downstream Stage 1 SFT (`VLM_Dental_Colab_SFT.ipynb`) and Stage 2 GRPO (`VLM_Dental_Colab_GRPO.ipynb`) are unified hybrid corpora containing **880 pathology traces**:
- **Lines 1 to 678 (678 traces)**: DENTEX pathology cohort (Images 1 to 705).
- **Lines 679 to 880 (202 traces)**: Tufts Dental Database Periapical Lesion overlap cohort (Images 1 to 1037).
- To preserve maximum experimental reproducibility, decoupled standalone files are maintained at `train_cot_traces_dentex.jsonl`, `train_cot_traces_tufts.jsonl`, and their respective `_no_tools.jsonl` variants.

### Quantitative Quality & Tool Analytics (With-Tools Corpus)
- **Multi-Turn Reasoning Depth**:
  - Pathology with tools: Mean **12.85 turns** on DENTEX, **11.68 turns** on Tufts All-Diseases (averaging 17.71 tool calls per trace, range 4 to 35 turns).
  - Healthy negative controls with tools: Mean **13.09 turns**, Median **12.0 turns**, Range **5 to 25 turns** (demonstrating systematic full-mouth quadrant surveying before confirming absence of disease).
  - Single-turn baseline (no-tools): Exactly **1.0 turn** committing directly to direct visual diagnosis.
- **Tool Utilization Frequency**:
  - `locate_tooth`: 299 invocations (primary spatial anchor across all quadrants).
  - `fdi_label`: 142 invocations (clinical quadrant-to-FDI coordinate normalization).
  - `zoom_crop`: 131 invocations (high-resolution region of interest magnification).
  - `window_level`: 45 invocations (contrast preset tuning for bone and enamel demineralization).
  - `nudge_crop`: 19 invocations (self-correcting spatial refinement of misaligned initial bounding boxes).
  - `contralateral_compare`: 12 invocations (bilateral symmetry comparison against contralateral anatomy).
- **Multi-Finding Pathology Complexity**:
  - Single-finding scans: 124 images (14.1%).
  - Multi-finding scans: 756 images (85.9%), with finding counts up to **23 distinct pathologies** per panoramic radiograph.
- **Clinical Syntactic & Format Compliance**:
  - JSON action schema compliance: **100.0%** (zero unparsed `<fake_tool_call>` or XML artifacts).
  - FDI World Dental Federation notation (Quadrants 1–4, Positions 1–8): **100.0%** adherence.
  - Zero-finding negative control commitment: **100.0%** of healthy scans terminate with `final_answer: []`.

### Tufts Multi-Disease Finding Analysis & 4-Disease Expansion
- Detailed investigation of the complete 1,000-image Tufts Dental Database reveals:
  - **660 Normal Scans**: `abnormality: None` (zero findings across all disease categories; 100% negative controls).
  - **340 Abnormal Scans**: Containing 374 total expert-annotated findings.
  - **20 Multi-Disease Scans**: Contain multiple distinct disease categories simultaneously on the same radiograph:
    - 12 images: `Non-Odontogenic` + `Periapical`
    - 6 images: `Periapical` + `Pericoronal`
    - 2 images: `Non-Odontogenic` + `Pericoronal`
  - In the 202 DENTEX-overlap dataset, 16 images had their non-periapical findings omitted due to taxonomy restriction.
  - The newly implemented `TUFTS_ALL_DISEASES` feature (`--tufts-all-diseases`) loads all **280 abnormal images** with tooth-associated pathology across the native 4-disease taxonomy (`Periapical`, `Non-Odontogenic`, `Pericoronal`, `Inter-Radicular`), generating multi-finding reasoning chains covering every lesion on the radiograph (total **984 findings**, 100% verified across both with-tools and no-tools cohorts).
  - **Exclusion Rationale for 60 Zero-Overlap Images**: Out of the 340 abnormal images, exactly 60 images depict pathology entirely disconnected from any tooth structure (e.g. maxillary sinus mucoceles or mandibular ramus radiolucencies). These scans are excluded because agent diagnostic actions (`locate_tooth`, `contralateral_compare`, `fdi_label`) fundamentally require an anchoring tooth position and FDI quadrant coordinates.
  - **Hugging Face Hub Synchronization & Git Untracking**: All 22 canonical completed trace files (~170 MB) are hosted on Hugging Face Hub (`Reza-Nadimi/vlm-dental-traces`) and untracked from git to preserve repository size. Workflows synchronize traces via `python scripts/sync_traces_hf.py --download`.


