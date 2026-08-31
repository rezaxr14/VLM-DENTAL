---
trigger: always_on
---

# Clinical Dataset Semantics & FDI Notation Rules

## 1. DENTEX "0-Index" Quirk (CRITICAL)
The DENTEX JSON labels map `category_id_1` to quadrants and `category_id_2` to tooth positions using a **0-indexed system** (Quadrant: 0=Upper Right ... 3=Lower Right. Position: 0 to 7).

**Rule:** The LLM and all Agent Prompts explicitly demand the use of **FDI Two-Digit Notation** (Quadrants 1-4, Positions 1-8).
- **DO NOT** pass 0-indexed quadrants to the Verifier or LLM. It will reject perfectly valid traces.
- **`dentex_row_to_fdi(row)` in `dental_agent/data/dentex.py` is the single source of truth for this conversion.** Always call `dentex_row_to_fdi()`.
- **DO NOT hand-write `+ 1` (or any other index-shift arithmetic) against `category_id_1`/`category_id_2` anywhere in this codebase.**
- This conversion is DENTEX-specific, not universal. Other dataset loaders (Tufts, Tunisia, etc.) hand back already-correct 1-indexed FDI values directly. `prepare_yolo_dataset.py`'s `DATASET_LOADERS` registry gives each dataset its own `quadrant_position_fn`.

## 2. Dataset Annotation Semantics: Honest Stop, Not a Guess (CRITICAL)
This project trains a medical diagnostic pipeline. A wrong label that looks plausible is worse than no label.
- **Rule:** When writing or extending a dataset loader (`dental_agent/data/*.py`), if a label's meaning, numbering convention, or category mapping is not confirmed directly against the real annotation file or documentation, do NOT guess. Raise `NotImplementedError` explaining what to check.
- **Rule:** Keep verifiable format parsing and unverifiable semantic mappings separable.
- **Rule:** A dataset with `has_diagnosis_labels=False` in `dental_agent/data/dataset_catalog.py` must never be wired into anything expecting diagnosis categories (`category_id_3`).

## 3. Multi-Finding Completeness: Never Truncate Ground Truth with `.iloc[0]` (CRITICAL)
Dental panoramic radiographs frequently contain multiple labeled abnormalities per image (1 to 7 findings).
- **Rule:** NEVER take `.iloc[0]` on an annotations DataFrame (`annots_df[annots_df["image_id"] == id]`) or assume single-finding ground truth.
- **Rule:** Every evaluation, reward calculation, and verification pass MUST process the **full list of ground truth findings** using set-level matching (`match_multi_findings` in `dental_agent.evaluation.metrics`).
- **Rule:** Compute full **FDI Localization Precision/Recall/F1** and **Diagnostic Match Precision/Recall/F1** over the complete ground-truth set.
