# Baseline #1: Zero-Shot VLM Evaluation Protocol & Architecture Reference

This document serves as the master technical specification for **Baseline #1: Zero-Shot Vision-Language Model Evaluation** in **VLM-DENTAL** (§6, Baseline #1). It outlines the benchmark protocol, model rosters, clinical prompting formulations, evaluation metrics, complete multi-finding bipartite matching (Rule 13), horizontal parallel slicing, and execution workflows across cloud and local backends.

---

## 1. Overview & Research Protocol Role

In the **VLM-DENTAL** clinical study, **Baseline #1 (Zero-Shot VLM)** establishes the empirical performance floor of general-purpose, pre-trained Vision-Language Models on panoramic dental radiographs (OPGs) prior to:
1. **Domain Adaptation** (Stage 1 SFT on specialist clinical reasoning traces).
2. **Reinforcement Learning** (Stage 2 GRPO for tool-augmented and direct reasoning).
3. **Radiological Workstation Tools** (Dynamic zoom, window leveling, contrast enhancement, tooth localization).

Panoramic radiographs present extreme spatial density, overlapping anatomical structures, bilateral symmetries, and tiny pathological features (such as subtle interproximal enamel demineralization or early periapical radiolucencies). Standard VLMs evaluated in this baseline are presented with the global panoramic radiograph and asked to detect, localize, and diagnose all dental pathologies in a single forward pass without tool assistance.

---

## 2. Evaluated Model Rosters & Provider Architecture

The zero-shot evaluation engine (`scripts/run_zero_shot.py`) routes through a unified multimodal interface (`api_pool.py`), supporting both frontier commercial APIs and open-weight backbones:

| Provider | Model Identifier | Model Size / Type | Deployment Mode |
| :--- | :--- | :---: | :--- |
| **Google Gemini** | `gemini-3.5-flash-lite` | Frontier Lightweight Multimodal | Google AI Studio / Gemini API |
| **Google Gemini** | `gemini-3.5-flash` | Frontier Flagship Multimodal | Google AI Studio / Gemini API |
| **Google Gemini** | `gemini-3.1-flash-lite` | Preceding Generation Lightweight | Google AI Studio / Gemini API |
| **NVIDIA NIM** | `meta/llama-3.2-90b-vision-instruct` | 90B Dense Vision-Language | Hosted NVIDIA NIM API |
| **NVIDIA NIM** | `meta/llama-3.2-11b-vision-instruct` | 11B Dense Vision-Language | Hosted NVIDIA NIM API |
| **NVIDIA NIM** | `moonshotai/kimi-k3` | Frontier Multimodal Reasoning | Hosted NVIDIA NIM API |
| **OpenRouter** | `minimax/minimax-m3-free` | General Multimodal VLM | Hosted OpenRouter API |
| **Transformers / vLLM** | `Qwen/Qwen3.5-9B` | 9B Student Backbone (Base Model) | Local GPU / TPU v5e-8 (Colab/Kaggle) |
| **Majority Baseline** | `Majority-Class Floor (§20)` | Statistical Empirical Floor | Static Cohort Frequency Mode |

---

## 3. Clinical Task Framing & Zero-Shot Prompt Formulation

### 3.1 FDI Two-Digit Notation Requirement (Rule 1)
All models are explicitly instructed to report tooth locations using **FDI Two-Digit Dental Notation**:
- **Quadrant** $\in \{1, 2, 3, 4\}$:
  - $1$: Upper Right (Patient Right / Viewer Left)
  - $2$: Upper Left (Patient Left / Viewer Right)
  - $3$: Lower Left (Patient Left / Viewer Right)
  - $4$: Lower Right (Patient Right / Viewer Left)
- **Tooth Position** $\in \{1, 2, \dots, 8\}$:
  - $1$: Central Incisor, $2$: Lateral Incisor, $3$: Canine, $4$: First Premolar,
  - $5$: Second Premolar, $6$: First Molar, $7$: Second Molar, $8$: Third Molar (Wisdom Tooth).

> [!CAUTION]
> DENTEX internal labels use 0-indexed quadrants (0–3) and positions (0–7). Zero-shot prompts strictly forbid 0-indexing. All ground truth is canonicalized through `dentex_row_to_fdi()` / `row_to_fdi()`.

### 3.2 Standard Zero-Shot Prompt Template (`ZERO_SHOT_PROMPT`)
```text
You are an expert dental radiologist evaluating a panoramic dental radiograph (OPG).

Carefully examine all four quadrants for dental and periapical pathologies.
For each finding, provide:
1. Quadrant (FDI notation: 1=Upper Right, 2=Upper Left, 3=Lower Left, 4=Lower Right)
2. Tooth position within quadrant (1=Central Incisor to 8=Third Molar)
3. Diagnosis from: 'Caries', 'Deep Caries', 'Periapical Lesion', 'Impacted Tooth'
4. Confidence score (0.0 to 1.0)

A single panoramic radiograph typically contains multiple distinct findings across different quadrants.
Identify ALL abnormal teeth visible in the radiograph.

Return your analysis strictly in the following JSON format:
{
  "thought": "<Brief clinical reasoning and anatomical scan>",
  "findings": [
    {
      "quadrant": <1-4>,
      "tooth_position": <1-8>,
      "diagnosis": "<Caries | Deep Caries | Periapical Lesion | Impacted Tooth>",
      "confidence": <0.0-1.0>
    }
  ]
}

If the radiograph shows no pathological findings (clinically normal), return:
{
  "thought": "<Brief explanation that no pathology is detected>",
  "findings": []
}
```

---

## 4. Evaluation Datasets & Held-Out Splits

1. **DENTEX Test Split**:
   - $50$ held-out panoramic radiographs strictly excluded from SFT trace generation and RL rollouts.
   - Contains single-finding and complex multi-finding cases (1 to 7 findings per radiograph).
2. **Tufts Held-Out Benchmark**:
   - Held-out panoramic radiographs evaluated for cross-dataset generalization.
3. **Tufts Healthy Negative Controls**:
   - Clinically verified normal panoramic radiographs used to evaluate False Positive Rate (FPR) and specificity.

---

## 5. Evaluation Metrics & Complete Multi-Finding Bipartite Matching (Rule 13)

Dental radiographs frequently contain multiple labeled abnormalities ($1$ to $7$ findings per image). In accordance with **Rule 13**, zero-shot evaluation **never** truncates ground truth via `.iloc[0]`.

### 5.1 Maximum Bipartite Matching (`match_multi_findings`)
Every evaluation record compares the set of model predictions $P = \{p_1, \dots, p_M\}$ against the full set of ground-truth findings $G = \{g_1, \dots, g_N\}$ using optimal Hungarian bipartite matching:
- **Cost Matrix**: Computed based on exact FDI match, spatial tooth proximity across the arch, and diagnostic category similarity.
- **Precision, Recall, and F1**:
  $$\text{FDI Precision} = \frac{|\text{Matched FDI Pairs}|}{M}, \quad \text{FDI Recall} = \frac{|\text{Matched FDI Pairs}|}{N}$$
  $$\text{FDI F1} = \frac{2 \cdot \text{FDI Precision} \cdot \text{FDI Recall}}{\text{FDI Precision} + \text{FDI Recall} + \epsilon}$$
  $$\text{Exact Precision} = \frac{|\text{Matched (FDI + Diag) Pairs}|}{M}, \quad \text{Exact Recall} = \frac{|\text{Matched (FDI + Diag) Pairs}|}{N}$$
  $$\text{Exact F1} = \frac{2 \cdot \text{Exact Precision} \cdot \text{Exact Recall}}{\text{Exact Precision} + \text{Exact Recall} + \epsilon}$$

### 5.2 Continuous Clinical Metrics
- **Spatial Proximity**: Continuous penalty based on geodesic dental distance along the dental arch:
  $$\Delta_{\text{spatial}} = 1.0 - \frac{|\text{FDI}_{\text{pred}} - \text{FDI}_{\text{gt}}|}{8.0}$$
- **Diagnostic Similarity**: Weighted similarity based on clinical taxonomy (e.g., `Caries` $\leftrightarrow$ `Deep Caries` has similarity $0.7$, while `Caries` $\leftrightarrow$ `Impacted` has similarity $0.0$).
- **Continuous Closeness Score**: Harmonic combination of spatial proximity and diagnostic similarity.
- **Expected Calibration Error (ECE)**: Measures reliability of predicted confidence scores partitioned into $B=10$ confidence bins:
  $$\text{ECE} = \sum_{b=1}^{B} \frac{|B_b|}{M} |\text{acc}(B_b) - \text{conf}(B_b)|$$
- **Bootstrap 95% Confidence Intervals**: Empirical 1,000-sample bootstrap for exact match accuracy.

---

## 6. Execution Modes & Distributed Parallel Slicing

### 6.1 Horizontal Parallel Slicing (`--total-slices`, `--slice-index`)
To evaluate large benchmarks quickly across multiple Google Colab / Kaggle instances without API rate exhaustion:
- The dataset is partitioned deterministically using `--slice-seed 42`:
  ```bash
  # Worker 1 (Slice 1 of 4)
  python scripts/run_zero_shot.py --provider gemini --model gemini-3.5-flash-lite --total-slices 4 --slice-index 1 --git-sync-every 5
  
  # Worker 2 (Slice 2 of 4)
  python scripts/run_zero_shot.py --provider gemini --model gemini-3.5-flash-lite --total-slices 4 --slice-index 2 --git-sync-every 5
  ```
- Each worker evaluates its slice independently, appending records to its local JSONL.
- The Git synchronization engine (`git_sync.py`) pulls remote updates, union-merges completed image IDs, and commits progress without collision.

### 6.2 Rate Pacing & Fail-Fast 429 Protocol (Rule 3)
- Evaluator enforces `--pacing-delay 1.5` (seconds) between successive calls.
- In accordance with **Rule 3**, on any 429 rate limit or quota exhaustion, the engine terminates immediately (or advances to the next provider) unless `--ignore-429` is explicitly set.

---

## 7. CLI Reference & Execution Commands

### Evaluating Hosted Commercial APIs
```bash
# Google Gemini 3.5 Flash Lite
python scripts/run_zero_shot.py --provider gemini --model gemini-3.5-flash-lite --dataset dentex --split test

# NVIDIA NIM LLaMA 3.2 90B Vision
python scripts/run_zero_shot.py --provider nvidia_nim --model meta/llama-3.2-90b-vision-instruct --dataset dentex --split test

# NVIDIA NIM Moonshot Kimi-k3
python scripts/run_zero_shot.py --provider nvidia_nim --model moonshotai/kimi-k3 --dataset dentex --split test

# OpenRouter MiniMax M3
python scripts/run_zero_shot.py --provider openrouter --model minimax/minimax-m3-free --dataset dentex --split test
```

### Evaluating Open-Weight Student Backbone (Qwen 3.5 9B Base)
```bash
# Direct PyTorch Transformers with SDPA Attention
python scripts/run_zero_shot.py --provider transformers --model Qwen/Qwen3.5-9B --dataset dentex --split test --repetition-penalty 1.10
```

### Resuming Incomplete Runs
```bash
# Resumes automatically, skipping already completed image IDs
python scripts/run_zero_shot.py --provider gemini --model gemini-3.5-flash-lite --retry-empty
```
