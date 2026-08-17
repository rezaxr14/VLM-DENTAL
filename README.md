# Dental-Agent: Tool-Augmented Agentic VLM for Panoramic Dental Radiographs

An Agentic, Tool-Augmented Vision-Language Model framework for panoramic dental radiograph diagnosis on the **DENTEX** benchmark. Built on the **Qwen/Qwen3.5-9B** backbone with multi-turn tool calling and **GRPO** (Group Relative Policy Optimization) reinforcement learning.

See [dentex-agentic-vlm-proposal.md](dentex-agentic-vlm-proposal.md) for the full research proposal and methodology.

---

## 🌟 Key Features

- **Agentic Multi-Turn Diagnostic Loop**: Autonomous tool usage (zoom/crop, contrast enhancement, FDI numbering, abnormal tooth locating) before delivering a structured diagnosis.
- **Hierarchical DENTEX Grounding**: Support for FDI World Dental Federation notation (quadrants 1–4, tooth positions 1–8) and multi-class pathology classification (Caries, Deep Caries, Periapical Lesions, Impacted Teeth).
- **Two-Stage Fine-Tuning Pipeline**:
  - **Aim 1 / Stage 1 (SFT)**: Multi-turn expert demonstration distillation — generated locally with Qwen/Qwen3.5-9B via a real LangGraph tool-execution loop (ground-truth-directed, Kaggle/Colab), cross-family verified (a different model family than the generator, to reduce correlated blind spots).
  - **Aim 2 / Stage 2 (GRPO)**: Direct policy gradient optimization against multi-objective rewards (FDI accuracy + pathology diagnosis + format adherence + tool efficiency).
- **Stage 0 Detector**: Dedicated YOLOv8m tooth localization model, trained with 5-fold cross-validation, to replace oracle grounding — validation mAP50 ≈ 0.647 (R ≈ 0.90, P ≈ 0.588), past the detection-quality bar set for live use.
- **Robust Evaluation & Ablation Harness**: H1 (tool-use vs. direct reasoning) and H2 (GRPO vs. SFT vs. zero-shot GPT-4o) evaluation suites with bootstrap confidence intervals and calibration (ECE) metrics.
- **Unified & Environment-Agnostic**: One-click execution on Local Workstations (RTX 4090), Kaggle, and Google Colab with HuggingFace Hub artifact persistence.

---

## 📁 Repository Structure

```
VLM-DENTAL/
├── configs/                      # YAML configuration files (default, rtx4090, kaggle, colab)
├── dental_agent/                 # Core Python package
│   ├── agent/                    # Orchestration loop, prompts, parsing, trajectory visualization
│   ├── data/                     # DENTEX download, COCO parsing, preprocessing, split handling
│   ├── evaluation/               # Metrics, baselines, ablations, sweeps, failure analysis
│   ├── model/                    # Qwen3.5-9B loading, QLoRA wrapping, checkpoints, generation
│   ├── rewards/                  # Graded accuracy, format, tool validity, efficiency, LLM judge
│   ├── tools/                    # Deterministic tools (zoom, contrast, FDI) and detectors
│   ├── training/                 # Aim 1 trace generation, Stage 1 SFT, Stage 2 GRPO, Stage 0
│   ├── utils/                    # Environment detection, HF persistence, reproducibility
│   ├── cli.py                    # Unified CLI entrypoint (`dental-agent`)
│   └── config.py                 # Dataclass-based configuration loader
├── notebooks/                    # Dedicated Colab/Kaggle execution notebooks (TraceGen, YOLO, SFT, GRPO)
├── scripts/                      # Standalone execution scripts
├── tests/                        # Offline pytest suite
├── dentex-agentic-vlm-proposal.md# Research proposal & theoretical formulation
├── deprecated_dentex_agentic_vlm_starter.ipynb # Legacy exploration notebook (deprecated)
├── pyproject.toml                # Build & packaging specification
└── requirements.txt              # Exact pinned dependencies
```

---

## 🚀 Quickstart

### 1. Installation

```bash
# Clone and enter the repo
cd VLM-DENTAL

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate      # On Windows
# source .venv/bin/activate # On Linux/macOS

# Install package in editable mode
pip install -e ".[all]"
```

### 2. Running Offline Tests

Verify all tools, parsing, reward calculation, and geometry operations without needing a GPU or dataset download:

```bash
pytest tests/ -v
```

### 3. CLI Commands

```bash
# Check configuration & environment
dental-agent info

# Run offline self-tests
dental-agent test

# Run Aim 1 Synthetic Trace Generation
python scripts/run_trace_gen.py --mode generate --max-images 20

# Run Stage 1 SFT Training
dental-agent train-sft --data data/synthetic_traces.jsonl --epochs 3

# Run Stage 2 GRPO Training
dental-agent train-grpo --group-size 4 --epochs-per-batch 2

# Run Full Evaluation Suite
dental-agent evaluate --checkpoint checkpoints/grpo-final --output results/
```

---

## ⚙️ Configuration & Compute Tiers

Configuration is managed via YAML under `configs/`:
- `configs/default.yaml`: Base configuration for Kaggle / Colab (16GB VRAM, **Qwen/Qwen3.5-9B**, 4-bit NF4 via Unsloth/bitsandbytes, GRPO Group Size = 4).
- `configs/rtx4090.yaml`: High-VRAM workstation configuration (24GB VRAM, **same Qwen/Qwen3.5-9B backbone**, GRPO Group Size = 8) — this tier scales group size and throughput, not model size.

> **Note on Configuration Precedence:** For all pipeline scripts (like `run_trace_gen.py`), parameters follow a strict priority order: **CLI Arguments > `.env` variables > Provider Defaults**. This allows you to set safe fallbacks in `.env` (like `MAX_TOKENS`) but seamlessly scale up runs dynamically via the command line (e.g. `--max-tokens 16384`) without changing configuration files.

Override settings via CLI or by passing a custom config file:
```bash
dental-agent --config configs/rtx4090.yaml train-grpo
```

---

## 🔬 Citation & License

This project is licensed under the MIT License. If you use this implementation or research design, please cite the accompanying DENTEX Agentic VLM proposal.
