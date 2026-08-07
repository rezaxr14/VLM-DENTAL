# Dental-Agent: Tool-Augmented Agentic VLM for Panoramic Dental Radiographs

An Agentic, Tool-Augmented Vision-Language Model framework for panoramic dental radiograph diagnosis on the **DENTEX** benchmark. Built on the **Qwen2.5-VL** backbone with multi-turn tool calling and **GRPO** (Group Relative Policy Optimization) reinforcement learning.

See [dentex-agentic-vlm-proposal.md](dentex-agentic-vlm-proposal.md) for the full research proposal and methodology.

---

## 🌟 Key Features

- **Agentic Multi-Turn Diagnostic Loop**: Autonomous tool usage (zoom/crop, contrast enhancement, FDI numbering, abnormal tooth locating) before delivering a structured diagnosis.
- **Hierarchical DENTEX Grounding**: Support for FDI World Dental Federation notation (quadrants 1–4, tooth positions 1–8) and multi-class pathology classification (Caries, Deep Caries, Periapical Lesions, Impacted Teeth).
- **Two-Stage Fine-Tuning Pipeline**:
  - **Aim 1 / Stage 1 (SFT)**: Multi-turn expert demonstration distillation with cross-family LLM verification (Gemini 2.5 Flash + Claude 3.5 Sonnet).
  - **Aim 2 / Stage 2 (GRPO)**: Direct policy gradient optimization against multi-objective rewards (FDI accuracy + pathology diagnosis + format adherence + tool efficiency).
- **Stage 0 Detector**: Dedicated Faster R-CNN MobileNetV3 tooth localization model to replace oracle grounding.
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
│   ├── model/                    # Qwen2.5-VL loading, QLoRA wrapping, checkpoints, generation
│   ├── rewards/                  # Graded accuracy, format, tool validity, efficiency, LLM judge
│   ├── tools/                    # Deterministic tools (zoom, contrast, FDI) and detectors
│   ├── training/                 # Aim 1 trace generation, Stage 1 SFT, Stage 2 GRPO, Stage 0
│   ├── utils/                    # Environment detection, HF persistence, reproducibility
│   ├── cli.py                    # Unified CLI entrypoint (`dental-agent`)
│   └── config.py                 # Dataclass-based configuration loader
├── scripts/                      # Standalone execution scripts
├── tests/                        # Offline pytest suite
├── dentex-agentic-vlm-proposal.md# Research proposal & theoretical formulation
├── dentex_agentic_vlm_starter.ipynb # Interactive exploration and demo notebook
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
dental-agent generate-traces --n-samples 20 --output data/synthetic_traces.jsonl

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
- `configs/default.yaml`: Base configuration for Kaggle / Colab (16GB VRAM, Qwen2.5-VL-3B, 4-bit NF4, GRPO Group Size = 4).
- `configs/rtx4090.yaml`: High-VRAM workstation configuration (24GB VRAM, Qwen2.5-VL-7B, GRPO Group Size = 8).

Override settings via CLI or by passing a custom config file:
```bash
dental-agent --config configs/rtx4090.yaml train-grpo
```

---

## 🔬 Citation & License

This project is licensed under the MIT License. If you use this implementation or research design, please cite the accompanying DENTEX Agentic VLM proposal.
