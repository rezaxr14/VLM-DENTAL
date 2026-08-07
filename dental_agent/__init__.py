"""
dental_agent — An Agentic, Tool-Augmented Vision-Language Model
for Panoramic Dental Radiograph Diagnosis.

Built on Qwen2.5-VL with GRPO reinforcement learning and a suite of
callable diagnostic tools (zoom/crop, contrast enhancement, FDI numbering,
grounding/segmentation). Trained and evaluated on the DENTEX benchmark.

See: dentex-agentic-vlm-proposal.md for the full research proposal.
"""

from dental_agent.config import load_config, load_env, ProjectConfig

# Automatically load .env on package import
load_env()

__version__ = "0.1.0"
__all__ = ["load_config", "load_env", "ProjectConfig", "__version__"]
