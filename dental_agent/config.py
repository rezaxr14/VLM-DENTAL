"""
Configuration system for dental_agent.

Loads settings from YAML config files, with environment-aware defaults
(Kaggle/Colab/local) and override support. All notebook-scattered CONFIG,
PERSIST_DIR, and env-var assignments are consolidated here.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class DataConfig:
    """Dataset-related settings."""

    dataset_repo: str = "ibrahimhamamci/DENTEX"
    target_size: tuple[int, int] = (1024, 512)
    holdout_fraction: float = 0.20
    parquet_cache: bool = True


@dataclass
class ModelConfig:
    """VLM backbone settings."""

    name: str = "Qwen/Qwen3.5-9B"
    load_in_4bit: bool = True
    bnb_quant_type: str = "nf4"
    bnb_compute_dtype: str = "bfloat16"
    bnb_double_quant: bool = True


@dataclass
class LoraConfig:
    """LoRA / QLoRA fine-tuning settings."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: str = "none"
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ]
    )


@dataclass
class RewardWeights:
    """Weights for the composite GRPO reward (§5.5)."""

    accuracy: float = 1.0
    format: float = 0.2
    tool_validity: float = 0.2
    efficiency: float = 0.1
    judge: float = 0.0  # Optional, requires LLM-judge API


@dataclass
class TrainingConfig:
    """Training pipeline settings."""

    lora: LoraConfig = field(default_factory=LoraConfig)
    reward_weights: RewardWeights = field(default_factory=RewardWeights)

    # SFT (Stage 1)
    sft_epochs: int = 1
    sft_lr: float = 1e-4
    sft_log_every: int = 1

    # GRPO (Stage 2)
    grpo_group_size: int = 4
    grpo_lr: float = 1e-5
    grpo_epochs_per_batch: int = 1
    grpo_clip_eps: float = 0.2
    grpo_kl_beta: float = 0.04
    # NOTE: a grpo_max_tool_calls field used to live here (default 4) but was
    # never actually read anywhere -- every real max_tool_calls default at
    # runtime (grpo.py, agent/loop.py, training/trace_generation.py,
    # rewards/composite.py, run_trace_gen) is consistently 50, and
    # R_efficiency's complexity-scaled reference budget was designed against
    # that real number, not 4. Removed rather than wired in or left stale --
    # see roadmap.md's changelog for this cleanup. If GRPO's tool-call budget
    # ever needs to be config-driven, add it back wired to an actual call
    # site, with 50 as the default to match everywhere else, not 4.

    # Stage 0 (grounding detector)
    detector_epochs: int = 1
    detector_batch_size: int = 2
    detector_lr: float = 5e-4


@dataclass
class APIConfig:
    """LLM API settings for trace generation & verification (§5.2)."""

    generator_provider: str = "local"
    generator_model: str = "Qwen/Qwen3.5-9B"
    verifier_provider: str = "auto_verifier"
    verifier_model: str = "auto_model"
    gemini_rpm_limit: int = 5
    gemini_rpd_limit: int = 20
    gemini_safety_margin: float = 0.95
    trace_candidates_k: int = 1
    max_retries: int = 3
    retry_delay: float = 5.0


@dataclass
class EvaluationConfig:
    """Evaluation settings."""

    judge_sample_n: int = 10
    bootstrap_n: int = 2000
    bootstrap_ci: float = 0.95
    ece_n_bins: int = 10


def load_env(env_path: str | Path | None = None) -> bool:
    """Load environment variables from a .env file into os.environ.

    Searches current working directory and parent directory tree by default.
    """
    try:
        import dotenv
        if env_path is not None:
            return dotenv.load_dotenv(dotenv_path=str(env_path), override=True)
        found = dotenv.find_dotenv(usecwd=True)
        if found:
            return dotenv.load_dotenv(found, override=True)
        return False
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class ProjectConfig:
    """Root configuration for the entire project."""

    seed: int = 42
    environment: str = "auto"  # auto | kaggle | colab | local
    persist_dir: str = ""  # Set automatically if empty

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)

    # HF artifact repo for cross-session sync
    hf_artifact_repo: Optional[str] = None

    def __post_init__(self) -> None:
        load_env()
        if self.environment == "auto":
            self.environment = detect_environment()
        if not self.persist_dir:
            self.persist_dir = os.environ.get("DENTAL_AGENT_PERSIST_DIR") or _default_persist_dir(self.environment)
        if not self.hf_artifact_repo:
            self.hf_artifact_repo = os.environ.get("HF_ARTIFACT_REPO")
        env_model = os.environ.get("MODEL_NAME") or os.environ.get("BACKBONE_MODEL")
        if env_model:
            self.model.name = env_model

        if os.environ.get("GENERATOR_PROVIDER"):
            self.api.generator_provider = os.environ["GENERATOR_PROVIDER"]
        if os.environ.get("GENERATOR_MODEL"):
            self.api.generator_model = os.environ["GENERATOR_MODEL"]
        elif self.api.generator_provider == "gemini" and os.environ.get("GEMINI_GENERATOR_MODEL"):
            self.api.generator_model = os.environ["GEMINI_GENERATOR_MODEL"]

        if os.environ.get("VERIFIER_PROVIDER"):
            self.api.verifier_provider = os.environ["VERIFIER_PROVIDER"]
        if os.environ.get("VERIFIER_MODEL"):
            self.api.verifier_model = os.environ["VERIFIER_MODEL"]

    @property
    def hf_cache_dir(self) -> str:
        return os.path.join(self.persist_dir, "hf_cache")

    @property
    def checkpoint_dir(self) -> str:
        return os.path.join(self.persist_dir, "checkpoints")

    @property
    def data_dir(self) -> str:
        return os.path.join(self.persist_dir, "data")

    def ensure_dirs(self) -> None:
        """Create all required persistent directories."""
        for d in (self.persist_dir, self.hf_cache_dir,
                  self.checkpoint_dir, self.data_dir):
            os.makedirs(d, exist_ok=True)

    def setup_hf_env(self) -> None:
        """Set HuggingFace environment variables for caching.

        Must be called BEFORE the first import of transformers / datasets /
        huggingface_hub, or those libraries will already have picked a
        default (non-persistent) cache location.
        """
        hub_cache = os.path.join(self.hf_cache_dir, "hub")
        os.environ["HF_HOME"] = self.hf_cache_dir
        os.environ["HF_HUB_CACHE"] = hub_cache
        os.environ["TRANSFORMERS_CACHE"] = hub_cache
        os.environ["HF_DATASETS_CACHE"] = os.path.join(
            self.hf_cache_dir, "datasets"
        )


# ---------------------------------------------------------------------------
# Environment detection
# ---------------------------------------------------------------------------

def detect_environment() -> str:
    """Detect whether we're running on Kaggle, Colab, or locally."""
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or os.path.exists("/kaggle"):
        return "kaggle"
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass
    return "local"


def _default_persist_dir(env: str) -> str:
    """Pick a sensible default persistent storage directory per environment."""
    if env == "colab":
        return "/content/drive/MyDrive/dental_agent"
    if env == "kaggle":
        return "/kaggle/working/dental_agent"
    return str(Path.home() / "dental_agent_cache")


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base*, mutating *base*."""
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


def _dict_to_config(d: dict) -> ProjectConfig:
    """Build a ProjectConfig from a flat/nested dict (e.g. from YAML)."""
    cfg = ProjectConfig()
    for key, val in d.items():
        if key == "data" and isinstance(val, dict):
            cfg.data = DataConfig(**val)
        elif key == "model" and isinstance(val, dict):
            cfg.model = ModelConfig(**val)
        elif key == "training" and isinstance(val, dict):
            lora = val.pop("lora", None)
            rw = val.pop("reward_weights", None)
            cfg.training = TrainingConfig(**val)
            if lora:
                cfg.training.lora = LoraConfig(**lora)
            if rw:
                cfg.training.reward_weights = RewardWeights(**rw)
        elif key == "api" and isinstance(val, dict):
            cfg.api = APIConfig(**val)
        elif key == "evaluation" and isinstance(val, dict):
            cfg.evaluation = EvaluationConfig(**val)
        elif hasattr(cfg, key):
            setattr(cfg, key, val)
    return cfg


def load_config(path: str | Path | None = None) -> ProjectConfig:
    """Load a ProjectConfig from a YAML file, falling back to defaults.

    Parameters
    ----------
    path : str or Path, optional
        Path to a YAML config file. If ``None``, returns the default config.
    """
    if path is None:
        cfg = ProjectConfig()
    else:
        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        cfg = _dict_to_config(raw)

    cfg.ensure_dirs()
    return cfg
