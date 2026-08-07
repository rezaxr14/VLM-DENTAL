"""
Reproducibility and experiment logging helpers.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from dental_agent.utils.environment import get_system_summary


def set_seed(seed: int = 42) -> None:
    """Set global deterministic random seeds across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def log_run_metadata(
    output_dir: str | Path,
    config: Any = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write run metadata (system, timestamps, git info, config) to run_log.json."""
    os.makedirs(output_dir, exist_ok=True)
    log_path = Path(output_dir) / "run_metadata.json"

    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "system": get_system_summary(),
        "config": str(config) if config else None,
        "extra": extra or {},
    }

    with open(log_path, "w") as f:
        json.dump(meta, f, indent=2)

    return log_path


def get_pip_freeze() -> str:
    """Capture currently installed Python package versions via pip freeze."""
    import subprocess
    import sys
    try:
        return subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    except Exception as e:
        return f"Error capturing pip freeze: {e}"

