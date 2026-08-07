"""
JSON serialization helpers for NumPy and PyTorch objects.
"""

from __future__ import annotations

from typing import Any
import numpy as np


def to_jsonable(obj: Any) -> Any:
    """Recursively convert NumPy scalars, arrays, and sets into standard JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(x) for x in obj]
    if isinstance(obj, (set, frozenset)):
        return [to_jsonable(x) for x in sorted(obj, key=str)]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)

    # Check for PyTorch Tensor if torch is imported
    if hasattr(obj, "detach") and hasattr(obj, "cpu") and hasattr(obj, "numpy"):
        return to_jsonable(obj.detach().cpu().numpy())

    return obj
