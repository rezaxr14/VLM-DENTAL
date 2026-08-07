"""
Checkpoint saving, loading, and inspection for fine-tuned LoRA adapters.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional

from dental_agent.model.backbone import get_model_classes


def save_checkpoint(
    model: Any,
    processor: Any,
    tag: str,
    checkpoint_dir: str | Path,
    extra_metadata: dict[str, Any] | None = None,
) -> str:
    """Save model adapter weights, processor, and metadata under checkpoint_dir / tag."""
    out_dir = os.path.join(str(checkpoint_dir), tag)
    os.makedirs(out_dir, exist_ok=True)

    if hasattr(model, "save_pretrained"):
        model.save_pretrained(out_dir)
    if hasattr(processor, "save_pretrained"):
        processor.save_pretrained(out_dir)

    meta = {
        "tag": tag,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "extra": extra_metadata or {},
    }
    with open(os.path.join(out_dir, "checkpoint_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    return out_dir


def list_checkpoints(checkpoint_dir: str | Path) -> list[dict[str, Any]]:
    """List all saved checkpoints and their metadata in checkpoint_dir."""
    if not os.path.exists(checkpoint_dir):
        return []

    checkpoints = []
    for item in sorted(os.listdir(checkpoint_dir)):
        item_path = os.path.join(str(checkpoint_dir), item)
        if os.path.isdir(item_path):
            meta_path = os.path.join(item_path, "checkpoint_meta.json")
            meta = {}
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
            checkpoints.append({
                "tag": item,
                "path": item_path,
                "timestamp": meta.get("timestamp", "unknown"),
                "extra": meta.get("extra", {}),
            })
    return checkpoints


def load_checkpoint(
    checkpoint_dir: str | Path,
    tag: str,
    device_map: str = "auto",
    quantization_config: Any = None,
) -> tuple[Any, Any]:
    """Reload a saved model checkpoint and processor."""
    ckpt_path = os.path.join(str(checkpoint_dir), tag)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    import torch
    from transformers import AutoProcessor

    ModelClass = get_model_classes()
    model = ModelClass.from_pretrained(
        ckpt_path,
        quantization_config=quantization_config,
        device_map=device_map if torch.cuda.is_available() else None,
    )
    processor = AutoProcessor.from_pretrained(ckpt_path)
    return model, processor


def load_latest_checkpoint(
    checkpoint_dir: str | Path,
    device_map: str = "auto",
    quantization_config: Any = None,
) -> tuple[Any, Any]:
    """Find and reload the most recently saved checkpoint in checkpoint_dir."""
    ckpts = list_checkpoints(checkpoint_dir)
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in directory: {checkpoint_dir}")
    latest_tag = ckpts[-1]["tag"]
    return load_checkpoint(checkpoint_dir, latest_tag, device_map=device_map, quantization_config=quantization_config)

