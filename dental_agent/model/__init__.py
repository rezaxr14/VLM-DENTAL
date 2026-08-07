"""
VLM backbone loading, LoRA configuration, generation utilities, and checkpoint management.
"""

from dental_agent.model.backbone import load_model, estimate_grpo_memory, apply_lora
from dental_agent.model.checkpoints import (
    save_checkpoint,
    load_checkpoint,
    load_latest_checkpoint,
    list_checkpoints,
)
from dental_agent.model.inference import generate_agent_reply, probe_vision_tokens

__all__ = [
    "load_model",
    "estimate_grpo_memory",
    "apply_lora",
    "save_checkpoint",
    "load_checkpoint",
    "load_latest_checkpoint",
    "list_checkpoints",
    "generate_agent_reply",
    "probe_vision_tokens",
]
