"""
Export a standalone, self-contained Python deployment module for the trained Dental Agent (§32).
"""

from __future__ import annotations

import os
from pathlib import Path
import click

STANDALONE_TEMPLATE = '''"""
Standalone Dental Radiograph Diagnostic Agent.
Auto-generated from DENTEX Agentic VLM project.
"""

import json
from PIL import Image
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "{model_id}"
ADAPTER_PATH = "{adapter_path}"

def load_standalone_agent(device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if ADAPTER_PATH and ADAPTER_PATH != "None":
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    model.eval()
    return model, processor

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python standalone_agent.py <image_path>")
        sys.exit(1)
    print("Standalone agent ready.")
'''


def export_standalone_agent_module(
    output_path: str | Path = "standalone_agent.py",
    model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    adapter_path: str | Path | None = "checkpoints/grpo-final",
) -> Path:
    """Generate and write a standalone single-file agent runner module."""
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    content = STANDALONE_TEMPLATE.format(
        model_id=model_id,
        adapter_path=str(adapter_path) if adapter_path else "None",
    )
    with open(out_p, "w") as f:
        f.write(content)
    print(f"Standalone agent exported to: {out_p.resolve()}")
    return out_p


@click.command()
@click.option("--output", "-o", default="standalone_agent.py", help="Output python file path.")
@click.option("--adapter", "-a", default="checkpoints/grpo-final", help="Path to LoRA adapter weights.")
@click.option("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct", help="Base HuggingFace model ID.")
def main(output: str, adapter: str, model_id: str) -> None:
    export_standalone_agent_module(output_path=output, model_id=model_id, adapter_path=adapter)


if __name__ == "__main__":
    main()
