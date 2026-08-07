"""
Standalone Dental Radiograph Diagnostic Agent.
Auto-generated from DENTEX Agentic VLM project.
"""

import json
from PIL import Image
import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
ADAPTER_PATH = "checkpoints/grpo-final"

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
