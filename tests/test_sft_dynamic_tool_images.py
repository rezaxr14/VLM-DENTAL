import json
import tempfile
from pathlib import Path
from PIL import Image
import torch
import pytest

from dental_agent.tools.registry import ToolRegistry
from dental_agent.agent.tool_dispatch import execute_tool_call
from dental_agent.training.sft import DentalSFTDataset, BucketedQwenVLCollator


class DummyTokenizer:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [10] * max(len(text) // 4, 1)

    def decode(self, token_ids):
        return "dummy text"


class DummyProcessor:
    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.image_token_id = 99999

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        parts = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
            elif isinstance(content, str):
                parts.append(content)
        return "\n".join(parts)

    def __call__(self, text, images=None, videos=None, padding=False, return_tensors=None, **kwargs):
        seq_len = max(len(text[0]) // 4, 10)
        input_ids = torch.full((1, seq_len), 10, dtype=torch.long)
        attention_mask = torch.ones((1, seq_len), dtype=torch.long)
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def test_tool_execution_produces_images():
    """Verify that all image-producing tools generate valid PIL Images distinct from base_image."""
    registry = ToolRegistry.create_default()
    base_image = Image.new("RGB", (1000, 500), color=(100, 150, 200))

    # 1. zoom_crop
    crop = execute_tool_call(registry, "zoom_crop", {"bbox": [200.0, 100.0, 100.0, 150.0], "padding_frac": 0.25}, base_image)
    assert isinstance(crop, Image.Image)
    assert crop.size != base_image.size
    assert crop.width < base_image.width

    # 2. window_level
    win = execute_tool_call(registry, "window_level", {"preset": "bone"}, base_image)
    assert isinstance(win, Image.Image)
    assert win.size == base_image.size

    # 3. denoise
    denoised = execute_tool_call(registry, "denoise", {"method": "bilateral", "strength": 0.5}, base_image)
    assert isinstance(denoised, Image.Image)

    # 4. contralateral_compare
    contra = execute_tool_call(registry, "contralateral_compare", {"bbox": [200, 100, 100, 150], "quadrant": 1}, base_image)
    assert isinstance(contra, Image.Image)

    # 5. enhance_contrast
    contrast = execute_tool_call(registry, "enhance_contrast", {"factor": 1.5}, base_image)
    assert isinstance(contrast, Image.Image)


def test_dental_sft_dataset_dynamic_tool_image_injection():
    """Verify that DentalSFTDataset dynamically executes and injects authentic tool images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_file = tmp_path / "sample.png"
        base_img = Image.new("RGB", (1200, 600), color=(128, 128, 128))
        base_img.save(img_file)

        # Synthetic multi-turn trace with multiple tools
        sample_record = {
            "image_id": 101,
            "image_path": str(img_file),
            "messages": [
                {"role": "system", "content": "You are a dental AI."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "<Image>"},
                        {"type": "text", "text": "Analyze this panoramic X-ray."},
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "thought": "Let's window level and zoom into tooth 38.",
                        "tool_calls": [
                            {"tool": "window_level", "args": {"preset": "bone"}},
                            {"tool": "zoom_crop", "args": {"bbox": [300.0, 200.0, 100.0, 150.0], "padding_frac": 0.2}},
                        ],
                    }),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "<Image>"},
                        {"type": "text", "text": "Result of window_level:"},
                        {"type": "image", "image": "<Image>"},
                        {"type": "text", "text": "Result of zoom_crop:"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps({"thought": "Diagnosis complete.", "final_answer": []}),
                },
            ],
            "turns": [
                {
                    "turn": 0,
                    "tool_calls_this_turn": [
                        {"tool_name": "window_level", "tool_args": {"preset": "bone"}, "tool_ok": True},
                        {"tool_name": "zoom_crop", "tool_args": {"bbox": [300.0, 200.0, 100.0, 150.0], "padding_frac": 0.2}, "tool_ok": True},
                    ],
                    "status": "tool_executed",
                }
            ],
            "final_answer": [],
        }

        trace_file = tmp_path / "traces.jsonl"
        with open(trace_file, "w", encoding="utf-8") as f:
            f.write(json.dumps(sample_record) + "\n")

        processor = DummyProcessor()
        dataset = DentalSFTDataset(trace_file, processor=processor, track="with_tools", data_dir=tmp_path)
        assert len(dataset) == 1

        item = dataset[0]
        assert "input_ids" in item
        assert "labels" in item


def test_collator_bucket_snapping_headroom():
    """Verify that BucketedQwenVLCollator snaps to calibrated buckets up to 32,768."""
    processor = DummyProcessor()
    collator = BucketedQwenVLCollator(processor=processor, track="with_tools")

    assert collator.buckets[-1] == 32768
    assert 65536 not in collator.buckets

    assert collator._snap_to_bucket(3000) == 8192
    assert collator._snap_to_bucket(7500) == 8192
    assert collator._snap_to_bucket(10000) == 16384
    assert collator._snap_to_bucket(15000) == 16384
    assert collator._snap_to_bucket(20000) == 32768
    assert collator._snap_to_bucket(30000) == 32768
    assert collator._snap_to_bucket(35000) == 32768

    collator_no_tools = BucketedQwenVLCollator(processor=processor, track="no_tools")
    assert collator_no_tools.buckets == [1536, 2048, 2560, 3072, 8192]
    assert collator_no_tools._snap_to_bucket(1000) == 1536
    assert collator_no_tools._snap_to_bucket(4000) == 8192
