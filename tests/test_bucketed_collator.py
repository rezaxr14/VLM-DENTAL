"""
Unit test for BucketedQwenVLCollator and 3D MRoPE Right-Padding Invariant.

Verifies:
1. Sequences snap up to the nearest discrete bucket boundary.
2. Padding is applied strictly on the right (padding_side = "right").
3. Attention mask is 1 for valid tokens and 0 for padded positions.
4. Padding positions have labels = -100.
"""

import pytest
import torch
from unittest.mock import MagicMock

from dental_agent.training.sft import BucketedQwenVLCollator


def test_bucketed_collator_snapping_and_right_padding():
    mock_processor = MagicMock()
    mock_processor.tokenizer.pad_token_id = 0

    collator_tools = BucketedQwenVLCollator(mock_processor, track="with_tools")
    assert collator_tools.buckets == [10240]

    # Test snapping logic up to 10240 headroom
    assert collator_tools._snap_to_bucket(1000) == 10240
    assert collator_tools._snap_to_bucket(8192) == 10240
    assert collator_tools._snap_to_bucket(10240) == 10240
    assert collator_tools._snap_to_bucket(20000) == 10240
    assert collator_tools._snap_to_bucket(40000) == 10240

    collator_no_tools = BucketedQwenVLCollator(mock_processor, track="no_tools")
    assert collator_no_tools.buckets == [1536, 2048, 2560, 3072, 8192]
    assert collator_no_tools._snap_to_bucket(800) == 1536
    assert collator_no_tools._snap_to_bucket(1800) == 2048
    assert collator_no_tools._snap_to_bucket(3500) == 8192


def test_bucketed_collator_overlength_warning():
    mock_processor = MagicMock()
    mock_processor.tokenizer.pad_token_id = 0

    collator = BucketedQwenVLCollator(mock_processor, custom_buckets=[50, 100])
    seq_len = 120  # Exceeds max bucket 100
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)
    batch = [{"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}]

    with pytest.warns(UserWarning, match="exceeds maximum bucket"):
        collated = collator(batch)

    # Should be truncated to max bucket 100
    assert collated["input_ids"].shape == (1, 100)
    assert collated["labels"].shape == (1, 100)


def test_bucketed_collator_batch_padding():
    mock_processor = MagicMock()
    mock_processor.tokenizer.pad_token_id = 9999

    # Use custom smaller buckets for fast testing
    collator = BucketedQwenVLCollator(mock_processor, custom_buckets=[50, 100])

    seq_len = 35
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)

    batch = [{"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}]
    collated = collator(batch)

    # 35 tokens should snap to bucket 50
    assert collated["input_ids"].shape == (1, 50), f"Expected shape (1, 50), got {collated['input_ids'].shape}"
    assert collated["attention_mask"].shape == (1, 50)
    assert collated["labels"].shape == (1, 50)

    # First 35 tokens must be the original sequence
    assert (collated["input_ids"][0, :seq_len] == input_ids[0]).all()
    assert (collated["labels"][0, :seq_len] == labels[0]).all()
    assert (collated["attention_mask"][0, :seq_len] == 1).all()

    # Right padding invariant: tokens 35:50 must be padded on the RIGHT
    assert (collated["input_ids"][0, seq_len:] == 9999).all(), "Padding tokens must be placed at the end (right-padding)"
    assert (collated["labels"][0, seq_len:] == -100).all(), "Padded positions must have labels = -100"
    assert (collated["attention_mask"][0, seq_len:] == 0).all(), "Padded positions must have attention_mask = 0"


def test_bucketed_collator_mm_token_type_ids():
    mock_processor = MagicMock()
    mock_processor.tokenizer.pad_token_id = 0
    mock_processor.image_token_id = 151655

    collator = BucketedQwenVLCollator(mock_processor, custom_buckets=[20])

    seq_len = 10
    input_ids = torch.tensor([[100, 151655, 151655, 101, 102, 103, 104, 105, 106, 107]])
    labels = input_ids.clone()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)
    mm_types = torch.tensor([[0, 1, 1, 0, 0, 0, 0, 0, 0, 0]])

    batch = [{
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
        "mm_token_type_ids": mm_types,
        "pixel_values": torch.randn((2, 16)),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }]

    collated = collator(batch)

    assert "mm_token_type_ids" in collated
    assert collated["mm_token_type_ids"].shape == (1, 20)
    # Original positions preserved
    assert (collated["mm_token_type_ids"][0, :seq_len] == mm_types[0]).all()
    # Padded positions are 0 (text/pad type)
    assert "pixel_values" in collated
    assert "image_grid_thw" in collated


def test_dynamic_padding_collator():
    """Verify that dynamic_padding=True pads to max_batch_len without static bucket snapping."""
    mock_processor = MagicMock()
    mock_processor.tokenizer.pad_token_id = 0

    collator = BucketedQwenVLCollator(mock_processor, track="with_tools", dynamic_padding=True)
    assert collator.dynamic_padding is True

    # Sequence length 350 - if bucketed, it would snap to 8192
    seq_len = 350
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)

    batch = [{"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}]
    collated = collator(batch)

    # Must be exactly 350, NOT 8192!
    assert collated["input_ids"].shape == (1, 350)
    assert collated["labels"].shape == (1, 350)
    assert collated["attention_mask"].shape == (1, 350)


def test_single_static_max_seq_len_padding():
    """Verify that specifying max_seq_len pads all batches directly to that static length."""
    mock_processor = MagicMock()
    mock_processor.tokenizer.pad_token_id = 0

    collator = BucketedQwenVLCollator(mock_processor, max_seq_len=100)
    assert collator.max_seq_len == 100

    seq_len = 42
    input_ids = torch.arange(1, seq_len + 1, dtype=torch.long).unsqueeze(0)
    labels = input_ids.clone()
    attention_mask = torch.ones((1, seq_len), dtype=torch.long)

    batch = [{"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}]
    collated = collator(batch)

    # Must be padded directly to max_seq_len (100)
    assert collated["input_ids"].shape == (1, 100)
    assert collated["labels"].shape == (1, 100)
    assert collated["attention_mask"].shape == (1, 100)
    assert (collated["attention_mask"][0, :seq_len] == 1).all()
    assert (collated["attention_mask"][0, seq_len:] == 0).all()


def test_dataset_token_lengths_manifest_masking(tmp_path):
    """Verify that DentalSFTDataset filters overlength traces using the pre-computed token manifest."""
    import json
    from dental_agent.training.sft import DentalSFTDataset

    # Create dummy traces file with 3 samples
    traces_file = tmp_path / "test_traces.jsonl"
    with open(traces_file, "w", encoding="utf-8") as f:
        f.write(json.dumps({"image_id": "img_short", "messages": []}) + "\n")
        f.write(json.dumps({"image_id": "img_medium", "messages": []}) + "\n")
        f.write(json.dumps({"image_id": "img_overlength", "messages": []}) + "\n")

    # Create dummy token lengths manifest
    manifest_file = tmp_path / "trace_token_lengths.json"
    manifest_data = {
        "lengths_by_image_id": {
            "img_short": 5000,
            "img_medium": 15000,
            "img_overlength": 35000,
        }
    }
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    mock_processor = MagicMock()
    mock_processor.tokenizer.pad_token_id = 0

    # Test with max_seq_len = 32768 (img_overlength should be filtered out)
    ds_32k = DentalSFTDataset(
        data_path=str(traces_file),
        processor=mock_processor,
        max_seq_len=32768,
        token_lengths_manifest=str(manifest_file),
    )
    assert len(ds_32k) == 2
    remaining_ids = [r["image_id"] for r in ds_32k.records]
    assert "img_short" in remaining_ids
    assert "img_medium" in remaining_ids
    assert "img_overlength" not in remaining_ids

    # Test backtrack to max_seq_len = 10000 (only img_short should remain)
    ds_10k = DentalSFTDataset(
        data_path=str(traces_file),
        processor=mock_processor,
        max_seq_len=10000,
        token_lengths_manifest=str(manifest_file),
    )
    assert len(ds_10k) == 1
    assert ds_10k.records[0]["image_id"] == "img_short"


