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
    assert collator_tools.buckets == [4096, 6144, 8192, 10240]

    # Test snapping logic
    assert collator_tools._snap_to_bucket(1000) == 4096
    assert collator_tools._snap_to_bucket(4096) == 4096
    assert collator_tools._snap_to_bucket(4097) == 6144
    assert collator_tools._snap_to_bucket(7000) == 8192
    assert collator_tools._snap_to_bucket(9000) == 10240
    assert collator_tools._snap_to_bucket(12000) == 10240

    collator_no_tools = BucketedQwenVLCollator(mock_processor, track="no_tools")
    assert collator_no_tools.buckets == [1536, 2048, 2560, 3072]
    assert collator_no_tools._snap_to_bucket(800) == 1536
    assert collator_no_tools._snap_to_bucket(1800) == 2048


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
