"""Unit tests for two-tier trace generation retry logic (§6 / LangGraph Trace Gen)."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

import pandas as pd
from scripts.run_trace_gen import parse_args, _run_generate_for_dataset


def test_parse_args_retry_defaults(monkeypatch):
    """Test CLI argument parsing for retry flags."""
    monkeypatch.setattr("sys.argv", ["run_trace_gen.py", "--mode", "generate", "--retry-failed"])
    args = parse_args()
    assert args.retry_failed is True
    assert args.max_retries_per_image is None  # Defaults to 3 dynamically in runner
    assert args.max_second_pass_retries is None


def test_parse_args_custom_retries(monkeypatch):
    """Test custom CLI retry parameters."""
    monkeypatch.setattr(
        "sys.argv",
        [
            "run_trace_gen.py",
            "--mode",
            "generate",
            "--retry-failed",
            "--max-retries-per-image",
            "2",
            "--max-second-pass-retries",
            "4",
        ],
    )
    args = parse_args()
    assert args.retry_failed is True
    assert args.max_retries_per_image == 2
    assert args.max_second_pass_retries == 4


def test_generation_retry_flow(tmp_path, monkeypatch):
    """Verify that a generator failure is retried up to max_retries and recovered."""
    output_file = tmp_path / "test_unverified.jsonl"

    # Mock dataset
    imgs_df = pd.DataFrame([{"id": 101, "local_path": str(tmp_path / "101.jpg")}])
    annots_df = pd.DataFrame([{"image_id": 101, "category_id_1": 0, "category_id_2": 7, "category_id_3": 0}])
    cats_df = pd.DataFrame([{"id": 0, "name": "Impacted"}])

    # Create dummy image
    from PIL import Image
    dummy_img = Image.new("RGB", (100, 100))
    dummy_img.save(str(tmp_path / "101.jpg"))

    # Mock generator attempts: attempt 1 fails, attempt 2 succeeds
    attempts = [0]
    def mock_generate_only(*args, **kwargs):
        attempts[0] += 1
        if attempts[0] == 1:
            return {"image_id": 101, "status": "generation_failed", "failure_reason": "Context limit reached"}
        else:
            return {"image_id": 101, "status": "unverified", "trajectory": {"messages": [{"role": "assistant", "content": "Done"}]}}

    monkeypatch.setattr("scripts.run_trace_gen.generate_only", mock_generate_only)
    monkeypatch.setattr("scripts.run_trace_gen.load_dentex_dataset", lambda **kw: (imgs_df, annots_df, cats_df))
    monkeypatch.setattr("dental_agent.data.dentex.download_dentex_slice", lambda ids, **kw: {i: str(tmp_path / f"{i}.jpg") for i in ids})
    monkeypatch.setenv("GENERATOR_PROVIDER", "nvidia_nim")
    monkeypatch.setattr("scripts.run_trace_gen.verify_local_server_health", lambda **kw: True)
    monkeypatch.delenv("DENTEX_IMAGES_REPO", raising=False)




    # Build mock args
    args = MagicMock()
    args.dataset = "dentex"
    args.split = "train"
    args.total_slices = 1
    args.slice_index = 1
    args.slice_seed = 42
    args.pacing_delay = 0.0
    args.git_sync_every = 0
    args.max_images = None
    args.no_tools = False
    args.status_only = False
    args.retry_failed = True
    args.max_retries_per_image = 3
    args.max_second_pass_retries = 3

    cfg = MagicMock()
    cfg.data_dir = str(tmp_path)

    _run_generate_for_dataset(args, cfg, "dentex", output_file)

    # Must have attempted twice and succeeded on attempt 2
    assert attempts[0] == 2
    assert output_file.exists()

    with open(output_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    # Traces were appended
    assert len(lines) == 1
    assert lines[-1]["status"] == "unverified"
