import pytest
from pathlib import Path
from unittest.mock import patch
from scripts.sync_traces_hf import (
    DEFAULT_TRACES_REPO,
    CANONICAL_TRACE_FILES,
    download_traces,
    status_traces,
)


def test_canonical_trace_files_list():
    assert "train_cot_traces.jsonl" in CANONICAL_TRACE_FILES
    assert "train_cot_traces_dentex.jsonl" in CANONICAL_TRACE_FILES
    assert "train_cot_traces_tufts.jsonl" in CANONICAL_TRACE_FILES
    assert "train_cot_traces_healthy_tufts.jsonl" in CANONICAL_TRACE_FILES
    assert len(CANONICAL_TRACE_FILES) == 18


@patch("huggingface_hub.list_repo_files")
def test_download_traces_local_cache_skip(mock_list, tmp_path: Path):
    mock_list.return_value = ["train_cot_traces.jsonl", "other.jsonl"]

    # Simulate an existing local file
    test_file = tmp_path / "train_cot_traces.jsonl"
    test_file.write_text('{"image_id": 1}\n', encoding="utf-8")

    # Call download_traces with force=False pointing to tmp_path
    # Since the file already exists, it should be marked 'cached' without network download
    results = download_traces(
        repo_id=DEFAULT_TRACES_REPO,
        target_dir=tmp_path,
        files=["train_cot_traces.jsonl"],
        force=False,
    )

    assert "train_cot_traces.jsonl" in results
    assert results["train_cot_traces.jsonl"] == "cached"
    assert test_file.read_text(encoding="utf-8") == '{"image_id": 1}\n'


@patch("huggingface_hub.list_repo_files")
def test_status_traces_local_only(mock_list, tmp_path: Path, capsys):
    mock_list.return_value = ["train_cot_traces.jsonl"]
    (tmp_path / "train_cot_traces.jsonl").write_text('{"image_id": 1}\n')
    status_traces(repo_id="test/repo", source_dir=tmp_path)
    captured = capsys.readouterr()
    assert "train_cot_traces.jsonl" in captured.out
