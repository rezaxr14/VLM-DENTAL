import json
import pytest
from pathlib import Path
from dental_agent.training.git_sync import check_for_duplicate_ids


def test_check_duplicate_ids_ignores_non_jsonl(tmp_path: Path):
    """Verifies that formatted JSON, markdown, LaTeX, and text files are skipped safely."""
    # Multi-line formatted JSON with floats and numbers (causes scalar json.loads)
    json_path = tmp_path / "zero_shot_metrics.json"
    json_data = {
        "model_a": {
            "format_adherence": 1.0,
            "pathology_f1": 0.85,
            "total_samples": 50.0
        }
    }
    json_path.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    assert check_for_duplicate_ids(json_path) == {}

    # Markdown file
    md_path = tmp_path / "summary.md"
    md_path.write_text("# Summary\n\n| Model | Score |\n|---|---|\n| A | 1.0 |\n", encoding="utf-8")
    assert check_for_duplicate_ids(md_path) == {}

    # LaTeX file
    tex_path = tmp_path / "table.tex"
    tex_path.write_text(r"\begin{tabular}{ll} Model & 1.0 \end{tabular}", encoding="utf-8")
    assert check_for_duplicate_ids(tex_path) == {}


def test_check_duplicate_ids_valid_jsonl_no_duplicates(tmp_path: Path):
    """Verifies clean jsonl files report zero duplicates."""
    jsonl_path = tmp_path / "clean_traces.jsonl"
    lines = [
        json.dumps({"dataset": "dentex", "image_id": 1, "text": "sample 1"}),
        json.dumps({"dataset": "dentex", "image_id": 2, "text": "sample 2"}),
        json.dumps({"dataset": "tufts", "image_id": 1, "text": "sample tufts 1"}),
    ]
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")
    assert check_for_duplicate_ids(jsonl_path) == {}


def test_check_duplicate_ids_detects_duplicates(tmp_path: Path):
    """Verifies that duplicate (dataset, image_id) entries are counted accurately."""
    jsonl_path = tmp_path / "duplicate_traces.jsonl"
    lines = [
        json.dumps({"dataset": "dentex", "image_id": 10, "step": 1}),
        json.dumps({"dataset": "dentex", "image_id": 20, "step": 1}),
        json.dumps({"dataset": "dentex", "image_id": 10, "step": 2}),
        json.dumps({"dataset": "dentex", "image_id": 10, "step": 3}),
        json.dumps({"dataset": "tufts", "image_id": 20, "step": 1}),
    ]
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")
    dupes = check_for_duplicate_ids(jsonl_path)
    assert dupes == {("dentex", 10): 3}


def test_check_duplicate_ids_handles_corrupt_and_non_dict_lines(tmp_path: Path):
    """Verifies that corrupt JSON, scalar JSON values, lists, and missing fields do not crash."""
    jsonl_path = tmp_path / "corrupt_traces.jsonl"
    lines = [
        "1.0",                          # scalar float
        "[1, 2, 3]",                    # JSON list
        '"hello world"',                # JSON string
        "true",                         # JSON boolean
        "{invalid json",                # malformed JSON
        json.dumps({"not_image_id": 1}), # missing image_id
        json.dumps({"dataset": "dentex", "image_id": "invalid_int"}), # unparseable int
        json.dumps({"dataset": "dentex", "image_id": 99}),
    ]
    jsonl_path.write_text("\n".join(lines), encoding="utf-8")
    assert check_for_duplicate_ids(jsonl_path) == {}


def test_check_duplicate_ids_nonexistent_file(tmp_path: Path):
    """Verifies non-existent path returns empty dict without throwing."""
    assert check_for_duplicate_ids(tmp_path / "does_not_exist.jsonl") == {}
