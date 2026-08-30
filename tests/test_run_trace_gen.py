import pytest
import os
import argparse
from unittest.mock import patch, MagicMock
from scripts.run_trace_gen import parse_args, main
import dental_agent.training.trace_generation as tg

@patch("sys.argv", ["run_trace_gen.py", "--mode", "generate", "--generator-provider", "groq", "--verifier-provider", "local"])
@patch("scripts.run_trace_gen.run_generate")
def test_main_cli_override(mock_run_generate):
    """Test that CLI override forces the provider module-wide."""
    main()
    assert os.environ["GENERATOR_PROVIDER"] == "groq"
    assert tg.GENERATOR_PROVIDER == "groq"
    mock_run_generate.assert_called_once()


@patch("sys.argv", ["run_trace_gen.py", "--mode", "generate"])
@patch.dict(os.environ, {})
@patch("dotenv.load_dotenv")
@patch("scripts.run_trace_gen.interactive_prompt")
@patch("scripts.run_trace_gen.run_generate")
def test_main_interactive_prompt_fallback(mock_run_generate, mock_prompt, mock_load_dotenv):
    """Test that missing env and args trigger the interactive prompt."""
    
    # Mock user choosing openrouter for generator
    mock_prompt.side_effect = ["openrouter"]
    
    # We must explicitly delete the providers from env if they exist to trigger the prompt
    if "GENERATOR_PROVIDER" in os.environ:
        del os.environ["GENERATOR_PROVIDER"]
    if "VERIFIER_PROVIDER" in os.environ:
        del os.environ["VERIFIER_PROVIDER"]
        
    main()
    
    assert os.environ["GENERATOR_PROVIDER"] == "openrouter"
    assert "VERIFIER_PROVIDER" not in os.environ # It shouldn't prompt or set os.environ if mode is generate
    assert tg.GENERATOR_PROVIDER == "openrouter"
    assert tg.VERIFIER_PROVIDER == "local" # defaults to local if not prompted
    
    assert mock_prompt.call_count == 1
    mock_run_generate.assert_called_once()


from pathlib import Path
from scripts.run_trace_gen import resolve_trace_paths


def test_resolve_trace_paths_dentex_canonical():
    unverified, verified = resolve_trace_paths("dentex", no_tools=False)
    assert unverified == Path("data/traces/train_cot_traces_unverified_dentex.jsonl")
    assert verified == Path("data/traces/train_cot_traces.jsonl")


def test_resolve_trace_paths_dentex_no_tools():
    unverified, verified = resolve_trace_paths("dentex", no_tools=True)
    assert unverified == Path("data/traces/train_cot_traces_unverified_dentex_no_tools.jsonl")
    assert verified == Path("data/traces/train_cot_traces_no_tools.jsonl")


def test_resolve_trace_paths_tufts_canonical():
    unverified, verified = resolve_trace_paths("tufts", no_tools=False)
    assert unverified == Path("data/traces/train_cot_traces_unverified_tufts.jsonl")
    assert verified == Path("data/traces/train_cot_traces.jsonl")


def test_resolve_trace_paths_tufts_no_tools():
    unverified, verified = resolve_trace_paths("tufts", no_tools=True)
    assert unverified == Path("data/traces/train_cot_traces_unverified_tufts_no_tools.jsonl")
    assert verified == Path("data/traces/train_cot_traces_no_tools.jsonl")


def test_resolve_trace_paths_explicit_overrides():
    unverified, verified = resolve_trace_paths(
        "dentex",
        no_tools=False,
        explicit_output="custom/path/unverified.jsonl",
        explicit_verified_output="custom/path/verified.jsonl",
    )
    assert unverified == Path("custom/path/unverified.jsonl")
    assert verified == Path("custom/path/verified.jsonl")


def test_resolve_trace_paths_legacy_fallback(tmp_path, monkeypatch):
    # If canonical does not exist but legacy file does, it resolves to legacy
    legacy_file = tmp_path / "train_cot_traces_unverified.jsonl"
    legacy_file.touch()

    # Monkeypatch the cwd to tmp_path
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "traces").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "traces" / "train_cot_traces_unverified.jsonl").touch()

    unverified, _ = resolve_trace_paths("unknown_dataset", no_tools=False)
    assert unverified == Path("data/traces/train_cot_traces_unverified.jsonl")


def test_parse_args_verified_output():
    with patch("sys.argv", ["run_trace_gen.py", "--verified-output", "custom_verified.jsonl"]):
        args = parse_args()
        assert args.verified_output == "custom_verified.jsonl"


