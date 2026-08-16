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

