import pytest
import os
import pandas as pd
from unittest.mock import patch
from dental_agent.training.trace_generation import _resolve_generator, _resolve_verifier

@patch.dict(os.environ, {"GENERATOR_PROVIDER": "openrouter", "VERIFIER_PROVIDER": "gemini"}, clear=True)
def test_resolve_providers():
    """Test that it doesn't use the old 'auto_generator' logic."""
    import dental_agent.training.trace_generation as tg
    
    # Reload the globals that were loaded at import time
    tg.GENERATOR_PROVIDER = os.environ.get("GENERATOR_PROVIDER")
    tg.VERIFIER_PROVIDER = os.environ.get("VERIFIER_PROVIDER")
    
    assert tg._resolve_generator()[0] == "openrouter"
    assert tg._resolve_verifier()[0] == "gemini"


