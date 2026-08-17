import pytest
import os
import json
import datetime
from unittest.mock import patch, MagicMock
from dental_agent.training.api_pool import call_llm, APIUsageTracker, RPDLimitExhausted

@pytest.fixture
def mock_tracker_file(tmp_path):
    """Provides an isolated tracker file for testing."""
    test_state_file = tmp_path / "test_api_usage_state.json"
    tracker = APIUsageTracker(state_path=str(test_state_file))
    return tracker

@patch.dict(os.environ, {"GROQ_RPD_LIMIT": "2"}, clear=True)
def test_tracker_increments_and_blocks(mock_tracker_file):
    """Test that the tracker properly increments and stops at the limit."""
    
    # First call: should work
    mock_tracker_file.acquire_slot("groq")
    assert mock_tracker_file.state["groq"]["calls_today"] == 1
    
    # Second call: should work and hit limit
    mock_tracker_file.acquire_slot("groq")
    assert mock_tracker_file.state["groq"]["calls_today"] == 2
    
    # Third call: should raise RPDLimitExhausted
    with pytest.raises(RPDLimitExhausted, match="exhausted its 2 RPD limit"):
        mock_tracker_file.acquire_slot("groq")

@patch.dict(os.environ, {"GROQ_RPD_LIMIT": "1"})
@patch("dental_agent.training.api_pool._POOL.get_openai_compatible")
def test_call_llm_hits_rpd_limit_before_api(mock_get_client, tmp_path):
    """Test that call_llm respects the tracker and stops BEFORE calling openai."""
    from dental_agent.training.api_pool import _TRACKER
    _TRACKER.state_path = str(tmp_path / "test_api_usage_state.json")
    _TRACKER.state = {}
    
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello!"))]
    )
    mock_get_client.return_value = mock_client
    
    # First call works
    call_llm(provider="groq", model="groq", system_prompt="sys", user_content="usr")
    assert mock_client.chat.completions.create.call_count == 1
    
    # Second call hits RPD limit, raises exception, DOES NOT call api
    with pytest.raises(RuntimeError, match="Hard stop on API errors per rule"):
        call_llm(provider="groq", model="groq", system_prompt="sys", user_content="usr")
        
    assert mock_client.chat.completions.create.call_count == 1  # Still 1!
