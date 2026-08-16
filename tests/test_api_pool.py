import pytest
from unittest.mock import patch, MagicMock
from dental_agent.training.api_pool import call_llm
import os
from PIL import Image

@patch("dental_agent.training.api_pool._POOL.get_openai_compatible")
def test_call_llm_kwargs_absorption(mock_get_client):
    """Test that stream and label are absorbed via kwargs."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Hello!"))]
    )
    mock_get_client.return_value = mock_client
    
    # Should not crash on stream=True, label="verify_trace"
    res = call_llm(
        provider="groq",
        model="groq/llama-3.1",
        system_prompt="sys",
        user_content="user",
        stream=True,
        label="verify_trace"
    )
    assert res == "Hello!"
    mock_client.chat.completions.create.assert_called_once()

@patch("dental_agent.training.api_pool._POOL.get_openai_compatible")
def test_call_llm_no_retries(mock_get_client):
    """Test that a 429 Error raises RuntimeError without looping."""
    mock_client = MagicMock()
    # Mocking a rate limit error
    mock_client.chat.completions.create.side_effect = Exception("429 Rate Limit Exceeded")
    mock_get_client.return_value = mock_client
    
    with pytest.raises(RuntimeError, match="Hard stop on API errors per rule"):
        call_llm(
            provider="groq",
            model="groq/llama-3.1",
            system_prompt="sys",
            user_content="user"
        )
    # Should only be called once, proving it does not retry.
    assert mock_client.chat.completions.create.call_count == 1

@patch("google.genai.Client")
@patch.dict(os.environ, {"GEMINI_API_KEY": "fake_key"})
def test_call_llm_gemini_direct(mock_genai_client):
    """Test gemini is called directly."""
    mock_client_instance = MagicMock()
    mock_client_instance.models.generate_content.return_value = MagicMock(text="Gemini text")
    mock_genai_client.return_value = mock_client_instance
    
    res = call_llm(
        provider="gemini",
        model="gemini-1.5-flash",
        system_prompt="sys",
        user_content="user"
    )
    assert res == "Gemini text"
    mock_client_instance.models.generate_content.assert_called_once()
