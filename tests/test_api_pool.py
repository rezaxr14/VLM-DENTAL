import pytest
from unittest.mock import patch, MagicMock
from dental_agent.training.api_pool import call_llm
import os
from PIL import Image

@patch("dental_agent.training.api_pool._POOL.get_openai_compatible")
def test_call_llm_kwargs_absorption(mock_get_client):
    """Test that stream and label are absorbed via kwargs."""
    mock_client = MagicMock()
    mock_chunk = MagicMock()
    mock_chunk.choices = [MagicMock()]
    mock_chunk.choices[0].delta.content = "Hello!"
    mock_client.chat.completions.create.return_value = [mock_chunk]
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

@patch("dental_agent.training.api_pool.Image.Image.thumbnail")
@patch("dental_agent.training.api_pool._POOL.get_openai_compatible")
def test_call_llm_image_max_dim(mock_get_client, mock_thumbnail):
    """Test that IMAGE_MAX_DIM correctly scales images, defaults, and overrides."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="Image processed"))]
    )
    mock_get_client.return_value = mock_client
    
    dummy_img = Image.new('RGB', (2000, 2000))
    
    # 1. Groq uses default 1280 if not set in os.environ
    call_llm(provider="groq", model="groq", system_prompt="", user_content="test", image=dummy_img)
    mock_thumbnail.assert_called_with((1280, 1280), Image.Resampling.LANCZOS)
    mock_thumbnail.reset_mock()
    
    # 2. Local vLLM uses default 0 (no scaling)
    call_llm(provider="local", model="vllm", system_prompt="", user_content="test", image=dummy_img)
    mock_thumbnail.assert_not_called()
    mock_thumbnail.reset_mock()
    
    # 3. Environment overrides to 0 (no scaling) for Groq
    with patch.dict(os.environ, {"GROQ_IMAGE_MAX_DIM": "0"}):
        call_llm(provider="groq", model="groq", system_prompt="", user_content="test", image=dummy_img)
        mock_thumbnail.assert_not_called()
        mock_thumbnail.reset_mock()
        
    # 4. Environment overrides to custom size
    with patch.dict(os.environ, {"OPENROUTER_IMAGE_MAX_DIM": "500"}):
        call_llm(provider="openrouter", model="openrouter", system_prompt="", user_content="test", image=dummy_img)
        mock_thumbnail.assert_called_with((500, 500), Image.Resampling.LANCZOS)
