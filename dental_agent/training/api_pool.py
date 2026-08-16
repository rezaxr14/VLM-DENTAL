"""
API Key Management and API Inference logic.

No more rotating pools. A single provider is passed and executed.
If it fails, it fails immediately without retries per Rule 9.
"""

from __future__ import annotations

import base64
import io
import os
import urllib.request
import urllib.error
from typing import Any, Optional
from PIL import Image

from dental_agent.config import load_env
import json
import datetime

load_env()

class RPDLimitExhausted(Exception):
    """Raised when the requested provider has exhausted its daily API call limit."""
    pass

class APIUsageTracker:
    """Tracks daily API calls per provider and enforces limits."""
    
    def __init__(self, state_path: Optional[str] = None) -> None:
        self.state_path = state_path or os.path.join(
            os.environ.get("DENTAL_AGENT_DATA_DIR", "data"),
            "api_usage_state.json",
        )
        self.state: dict[str, Any] = self._load()
        self._reset_if_new_day()

    def _load(self) -> dict[str, Any]:
        if os.path.exists(self.state_path):
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except Exception:
                return {}
        return {}

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.state_path) or ".", exist_ok=True)
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def _reset_if_new_day(self) -> None:
        today = datetime.date.today().isoformat()
        changed = False
        for p in list(self.state.keys()):
            if self.state[p].get("date") != today:
                self.state[p] = {
                    "date": today,
                    "calls_today": 0,
                }
                changed = True
        if changed:
            self._save()

    def acquire_slot(self, provider: str) -> None:
        """Check if provider is under limit, increment usage, and save."""
        if provider in ("local", "auto_verifier", "auto_generator"):
            return # No limits for local

        self._reset_if_new_day()
        
        prefix = provider.upper().replace('_NIM', '')
        limit_str = os.environ.get(f"{prefix}_RPD_LIMIT")
        
        if not limit_str:
            return # No limit configured

        try:
            rpd_limit = int(limit_str)
        except ValueError:
            return

        today = datetime.date.today().isoformat()
        if provider not in self.state:
            self.state[provider] = {"date": today, "calls_today": 0}

        if self.state[provider]["calls_today"] >= rpd_limit:
            raise RPDLimitExhausted(f"Provider '{provider}' has exhausted its {rpd_limit} RPD limit.")
            
        self.state[provider]["calls_today"] += 1
        self._save()

    def get_stats(self, provider: str) -> str:
        """Return a string summary of the provider's usage."""
        if provider == "local":
            return "LOCAL vLLM: No rate limits"
            
        prefix = provider.upper().replace('_NIM', '')
        limit_str = os.environ.get(f"{prefix}_RPD_LIMIT", "Unknown")
        
        self._reset_if_new_day()
        calls = self.state.get(provider, {}).get("calls_today", 0)
        return f"{provider.upper()}: {calls}/{limit_str} RPD used today"

_TRACKER = APIUsageTracker()


# ---------------------------------------------------------------------------
# API Session Pool (cached provider clients)
# ---------------------------------------------------------------------------

class APISessionPool:
    """Manages cached client instances across API providers."""

    def __init__(self) -> None:
        self._compat_clients: dict[str, Any] = {}

    def get_openai_compatible(self, provider: str) -> Any:
        if provider in self._compat_clients:
            return self._compat_clients[provider]

        from openai import OpenAI

        configs = {
            "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
            "nvidia_nim": ("https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
            "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
            # Local vLLM server
            "local": (os.environ.get("LOCAL_VLLM_BASE_URL", "http://localhost:8000/v1"), None),
        }
        if provider not in configs:
            raise ValueError(f"Unknown OpenAI-compatible provider '{provider}'.")

        base_url, api_key_env = configs[provider]
        if api_key_env:
            api_key = os.environ.get(api_key_env)
            if not api_key:
                raise ValueError(f"{api_key_env} environment variable is not set.")
        else:
            api_key = "not-needed-for-local-vllm"

        client = OpenAI(base_url=base_url, api_key=api_key)
        self._compat_clients[provider] = client
        return client

_POOL = APISessionPool()

# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------

def _pil_to_base64_jpeg(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def image_to_b64(img: Image.Image) -> str:
    return _pil_to_base64_jpeg(img)

# ---------------------------------------------------------------------------
# Universal LLM caller
# ---------------------------------------------------------------------------

def verify_local_server_health(timeout: float = 5.0) -> bool:
    """
    Pings the local vLLM server to ensure it is responsive.
    Returns True if healthy, False otherwise.
    """
    base_url = os.environ.get("LOCAL_VLLM_BASE_URL", "http://localhost:8000/v1")
    health_url = base_url.replace("/v1", "/health")
    
    try:
        req = urllib.request.Request(health_url)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.getcode() == 200
    except Exception:
        # Fallback to checking /v1/models if /health doesn't exist or vLLM version varies
        try:
            models_url = f"{base_url}/models"
            req = urllib.request.Request(models_url)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.getcode() == 200
        except Exception:
            return False

def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_content: str | list[dict[str, Any]],
    image: Optional[Image.Image] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    response_mime_type: Optional[str] = None,
    **kwargs: Any,
) -> str:
    """Universal API caller for vision-language models without retries."""
    
    _TRACKER.acquire_slot(provider)
    
    try:
        if provider in ("google", "gemini"):
            from google import genai
            from google.genai import types

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
                
            client = genai.Client(api_key=api_key)

            if isinstance(user_content, list):
                contents = []
                extracted_system = []
                for msg in user_content:
                    if msg["role"] == "system":
                        extracted_system.append(msg["content"])
                        continue
                        
                    role = "user" if msg["role"] == "user" else "model"
                    parts = []
                    content_payload = msg["content"]
                    if isinstance(content_payload, str):
                        parts.append(types.Part.from_text(text=content_payload))
                    elif isinstance(content_payload, list):
                        for item in content_payload:
                            if item["type"] == "text":
                                parts.append(types.Part.from_text(text=item["text"]))
                            elif item["type"] == "image":
                                b64 = _pil_to_base64_jpeg(item["image"].convert("RGB"))
                                import base64
                                raw_bytes = base64.b64decode(b64)
                                parts.append(types.Part.from_bytes(data=raw_bytes, mime_type="image/jpeg"))
                    contents.append(types.Content(role=role, parts=parts))
                    
                if not system_prompt and extracted_system:
                    system_prompt = "\n".join(extracted_system)
            else:
                parts: list[Any] = [types.Part.from_text(text=user_content)]
                if image is not None:
                    b64 = _pil_to_base64_jpeg(image.convert("RGB"))
                    import base64
                    raw_bytes = base64.b64decode(b64)
                    parts.append(types.Part.from_bytes(data=raw_bytes, mime_type="image/jpeg"))
                contents = parts

            resp = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt if system_prompt else None,
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                    response_mime_type=response_mime_type,
                ),
            )
            return resp.text.strip() if resp.text else ""

        elif provider in ("groq", "nvidia_nim", "openrouter", "local"):
            client = _POOL.get_openai_compatible(provider)

            built_messages: list[dict[str, Any]] = []
            if system_prompt:
                built_messages.append({"role": "system", "content": system_prompt})

            if isinstance(user_content, list):
                for msg in user_content:
                    role = msg["role"]
                    if role == "system":
                        built_messages.insert(0, {"role": "system", "content": msg["content"]})
                        continue
                    payload = msg["content"]
                    if isinstance(payload, str):
                        built_messages.append({"role": role, "content": payload})
                        continue
                    parts: list[dict[str, Any]] = []
                    for item in payload:
                        if item["type"] == "text":
                            parts.append({"type": "text", "text": item["text"]})
                        elif item["type"] == "image":
                            b64 = _pil_to_base64_jpeg(item["image"].convert("RGB"))
                            parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            })
                    built_messages.append({"role": role, "content": parts})
            else:
                user_parts: list[dict[str, Any]] = []
                if image is not None:
                    b64 = _pil_to_base64_jpeg(image)
                    user_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                user_parts.append({"type": "text", "text": user_content})
                built_messages.append({"role": "user", "content": user_parts})

            extra_headers = None
            if provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://github.com/rezaxr14/VLM-DENTAL"),
                    "X-Title": os.environ.get("OPENROUTER_TITLE", "VLM-DENTAL"),
                }

            response = client.chat.completions.create(
                model=model,
                messages=built_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers=extra_headers,
            )
            return response.choices[0].message.content or ""

        else:
            raise ValueError(f"Unknown provider '{provider}'")

    except Exception as e:
        raise RuntimeError(f"API Error ({e}): Hard stop on API errors per rule. No retries allowed. Exiting.")
