"""
API Key Pool and Rate Limiting for Multi-Provider Vision-Language Model Inference.

Manages round-robin rotation, 5-minute cooling, and requests-per-day (RPD)
quotas across multiple providers (NVIDIA NIM, Groq, OpenRouter, Gemini), with automatic
provider fallback.

Groq, NVIDIA NIM, OpenRouter, and a local vLLM server are all OpenAI-compatible
(same /chat/completions surface, just a different base_url + API key), so they share
one client factory (APISessionPool.get_openai_compatible). Gemini uses the `google-genai` SDK.
"""

from __future__ import annotations

import base64
import datetime
import io
import json
import os
import time
from typing import Any, Optional
import pandas as pd
from PIL import Image

from dental_agent.config import load_env

# Automatically load .env if not already in environment
load_env()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class AllKeysExhaustedToday(Exception):
    """Raised when every provider in the pool has reached its daily request cap."""
    pass


# ---------------------------------------------------------------------------
# Provider Pool (Round-Robin with Cooling & Daily Caps)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Model env-var mapping helpers
# ---------------------------------------------------------------------------

def _provider_to_api_key_env(provider: str) -> str:
    """Map provider name to its API key env var."""
    return f"{provider.upper().replace('_NIM', '')}_API_KEY"


def _provider_to_model_env(provider: str, role: str = "VERIFIER") -> str:
    """Map provider name + role to its model env var."""
    return f"{provider.upper().replace('_NIM', '')}_{role}_MODEL"


def _is_valid_env(val: str | None) -> bool:
    """Return True if an env var value is set and not a placeholder."""
    if not val:
        return False
    v = val.strip().lower()
    return bool(v and not v.startswith("your_") and not v.startswith("placeholder") and v != "none")


# ---------------------------------------------------------------------------
# Base Pool (shared logic for ProviderPool and GeneratorPool)
# ---------------------------------------------------------------------------

class _BasePool:
    """Shared round-robin + cooldown + daily-cap logic.

    Subclasses set ``_role`` ("VERIFIER" or "GENERATOR") and optionally
    override ``_default_candidates``.
    """

    _role: str = "VERIFIER"  # overridden by subclass

    def __init__(
        self,
        providers: list[str] | None = None,
        cooldown_seconds: float = 300.0,
        rpd_limit: int = 10,
        state_path: str | None = None,
    ) -> None:
        if providers is None:
            candidates = ["nvidia_nim", "groq", "openrouter", "gemini"]
            active = []
            for p in candidates:
                key_env = _provider_to_api_key_env(p)
                model_env = _provider_to_model_env(p, self._role)
                key_val = os.environ.get(key_env, "").strip()
                model_val = os.environ.get(model_env, "").strip()
                # Activate only if BOTH the API key and the model are set
                if _is_valid_env(key_val) and _is_valid_env(model_val):
                    active.append(p)
            self.providers = active
        else:
            self.providers = providers

        self.cooldown = cooldown_seconds
        self.rpd_limit = rpd_limit
        self.state_path = state_path or os.path.join(
            os.environ.get("DENTAL_AGENT_DATA_DIR", "data"),
            f"{self._role.lower()}_pool_state.json",
        )
        self.state: dict[str, Any] = self._load()
        self._next_idx = 0
        self._reset_if_new_day()

    # -- persistence ---------------------------------------------------------

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
        for p in self.providers:
            entry = self.state.get(p)
            if entry is None or entry.get("date") != today:
                self.state[p] = {
                    "date": today,
                    "last_call_ts": 0.0,
                    "calls_today": 0,
                }
                changed = True
        if changed:
            self._save()

    # -- acquire -------------------------------------------------------------

    def acquire(self) -> tuple[str, str]:
        """Finds the next available provider. Blocks if all are on cooldown.

        Returns ``(provider, model)``.
        """
        if not self.providers:
            raise ValueError(
                f"No providers have both an API key and a {self._role}_MODEL set in .env!"
            )

        self._reset_if_new_day()

        # All exhausted for the day?
        if all(
            self.state.get(p, {}).get("calls_today", 0) >= self.rpd_limit
            for p in self.providers
        ):
            raise AllKeysExhaustedToday(
                f"All {len(self.providers)} {self._role.lower()} providers have "
                f"reached their {self.rpd_limit} RPD limit."
            )

        while True:
            closest_wait = float("inf")

            for _ in range(len(self.providers)):
                idx = self._next_idx % len(self.providers)
                self._next_idx += 1
                p = self.providers[idx]
                entry = self.state[p]

                if entry["calls_today"] >= self.rpd_limit:
                    continue

                elapsed = time.time() - entry.get("last_call_ts", 0.0)
                if elapsed >= self.cooldown:
                    entry["last_call_ts"] = time.time()
                    entry["calls_today"] += 1
                    self._save()

                    model_env = _provider_to_model_env(p, self._role)
                    model = os.environ.get(model_env, "").strip()
                    if not model:
                        raise ValueError(
                            f"Provider '{p}' was selected but {model_env} is not set in .env"
                        )
                    return p, model
                else:
                    wait = self.cooldown - elapsed
                    if wait < closest_wait:
                        closest_wait = wait

            print(
                f"All {self._role.lower()} providers on cooldown. "
                f"Sleeping for {closest_wait:.1f} seconds..."
            )
            time.sleep(closest_wait + 0.1)


# ---------------------------------------------------------------------------
# Concrete Pools
# ---------------------------------------------------------------------------

class ProviderPool(_BasePool):
    """Round-robins **verifier** calls across configured providers.

    Reads ``API_COOLDOWN_SECONDS`` and ``API_RPD_LIMIT`` from the environment.
    State persists to ``data/verifier_pool_state.json``.
    """

    _role = "VERIFIER"


class GeneratorPool(_BasePool):
    """Round-robins **generator** calls across configured providers.

    Only used when ``GENERATOR_PROVIDER`` is NOT ``local``.
    Reads ``GENERATOR_COOLDOWN_SECONDS`` and ``GENERATOR_RPD_LIMIT``.
    State persists to ``data/generator_pool_state.json``.
    """

    _role = "GENERATOR"


# ---------------------------------------------------------------------------
# Global pool singletons
# ---------------------------------------------------------------------------

_provider_pool: ProviderPool | None = None
_generator_pool: GeneratorPool | None = None


def get_provider_pool() -> ProviderPool:
    """Return the global verifier ProviderPool singleton."""
    global _provider_pool
    if _provider_pool is None:
        cooldown = float(os.environ.get("API_COOLDOWN_SECONDS", "300"))
        rpd = int(os.environ.get("API_RPD_LIMIT", "10"))
        _provider_pool = ProviderPool(cooldown_seconds=cooldown, rpd_limit=rpd)
    return _provider_pool


def get_generator_pool() -> GeneratorPool:
    """Return the global GeneratorPool singleton.

    Only meaningful when ``GENERATOR_PROVIDER`` is an external API.
    """
    global _generator_pool
    if _generator_pool is None:
        cooldown = float(os.environ.get("GENERATOR_COOLDOWN_SECONDS", "60"))
        rpd = int(os.environ.get("GENERATOR_RPD_LIMIT", "50"))
        _generator_pool = GeneratorPool(cooldown_seconds=cooldown, rpd_limit=rpd)
    return _generator_pool


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

import urllib.request
import urllib.error

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
    max_retries: int = 3,
    retry_delay: float = 2.0,
    response_mime_type: Optional[str] = None,
) -> str:
    """Universal API caller for vision-language models with pool-based fallback routing."""
    
    # Auto-route through the verifier pool
    if provider == "auto_verifier":
        pool = get_provider_pool()
        provider, model = pool.acquire()
        print(f"  [call_llm] Auto-routed to verifier provider '{provider}' (model: {model})")

    # Auto-route through the generator pool (external API generation)
    elif provider == "auto_generator":
        pool = get_generator_pool()
        provider, model = pool.acquire()
        print(f"  [call_llm] Auto-routed to generator provider '{provider}' (model: {model})")

    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
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

        except AllKeysExhaustedToday:
            raise
        except Exception as e:
            last_error = e
            err_str = str(e).lower()
            if attempt < max_retries:
                backoff = (retry_delay * 2 ** (attempt - 1)) if "503" not in err_str else (retry_delay + attempt * 2)
                time.sleep(backoff)

    raise RuntimeError(
        f"Failed to call {provider}/{model} after {max_retries} attempts: {last_error}"
    )
