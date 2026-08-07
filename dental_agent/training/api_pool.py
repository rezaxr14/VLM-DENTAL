"""
API Key Pool and Rate Limiting for Multi-Provider Vision-Language Model Inference.

Manages round-robin rotation, requests-per-minute (RPM), and requests-per-day (RPD)
quotas across Google Gemini keys, with automatic multi-model fallback (e.g., primary
gemini-2.5-flash -> fallback gemini-1.5-flash for 40 RPD/key), and cached clients
for Anthropic and OpenAI.
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
    """Raised when every key in the pool has reached its daily request cap across all models."""
    pass


# ---------------------------------------------------------------------------
# Gemini Key Pool with Multi-Model Fallback (e.g. 20 + 20 = 40 RPD per key)
# ---------------------------------------------------------------------------

class GeminiKeyPool:
    """Round-robins calls across GEMINI_API_KEYS, respecting each key's own RPM and RPD
    limits with automatic multi-model fallback.
    
    When the primary model (e.g. gemini-2.5-flash) reaches its 20 RPD cap on all keys,
    it automatically falls back to the secondary model (e.g. gemini-1.5-flash) for
    another 20 RPD, providing 40 total daily requests per key.
    
    Daily-usage state persists to disk so it survives session restarts and resets
    automatically on new calendar days.
    """

    def __init__(
        self,
        keys: list[str],
        models: list[str] | None = None,
        rpm: int = 5,
        rpd: int = 20,
        safety_margin: float = 1.0,
        state_path: str | None = None,
    ) -> None:
        self.keys = keys
        self.models = models or [
            os.environ.get("GEMINI_PRIMARY_MODEL", "gemini-3.6-flash"),
            os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash"),
        ]
        self.rpm_limit = max(1, int(rpm * safety_margin))
        self.rpd_limit = max(1, int(rpd * safety_margin))
        self.state_path = state_path or os.path.join(
            os.environ.get("DENTAL_AGENT_DATA_DIR", "data"), "gemini_key_state.json"
        )
        self.state: dict[str, Any] = self._load()
        self._next_idx = 0
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
        for i in range(len(self.keys)):
            kid = str(i)
            entry = self.state.get(kid)
            if entry is None or entry.get("date") != today:
                self.state[kid] = {
                    "date": today,
                    "last_call_ts": 0.0,
                    "models": {m: 0 for m in self.models},
                    "calls_today": 0,
                }
                changed = True
            else:
                # Ensure all models are present in subdict
                if "models" not in entry:
                    entry["models"] = {m: entry.get("calls_today", 0) for m in self.models}
                    changed = True
                for m in self.models:
                    if m not in entry["models"]:
                        entry["models"][m] = 0
                        changed = True
        if changed:
            self._save()

    def acquire(self, model: str | None = None) -> tuple[int, str, str]:
        """Blocks for RPM spacing, returns (key_index, key_str, effective_model).
        
        If `model` is provided, tries that model first; if exhausted across all keys,
        falls back to remaining models in self.models.
        """
        if not self.keys:
            raise ValueError("GEMINI_API_KEYS is empty — set GEMINI_API_KEY in .env before calling.")
        self._reset_if_new_day()

        # Build candidate model list with requested model first
        candidate_models = list(self.models)
        if model and model in candidate_models:
            candidate_models.remove(model)
            candidate_models.insert(0, model)
        elif model:
            candidate_models.insert(0, model)

        for target_model in candidate_models:
            for _ in range(len(self.keys)):
                idx = self._next_idx % len(self.keys)
                self._next_idx += 1
                entry = self.state[str(idx)]
                model_calls = entry.get("models", {}).get(target_model, 0)
                if model_calls >= self.rpd_limit:
                    continue  # key reached daily limit for this model

                elapsed = time.time() - entry.get("last_call_ts", 0.0)
                min_interval = 60.0 / self.rpm_limit
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

                entry["last_call_ts"] = time.time()
                entry["models"][target_model] = model_calls + 1
                entry["calls_today"] = sum(entry["models"].values())
                self._save()
                return idx, self.keys[idx], target_model

        total_cap = len(self.keys) * self.rpd_limit * len(self.models)
        raise AllKeysExhaustedToday(
            f"All {len(self.keys)} Gemini key(s) have reached today's total quota "
            f"({total_cap} total requests across models {self.models} at {self.rpd_limit} RPD/model/key). "
            f"Gemini free tier resets around midnight Pacific time — resume=True in run_aim1_batch "
            f"will pick up exactly where this left off."
        )

    def status(self) -> pd.DataFrame:
        """Return budget and calls used per key and per model."""
        self._reset_if_new_day()
        rows = []
        for i in range(len(self.keys)):
            entry = self.state.get(str(i), {})
            models_dict = entry.get("models", {})
            for m in self.models:
                used = models_dict.get(m, 0)
                rows.append({
                    "key_index": i,
                    "model": m,
                    "calls_today": used,
                    "rpd_limit": self.rpd_limit,
                    "remaining_today": max(0, self.rpd_limit - used),
                })
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Global pool singleton — lazy-initialized
# ---------------------------------------------------------------------------

_gemini_pool: GeminiKeyPool | None = None


def get_gemini_pool() -> GeminiKeyPool:
    """Return the global GeminiKeyPool, constructing it from GEMINI_API_KEYS or GEMINI_API_KEY."""
    global _gemini_pool
    if _gemini_pool is None:
        keys_raw = os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY", "")
        keys = [k.strip().strip("'\"") for k in keys_raw.split(",") if k.strip().strip("'\"")]
        models_raw = os.environ.get("GEMINI_MODELS", "")
        models = [m.strip() for m in models_raw.split(",") if m.strip()] or [
            os.environ.get("GEMINI_PRIMARY_MODEL", "gemini-2.5-flash"),
            os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-1.5-flash"),
        ]
        rpm = int(os.environ.get("GEMINI_RPM_LIMIT", "5"))
        rpd = int(os.environ.get("GEMINI_RPD_LIMIT", "20"))
        _gemini_pool = GeminiKeyPool(keys, models=models, rpm=rpm, rpd=rpd)
    return _gemini_pool


# ---------------------------------------------------------------------------
# API Session Pool (cached provider clients)
# ---------------------------------------------------------------------------

class APISessionPool:
    """Manages cached client instances across API providers."""

    def __init__(self) -> None:
        self._openai_client: Any = None
        self._anthropic_client: Any = None

    def get_openai(self) -> Any:
        if self._openai_client is None:
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set.")
            self._openai_client = OpenAI(api_key=api_key)
        return self._openai_client

    def get_anthropic(self) -> Any:
        if self._anthropic_client is None:
            from anthropic import Anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable is not set.")
            self._anthropic_client = Anthropic(api_key=api_key)
        return self._anthropic_client


_POOL = APISessionPool()


# ---------------------------------------------------------------------------
# Image encoding helpers
# ---------------------------------------------------------------------------

def _pil_to_base64_jpeg(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def image_to_b64(img: Image.Image) -> str:
    """Public alias for base64-encoding a PIL image to JPEG."""
    return _pil_to_base64_jpeg(img)


# ---------------------------------------------------------------------------
# Universal LLM caller with retry
# ---------------------------------------------------------------------------

def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_content: str,
    image: Optional[Image.Image] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    max_retries: int = 3,
    retry_delay: float = 2.0,
) -> str:
    """Universal API caller for vision-language models with retry logic and multi-model fallback."""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if provider in ("google", "gemini"):
                from google import genai
                from google.genai import types

                pool = get_gemini_pool()
                _, api_key, actual_model = pool.acquire(model=model)
                client = genai.Client(api_key=api_key)

                parts: list[Any] = [user_content]
                if image is not None:
                    parts.append(image.convert("RGB"))

                resp = client.models.generate_content(
                    model=actual_model,
                    contents=parts,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        max_output_tokens=max_tokens,
                    ),
                )
                return resp.text.strip() if resp.text else ""

            elif provider == "anthropic":
                client = _POOL.get_anthropic()
                msg_content: list[dict[str, Any]] = []
                if image is not None:
                    b64 = _pil_to_base64_jpeg(image)
                    msg_content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": b64,
                        },
                    })
                msg_content.append({"type": "text", "text": user_content})

                response = client.messages.create(
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    system=system_prompt,
                    messages=[{"role": "user", "content": msg_content}],
                )
                return "".join(
                    block.text for block in response.content if hasattr(block, "text")
                ).strip()

            elif provider == "openai":
                client = _POOL.get_openai()
                user_msg_content: list[dict[str, Any]] = []
                if image is not None:
                    b64 = _pil_to_base64_jpeg(image)
                    user_msg_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                user_msg_content.append({"type": "text", "text": user_content})

                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_msg_content},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
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
                # If 503 (high demand) or 429 (rate limit), increase backoff and allow rotating keys
                backoff = (retry_delay * 2 ** (attempt - 1)) if "503" not in err_str else (retry_delay + attempt * 2)
                time.sleep(backoff)

    raise RuntimeError(
        f"Failed to call {provider}/{model} after {max_retries} attempts: {last_error}"
    )
