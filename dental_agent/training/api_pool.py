"""
Multi-provider LLM/VLM API client pool with retry logic and multimodal support.
Supports Gemini (Google GenAI), Anthropic Claude, and OpenAI (GPT-4o).
Includes GeminiKeyPool for multi-key rotation with RPM/RPD rate-limit tracking.
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


# ---------------------------------------------------------------------------
# GeminiKeyPool: multi-key rotation with per-key RPM/RPD tracking
# ---------------------------------------------------------------------------

class AllKeysExhaustedToday(Exception):
    """Raised by GeminiKeyPool when every configured key has hit today's request cap.
    run_aim1_batch catches this specifically and stops cleanly rather than
    retrying with backoff — a daily quota won't reset in seconds, only overnight."""
    pass


class GeminiKeyPool:
    """Round-robins calls across GEMINI_API_KEYS, respecting each key's own RPM and RPD
    limits, persisting daily-usage state to disk so it survives a session restart —
    including into a new calendar day, at which point each key's counter resets
    automatically (detected by comparing today's date to the saved state).

    Keys are identified by their POSITION in the list, never by their value,
    so this state file is safe to sync without leaking anything.
    """

    def __init__(
        self,
        keys: list[str],
        rpm: int = 5,
        rpd: int = 20,
        safety_margin: float = 0.95,
        state_path: str | None = None,
    ) -> None:
        self.keys = keys
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
            with open(self.state_path) as f:
                return json.load(f)
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
                self.state[kid] = {"date": today, "calls_today": 0, "last_call_ts": 0.0}
                changed = True
        if changed:
            self._save()

    def acquire(self) -> tuple[int, str]:
        """Blocks only as long as needed for the chosen key's own RPM spacing, then
        returns (index, key). Raises AllKeysExhaustedToday if every key has hit its
        daily cap."""
        if not self.keys:
            raise ValueError("GEMINI_API_KEYS is empty — set it before calling provider='gemini'.")
        self._reset_if_new_day()

        for _ in range(len(self.keys)):
            idx = self._next_idx % len(self.keys)
            self._next_idx += 1
            entry = self.state[str(idx)]
            if entry["calls_today"] >= self.rpd_limit:
                continue  # this key is done for today
            elapsed = time.time() - entry["last_call_ts"]
            min_interval = 60.0 / self.rpm_limit
            if elapsed < min_interval:
                time.sleep(min_interval - elapsed)
            entry["last_call_ts"] = time.time()
            entry["calls_today"] += 1
            self._save()
            return idx, self.keys[idx]

        raise AllKeysExhaustedToday(
            f"All {len(self.keys)} Gemini key(s) have hit today's cap "
            f"({self.rpd_limit} calls/key after the safety margin). Gemini's free tier "
            f"resets around midnight Pacific time — re-run tomorrow; "
            f"resume=True in run_aim1_batch will pick up exactly where this left off."
        )

    def status(self) -> pd.DataFrame:
        """How much of today's budget is left per key."""
        self._reset_if_new_day()
        return pd.DataFrame([{
            "key_index": i,
            "calls_today": self.state[str(i)]["calls_today"],
            "rpd_limit": self.rpd_limit,
            "remaining_today": max(0, self.rpd_limit - self.state[str(i)]["calls_today"]),
        } for i in range(len(self.keys))])


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
        rpm = int(os.environ.get("GEMINI_RPM_LIMIT", "5"))
        rpd = int(os.environ.get("GEMINI_RPD_LIMIT", "20"))
        _gemini_pool = GeminiKeyPool(keys, rpm=rpm, rpd=rpd)
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
    """Universal API caller for proprietary vision-language models with retry logic.
    Uses GeminiKeyPool for the 'gemini' provider to respect RPM/RPD limits."""
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            if provider in ("google", "gemini"):
                from google import genai
                from google.genai import types

                pool = get_gemini_pool()
                _, api_key = pool.acquire()
                client = genai.Client(api_key=api_key)

                parts: list[Any] = [user_content]
                if image is not None:
                    parts.append(image.convert("RGB"))

                resp = client.models.generate_content(
                    model=model,
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
                user_blocks: list[dict[str, Any]] = []
                if image is not None:
                    b64 = _pil_to_base64_jpeg(image)
                    user_blocks.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                user_blocks.append({"type": "text", "text": user_content})

                response = client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_blocks},
                    ],
                )
                return response.choices[0].message.content.strip()

            else:
                raise ValueError(f"Unknown API provider: {provider}")

        except AllKeysExhaustedToday:
            raise  # bubble straight out — don't retry a daily quota
        except Exception as e:
            last_error = e
            if attempt < max_retries:
                time.sleep(retry_delay * attempt)
            else:
                raise RuntimeError(
                    f"API call to {provider}/{model} failed after {max_retries} attempts: {e}"
                ) from last_error

    return ""
