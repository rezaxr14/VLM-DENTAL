"""
API Key Management and API Inference logic.

No more rotating pools. A single provider is passed and executed.
If it fails, it fails immediately without retries per Rule 9.
"""

from __future__ import annotations

import base64
import io
import os
import time
import threading
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
        self._lock = threading.Lock()
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
        # Only the daily call count resets at midnight -- cooldown/RPM tracking
        # (last_call_time, recent_call_times) is NOT a daily concept and must
        # survive the day boundary, so this must not wipe the whole entry.
        today = datetime.date.today().isoformat()
        changed = False
        for p in list(self.state.keys()):
            if self.state[p].get("date") != today:
                self.state[p]["date"] = today
                self.state[p]["calls_today"] = 0
                changed = True
        if changed:
            self._save()

    def acquire_slot(self, provider: str, role: str = "generator") -> None:
        """Block (sleep) as needed, then reserve a slot for this provider/role.

        Enforces three independent constraints, all `.env`-driven, all optional
        (unset = unlimited for that dimension):
          - {PREFIX}_COOLDOWN_SECONDS: minimum gap since this provider's last call,
            regardless of role. This is the PRIMARY, deliberately conservative pacing
            knob -- per explicit instruction, requests must be spread out across the
            full window, never rushed/bursted, and should not approach anywhere near
            50% of the stated RPM even when RPM alone would allow tighter spacing.
          - {PREFIX}_{ROLE}_RPM_LIMIT (ROLE = GENERATOR or VERIFIER): rolling 60s
            request cap, tracked independently per role since generator and verifier
            traffic are paced separately by design.
          - {PREFIX}_RPD_LIMIT: hard daily cap, shared across roles (same real
            account), raises immediately with no wait -- exhausting this excludes
            the provider for the rest of the day, it does not throttle-and-continue.
        """
        if provider in ("local", "auto_verifier", "auto_generator"):
            return

        with self._lock:
            self._reset_if_new_day()
            prefix = provider.upper().replace('_NIM', '')
            role_upper = role.upper()
            today = datetime.date.today().isoformat()
            entry = self.state.setdefault(provider, {"date": today, "calls_today": 0})

            rpd_str = os.environ.get(f"{prefix}_RPD_LIMIT")
            if rpd_str:
                try:
                    rpd_limit = int(rpd_str)
                except ValueError:
                    rpd_limit = 0
                if rpd_limit and entry.get("calls_today", 0) >= rpd_limit:
                    raise RPDLimitExhausted(
                        f"Provider '{provider}' has exhausted its {rpd_limit} RPD limit. "
                        "Excluded for the rest of the day -- use a different provider."
                    )

            cooldown = float(os.environ.get(f"{prefix}_COOLDOWN_SECONDS", 0) or 0)
            rpm_str = os.environ.get(f"{prefix}_{role_upper}_RPM_LIMIT") or os.environ.get(f"{prefix}_RPM_LIMIT")
            rpm_limit = int(rpm_str) if rpm_str else 0
            hard_cap_str = os.environ.get(f"{prefix}_RPM_HARD_CAP")
            if hard_cap_str:
                # Never effectively exceed this regardless of what GENERATOR/VERIFIER_RPM_LIMIT
                # is set to -- e.g. Gemini's real ceiling is 5 RPM; setting the operator-tunable
                # limit to 10 by mistake still only ever dispatches at the hard cap.
                hard_cap = int(hard_cap_str)
                rpm_limit = min(rpm_limit, hard_cap) if rpm_limit else hard_cap

            now = time.time()
            recent_by_role: dict[str, list[float]] = entry.setdefault("recent_call_times", {})
            recent = [t for t in recent_by_role.get(role, []) if now - t < 60]

            wait_for = 0.0
            last_call_time = entry.get("last_call_time")
            if cooldown and last_call_time is not None:
                wait_for = max(wait_for, cooldown - (now - last_call_time))
            if rpm_limit and len(recent) >= rpm_limit:
                # Oldest call in the current 60s window falls out of it; wait until it does.
                wait_for = max(wait_for, 60.0 - (now - recent[0]))

            if wait_for > 0:
                print(f"  [throttle] {provider}/{role}: spacing out requests, waiting {wait_for:.1f}s...", flush=True)
                time.sleep(wait_for)
                now = time.time()

            recent.append(now)
            recent_by_role[role] = recent[-max(rpm_limit, 1):] if rpm_limit else recent
            entry["last_call_time"] = now
            entry["calls_today"] = entry.get("calls_today", 0) + 1
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

def _call_llm_once(
    provider: str,
    model: str,
    system_prompt: str,
    user_content: str | list[dict[str, Any]],
    image: Optional[Image.Image] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    response_mime_type: Optional[str] = None,
    role: str = "generator",
    **kwargs: Any,
) -> str:
    """Single-attempt universal API caller. Called by call_llm, which adds the
    one-retry-for-5xx-only policy around this. Do not call this directly from
    trace_generation.py / langgraph_loop.py -- call `call_llm` instead."""

    _TRACKER.acquire_slot(provider, role=role)
    
    try:
        if provider in ("google", "gemini"):
            from google import genai
            from google.genai import types

            api_key = os.environ.get("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY environment variable is not set.")
                
            client = genai.Client(api_key=api_key)
            max_dim_str = os.environ.get("GEMINI_IMAGE_MAX_DIM")
            max_dim = int(max_dim_str) if max_dim_str is not None else 0

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
                                scaled_img = item["image"].copy().convert("RGB")
                                if max_dim > 0:
                                    scaled_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                                b64 = _pil_to_base64_jpeg(scaled_img)
                                import base64
                                raw_bytes = base64.b64decode(b64)
                                parts.append(types.Part.from_bytes(data=raw_bytes, mime_type="image/jpeg"))
                    contents.append(types.Content(role=role, parts=parts))
                    
                if not system_prompt and extracted_system:
                    system_prompt = "\n".join(extracted_system)
            else:
                parts: list[Any] = [types.Part.from_text(text=user_content)]
                if image is not None:
                    scaled_img = image.copy().convert("RGB")
                    if max_dim > 0:
                        scaled_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                    b64 = _pil_to_base64_jpeg(scaled_img)
                    import base64
                    raw_bytes = base64.b64decode(b64)
                    parts.append(types.Part.from_bytes(data=raw_bytes, mime_type="image/jpeg"))
                contents = parts

            gen_config = types.GenerateContentConfig(
                system_instruction=system_prompt if system_prompt else None,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type=response_mime_type,
            )

            if kwargs.get("stream", False):
                label = kwargs.get("label", "")
                if label:
                    print(f"\n--- [{provider}/{model}] {label} ---", flush=True)
                collected = []
                for chunk in client.models.generate_content_stream(
                    model=model, contents=contents, config=gen_config,
                ):
                    if getattr(chunk, "text", None):
                        print(chunk.text, end="", flush=True)
                        collected.append(chunk.text)
                print()
                return "".join(collected).strip()

            resp = client.models.generate_content(
                model=model, contents=contents, config=gen_config,
            )
            return resp.text.strip() if resp.text else ""

        elif provider in ("groq", "nvidia_nim", "openrouter", "local"):
            client = _POOL.get_openai_compatible(provider)
            prefix = provider.upper().replace('_NIM', '')
            default_max_dim = {"GROQ": 1280, "OPENROUTER": 1600}.get(prefix, 0)
            max_dim_str = os.environ.get(f"{prefix}_IMAGE_MAX_DIM")
            max_dim = int(max_dim_str) if max_dim_str is not None else default_max_dim

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
                            scaled_img = item["image"].copy()
                            if max_dim > 0:
                                scaled_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                            b64 = _pil_to_base64_jpeg(scaled_img)
                            parts.append({
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                            })
                    built_messages.append({"role": role, "content": parts})
            else:
                user_parts: list[dict[str, Any]] = []
                if image is not None:
                    scaled_img = image.copy()
                    if max_dim > 0:
                        scaled_img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                    b64 = _pil_to_base64_jpeg(scaled_img)
                    user_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })
                user_parts.append({"type": "text", "text": user_content})
                built_messages.append({"role": "user", "content": user_parts})

            extra_headers = None
            extra_body = kwargs.get("extra_body", {})
            if provider == "openrouter":
                extra_headers = {
                    "HTTP-Referer": os.environ.get("OPENROUTER_REFERER", "https://github.com/rezaxr14/VLM-DENTAL"),
                    "X-Title": os.environ.get("OPENROUTER_TITLE", "VLM-DENTAL"),
                }
                extra_body = dict(extra_body)
                if "include_reasoning" not in extra_body:
                    extra_body["include_reasoning"] = True

            stream_requested = kwargs.get("stream", False)
            if stream_requested:
                label = kwargs.get("label", "")
                if label:
                    print(f"\n--- [{provider}/{model}] {label} ---", flush=True)
            response = client.chat.completions.create(
                model=model,
                messages=built_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                extra_headers=extra_headers,
                extra_body=extra_body if extra_body else None,
                stream=stream_requested,
            )
            
            if stream_requested:
                collected = []
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta:
                        delta = chunk.choices[0].delta
                        delta_extra = getattr(delta, "model_extra", None) or {}
                        text = (
                            getattr(delta, "content", None)
                            or getattr(delta, "reasoning", None)
                            or getattr(delta, "reasoning_content", None)
                            or delta_extra.get("reasoning")
                            or delta_extra.get("reasoning_content")
                        )
                        if text:
                            print(text, end="", flush=True)
                            collected.append(str(text))
                print() # newline
                full_text = "".join(collected)
                if kwargs.get("return_metadata", False):
                    return full_text, {"finish_reason": "stop", "usage": {}}
                return full_text
            else:
                choice = response.choices[0] if response.choices else None
                finish_reason = getattr(choice, "finish_reason", "stop") if choice else "stop"
                content = choice.message.content or "" if choice and choice.message else ""
                
                # Extract reasoning/thinking tokens for reasoning models (e.g. OpenRouter Nemotron / DeepSeek)
                if choice and choice.message:
                    msg = choice.message
                    msg_dict = {}
                    if hasattr(msg, "model_dump"):
                        try:
                            msg_dict = msg.model_dump() or {}
                        except Exception:
                            pass
                    elif hasattr(msg, "to_dict"):
                        try:
                            msg_dict = msg.to_dict() or {}
                        except Exception:
                            pass
                    msg_extra = getattr(msg, "model_extra", None) or {}
                    reasoning = (
                        getattr(msg, "reasoning", None)
                        or getattr(msg, "reasoning_content", None)
                        or msg_extra.get("reasoning")
                        or msg_extra.get("reasoning_content")
                        or msg_dict.get("reasoning")
                        or msg_dict.get("reasoning_content")
                        or msg_dict.get("thought")
                        or msg_dict.get("thinking")
                    )
                    if reasoning:
                        if content:
                            content = f"{reasoning}\n\n{content}"
                        else:
                            content = str(reasoning)

                usage = getattr(response, "usage", None)
                usage_dict = {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                }
                if kwargs.get("return_metadata", False):
                    return content, {"finish_reason": finish_reason, "usage": usage_dict}
                return content

        else:
            raise ValueError(f"Unknown provider '{provider}'")

    except Exception as e:
        err_str = str(e)
        if "Rate limit reached" in err_str or "rate_limit_exceeded" in err_str or "429" in err_str:
            import re
            tpm_match = re.search(r"Limit\s+(\d+),\s*Used\s+(\d+),\s*Requested\s+(\d+)", err_str)
            retry_match = re.search(r"try again in\s+([\d\.]+s)", err_str)
            if tpm_match:
                lim, used, req = tpm_match.groups()
                retry_s = retry_match.group(1) if retry_match else "15s"
                print(f"\n🛑 [{provider.upper()} 429 RATE LIMIT] Limit: {lim} TPM | Used: {used} TPM | Requested: {req} TPM | Retry after: {retry_s}", flush=True)
        raise RuntimeError(f"API Error ({e})")


def call_llm(
    provider: str,
    model: str,
    system_prompt: str,
    user_content: str | list[dict[str, Any]],
    image: Optional[Image.Image] = None,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    response_mime_type: Optional[str] = None,
    role: str = "generator",
    **kwargs: Any,
) -> str:
    """Universal API caller. Retries exactly once, only for a 5xx-class (server-side,
    transient) failure, with a short backoff -- e.g. NVIDIA's occasional
    "Internal server error". Every other failure (429 rate-limited, 413 too-large,
    anything else) keeps the original hard-stop-no-retries behavior, since retrying
    those unchanged just fails the same way again and burns quota for nothing.
    (Exception: if IGNORE_429 is set in env, allows up to 10 retries for 429 errors.)
    """
    retries_429 = 0
    retries_5xx = 0
    max_429_retries = 10
    
    while True:
        try:
            return _call_llm_once(
                provider, model, system_prompt, user_content, image=image,
                max_tokens=max_tokens, temperature=temperature,
                response_mime_type=response_mime_type, role=role, **kwargs,
            )
        except Exception as e:
            msg = str(e)
            status_code = getattr(getattr(e, "response", None), "status_code", None) or getattr(e, "status_code", None)
            msg_lower = msg.lower()

            ignore_all_errors = os.environ.get("IGNORE_API_ERRORS", "false").lower() == "true"
            
            is_5xx = (
                "internal server error" in msg_lower
                or "unavailable" in msg_lower  # Gemini: "503 UNAVAILABLE. ... high demand ..."
                or any(f"{code} " in msg or f"{code}." in msg or msg.strip().startswith(str(code))
                       for code in (500, 502, 503, 504))
                or (isinstance(status_code, int) and 500 <= status_code < 600)
            )
            
            is_429 = (
                "429" in msg or
                "rate limit" in msg_lower or
                "too many requests" in msg_lower or
                (isinstance(status_code, int) and status_code == 429)
            )
            
            if ignore_all_errors:
                if retries_429 < max_429_retries:
                    print(f"  [retry] {provider}: API Error Hit (attempt {retries_429+1}/{max_429_retries}). Sleeping 5s before retrying...", flush=True)
                    time.sleep(5)
                    retries_429 += 1
                    continue
                else:
                    raise RuntimeError(f"{msg}: Hard stop on API errors per rule. Max retries ({max_429_retries}) exceeded. Exiting.")

            if is_429 and os.environ.get("IGNORE_429", "false").lower() == "true":
                if retries_429 < max_429_retries:
                    print(f"  [retry] {provider}: 429 Rate Limit Hit (attempt {retries_429+1}/{max_429_retries}). Sleeping 5s before retrying...", flush=True)
                    time.sleep(5)
                    retries_429 += 1
                    continue
                else:
                    raise RuntimeError(f"{msg}: Hard stop on API errors per rule. 429 Max retries ({max_429_retries}) exceeded. Exiting.")
            
            if not is_5xx:
                raise RuntimeError(f"{msg}: Hard stop on API errors per rule. No retries allowed. Exiting.")

            if retries_5xx == 0:
                print(f"  [retry] {provider}: transient 5xx-class error, retrying once in 5s...", flush=True)
                time.sleep(5)
                retries_5xx += 1
                continue
            
            raise RuntimeError(f"{msg}: Hard stop after one retry. No further retries allowed. Exiting.")
