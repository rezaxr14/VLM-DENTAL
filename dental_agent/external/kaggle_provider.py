"""``reply_fn`` adapter that lets the project's agent loop talk to a hosted model.

Contract (``run_agent`` / ``run_agent_no_tools``): ``reply_fn(model, processor, messages, max_new_tokens=, temperature=,
return_ids=True, past_key_values=, cache_state=, return_cache=True)`` returns ``(text, prompt_len, token_ids, pkv,
cache_state)``. A remote model has no token ids and no KV cache, so those are ``[]`` / ``None`` (never invented), exactly
like ``harness.specdec.ServingReply``.

The adapter owns ONE chat per image. A ``ChatBackend`` is whatever carries a chat to the model:

* ``OpenAICompatBackend``: Kaggle's model proxy, OpenAI-compatible route ``{MODEL_PROXY_URL}/openapi/chat/completions``
  (verified by probe v5 on 2026-11-14 for Gemini and GPT: base64 ``image_url`` parts, history re-sent every turn,
  ``max_tokens`` enforced with ``finish_reason == "length"``, ``Authorization: Bearer``). It is stateless, so it keeps the
  history itself and re-sends it each turn (images are re-billed every turn). Usage arrives as ``usage.prompt_tokens`` /
  ``completion_tokens`` plus ``usage.cost.{input,output}_tokens_cost_nanodollars`` and ``total_backend_latency_ms``. Not
  yet verified: ``claude-*`` slugs (the bare id probed was "not found"), reasoning models that may require
  ``max_completion_tokens`` (``token_param``), and whether ``temperature`` is accepted for every model (the Kaggle SDK
  itself disables it on this route, so ``send_temperature`` is overridable and recorded per turn).
* any object with ``open_chat(system) -> chat`` and ``chat.send(parts, max_tokens, temperature) -> BackendReply``
  (tests use a fake; a stateful Kaggle SDK chat plugs in here once gate G1 verifies it).

Failure policy: no automatic retry. Infrastructure errors raise ``ProviderError`` and propagate (the coordinator stores them
as evidence and re-runs the image on resume); a refusal or empty reply is returned as an empty text, which the loop scores
as an answerless episode, and counted in ``usage``.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from dental_agent.external.transcript_sync import HistoryMismatch, TranscriptSync, to_parts


class ProviderError(RuntimeError):
    """Infrastructure failure talking to the provider (timeout, 5xx, quota, auth, network)."""

    def __init__(self, message: str, kind: str = "infra", status: int | None = None):
        super().__init__(message)
        self.kind = kind
        self.status = status


@dataclass
class BackendReply:
    text: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    finish_reason: str | None = None
    refused: bool = False
    max_tokens_enforced: bool | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class ChatHandle(Protocol):
    def send(self, parts: list[dict[str, Any]], max_tokens: int, temperature: float) -> BackendReply: ...

    def replay(self, parts: list[dict[str, Any]], text: str) -> None:
        """Append a turn that was answered earlier (from the journal) to the history WITHOUT any request."""


class ChatBackend(Protocol):
    model_slug: str

    def open_chat(self, system: str | None) -> ChatHandle: ...


def _expiry_epoch(value: str | None) -> float | None:
    """``MODEL_PROXY_EXPIRY_TIME`` (ISO, e.g. 2026-11-14T10:48:46.777000Z) as epoch seconds; None when absent or unparsable."""
    if not value:
        return None
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
    except ValueError:
        return None


def load_env_file(path: str) -> dict[str, str]:
    """Load KEY=VALUE lines (as written by ``kaggle benchmarks init --env-file``) into ``os.environ``; returns what was loaded."""
    import re
    loaded: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            m = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)", line.rstrip("\n"))
            if m:
                loaded[m.group(1)] = m.group(2).strip().strip("'\"")
    os.environ.update(loaded)
    return loaded


def kaggle_cli_refresher(env_file: str = "kaggle_proxy.env", runner=None):
    """Refresher that re-runs ``kaggle benchmarks init -y --env-file`` (the proxy key lives about one hour) and reloads the env."""
    import subprocess
    run = runner or (lambda cmd: subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=120))

    def refresh() -> None:
        run(["kaggle", "benchmarks", "init", "-y", "--env-file", env_file])
        load_env_file(env_file)
    return refresh


def _normalize_base_url(url: str) -> str:
    """Same normalisation as ``kaggle_benchmarks.kaggle.model_proxy.ModelProxy``: strip a trailing /openapi or /genai."""
    url = url.strip().rstrip("/")
    for suffix in ("/openapi", "/genai"):
        if url.endswith(suffix):
            return url[: -len(suffix)]
    return url


def cost_usd_from_usage(usage: dict[str, Any]) -> float | None:
    """Model-proxy cost is ``usage.cost = {input_tokens_cost_nanodollars, output_tokens_cost_nanodollars}``.

    Unknown stays ``None`` (never 0); a bare numeric ``cost`` is not assumed to be dollars."""
    cost = usage.get("cost")
    if not isinstance(cost, dict):
        return None
    parts = [cost.get("input_tokens_cost_nanodollars"), cost.get("output_tokens_cost_nanodollars")]
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) for v in parts):
        return None
    return (parts[0] + parts[1]) / 1e9


class OpenAICompatBackend:
    """Stateless OpenAI-compatible chat completions; the chat object holds the history and re-sends it."""

    def __init__(self, model_slug: str, base_url: str | None = None, api_key: str | None = None,
                 path: str = "/openapi/chat/completions", timeout: float = 600.0, extra_payload: dict[str, Any] | None = None,
                 post=None, token_param: str = "max_tokens", send_temperature: bool = True, refresher=None,
                 refresh_margin_s: float = 300.0, clock=time.time, adaptive: bool = False):
        self.model_slug = model_slug
        self.base_url = _normalize_base_url(base_url or os.environ.get("MODEL_PROXY_URL") or "")
        self.token_param = token_param
        self.send_temperature = bool(send_temperature)
        self._explicit_key = api_key
        self._refresher, self._margin, self._clock = refresher, float(refresh_margin_s), clock
        self.path = path
        self.timeout = timeout
        self.extra_payload = dict(extra_payload or {})
        self._post = post or self._http_post
        self.adaptive = bool(adaptive)             # unknown model: learn temperature / token-parameter support from a 400
        self.adaptations: list[str] = []
        self._auth_lock = threading.Lock()
        self._last_reactive_refresh = 0.0
        if not self.base_url and post is None:
            raise ProviderError("MODEL_PROXY_URL is not set", kind="config")

    @property
    def api_key(self) -> str:
        """An explicit key wins; otherwise the current ``MODEL_PROXY_API_KEY``, refreshed before it expires (a run outlives the
        one-hour key). The refresh is proactive, never a retry of a failed request."""
        if self._explicit_key:
            return self._explicit_key
        if self._refresher is not None:
            expiry = _expiry_epoch(os.environ.get("MODEL_PROXY_EXPIRY_TIME"))
            if expiry is not None and expiry - self._clock() < self._margin:
                self._refresher()
        return os.environ.get("MODEL_PROXY_API_KEY") or ""

    def _http_post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            return self._http_post_once(url, payload)
        except ProviderError as exc:
            # A 401 means the key expired (or was rotated) between the proactive check and this request. The request was not
            # processed, so renewing the key and sending it again cannot double-spend. At most once a minute, never in a loop.
            if exc.status == 401 and self._refresher is not None and not self._explicit_key:
                with self._auth_lock:
                    if self._clock() - self._last_reactive_refresh > 60.0:
                        self._last_reactive_refresh = self._clock()
                        self._refresher()
                        return self._http_post_once(url, payload)
            raise

    def _http_post_once(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            kind = "auth" if exc.code in (401, 403) else "rate_limit" if exc.code == 429 else "infra"
            raise ProviderError(f"HTTP {exc.code}: {body}", kind=kind, status=exc.code) from exc
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as exc:
            raise ProviderError(f"{type(exc).__name__}: {exc}", kind="infra") from exc

    def _adapt(self, exc: "ProviderError", payload: dict[str, Any]) -> dict[str, Any] | None:
        """A 400 naming ``temperature`` or ``max_tokens`` means this model does not accept it: stop sending / switch the parameter
        for the rest of the run and return the corrected payload (None when the error is something else)."""
        if exc.status != 400:
            return None
        msg = str(exc).lower()
        fixed = dict(payload)
        if "temperature" in msg and "temperature" in fixed:
            self.send_temperature = False
            fixed.pop("temperature")
            self.adaptations.append("temperature_not_sent")
            return fixed
        if "max_completion_tokens" in msg and "max_tokens" in fixed:
            self.token_param = "max_completion_tokens"
            fixed["max_completion_tokens"] = fixed.pop("max_tokens")
            self.adaptations.append("token_param=max_completion_tokens")
            return fixed
        return None

    def open_chat(self, system: str | None) -> "_StatelessChat":
        return _StatelessChat(self, system)


def _openai_content(parts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for p in parts:
        if p["type"] == "text":
            out.append({"type": "text", "text": p["text"]})
        else:
            out.append({"type": "image_url", "image_url": {"url": "data:image/png;base64," + p["png_b64"]}})
    return out


class _StatelessChat:
    def __init__(self, backend: OpenAICompatBackend, system: str | None):
        self.backend = backend
        self.history: list[dict[str, Any]] = [{"role": "system", "content": system}] if system else []

    def replay(self, parts, text):
        self.history = self.history + [{"role": "user", "content": _openai_content(parts)}, {"role": "assistant", "content": text}]

    def send(self, parts, max_tokens, temperature):
        b = self.backend
        messages = self.history + [{"role": "user", "content": _openai_content(parts)}]
        payload = {"model": b.model_slug, "messages": messages, b.token_param: int(max_tokens), "n": 1, "stream": False}
        if b.send_temperature:
            payload["temperature"] = float(temperature)
        payload.update(b.extra_payload)
        started = time.perf_counter()
        try:
            resp = b._post(b.base_url + b.path, payload)
        except ProviderError as exc:
            fix = b._adapt(exc, payload) if b.adaptive else None
            if fix is None:
                raise
            resp = b._post(b.base_url + b.path, fix)            # the rejected request produced no answer and no charge
        choice = (resp.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        text = message.get("content")
        usage = resp.get("usage") or {}
        refused = bool(message.get("refusal")) or choice.get("finish_reason") == "content_filter"
        if not isinstance(text, str):
            text = ""
            refused = refused or bool(message.get("refusal"))
        self.history = messages + [{"role": "assistant", "content": text}]
        finish = choice.get("finish_reason")
        return BackendReply(text=text, prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
                            cost_usd=cost_usd_from_usage(usage), finish_reason=finish, refused=refused,
                            max_tokens_enforced=True if finish == "length" else None,
                            raw={"latency_s": time.perf_counter() - started, "backend_latency_ms": usage.get("total_backend_latency_ms"),
                                 "temperature_sent": b.send_temperature, "token_param": b.token_param})


class KaggleReply:
    """One instance per image (a second image on the same instance is refused)."""

    def __init__(self, backend: ChatBackend, image_id: int | None = None, usage_sink=None, journal=None):
        self.backend = backend
        self.image_id = image_id
        self._sync = TranscriptSync()
        self._chat: ChatHandle | None = None
        self.usage: list[dict[str, Any]] = []
        self._sink = usage_sink
        self._journal = journal                  # optional TurnJournal: replay answered turns, store new ones

    def __call__(self, model, processor, messages, **kwargs):
        system, tail = self._sync.new_user_messages(messages)      # raises HistoryMismatch: fail closed, no request is made
        if self._chat is None:
            self._chat = self.backend.open_chat(system)
        parts = [p for m in tail for p in to_parts(m)]
        max_tokens = int(kwargs.get("max_new_tokens", 2048))
        temperature = float(kwargs.get("temperature", 0.0))
        turn = len(self.usage)
        key = None
        if self._journal is not None and self.image_id is not None:
            from dental_agent.external.journal import request_key
            key = request_key(self.backend.model_slug, system, parts, max_tokens, temperature)
            stored = self._journal.get(self.image_id, turn)
            if stored is not None:
                if stored["request_sha"] != key:
                    raise HistoryMismatch(f"journal turn {turn} of image {self.image_id} was answered for a different request; "
                                          "refusing to replay it (delete the journal to start this image over)")
                if not hasattr(self._chat, "replay"):
                    raise HistoryMismatch("this chat backend cannot replay journalled turns")
                self._chat.replay(parts, stored["text"])                 # no request, no cost
                self._sync.commit(system, tail, stored["text"])
                self.usage.append({**stored["usage"], "replayed": True})
                return stored["text"], int((stored["usage"] or {}).get("prompt_tokens") or 0), [], None, None
        started = time.perf_counter()
        reply = self._chat.send(parts, max_tokens, temperature)    # ProviderError propagates, nothing is retried
        wall = time.perf_counter() - started
        text = reply.text if isinstance(reply.text, str) else ""
        self._sync.commit(system, tail, text)
        record = {"turn": len(self.usage), "image_id": self.image_id, "model": self.backend.model_slug,
                  "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens,
                  "cost_usd": reply.cost_usd, "finish_reason": reply.finish_reason, "refused": bool(reply.refused),
                  "empty": not text.strip(), "max_tokens": max_tokens, "max_tokens_enforced": reply.max_tokens_enforced,
                  "n_new_messages": len(tail), "wall_seconds": wall,
                  "backend_latency_ms": reply.raw.get("backend_latency_ms"), "temperature_sent": reply.raw.get("temperature_sent")}
        self.usage.append(record)
        if key is not None:
            self._journal.record(self.image_id, turn, key, text, record)     # saved before the loop sees the reply
        if self._sink is not None:
            self._sink(record)
        return text, int(reply.prompt_tokens or 0), [], None, None


__all__ = ["BackendReply", "ChatBackend", "HistoryMismatch", "KaggleReply", "OpenAICompatBackend", "ProviderError"]
