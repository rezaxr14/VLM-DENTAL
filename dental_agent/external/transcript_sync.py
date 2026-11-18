"""Keep a remote chat in step with the agent loop's transcript, and fail closed when they drift.

``run_agent`` hands its reply function the FULL transcript every turn and appends the assistant reply itself afterwards.
A remote chat that already holds the earlier turns must therefore be sent only the NEW user messages. ``TranscriptSync``
remembers a fingerprint of everything already exchanged (user messages sent, assistant texts received) and, on each call,
verifies the transcript still starts with exactly that history before returning the new tail.

Mismatch (an edited assistant text, a changed image, a reordered or dropped message, a changed system prompt, or more than
one image-bearing change) raises ``HistoryMismatch``; the caller must not guess.
"""
from __future__ import annotations

import base64
import hashlib
import io
from typing import Any


class HistoryMismatch(RuntimeError):
    """The loop's transcript no longer extends what the remote chat has seen."""


def image_fingerprint(image: Any) -> str:
    h = hashlib.sha256()
    h.update(f"{getattr(image, 'mode', '?')}|{getattr(image, 'size', '?')}|".encode())
    h.update(image.tobytes())
    return h.hexdigest()


def message_fingerprint(message: dict[str, Any]) -> str:
    h = hashlib.sha256()
    h.update(str(message.get("role")).encode() + b"\0")
    content = message.get("content")
    if isinstance(content, str):
        h.update(b"T" + content.encode("utf-8"))
        return h.hexdigest()
    for part in content or []:
        kind = part.get("type")
        if kind == "text":
            h.update(b"T" + str(part.get("text", "")).encode("utf-8") + b"\0")
        elif kind == "image":
            h.update(b"I" + image_fingerprint(part["image"]).encode() + b"\0")
        else:
            raise HistoryMismatch(f"unsupported message part '{kind}'")
    return h.hexdigest()


def to_parts(message: dict[str, Any]) -> list[dict[str, Any]]:
    """Provider-neutral parts of one user message: ``{'type':'text','text':..}`` / ``{'type':'image','png_b64':..}``."""
    content = message.get("content")
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    parts: list[dict[str, Any]] = []
    for part in content or []:
        if part.get("type") == "text":
            parts.append({"type": "text", "text": str(part.get("text", ""))})
        elif part.get("type") == "image":
            buf = io.BytesIO()
            part["image"].convert("RGB").save(buf, format="PNG")
            parts.append({"type": "image", "png_b64": base64.b64encode(buf.getvalue()).decode()})
        else:
            raise HistoryMismatch(f"unsupported message part '{part.get('type')}'")
    return parts


class TranscriptSync:
    def __init__(self) -> None:
        self._history: list[str] = []      # fingerprints of non-system messages already exchanged, in order
        self._system: str | None = None
        self.calls = 0

    @property
    def exchanged(self) -> int:
        return len(self._history)

    def new_user_messages(self, messages: list[dict[str, Any]]) -> tuple[str | None, list[dict[str, Any]]]:
        """Return ``(system_text, new_user_messages)`` after verifying the prefix. Does not record anything yet."""
        system_msgs = [m for m in messages if m.get("role") == "system"]
        if len(system_msgs) > 1 or (system_msgs and messages[0].get("role") != "system"):
            raise HistoryMismatch("system message must be single and first")
        system = system_msgs[0].get("content") if system_msgs else None
        if self.calls and system != self._system:
            raise HistoryMismatch("system prompt changed between calls")
        body = [m for m in messages if m.get("role") != "system"]
        n = len(self._history)
        if len(body) <= n:
            raise HistoryMismatch(f"transcript has {len(body)} messages, remote chat already holds {n}")
        for i in range(n):
            if message_fingerprint(body[i]) != self._history[i]:
                raise HistoryMismatch(f"message {i} differs from what the remote chat holds")
        tail = body[n:]
        if any(m.get("role") != "user" for m in tail):
            raise HistoryMismatch("new messages must all be user messages (the loop appends the assistant reply itself)")
        return system, tail

    def commit(self, system: str | None, tail: list[dict[str, Any]], assistant_text: str) -> None:
        """Record that ``tail`` was sent and ``assistant_text`` came back."""
        if not self.calls:
            self._system = system
        self._history.extend(message_fingerprint(m) for m in tail)
        self._history.append(message_fingerprint({"role": "assistant", "content": assistant_text}))
        self.calls += 1
