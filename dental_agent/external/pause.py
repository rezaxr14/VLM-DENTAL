"""Pause instead of fail: a spend cap or an expired key stops a run cleanly, and the run is resumed later at no extra cost.

``classify`` turns a provider failure into ``auth`` / ``quota`` / ``rate_limit`` / ``infra``. ``auth`` and ``quota`` stop the
run with ``RunPaused`` (no error record is written for the image in flight: its answered turns are in the journal and its
next turn is simply sent on resume). ``QuotaGuard`` reads ``kaggle benchmarks quota`` before each image so a run stops at an
image boundary when the daily or monthly allowance is nearly spent, and reports when it refills.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from typing import Any, Callable

QUOTA_WORDS = re.compile(r"quota|daily|monthly|budget|exceed|insufficient|credit|spend|limit reached", re.I)


class RunPaused(RuntimeError):
    """The run stopped on purpose and can be resumed. ``kind`` is auth | quota | rate_limit; ``refill_at`` is set for quota."""

    def __init__(self, kind: str, message: str, refill_at: str | None = None):
        super().__init__(message)
        self.kind = kind
        self.refill_at = refill_at


def classify(exc: BaseException) -> str:
    """auth | quota | rate_limit | infra for a ``ProviderError`` (anything else is infra)."""
    status = getattr(exc, "status", None)
    kind = getattr(exc, "kind", "infra")
    text = str(exc)
    if status in (402, 403, 429) and QUOTA_WORDS.search(text):
        return "quota"
    if kind == "auth" or status in (401, 403):
        return "auth"
    if kind == "rate_limit" or status == 429:
        return "rate_limit"
    return "infra"


def _usd(text: Any) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))
    return float(m.group(0)) if m else None


def parse_quota(payload: Any) -> dict[str, dict[str, Any]]:
    """``kaggle benchmarks quota --format json`` -> ``{'Daily': {'remaining': 9.91, 'total': 10.0, 'refill_at': '...'}, ...}``."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    out: dict[str, dict[str, Any]] = {}
    for row in payload:
        period = str(row.get("period"))
        out[period] = {"remaining": _usd(row.get("remaining")), "used": _usd(row.get("used")), "total": _usd(row.get("total")),
                       "refill_at": row.get("refillAt")}
    return out


def kaggle_quota_runner() -> str:
    r = subprocess.run(["kaggle", "benchmarks", "quota", "--format", "json"], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout)[:300])
    out = [ln for ln in r.stdout.splitlines() if not ln.lower().startswith(("warning", "kaggle cli"))]
    return "\n".join(out)


class QuotaGuard:
    """Raises ``RunPaused('quota')`` when any allowance period has less than ``reserve_usd`` left.

    The reserve should cover the most one image can cost (a tools run on a frontier model can reach a few dollars). When the
    quota command is unavailable the guard stays silent and the error classification remains the safety net."""

    def __init__(self, runner: Callable[[], str] | None = None, reserve_usd: float = 1.0, ttl_s: float = 45.0,
                 clock: Callable[[], float] = time.time):
        self._runner = runner or kaggle_quota_runner
        self.reserve_usd, self.ttl_s, self._clock = float(reserve_usd), float(ttl_s), clock
        self._cache: tuple[float, dict[str, dict[str, Any]]] | None = None

    def snapshot(self, force: bool = False) -> dict[str, dict[str, Any]] | None:
        now = self._clock()
        if not force and self._cache and now - self._cache[0] < self.ttl_s:
            return self._cache[1]
        try:
            snap = parse_quota(self._runner())
        except Exception:
            return self._cache[1] if self._cache else None
        self._cache = (now, snap)
        return snap

    def check(self) -> None:
        snap = self.snapshot()
        if not snap:
            return
        low = {p: v for p, v in snap.items() if v.get("remaining") is not None and v["remaining"] < self.reserve_usd}
        if low:
            # the binding period is the one that refills LAST (a spent month outlasts a spent day)
            period, v = max(low.items(), key=lambda kv: str(kv[1].get("refill_at") or ""))
            raise RunPaused("quota", f"{period} allowance has ${v['remaining']:.2f} left (reserve ${self.reserve_usd:.2f})",
                            refill_at=v.get("refill_at"))
