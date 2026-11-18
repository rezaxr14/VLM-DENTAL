"""Verified model profiles for Kaggle's model proxy (probe v7, 2026-11-14, interactive session, ``mp-staging`` proxy).

``kaggle benchmarks tasks models`` lists 37 models, but the proxy this key can call serves only the ones below (everything
else answered "model not found" under every slug form). A model that is not in ``PROFILES`` is never run silently: the
settings here (token-limit parameter, whether ``temperature`` is accepted, prices) were each measured, not assumed.

Measured facts that shape the benchmark (see docs/KAGGLE_EXTERNAL_MODELS.md):
* ``claude-sonnet-5`` rejects ``temperature``: its decoding is the provider default, not greedy (recorded in the run identity).
* Gemini 3.6 Flash is not repeatable at temperature 0 (hidden thinking); the two flash-lite models were.
* Gemini and Claude/GPT see images differently: Gemini spends a fixed ~1.1k tokens per image at any size, GPT and Claude scale
  with pixels (2048x1024: 2466 and 2753 tokens). The proxy ignored ``detail`` and ``media_resolution``.
* ``completion_tokens`` includes hidden reasoning tokens, so ``usage.cost`` is the true spend.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dental_agent.external.cost_guard import Price
from dental_agent.external.kaggle_provider import OpenAICompatBackend

VERIFIED_ON = "2026-11-14 probe v7"


class UnknownModel(KeyError):
    pass


@dataclass(frozen=True)
class ModelProfile:
    slug: str                       # exactly what the proxy accepts
    provider: str
    usd_per_mtok_in: float
    usd_per_mtok_out: float
    token_param: str = "max_tokens"
    send_temperature: bool = True   # False: the proxy/model rejects temperature, so decoding is the provider default
    hidden_reasoning: bool = False  # reasoning tokens are generated and billed by default
    image_tokens: str = "scales"    # "fixed" (about 1.1k at any size) or "scales" with pixels
    note: str = ""

    @property
    def price(self) -> Price:
        return Price(self.usd_per_mtok_in, self.usd_per_mtok_out)

    def decoding_identity(self) -> dict[str, Any]:
        """What changes the meaning of this model's numbers; stored in the run identity."""
        return {"profile_verified": VERIFIED_ON, "token_param": self.token_param, "temperature_sent": self.send_temperature,
                "hidden_reasoning_default": self.hidden_reasoning, "image_tokens": self.image_tokens}


PROFILES: dict[str, ModelProfile] = {p.slug: p for p in [
    ModelProfile("anthropic/claude-sonnet-5@default", "anthropic", 2.00, 10.00, send_temperature=False,
                 note="rejects temperature; not greedy"),
    ModelProfile("gemini-3-flash-preview", "google", 0.50, 3.00, hidden_reasoning=True, image_tokens="fixed"),
    ModelProfile("gemini-3.1-flash-lite-preview", "google", 0.25, 1.50, image_tokens="fixed"),
    ModelProfile("gemini-3.5-flash-lite", "google", 0.30, 2.50, image_tokens="fixed"),
    ModelProfile("gemini-3.6-flash", "google", 0.75, 3.75, hidden_reasoning=True, image_tokens="fixed",
                 note="not repeatable at temperature 0"),
    ModelProfile("gpt-5.4-nano-2026-03-17", "openai", 0.20, 1.25,
                 note="accepts temperature only while reasoning_effort is unset"),
]}

# served by the proxy but cannot read images (kept so a roster listing explains itself)
TEXT_ONLY = ("gpt-oss-120b", "ibm/granite-4.0-h-small")


def get_profile(slug: str) -> ModelProfile:
    try:
        return PROFILES[slug]
    except KeyError:
        known = ", ".join(sorted(PROFILES))
        raise UnknownModel(f"{slug!r} has no verified profile (text-only: {', '.join(TEXT_ONLY)}). Verified: {known}") from None


def backend_for(slug: str, **kwargs) -> OpenAICompatBackend:
    """Backend with the model's measured token parameter and temperature policy (``kwargs`` go to ``OpenAICompatBackend``)."""
    p = get_profile(slug)
    kwargs.setdefault("token_param", p.token_param)
    kwargs.setdefault("send_temperature", p.send_temperature)
    return OpenAICompatBackend(p.slug, **kwargs)


def as_roster():
    from dental_agent.external.roster import RosterEntry
    return [RosterEntry(p.slug, vision=True, source="profile:" + VERIFIED_ON, note=p.note or None) for p in PROFILES.values()]


# --------------------------------------------------------------------------------------------------------------------
# Hosted (non-live) runs: ``kaggle benchmarks tasks run`` executes the task server-side against the FULL catalogue.
# Probe v9 (2026-11-14) scheduled every catalogue model on its own and asked each to read a two-colour image; this table is
# that outcome. Inside a hosted run the proxy slug is provider-prefixed (``raw_slug``); the CLI slug (``cli_slug``) is what
# ``tasks run -m`` takes. Prices are $ per million tokens, derived from ONE tiny request's ``usage.cost`` (indicative only;
# the run ledger records the real per-request cost). Interactive keys reach only ``PROFILES`` above.
# --------------------------------------------------------------------------------------------------------------------
HOSTED_VERIFIED_ON = "2026-11-14 probe v9"
# decoding policy for hosted runs: temperature 0 is requested and dropped for a model that rejects it, the token parameter is
# switched for a model that wants max_completion_tokens; both are learned from the provider's 400 and recorded per request
HOSTED_DECODING = {"profile_verified": HOSTED_VERIFIED_ON, "mode": "adaptive", "temperature_requested": 0.0}


@dataclass(frozen=True)
class HostedModel:
    cli_slug: str                   # ``kaggle benchmarks tasks run -m`` / ``tasks models``
    raw_slug: str                   # the slug the proxy accepts INSIDE a hosted run (``llm.model``)
    vision: str                     # ok | unreliable (answered but wrong on the probe) | text_only | unserved (listed, 404)
    usd_per_mtok_in: float | None
    usd_per_mtok_out: float | None

    @property
    def price(self) -> Price | None:
        if self.usd_per_mtok_in is None or self.usd_per_mtok_out is None:
            return None
        return Price(self.usd_per_mtok_in, self.usd_per_mtok_out)


HOSTED_CATALOG: tuple[HostedModel, ...] = (
    HostedModel("gemini-3.8-flash", "google/gemini-3.8-flash", "ok", 0.75, 3.75),
    HostedModel("claude-haiku-4-5-20251001", "anthropic/claude-haiku-4-5@20251001", "ok", 1.0, 5.0),
    HostedModel("claude-haiku-5-5", "anthropic/claude-haiku-5-5", "ok", 0.1, 0.5),
    HostedModel("claude-opus-4-5-20251101", "anthropic/claude-opus-4-5@20251101", "ok", 5.0, 25.0),
    HostedModel("claude-opus-4-6-default", "anthropic/claude-opus-4-6@default", "ok", 5.0, 25.0),
    HostedModel("claude-opus-4-7-default", "anthropic/claude-opus-4-7@default", "ok", 5.0, 25.0),
    HostedModel("claude-opus-4-8-default", "anthropic/claude-opus-4-8@default", "ok", 5.0, 25.0),
    HostedModel("claude-opus-5-default", "anthropic/claude-opus-5@default", "ok", 5.0, 25.0),
    HostedModel("claude-opus-5-5-default", "anthropic/claude-opus-5-5@default", "ok", 4.0, 20.0),
    HostedModel("claude-sonnet-4-6-default", "anthropic/claude-sonnet-4-6@default", "ok", 3.0, 15.0),
    HostedModel("claude-sonnet-5-default", "anthropic/claude-sonnet-5@default", "ok", 2.0, 10.0),
    HostedModel("claude-sonnet-5-5-default", "anthropic/claude-sonnet-5-5@default", "ok", 2.0, 10.0),
    HostedModel("gemini-2.5-flash", "google/gemini-2.5-flash", "ok", 0.3, 2.5),
    HostedModel("gemini-2.5-pro", "google/gemini-2.5-pro", "ok", 1.25, 10.0),
    HostedModel("gemini-3-flash-preview", "google/gemini-3-flash-preview", "ok", 0.5, 3.0),
    HostedModel("gemini-3.1-flash-lite-preview", "google/gemini-3.1-flash-lite-preview", "ok", 0.25, 1.5),
    HostedModel("gemini-3.1-pro-preview", "google/gemini-3.1-pro-preview", "ok", 2.0, 12.0),
    HostedModel("gemini-3.5-flash-lite", "google/gemini-3.5-flash-lite", "ok", 0.3, 2.5),
    HostedModel("gemini-3.6-flash", "google/gemini-3.6-flash", "ok", 0.75, 3.75),
    HostedModel("gemini-4-argon", "google/gemini-4-argon", "ok", 2.0, 10.0),
    HostedModel("gemma-4-26b-a4b-it", "google/gemma-4-26b-a4b", "ok", 0.15, 0.6),
    HostedModel("gemma-4-31b-it", "google/gemma-4-31b", "ok", 0.15, 0.6),
    HostedModel("gpt-5.4-2026-03-05", "openai/gpt-5.4-2026-03-05", "ok", 2.5, 15.0),
    HostedModel("gpt-5.4-mini-2026-03-17", "openai/gpt-5.4-mini-2026-03-17", "ok", 0.75, 4.5),
    HostedModel("gpt-5.4-nano-2026-03-17", "openai/gpt-5.4-nano-2026-03-17", "ok", 0.2, 1.25),
    HostedModel("gpt-5.5-2026-04-23", "openai/gpt-5.5-2026-04-23", "ok", 5.0, 30.0),
    HostedModel("gpt-5.6-luna", "openai/gpt-5.6-luna", "ok", 0.2, 1.2),
    HostedModel("gpt-5.6-sol", "openai/gpt-5.6-sol", "ok", 4.0, 20.0),
    HostedModel("gpt-5.6-terra", "openai/gpt-5.6-terra", "ok", 2.0, 12.0),
    HostedModel("gpt-6-astra", "openai/gpt-6-astra", "ok", 10.0, 50.0),
    HostedModel("gpt-6-luna", "openai/gpt-6-luna", "unreliable", 0.1, 0.5),
    HostedModel("gpt-6-sol", "openai/gpt-6-sol", "ok", 2.0, 10.0),
    HostedModel("gpt-6.1-sol", "openai/gpt-6.1-sol", "ok", 2.0, 10.0),
    HostedModel("gpt-oss-120b", "openai/gpt-oss-120b", "text_only", None, None),
    HostedModel("grok-4.5-0708", "xai/grok-4.5-0708", "unserved", None, None),
    HostedModel("grok-4.6", "xai/grok-4.6", "ok", 2.0, 6.0),
    HostedModel("grok-4.7", "xai/grok-4.7", "ok", 1.302, 6.0),
)
_HOSTED = {m.cli_slug: m for m in HOSTED_CATALOG}


def hosted_model(cli_slug: str) -> HostedModel:
    try:
        return _HOSTED[cli_slug]
    except KeyError:
        raise UnknownModel(f"{cli_slug!r} is not in the hosted catalogue verified on {HOSTED_VERIFIED_ON}") from None


def hosted_vision_models(include_unreliable: bool = False) -> list[str]:
    """CLI slugs of the hosted models that read images (``unreliable`` only on request)."""
    ok = {"ok", "unreliable"} if include_unreliable else {"ok"}
    return [m.cli_slug for m in HOSTED_CATALOG if m.vision in ok]
