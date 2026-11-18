"""Pause / resume behaviour of the external-model runner, tested offline (scripted HTTP layer, fake clocks, no network, no quota)."""
import json
import threading

import pytest

from dental_agent.external import coordinator, importer, journal, pause
from dental_agent.external.kaggle_provider import OpenAICompatBackend, ProviderError
from dental_agent.external.transcript_sync import HistoryMismatch
from test_external_kaggle import EMPTY, FINAL, TOOL, cohort, settings_for  # noqa: F401  (fixtures and helpers)


def ok(text, prompt=100, completion=5):
    return {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "cost": {"input_tokens_cost_nanodollars": 1000, "output_tokens_cost_nanodollars": 500}}}


class Scripted:
    """The HTTP layer: replays a script of responses / exceptions and remembers every payload it was sent."""

    def __init__(self, script):
        self.script, self.payloads, self.lock = list(script), [], threading.Lock()

    def __call__(self, url, payload):
        with self.lock:
            self.payloads.append(payload)
            item = self.script.pop(0) if len(self.script) > 1 else self.script[0]
        if isinstance(item, Exception):
            raise item
        return item


def backend_for(script, **kw):
    post = Scripted(script)
    return OpenAICompatBackend("fake/model-1", base_url="http://x", api_key="k", post=post, **kw), post


def run(backend, cohort, tmp_path, cond="kaggle_zero_shot", ids=(100, 101, 102), **kw):
    images, _, _ = cohort
    ids = list(ids)
    return coordinator.run_cell(backend=backend, condition=cond, images_df=images, image_ids=ids, out_dir=tmp_path,
                                settings=settings_for(cond, ids), log=lambda s: None, **kw)


# ----------------------------------------------------------------------------- classification

@pytest.mark.parametrize("exc,expected", [
    (ProviderError("HTTP 401: expired", kind="auth", status=401), "auth"),
    (ProviderError("HTTP 403: forbidden", kind="auth", status=403), "auth"),
    (ProviderError("HTTP 429: slow down", kind="rate_limit", status=429), "rate_limit"),
    (ProviderError("HTTP 429: daily quota exceeded", kind="rate_limit", status=429), "quota"),
    (ProviderError("HTTP 403: monthly budget exceeded", kind="auth", status=403), "quota"),
    (ProviderError("HTTP 503: unavailable", kind="infra", status=503), "infra"),
    (ValueError("anything else"), "infra"),
])
def test_classify(exc, expected):
    assert pause.classify(exc) == expected


# ----------------------------------------------------------------------------- journal: pause mid-image, resume for free

def test_auth_error_pauses_and_resume_only_pays_for_the_unanswered_turn(cohort, tmp_path, ):
    first, post1 = backend_for([ok(TOOL), ProviderError("HTTP 401: expired", kind="auth", status=401)])
    stats = run(first, cohort, tmp_path, cond="kaggle_tools", ids=[100])
    assert stats["paused"]["kind"] == "auth" and stats["ran"] == 0 and stats["errors"] == 0
    paths = importer.cell_paths(tmp_path, "fake/model-1", "kaggle_tools")
    assert importer.read_jsonl(paths["transport"]) == []                       # nothing recorded for the interrupted image
    assert len(journal.TurnJournal(paths["journal"])) == 1                     # but its answered turn is saved

    second, post2 = backend_for([ok(FINAL)])
    stats2 = run(second, cohort, tmp_path, cond="kaggle_tools", ids=[100])
    assert stats2["ran"] == 1 and stats2["replayed_turns"] == 1
    assert len(post2.payloads) == 1                                            # only the unanswered turn is sent
    roles = [m["role"] for m in post2.payloads[0]["messages"]]
    assert roles == ["system", "user", "assistant", "user"]                    # the replayed turn is part of the history
    assert post2.payloads[0]["messages"][2]["content"] == TOOL
    rec = importer.read_jsonl(paths["transport"])[0]
    assert not rec.get("error") and [u.get("replayed", False) for u in rec["usage"]] == [True, False]


def test_changed_request_fails_closed_instead_of_replaying(cohort, tmp_path):
    first, _ = backend_for([ok(FINAL)])
    run(first, cohort, tmp_path, ids=[100])
    paths = importer.cell_paths(tmp_path, "fake/model-1", "kaggle_zero_shot")
    # forget the transport record but keep a journal that no longer matches the request
    paths["transport"].unlink()
    rows = [json.loads(line) for line in paths["journal"].read_text().splitlines()]
    rows[0]["request_sha"] = "0" * 64
    paths["journal"].write_text("".join(json.dumps(r) + "\n" for r in rows))
    second, post = backend_for([ok(FINAL)])
    stats = run(second, cohort, tmp_path, ids=[100])
    rec = importer.read_jsonl(paths["transport"])[0]
    assert rec["error"]["type"] == "HistoryMismatch" and not post.payloads and stats["errors"] == 1


# ----------------------------------------------------------------------------- quota guard and quota errors

QUOTA_JSON = json.dumps([
    {"period": "Daily", "used": "$9.50", "remaining": "$0.50", "total": "$10.00", "refillAt": "2026-11-15T08:48:59.524000"},
    {"period": "Monthly", "used": "$20.00", "remaining": "$80.00", "total": "$100.00", "refillAt": "2026-12-14T08:48:59.523000"}])


def test_quota_guard_pauses_before_any_request(cohort, tmp_path):
    backend, post = backend_for([ok(FINAL)])
    stats = run(backend, cohort, tmp_path, guard=pause.QuotaGuard(runner=lambda: QUOTA_JSON, reserve_usd=1.0))
    assert not post.payloads and stats["ran"] == 0
    assert stats["paused"]["kind"] == "quota" and stats["paused"]["refill_at"] == "2026-11-15T08:48:59.524000"
    assert "paused" in stats["stopped"]


def test_quota_guard_binding_period_is_the_one_that_refills_last():
    both_low = json.loads(QUOTA_JSON)
    both_low[1]["remaining"] = "$0.10"
    with pytest.raises(pause.RunPaused) as e:
        pause.QuotaGuard(runner=lambda: json.dumps(both_low), reserve_usd=1.0).check()
    assert e.value.refill_at.startswith("2026-12-14")


def test_quota_guard_is_silent_when_the_command_fails_or_there_is_room():
    def boom():
        raise RuntimeError("no cli")
    pause.QuotaGuard(runner=boom).check()
    pause.QuotaGuard(runner=lambda: QUOTA_JSON, reserve_usd=0.25).check()


def test_quota_guard_caches_for_its_ttl():
    calls, now = [], [0.0]
    g = pause.QuotaGuard(runner=lambda: calls.append(1) or QUOTA_JSON, reserve_usd=0.25, ttl_s=60, clock=lambda: now[0])
    g.check(); g.check()
    now[0] = 61.0
    g.check()
    assert len(calls) == 2


def test_provider_quota_error_pauses_without_an_error_record(cohort, tmp_path):
    backend, _ = backend_for([ProviderError("HTTP 429: daily quota exceeded", kind="rate_limit", status=429)])
    stats = run(backend, cohort, tmp_path)
    paths = importer.cell_paths(tmp_path, "fake/model-1", "kaggle_zero_shot")
    assert stats["paused"]["kind"] == "quota" and importer.read_jsonl(paths["transport"]) == []


# ----------------------------------------------------------------------------- rate limits, parallel workers

def test_rate_limit_backs_off_and_retries_the_same_image(cohort, tmp_path):
    backend, post = backend_for([ProviderError("HTTP 429: slow down", kind="rate_limit", status=429), ok(FINAL)])
    waits = []
    stats = run(backend, cohort, tmp_path, ids=[100], sleep=waits.append)
    assert waits == [30.0] and stats["ran"] == 1 and stats["errors"] == 0 and stats["paused"] is None


def test_persistent_rate_limit_pauses(cohort, tmp_path):
    backend, _ = backend_for([ProviderError("HTTP 429: slow down", kind="rate_limit", status=429)])
    stats = run(backend, cohort, tmp_path, ids=[100], sleep=lambda s: None, rate_limit_backoff=(1.0, 2.0))
    assert stats["paused"]["kind"] == "rate_limit" and stats["ran"] == 0


def test_parallel_workers_complete_every_image_once(cohort, tmp_path):
    backend, post = backend_for([ok(FINAL)])
    stats = run(backend, cohort, tmp_path, workers=3)
    paths = importer.cell_paths(tmp_path, "fake/model-1", "kaggle_zero_shot")
    rows = importer.read_jsonl(paths["transport"])
    assert stats["ran"] == 3 and sorted(r["image_id"] for r in rows) == [100, 101, 102] and len(post.payloads) == 3


# ----------------------------------------------------------------------------- key renewal and model quirks

def test_expired_key_is_renewed_once_and_the_request_resent():
    renewed = []
    b = OpenAICompatBackend("fake/model-1", base_url="http://x", refresher=lambda: renewed.append(1), clock=lambda: 1000.0)
    calls = []

    def once(url, payload):
        calls.append(1)
        if len(calls) == 1:
            raise ProviderError("HTTP 401: expired", kind="auth", status=401)
        return ok("hi")
    b._http_post_once = once
    assert b._http_post("http://x/v", {})["choices"][0]["message"]["content"] == "hi"
    assert renewed == [1] and len(calls) == 2


def test_explicit_key_is_never_renewed():
    b = OpenAICompatBackend("fake/model-1", base_url="http://x", api_key="k", refresher=lambda: pytest.fail("must not refresh"))
    b._http_post_once = lambda url, payload: (_ for _ in ()).throw(ProviderError("HTTP 401", kind="auth", status=401))
    with pytest.raises(ProviderError):
        b._http_post("http://x/v", {})


def test_adaptive_backend_learns_temperature_and_token_param_from_a_400():
    b, post = backend_for([ProviderError("HTTP 400: Unsupported value: 'temperature' does not support 0.0", status=400), ok("a"), ok("b")], adaptive=True)
    chat = b.open_chat(None)
    r = chat.send([{"type": "text", "text": "x"}], 10, 0.0)
    assert r.text == "a" and r.raw["temperature_sent"] is False and "temperature" not in post.payloads[1]
    assert b.adaptations == ["temperature_not_sent"]
    chat.send([{"type": "text", "text": "y"}], 10, 0.0)
    assert "temperature" not in post.payloads[2]

    b2, post2 = backend_for([ProviderError("HTTP 400: Use 'max_completion_tokens' instead of max_tokens", status=400), ok("a")], adaptive=True)
    b2.open_chat(None).send([{"type": "text", "text": "x"}], 10, 0.0)
    assert post2.payloads[1].get("max_completion_tokens") == 10 and "max_tokens" not in post2.payloads[1]


def test_non_adaptive_backend_does_not_swallow_a_400():
    b, _ = backend_for([ProviderError("HTTP 400: temperature unsupported", status=400)])
    with pytest.raises(ProviderError):
        b.open_chat(None).send([{"type": "text", "text": "x"}], 10, 0.0)


def test_misconfigured_endpoint_is_raised_not_recorded(cohort, tmp_path):
    backend, _ = backend_for([ProviderError("MODEL_PROXY_URL is not set", kind="config")])
    with pytest.raises(ProviderError):
        run(backend, cohort, tmp_path, ids=[100])
