"""Kaggle / external-model integration, tested offline with scripted fake backends (no network, no Kaggle SDK)."""
import io
import json
import os

import pandas as pd
import pytest
from PIL import Image

from dental_agent.external import conditions as C
from dental_agent.external import coordinator, cost_guard, identity, importer, preflight, records, roster, tables
from dental_agent.external.kaggle_provider import BackendReply, KaggleReply, OpenAICompatBackend, ProviderError
from dental_agent.external.transcript_sync import HistoryMismatch, TranscriptSync
from dental_agent.tools.synthetic import make_synthetic_dental_image

FINAL = json.dumps({"thought": "x", "final_answer": [{"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries", "confidence": 0.9}]})
EMPTY = json.dumps({"thought": "healthy", "final_answer": []})
TOOL = json.dumps({"thought": "zoom", "tool_calls": [{"tool": "zoom_crop", "args": {"bbox": [100, 100, 80, 80]}}]})


class FakeChat:
    def __init__(self, replies, log):
        self.replies, self.log, self.i = list(replies), log, 0

    def send(self, parts, max_tokens, temperature):
        self.log.append({"parts": parts, "max_tokens": max_tokens, "temperature": temperature})
        r = self.replies[self.i]
        self.i += 1
        if isinstance(r, Exception):
            raise r
        return BackendReply(text=r, prompt_tokens=10 * self.i, completion_tokens=5, cost_usd=None, finish_reason="stop")


class FakeBackend:
    model_slug = "fake/model-1"

    def __init__(self, per_chat):
        self.per_chat, self.chats, self.systems, self.sent = list(per_chat), 0, [], []

    def open_chat(self, system):
        self.systems.append(system)
        replies = self.per_chat[min(self.chats, len(self.per_chat) - 1)]
        self.chats += 1
        log = []
        self.sent.append(log)
        return FakeChat(replies, log)


@pytest.fixture
def harness_rules(monkeypatch):
    """The harness path runs against the real rule bindings (the check is Python-version independent since ``_stable_dump``)."""
    from dental_agent.harness import rules
    rules.validate_implementations()


@pytest.fixture
def cohort(tmp_path):
    rows = []
    for i in (100, 101, 102):
        p = tmp_path / f"{i}.png"
        make_synthetic_dental_image(width=1024, height=512, seed=i).save(p)
        rows.append({"id": i, "local_path": str(p)})
    images = pd.DataFrame(rows)
    annots = pd.DataFrame([
        {"id": 1, "image_id": 100, "category_id_1": 1, "category_id_2": 6, "category_id_3": 1, "bbox": [500, 200, 80, 100]},
        {"id": 2, "image_id": 101, "category_id_1": 3, "category_id_2": 8, "category_id_3": 4, "bbox": [800, 350, 90, 110]},
    ])
    cats = pd.DataFrame([{"id": 1, "name": "Caries"}, {"id": 2, "name": "Deep Caries"}, {"id": 3, "name": "Periapical Lesion"}, {"id": 4, "name": "Impacted Tooth"}])
    return images, annots, cats


def settings_for(cond, ids, slug="fake/model-1"):
    return identity.run_settings(C.get_condition(cond), slug, cohort=identity.cohort_sha(ids), split="test", grounder_sha256="g" * 64)


# ----------------------------------------------------------------------------- transcript sync

def msg(role, *parts):
    return {"role": role, "content": [p if isinstance(p, dict) else {"type": "text", "text": p} for p in parts] if role == "user" else parts[0]}


def img(color):
    return {"type": "image", "image": Image.new("RGB", (8, 8), color)}


def test_sync_sends_only_new_user_messages():
    s = TranscriptSync()
    sysm = {"role": "system", "content": "S"}
    m1 = msg("user", img((1, 2, 3)), "hello")
    system, tail = s.new_user_messages([sysm, m1])
    assert system == "S" and tail == [m1]
    s.commit(system, tail, "A1")
    m2 = msg("user", "obs")
    _, tail2 = s.new_user_messages([sysm, m1, {"role": "assistant", "content": "A1"}, m2])
    assert tail2 == [m2]


@pytest.mark.parametrize("mutate", ["assistant", "image", "system", "assistant_in_tail", "no_new", "reorder"])
def test_sync_fails_closed(mutate):
    s = TranscriptSync()
    sysm = {"role": "system", "content": "S"}
    m1 = msg("user", img((1, 2, 3)), "hello")
    s.commit("S", [m1], "A1")
    a1 = {"role": "assistant", "content": "A1"}
    m2 = msg("user", "obs")
    cases = {
        "assistant": [sysm, m1, {"role": "assistant", "content": "A1 edited"}, m2],
        "image": [sysm, msg("user", img((9, 9, 9)), "hello"), a1, m2],
        "system": [{"role": "system", "content": "other"}, m1, a1, m2],
        "assistant_in_tail": [sysm, m1, a1, m2, {"role": "assistant", "content": "x"}],
        "no_new": [sysm, m1, a1],
        "reorder": [sysm, a1, m1, m2],
    }
    with pytest.raises(HistoryMismatch):
        s.new_user_messages(cases[mutate])


# ----------------------------------------------------------------------------- reply_fn contract

def test_kaggle_reply_contract_and_one_chat(cohort):
    images, _, _ = cohort
    backend = FakeBackend([[FINAL]])
    reply = KaggleReply(backend, image_id=100)
    from dental_agent.agent.loop import run_agent_no_tools
    traj = run_agent_no_tools(100, images, None, None, verbose=False, canonical_resize=True, reply_fn=reply, max_new_tokens=777)
    assert traj.format_ok and isinstance(traj.final_answer, list)
    assert backend.chats == 1 and backend.sent[0][0]["max_tokens"] == 777 and backend.sent[0][0]["temperature"] == 0.0
    kinds = [p["type"] for p in backend.sent[0][0]["parts"]]
    assert kinds == ["image", "text"]
    assert traj.assistant_token_spans[0]["token_ids"] == []          # never invented
    assert reply.usage[0]["prompt_tokens"] == 10 and reply.usage[0]["cost_usd"] is None
    # reuse for another image / replay of the same history is refused
    with pytest.raises(HistoryMismatch):
        run_agent_no_tools(101, images, None, None, verbose=False, canonical_resize=True, reply_fn=reply)


def test_no_tools_default_path_unchanged(cohort, monkeypatch):
    images, _, _ = cohort
    import dental_agent.agent.loop as loop
    calls = []

    def fake(model, processor, messages, **kw):
        calls.append(kw)
        return FINAL, 5, [1, 2]
    monkeypatch.setattr(loop, "generate_agent_reply", fake)
    traj = loop.run_agent_no_tools(100, images, "m", "p", verbose=False)
    assert calls == [{"max_new_tokens": 2048, "temperature": 0.0, "return_ids": True}]
    assert traj.assistant_token_spans[0]["token_ids"] == [1, 2]


def test_multi_turn_tools_send_only_new_content(cohort):
    images, _, _ = cohort
    backend = FakeBackend([[TOOL, FINAL]])
    reply = KaggleReply(backend, image_id=100)
    from dental_agent.agent.loop import run_agent
    from dental_agent.utils.canonical import set_letterbox
    set_letterbox(False)
    traj = run_agent(100, images, None, None, canonical_resize=True, reply_fn=reply, verbose=False)
    assert traj.final_answer and traj.tool_calls == 1
    first, second = backend.sent[0]
    assert [p["type"] for p in first["parts"]] == ["image", "text"]
    assert [p["type"] for p in second["parts"]] == ["image", "text"]          # the zoom crop + "Result of zoom_crop:" only
    assert "Result of zoom_crop" in second["parts"][1]["text"]
    assert "FIRST" not in json.dumps(second)[:0]
    assert backend.systems[0] and "zoom_crop" in backend.systems[0]


def test_harness_changes_system_prompt_only(cohort, harness_rules):
    images, _, _ = cohort
    from dental_agent.agent.loop import run_agent
    from dental_agent.harness.configuration import harness_options
    systems = []
    for harness in (None, harness_options("rules_skills_index")):
        backend = FakeBackend([[FINAL]])
        run_agent(100, images, None, None, canonical_resize=True, reply_fn=KaggleReply(backend), harness=harness, verbose=False)
        systems.append(backend.systems[0])
    assert len(systems[1]) > len(systems[0]) and systems[1].startswith(systems[0][:200])


def test_openai_compat_backend_resends_history_and_keeps_unknown_usage():
    seen = []

    def post(url, payload):
        seen.append(json.loads(json.dumps(payload)))
        n = len(seen)
        return {"choices": [{"message": {"content": f"r{n}"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 7} if n == 1 else {}}

    b = OpenAICompatBackend("m/x", base_url="http://proxy", api_key="k", post=post)
    chat = b.open_chat("SYS")
    r1 = chat.send([{"type": "text", "text": "a"}], 100, 0.0)
    r2 = chat.send([{"type": "text", "text": "b"}], 100, 0.0)
    assert [m["role"] for m in seen[1]["messages"]] == ["system", "user", "assistant", "user"]
    assert r1.prompt_tokens == 7 and r2.prompt_tokens is None and r2.cost_usd is None
    assert seen[0]["max_tokens"] == 100 and seen[0]["temperature"] == 0.0


def test_proxy_backend_verified_route_usage_and_cost():
    """Shapes taken from the live probe (2026-11-14): /openapi route, nanodollar cost, backend latency."""
    seen = []

    def post(url, payload):
        seen.append((url, json.loads(json.dumps(payload))))
        return {"choices": [{"message": {"content": "1, 2", "refusal": ""}, "finish_reason": "length"}],
                "usage": {"completion_tokens": 4, "prompt_tokens": 14, "total_tokens": 18, "total_backend_latency_ms": 486,
                          "cost": {"input_tokens_cost_nanodollars": 3500, "output_tokens_cost_nanodollars": 6000}}}

    b = OpenAICompatBackend("gemini-3.1-flash-lite-preview", base_url="https://proxy.example/models/", api_key="k", post=post)
    r = b.open_chat(None).send([{"type": "text", "text": "x"}], 8, 0.0)
    assert seen[0][0] == "https://proxy.example/models/openapi/chat/completions"
    assert r.cost_usd == pytest.approx(9.5e-6) and r.prompt_tokens == 14 and r.completion_tokens == 4
    assert r.finish_reason == "length" and r.max_tokens_enforced is True and r.raw["backend_latency_ms"] == 486


def test_proxy_backend_base_url_suffix_and_overrides():
    assert OpenAICompatBackend("m", base_url="http://p/models/openapi", post=lambda *a: {}).base_url == "http://p/models"
    assert OpenAICompatBackend("m", base_url="http://p/models/genai/", post=lambda *a: {}).base_url == "http://p/models"
    seen = []
    b = OpenAICompatBackend("m", base_url="http://p", post=lambda u, p: seen.append(p) or {"choices": [{"message": {"content": "x"}}]},
                            token_param="max_completion_tokens", send_temperature=False)
    r = b.open_chat(None).send([{"type": "text", "text": "x"}], 50, 0.0)
    assert seen[0]["max_completion_tokens"] == 50 and "max_tokens" not in seen[0] and "temperature" not in seen[0]
    assert r.raw["temperature_sent"] is False and r.cost_usd is None and r.max_tokens_enforced is None


def test_cost_unknown_is_never_zero():
    from dental_agent.external.kaggle_provider import cost_usd_from_usage
    assert cost_usd_from_usage({}) is None
    assert cost_usd_from_usage({"cost": 0.5}) is None                         # bare number: unit unknown
    assert cost_usd_from_usage({"cost": {"input_tokens_cost_nanodollars": 1000}}) is None
    assert cost_usd_from_usage({"cost": {"input_tokens_cost_nanodollars": 0, "output_tokens_cost_nanodollars": 0}}) == 0.0


def test_proxy_backend_requires_url(monkeypatch):
    monkeypatch.delenv("MODEL_PROXY_URL", raising=False)
    with pytest.raises(ProviderError):
        OpenAICompatBackend("m")


# ----------------------------------------------------------------------------- coordinator, resume, errors

def test_run_cell_resume_error_and_import(cohort, tmp_path):
    images, annots, cats = cohort
    ids = [100, 101, 102]
    st = settings_for("kaggle_zero_shot", ids)
    gt_hit = FINAL
    backend = FakeBackend([[gt_hit], [ProviderError("boom 503", status=503)], [EMPTY]])
    stats = coordinator.run_cell(backend=backend, condition="kaggle_zero_shot", images_df=images, image_ids=ids, out_dir=tmp_path, settings=st, log=lambda s: None)
    assert (stats["ran"], stats["errors"]) == (2, 1)
    # resume: only image 101 is re-run
    backend2 = FakeBackend([[EMPTY]])
    stats2 = coordinator.run_cell(backend=backend2, condition="kaggle_zero_shot", images_df=images, image_ids=ids, out_dir=tmp_path, settings=st, log=lambda s: None)
    assert (stats2["done"], stats2["ran"]) == (2, 1) and backend2.chats == 1
    paths = importer.cell_paths(tmp_path, backend.model_slug, "kaggle_zero_shot")
    res = importer.import_transport(paths["transport"], annots, cats, settings=st, cohort_ids=set(ids), evaluated_path=paths["evaluated"])
    assert len(res["records"]) == 3 and not res["pending"] and not res["errors"]
    # the evaluated records go straight into the project's matrix code
    from dental_agent.evaluation.matrix import build_summary, load_records
    loaded = load_records(paths["evaluated"])
    summary = build_summary({"kaggle_zero_shot@canonical": loaded}, n_boot=50)
    assert summary["cells"]["kaggle_zero_shot@canonical"]["n_images"] == 3


def test_consecutive_errors_stop_cell(cohort, tmp_path):
    images, _, _ = cohort
    ids = [100, 101, 102]
    st = settings_for("kaggle_zero_shot", ids)
    backend = FakeBackend([[ProviderError("HTTP 503: unavailable", kind="infra", status=503)]])
    stats = coordinator.run_cell(backend=backend, condition="kaggle_zero_shot", images_df=images, image_ids=ids, out_dir=tmp_path, settings=st,
                                 max_consecutive_errors=2, log=lambda s: None)
    assert stats["errors"] == 2 and "consecutive" in stats["stopped"]


def test_budget_stops_at_image_boundary(cohort, tmp_path):
    images, _, _ = cohort
    ids = [100, 101, 102]
    st = settings_for("kaggle_zero_shot", ids)
    ledger = cost_guard.CostLedger(tmp_path / "ledger.jsonl", cost_guard.Budget(max_requests=1))
    stats = coordinator.run_cell(backend=FakeBackend([[EMPTY]]), condition="kaggle_zero_shot", images_df=images, image_ids=ids, out_dir=tmp_path,
                                 settings=st, ledger=ledger, log=lambda s: None)
    assert stats["ran"] == 1 and stats["stopped"].startswith("budget")


def test_resume_with_changed_settings_is_refused(cohort, tmp_path):
    images, _, _ = cohort
    ids = [100]
    st = settings_for("kaggle_zero_shot", ids)
    coordinator.run_cell(backend=FakeBackend([[EMPTY]]), condition="kaggle_zero_shot", images_df=images, image_ids=ids, out_dir=tmp_path, settings=st, log=lambda s: None)
    st2 = {**st, "max_new_tokens": 1}
    with pytest.raises(identity.IdentityMismatch):
        coordinator.run_cell(backend=FakeBackend([[EMPTY]]), condition="kaggle_zero_shot", images_df=images, image_ids=ids, out_dir=tmp_path, settings=st2, log=lambda s: None)


def test_tools_condition_end_to_end(cohort, tmp_path, harness_rules):
    images, _, _ = cohort
    ids = [100]
    for cond in ("kaggle_tools", "kaggle_tools_rules_skills"):
        st = settings_for(cond, ids)
        stats = coordinator.run_cell(backend=FakeBackend([[TOOL, FINAL]]), condition=cond, images_df=images, image_ids=ids, out_dir=tmp_path, settings=st, log=lambda s: None)
        assert stats["ran"] == 1
    assert settings_for("kaggle_tools", ids)["harness"] == {"mode": "off"}
    assert settings_for("kaggle_tools_rules_skills", ids)["harness"]["mode"] == "rules_skills_index"
    assert identity.fingerprint(settings_for("kaggle_tools", ids)) != identity.fingerprint(settings_for("kaggle_tools_rules_skills", ids))


# ----------------------------------------------------------------------------- record schema / importer safety

EVAL_KEYS = {"image_id", "condition", "dataset", "split", "model_id", "image_view", "canonical_letterbox", "adapter_path", "ground_truth", "predictions",
             "matched_pairs", "fdi_precision", "fdi_recall", "fdi_f1", "exact_precision", "exact_recall", "exact_f1", "closeness_score", "spatial_proximity",
             "diagnostic_similarity", "fdi_correct", "exact_match", "all_exact_match", "format_ok", "confidence", "turns", "tool_calls", "raw_output",
             "prompt", "seconds", "eval_settings", "trajectory", "ablation", "timestamp"}


def test_evaluated_record_schema_matches_evaluate_condition(cohort):
    _, annots, cats = cohort
    traj = type("T", (), {"final_answer": json.loads(FINAL)["final_answer"], "format_ok": True, "turns": [{"turn": 0, "raw_output": FINAL, "status": "final_answer"}],
                          "tool_calls": 0, "messages": [], "harness_events": None})()
    t = records.transport_record(image_id=100, condition="kaggle_zero_shot", model_slug="m", traj=traj, usage=[], seconds=1.0, identity={})
    rec = records.evaluated_record(t, annots, cats, dataset="dentex", split="test", settings={}, model_slug="m")
    assert EVAL_KEYS <= set(rec) and set(rec) - EVAL_KEYS == {"external"}
    assert rec["ground_truth"] and "bbox" in rec["ground_truth"][0]
    assert rec["canonical_letterbox"] is False and rec["image_view"] == "canonical"


def test_transport_with_scoring_fields_is_rejected(cohort):
    with pytest.raises(ValueError, match="scoring fields"):
        importer.validate_transport({"kind": "transport", "image_id": 1, "ground_truth": []}, cohort_ids=None, settings=None)


def test_transport_outside_cohort_is_rejected():
    with pytest.raises(ValueError, match="not in the cohort"):
        importer.validate_transport({"kind": "transport", "image_id": 5}, cohort_ids={1}, settings=None)


def test_usage_summary_keeps_unknown_as_none():
    s = records.usage_summary([{"prompt_tokens": 5, "completion_tokens": None, "cost_usd": None}, {"prompt_tokens": 7, "completion_tokens": 1, "cost_usd": None}])
    assert s["prompt_tokens"] == 12 and s["completion_tokens"] is None and s["cost_usd"] is None and s["n_unknown_completion_tokens"] == 1


def test_findings_parity_with_script():
    try:
        import scripts.evaluate_models as ev
    except ImportError as exc:
        pytest.skip(f"scripts.evaluate_models needs the training stack: {exc}")
    traj = type("T", (), {"final_answer": [{"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries", "confidence": 0.5}], "messages": [], "turns": []})()
    assert ev.findings_from_trajectory(traj) == records.findings_from_trajectory(traj)
    traj.final_answer = {"quadrant": 2, "tooth_position": 1, "diagnosis": "Caries"}
    assert ev.findings_from_trajectory(traj) == records.findings_from_trajectory(traj)
    traj.final_answer = None
    assert ev.findings_from_trajectory(traj) == records.findings_from_trajectory(traj) == []


# ----------------------------------------------------------------------------- cost, roster, preflight, tables

def test_cost_estimate_worst_case_is_the_loop_bound():
    e = cost_guard.estimate("kaggle_tools", 250)
    assert e["worst"]["requests"] == 250 * 65 and e["worst"]["cost_usd"] is None
    assert cost_guard.estimate("kaggle_zero_shot", 250)["worst"]["requests"] == 250
    quad = cost_guard.input_tokens_for(65, 100, 10)
    assert quad == 65 * 100 + 10 * 65 * 64 / 2
    priced = cost_guard.estimate("kaggle_tools", 10, price=cost_guard.Price(1.0, 2.0))
    assert priced["worst"]["cost_usd"] > 0


def test_ledger_unknown_cost_never_counts_as_zero(tmp_path):
    led = cost_guard.CostLedger(tmp_path / "l.jsonl", cost_guard.Budget(max_cost_usd=1.0))
    led.record({"prompt_tokens": None, "cost_usd": None})
    led.check()
    s = led.summary()
    assert s["cost_usd"] is None and s["input_tokens"] is None and s["n_unknown_cost"] == 1
    led2 = cost_guard.CostLedger(tmp_path / "l.jsonl")      # reloads from disk
    assert led2.requests == 1


def test_ledger_cost_budget(tmp_path):
    led = cost_guard.CostLedger(None, cost_guard.Budget(max_cost_usd=1.0))
    led.record({"prompt_tokens": 1, "cost_usd": 1.5})
    with pytest.raises(cost_guard.BudgetExceeded):
        led.check()


def test_roster_parse_and_probe():
    assert [e.slug for e in roster.parse_listing('["a/b", {"slug": "c/d", "vision": true}]')] == ["a/b", "c/d"]
    table = "SLUG  NAME\n----  ----\nmodel-a  A\nmodel-b  B\n"
    assert [e.slug for e in roster.parse_listing(table)] == ["model-a", "model-b"]
    ok = FakeBackend([['{"left": "red", "right": "blue"}']])
    bad = FakeBackend([['{"left": "blue", "right": "red"}']])
    assert roster.probe_vision(ok) and not roster.probe_vision(bad)
    entries = [roster.RosterEntry("a", vision=True), roster.RosterEntry("b")]
    assert [e.slug for e in roster.usable(entries)] == ["a"]


def test_preflight_checks(tmp_path, cohort):
    images, _, _ = cohort
    assert preflight.check_yolo_overlap([1, 2], [3])["ok"] and not preflight.check_yolo_overlap([1, 2], [2, 3])["ok"]
    assert preflight.check_images(images, [100, 101])["ok"] and not preflight.check_images(images, [999])["ok"]
    f = tmp_path / "w.pt"
    f.write_bytes(b"abc")
    g = type("G", (), {"model": object(), "model_path": f})()
    r = preflight.check_grounder(lambda: g)
    assert r["ok"] and len(r["sha256"]) == 64
    g2 = type("G", (), {"model": None, "model_path": f})()
    assert not preflight.check_grounder(lambda: g2)["ok"]
    with pytest.raises(preflight.PreflightFailed):
        preflight.require({"grounder": {"ok": False, "error": "nope"}})


def test_tables_and_harness_effect(cohort, tmp_path):
    images, annots, cats = cohort
    ids = [100, 101, 102]
    cells = {}
    for cond, reply in (("kaggle_zero_shot", EMPTY), ("kaggle_tools", FINAL)):
        st = settings_for(cond, ids)
        coordinator.run_cell(backend=FakeBackend([[reply]] if cond == "kaggle_zero_shot" else [[FINAL]]), condition=cond, images_df=images, image_ids=ids,
                             out_dir=tmp_path, settings=st, log=lambda s: None)
        p = importer.cell_paths(tmp_path, "fake/model-1", cond)
        cells[f"fake/model-1/{cond}"] = importer.import_transport(p["transport"], annots, cats, settings=st, cohort_ids=set(ids))["records"]
    rows = tables.table_rows(cells, n_boot=50)
    assert len(rows) == 2 and "| Cell |" in tables.markdown_table(rows)
    eff = tables.harness_effect(cells, tables.standard_pairs("fake/model-1"), n_boot=50)
    assert [e["label"] for e in eff] == ["fake/model-1: tools vs zero-shot"]      # pairs with a missing cell are skipped
    assert eff[0]["n_paired"] == 3
    assert "tools vs zero-shot" in tables.markdown_effects(eff)


# --- verified model profiles, credential refresh, identity --------------------------------------------------------------
def test_profiles_are_complete_and_backend_follows_them():
    from dental_agent.external import model_profiles as MP
    assert len(MP.PROFILES) == 6 and all(p.usd_per_mtok_in > 0 and p.usd_per_mtok_out > 0 for p in MP.PROFILES.values())
    claude = MP.get_profile("anthropic/claude-sonnet-5@default")
    assert claude.send_temperature is False and claude.decoding_identity()["temperature_sent"] is False
    seen = []
    b = MP.backend_for(claude.slug, base_url="http://p", api_key="k", post=lambda u, p: seen.append(p) or {"choices": [{"message": {"content": "x"}}]})
    b.open_chat(None).send([{"type": "text", "text": "x"}], 10, 0.0)
    assert "temperature" not in seen[0] and seen[0]["max_tokens"] == 10 and seen[0]["model"] == claude.slug
    assert MP.get_profile("gpt-5.4-nano-2026-03-17").price.input_per_mtok == 0.20
    assert all(e.vision is True for e in MP.as_roster())


def test_unverified_or_text_only_models_are_never_run_silently():
    from dental_agent.external import model_profiles as MP
    for slug in ("gpt-6-sol", "gpt-oss-120b", "claude-opus-5-default"):
        with pytest.raises(MP.UnknownModel):
            MP.backend_for(slug, base_url="http://p", post=lambda *a: {})


def test_key_is_refreshed_before_expiry_and_only_then(monkeypatch):
    from dental_agent.external.kaggle_provider import _expiry_epoch
    assert _expiry_epoch("2026-11-14T10:48:46.777000Z") == _expiry_epoch("2026-11-14T10:48:46.777000+00:00") and _expiry_epoch("nope") is None and _expiry_epoch(None) is None
    now = _expiry_epoch("2026-11-14T10:00:00Z")
    monkeypatch.setenv("MODEL_PROXY_API_KEY", "old")
    monkeypatch.setenv("MODEL_PROXY_EXPIRY_TIME", "2026-11-14T10:30:00Z")
    calls = []

    def refresher():
        calls.append(1)
        monkeypatch.setenv("MODEL_PROXY_API_KEY", "new")
        monkeypatch.setenv("MODEL_PROXY_EXPIRY_TIME", "2026-11-14T11:30:00Z")

    clock = {"t": now}
    b = OpenAICompatBackend("m", base_url="http://p", post=lambda *a: {}, refresher=refresher, clock=lambda: clock["t"])
    assert b.api_key == "old" and not calls                       # 30 min left: no refresh
    clock["t"] = now + 26 * 60                                    # 4 min left, inside the 5 min margin
    assert b.api_key == "new" and len(calls) == 1
    assert b.api_key == "new" and len(calls) == 1                 # fresh again: not refreshed twice
    assert OpenAICompatBackend("m", base_url="http://p", api_key="explicit", post=lambda *a: {}, refresher=refresher, clock=lambda: now + 99999).api_key == "explicit"


def test_env_file_loading_and_cli_refresher(tmp_path, monkeypatch):
    from dental_agent.external.kaggle_provider import kaggle_cli_refresher, load_env_file
    f = tmp_path / "p.env"
    f.write_text("MODEL_PROXY_URL='https://x/models'\nexport MODEL_PROXY_API_KEY=abc\n# c\nMODEL_PROXY_EXPIRY_TIME=2026-11-14T10:48:46Z\n")
    for k in ("MODEL_PROXY_URL", "MODEL_PROXY_API_KEY", "MODEL_PROXY_EXPIRY_TIME"):
        monkeypatch.delenv(k, raising=False)
    assert load_env_file(str(f))["MODEL_PROXY_API_KEY"] == "abc" and os.environ["MODEL_PROXY_URL"] == "https://x/models"
    ran = []
    kaggle_cli_refresher(str(f), runner=lambda cmd: ran.append(cmd))()
    assert ran == [["kaggle", "benchmarks", "init", "-y", "--env-file", str(f)]]


def test_http_errors_are_classified(monkeypatch):
    import urllib.error
    for code, kind in ((401, "auth"), (403, "auth"), (429, "rate_limit"), (500, "infra")):
        def boom(req, timeout=None, code=code):
            raise urllib.error.HTTPError("u", code, "m", {}, io.BytesIO(b"{}"))
        monkeypatch.setattr("urllib.request.urlopen", boom)
        b = OpenAICompatBackend("m", base_url="http://p", api_key="k")
        with pytest.raises(ProviderError) as ei:
            b.open_chat(None).send([{"type": "text", "text": "x"}], 5, 0.0)
        assert ei.value.kind == kind and ei.value.status == code


def test_identity_decoding_block_is_optional_and_changes_fingerprint(harness_rules):
    cond = C.get_condition("kaggle_zero_shot")
    base = identity.run_settings(cond, "m", cohort="c", split="test")
    assert "decoding" not in base
    with_dec = identity.run_settings(cond, "m", cohort="c", split="test", decoding={"temperature_sent": False})
    assert with_dec["decoding"] == {"temperature_sent": False} and identity.fingerprint(base) != identity.fingerprint(with_dec)
    with pytest.raises(identity.IdentityMismatch):
        identity.assert_same(base, with_dec)
