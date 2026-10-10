"""The first user prompt is one shared string, and the manifest reports GPU and TPU token costs separately."""

import json

import pandas as pd
from PIL import Image

import dental_agent.agent.loop as loop
from dental_agent.agent.prompts import FIRST_USER_PROMPT
from dental_agent.training.sft import CLEAN_FIRST_USER_PROMPT
from dental_agent.utils.canonical import SLOT_BUDGET, slot_totals
from scripts.compute_exact_trace_lengths import save_manifest, tpu_static_lengths


def test_inference_prompt_is_the_training_prompt_without_an_image_id(tmp_path, monkeypatch):
    seen = []

    def fake_generate(model, processor, messages, **kw):
        seen.append(messages[1]["content"][1]["text"])
        return "not json", 10, [1], None, None

    p = tmp_path / "x.png"
    Image.new("RGB", (64, 32), (9, 9, 9)).save(p)
    monkeypatch.setattr(loop, "generate_agent_reply", fake_generate)
    loop.run_agent(7, pd.DataFrame([{"id": 7, "local_path": str(p)}]), None, None, verbose=False)
    assert seen == [FIRST_USER_PROMPT] == [CLEAN_FIRST_USER_PROMPT]
    assert "image_id" not in FIRST_USER_PROMPT


def test_manifest_reports_gpu_and_tpu_cost_separately(tmp_path):
    static = slot_totals(SLOT_BUDGET)["tokens"]
    lengths = {"a.jsonl::tufts::1": 3000, "a.jsonl::tufts::2": 9000, "b.jsonl::dentex::3": 4000}
    vision = {"a.jsonl::tufts::1": 1152, "a.jsonl::tufts::2": 1152 * 2, "b.jsonl::dentex::3": 1152}
    tpu = tpu_static_lengths(lengths, vision, static)
    assert tpu == {k: lengths[k] - vision[k] + static for k in lengths}
    assert tpu_static_lengths(lengths, {}, static) == {}                       # no vision counts -> no TPU figure

    out = tmp_path / "m.json"
    res = save_manifest(out, "m", list(lengths.values()), {}, lengths, True, vision, dict(SLOT_BUDGET))
    meta = json.loads(out.read_text())["_meta"]
    assert meta["static_vision_tokens"] == static and meta["vision_slots"] == dict(SLOT_BUDGET)
    assert meta["compliance"]["16384"] == 3                                    # GPU: real tokens
    assert meta["compliance_tpu_static"]["16384"] == sum(v <= 16384 for v in tpu.values())
    assert meta["stats_tpu_static"]["max"] == max(tpu.values()) and "stats_tpu_static" in res

    # Original-size measurements carry no TPU figure (slot padding needs canonical views).
    save_manifest(out, "m", list(lengths.values()), {}, lengths, False, vision, dict(SLOT_BUDGET))
    assert "stats_tpu_static" not in json.loads(out.read_text())["_meta"]
