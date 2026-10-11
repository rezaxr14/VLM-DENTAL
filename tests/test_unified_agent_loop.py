"""One agent loop for GRPO rollouts and evaluation: budgets, healthy answers, malformed-reply handling, prompts."""

import argparse
import json
from types import SimpleNamespace

import pandas as pd
import pytest
from PIL import Image

from dental_agent.agent import loop as agent_loop
from dental_agent.agent.prompts import FIRST_USER_PROMPT, NO_TOOLS_SYSTEM_PROMPT, build_agent_system_prompt
from dental_agent.rewards.components import reward_format
from dental_agent.rewards.composite import combine_reward
from dental_agent.tools.registry import ToolRegistry

ZOOM = json.dumps({"thought": "look", "tool_calls": [{"tool": "zoom_crop", "args": {"bbox": [10, 10, 60, 60]}}]})
HEALTHY = json.dumps({"thought": "all normal", "final_answer": []})
FINDING = json.dumps({"thought": "lesion", "final_answer": [{"quadrant": 3, "tooth_position": 6,
                                                            "diagnosis": "Periapical Lesion", "confidence": 0.9}]})


@pytest.fixture()
def images_df(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGB", (400, 200), (90, 90, 90)).save(p)
    return pd.DataFrame({"id": [7], "local_path": [str(p)]})


def _script(monkeypatch, replies, calls):
    """Replace generation with scripted replies; record the kwargs and message list of every call."""
    it = iter(replies)

    def fake_generate(model, processor, messages, **kw):
        calls.append({"kw": kw, "n_messages": len(messages), "messages": list(messages)})
        reply = next(it)
        if kw.get("return_cache"):
            return reply, 10, [1, 2, 3], None, {"total_len": 10, "last_msg_count": len(messages)}
        return reply, 10, [1, 2, 3]

    monkeypatch.setattr(agent_loop, "generate_agent_reply", fake_generate)


def _run(images_df, **kw):
    return agent_loop.run_agent(7, images_df, model=None, processor=None, registry=ToolRegistry.create_default(),
                                verbose=False, **kw)


def test_healthy_empty_answer_is_a_valid_format(monkeypatch, images_df):
    calls = []
    _script(monkeypatch, [ZOOM, HEALTHY], calls)
    traj = _run(images_df, max_new_tokens=1234, dataset="dentex")
    assert traj.final_answer == [] and traj.format_ok is True
    assert all(c["kw"]["max_new_tokens"] == 1234 for c in calls)            # no hidden 512
    obs = calls[1]["messages"][-1]["content"]
    assert any(c.get("text") == "Result of zoom_crop:" for c in obs if c["type"] == "text")  # training format
    assert reward_format(traj.to_dict()) == 1.0


def test_empty_answer_scores_best_on_a_healthy_image_and_format_is_not_penalised():
    healthy = {"final_answer": [], "format_ok": True, "tool_calls": 0, "turns": []}
    garbage = {"final_answer": None, "format_ok": False, "tool_calls": 0, "turns": []}
    wrong = {"final_answer": [{"quadrant": 1, "tooth_position": 1, "diagnosis": "Caries"}], "format_ok": True,
             "tool_calls": 0, "turns": []}
    r_h, comp_h = combine_reward(healthy, [])
    assert comp_h["format"] == 1.0 if "format" in comp_h else True
    assert r_h > combine_reward(wrong, [])[0] > combine_reward(garbage, [])[0]
    assert reward_format({"final_answer": []}) == 1.0  # even without the precomputed flag


def test_malformed_reply_ends_episode_unless_retries_allowed(monkeypatch, images_df):
    calls = []
    _script(monkeypatch, ["not json at all", FINDING], calls)
    traj = _run(images_df, malformed_retries=0)
    assert traj.format_ok is False and traj.final_answer is None and len(traj.turns) == 1

    calls2 = []
    _script(monkeypatch, ["not json at all", FINDING], calls2)
    traj2 = _run(images_df, malformed_retries=1)
    assert traj2.format_ok is True and len(traj2.final_answer) == 1
    err = calls2[1]["messages"][-1]
    assert err["role"] == "user" and err["content"].startswith("Error:")


def test_max_turns_bounds_the_episode(monkeypatch, images_df):
    calls = []
    _script(monkeypatch, [ZOOM] * 10, calls)
    traj = _run(images_df, max_turns=3, max_tool_calls=50)
    assert len(traj.turns) == 3 and traj.final_answer is None and traj.format_ok is False

    calls2 = []
    _script(monkeypatch, [ZOOM] * 100, calls2)
    traj2 = _run(images_df, max_tool_calls=5)  # default bound = max_tool_calls + 15 turns, as GRPO always had
    assert len(traj2.turns) <= 5 + 15


def test_system_prompt_follows_dataset(monkeypatch, images_df):
    calls = []
    _script(monkeypatch, [HEALTHY], calls)
    _run(images_df, dataset="tufts")
    system = calls[0]["messages"][0]["content"]
    reg = ToolRegistry.create_default()
    assert system == build_agent_system_prompt(reg.format_tool_descriptions(), dataset="tufts")


def test_no_tools_uses_unified_prompt_and_empty_answer_is_valid(monkeypatch, images_df):
    calls = []
    _script(monkeypatch, [HEALTHY], calls)
    traj = agent_loop.run_agent_no_tools(7, images_df, model=None, processor=None, verbose=False, max_new_tokens=999)
    assert calls[0]["messages"][0]["content"] == NO_TOOLS_SYSTEM_PROMPT
    assert calls[0]["messages"][1]["content"][1]["text"] == FIRST_USER_PROMPT
    assert calls[0]["kw"]["max_new_tokens"] == 999
    assert traj.format_ok is True and traj.final_answer == []


def test_unified_no_tools_prompt_teaches_thought_then_final_answer():
    p = NO_TOOLS_SYSTEM_PROMPT
    assert '"thought"' in p and '"final_answer"' in p and "final_answer\": []" in p
    assert "directive" not in p.lower()  # the teacher prompt's generation-time scaffolding must not leak


def _args(**kw):
    base = dict(max_tool_calls=50, max_turns=None, max_new_tokens=2048, malformed_retries=0, canonical_resize=False,
                dataset="dentex")
    base.update(kw)
    return argparse.Namespace(**base)


def test_eval_adapter_uses_the_shared_loop(monkeypatch, images_df):
    import scripts.evaluate_models as ev

    seen = {}

    def fake_run_agent(**kw):
        seen["agent"] = kw
        return SimpleNamespace(
            final_answer=[{"quadrant": 3, "tooth_position": 6, "diagnosis": "Periapical Lesion", "confidence": 0.9}],
            format_ok=True, tool_calls=4, turns=[{"raw_output": "a"}, {"raw_output": "b"}])

    monkeypatch.setattr(ev, "run_agent", fake_run_agent)
    preds, raw, ok, turns, tools = ev.run_agent_condition(None, None, images_df, 7, ToolRegistry.create_default(), _args(max_turns=30))
    assert len(preds) == 1 and ok and turns == 2 and tools == 4 and raw == "a\n---\nb"
    assert seen["agent"]["temperature"] == 0.0 and seen["agent"]["max_turns"] == 30
    assert seen["agent"]["max_new_tokens"] == 2048 and seen["agent"]["max_tool_calls"] == 50

    monkeypatch.setattr(ev, "run_agent_no_tools", lambda **kw: SimpleNamespace(
        final_answer=[], format_ok=True, tool_calls=0, turns=[{"raw_output": "x"}]))
    preds, raw, ok, turns, tools = ev.run_agent_condition(None, None, images_df, 7, None, _args())
    assert preds == [] and ok is True and turns == 1 and tools == 0   # healthy answer is a correct, well-formed answer


# ---------------------------------------------------------------------------
# SFT trains under the prompt the model will see at GRPO / evaluation time
# ---------------------------------------------------------------------------

def _sft_dataset(tmp_path, monkeypatch, track, rec):
    import torch
    sft = pytest.importorskip("dental_agent.training.sft")
    traces = tmp_path / "t.jsonl"
    traces.write_text(json.dumps(rec) + "\n")
    captured = {}

    class Proc:
        tokenizer = None

        def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
            captured["messages"] = messages
            return "text"

        def __call__(self, **kw):
            return {"input_ids": torch.ones(1, 5, dtype=torch.long)}

    monkeypatch.setattr(sft, "safe_process_vision_info", lambda m: ([], None))
    monkeypatch.setattr(sft, "build_conversational_labels", lambda ids, tok, skip_assistant_turns=None: torch.ones_like(ids))
    ds = sft.DentalSFTDataset(traces, processor=Proc(), track=track, data_dir=str(tmp_path))
    ds[0]
    return captured["messages"]


def test_sft_substitutes_current_system_prompt_for_the_track(tmp_path, monkeypatch):
    old_teacher = "OLD TEACHER PROMPT: You will be told which finding(s) a case has (TEACHER DIRECTIVE)"
    no_tools = {"image_id": 1, "dataset": "dentex", "messages": [
        {"role": "system", "content": old_teacher},
        {"role": "user", "content": "TEACHER DIRECTIVE: This image has 1 finding(s): Q3T6:Caries"},
        {"role": "assistant", "content": HEALTHY},
    ]}
    msgs = _sft_dataset(tmp_path, monkeypatch, "no_tools", no_tools)
    assert msgs[0]["content"] == NO_TOOLS_SYSTEM_PROMPT
    assert "Q3T6" not in json.dumps(msgs[1], default=str)  # directive still stripped from the user turn

    old_agent = "OLD AGENT PROMPT without the healthy-scan instruction"
    with_tools = {"image_id": 2, "dataset": "dentex", "messages": [
        {"role": "system", "content": old_agent},
        {"role": "user", "content": [{"type": "text", "text": FIRST_USER_PROMPT}]},
        {"role": "assistant", "content": HEALTHY},
    ]}
    msgs = _sft_dataset(tmp_path, monkeypatch, "with_tools", with_tools)
    expected = build_agent_system_prompt(ToolRegistry.create_default().format_tool_descriptions(), dataset="dentex")
    assert msgs[0]["content"] == expected and old_agent not in msgs[0]["content"]

    tufts = {**with_tools, "image_id": 3, "dataset": "tufts"}
    msgs = _sft_dataset(tmp_path, monkeypatch, "with_tools", tufts)
    assert msgs[0]["content"] == old_agent  # which Tufts class list a record used is not recorded: keep what it has
