"""run_agent: sampling temperature is explicit, and an unparseable reply stays in the transcript."""

import pandas as pd
from PIL import Image

import dental_agent.agent.loop as loop


def _df(tmp_path):
    p = tmp_path / "x.png"
    Image.new("RGB", (400, 200), (50, 50, 50)).save(p)
    return pd.DataFrame([{"id": 1, "local_path": str(p)}])


def test_temperature_reaches_every_generation_and_defaults_to_greedy(tmp_path, monkeypatch):
    seen = []

    def fake_generate(model, processor, messages, **kw):
        seen.append(kw.get("temperature"))
        return "not json", 100, [1, 2, 3], None, None

    monkeypatch.setattr(loop, "generate_agent_reply", fake_generate)
    loop.run_agent(1, _df(tmp_path), None, None, verbose=False)
    loop.run_agent(1, _df(tmp_path), None, None, verbose=False, temperature=0.7)
    assert seen == [0.0, 0.7]


def test_unparseable_reply_is_kept_so_its_tokens_can_be_trained_on(tmp_path, monkeypatch):
    monkeypatch.setattr(loop, "generate_agent_reply", lambda *a, **k: ("not json at all", 100, [1, 2, 3], None, None))
    traj = loop.run_agent(1, _df(tmp_path), None, None, verbose=False, temperature=0.7)
    assert traj.messages[-1] == {"role": "assistant", "content": "not json at all"}
    assert traj.turns[-1]["status"] == "unparseable_json" and len(traj.assistant_token_spans) == 1
