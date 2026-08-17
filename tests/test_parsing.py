"""
Unit tests for robust JSON parsing of model responses.
"""

from dental_agent.agent.parsing import parse_agent_json


def test_parse_clean_json() -> None:
    raw = '{"tool": "zoom_crop", "args": {"bbox": [100, 200, 50, 50]}}'
    parsed = parse_agent_json(raw)
    assert parsed is not None
    assert parsed["tool_calls"][0]["tool"] == "zoom_crop"
    assert parsed["tool_calls"][0]["args"]["bbox"] == [100, 200, 50, 50]


def test_parse_markdown_code_block() -> None:
    raw = """Here is my decision:
```json
{
  "thought": "I notice a radiolucency on tooth 36.",
  "tool": "zoom_crop",
  "args": {"bbox": [500, 300, 80, 80]}
}
```
"""
    parsed = parse_agent_json(raw)
    assert parsed is not None
    assert parsed["tool_calls"][0]["tool"] == "zoom_crop"
    assert "thought" in parsed


def test_parse_surrounding_commentary() -> None:
    raw = 'Sure, calling tool: {"final_answer": {"quadrant": 2, "tooth_position": 4, "diagnosis": "Caries"}} hope this helps!'
    parsed = parse_agent_json(raw)
    assert parsed is not None
    assert "final_answer" in parsed
    assert parsed["final_answer"]["quadrant"] == 2


def test_parse_trailing_comma() -> None:
    raw = '{"final_answer": {"quadrant": 1, "tooth_position": 6, "diagnosis": "Deep Caries",}}'
    parsed = parse_agent_json(raw)
    assert parsed is not None
    assert parsed["final_answer"]["diagnosis"] == "Deep Caries"


def test_parse_invalid_text() -> None:
    assert parse_agent_json("This is definitely not JSON.") is None
    assert parse_agent_json("") is None
    assert parse_agent_json(None) is None


def test_parse_think_block() -> None:
    raw = """<think>
This is my reasoning process. I am thinking about how to solve the problem.
{ "tool": "this_should_be_ignored" }
</think>
```json
{
  "tool": "zoom_crop",
  "args": {"bbox": [10, 20, 30, 40]}
}
```"""
    parsed = parse_agent_json(raw)
    assert parsed is not None
    assert parsed["tool_calls"][0]["tool"] == "zoom_crop"

