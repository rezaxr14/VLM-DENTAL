import pytest
import os
import json
import pandas as pd
from unittest.mock import patch, MagicMock
from PIL import Image
from dental_agent.training.trace_generation import _resolve_generator, _resolve_verifier
from dental_agent.agent.langgraph_loop import run_trace_gen
from dental_agent.tools.registry import ToolRegistry

@patch.dict(os.environ, {"GENERATOR_PROVIDER": "openrouter", "VERIFIER_PROVIDER": "gemini"}, clear=True)
def test_resolve_providers():
    """Test that it doesn't use the old 'auto_generator' logic."""
    import dental_agent.training.trace_generation as tg
    
    # Reload the globals that were loaded at import time
    tg.GENERATOR_PROVIDER = os.environ.get("GENERATOR_PROVIDER")
    tg.VERIFIER_PROVIDER = os.environ.get("VERIFIER_PROVIDER")
    
    assert tg._resolve_generator()[0] == "openrouter"
    assert tg._resolve_verifier()[0] == "gemini"


def test_multi_blob_dump_guard():
    """Test that a response with >2 action blobs fails fast immediately."""
    registry = ToolRegistry.create_default()
    image = Image.new("RGB", (100, 100), color="white")
    gt = [{"quadrant": 1, "tooth_position": 8, "diagnosis": "Caries", "bbox": [10, 10, 20, 20]}]

    # Dump of 4 action blocks
    dump_response = "\n".join([
        f'{{"thought": "dump {i}", "tool": "locate_tooth", "args": {{"tooth": 18}}}}'
        for i in range(4)
    ])

    with patch("dental_agent.agent.langgraph_loop.call_llm", return_value=dump_response):
        traj, error = run_trace_gen(
            image=image,
            ground_truth=gt,
            registry=registry,
            system_prompt="",
            max_turns=10,
            max_blobs_per_turn=2,
        )
        assert traj is not None
        assert traj["final_answer"] is None
        assert "multi_blob_dump" in error


def test_padding_loop_guard():
    """Test that repetitive mechanical filler thoughts trigger padding_loop termination."""
    registry = ToolRegistry.create_default()
    image = Image.new("RGB", (100, 100), color="white")
    gt = [{"quadrant": 1, "tooth_position": 8, "diagnosis": "Caries", "bbox": [10, 10, 20, 20]}]

    # Emits mechanical filler thought repeatedly with varying tools
    responses = [
        '{"thought": "Making a tool call before final answer.", "tool": "denoise", "args": {"strength": 0.1}}',
        '{"thought": "Making a tool call before final answer.", "tool": "window_level", "args": {"preset": "bone"}}',
        '{"thought": "Performing additional tool work to satisfy the requirement.", "tool": "enhance_contrast", "args": {"factor": 1.2}}',
    ]
    resp_iter = iter(responses)

    with patch("dental_agent.agent.langgraph_loop.call_llm", side_effect=lambda **kw: next(resp_iter)):
        traj, error = run_trace_gen(
            image=image,
            ground_truth=gt,
            registry=registry,
            system_prompt="",
            max_turns=10,
            max_padding_turns=3,
        )
        assert traj is not None
        assert traj["final_answer"] is None
        assert "padding_loop" in error


def test_identical_tool_loop_guard():
    """Test that calling the exact same tool with exact same arguments 3 times terminates."""
    registry = ToolRegistry.create_default()
    image = Image.new("RGB", (100, 100), color="white")
    gt = [{"quadrant": 1, "tooth_position": 8, "diagnosis": "Caries", "bbox": [10, 10, 20, 20]}]

    identical_resp = '{"thought": "Checking tooth", "tool": "locate_tooth", "args": {"tooth": 18}}'

    with patch("dental_agent.agent.langgraph_loop.call_llm", return_value=identical_resp):
        traj, error = run_trace_gen(
            image=image,
            ground_truth=gt,
            registry=registry,
            system_prompt="",
            max_turns=10,
            max_identical_repeats=3,
        )
        assert traj is not None
        assert traj["final_answer"] is None
        assert "identical_tool_loop" in error


def test_healthy_scan_final_answer_enforcement():
    """Test that on a healthy scan (gt=[]), non-empty final_answer is rejected until corrected to []."""
    registry = ToolRegistry.create_default()
    image = Image.new("RGB", (100, 100), color="white")
    gt = []

    responses = [
        '{"thought": "Bone window", "tool_calls": [{"tool": "window_level", "args": {"preset": "bone"}}]}',
        '{"thought": "Locating 18", "tool_calls": [{"tool": "locate_tooth", "args": {"tooth": 18}}]}',
        '{"thought": "Zooming", "tool_calls": [{"tool": "zoom_crop", "args": {"bbox": [10, 10, 20, 20]}}]}',
        '{"thought": "Tooth 18 is impacted", "final_answer": [{"quadrant": 1, "tooth_position": 8, "diagnosis": "Impacted Tooth", "confidence": 0.9}]}',
        '{"thought": "Re-evaluating confirms normal physiological anatomy with no pathology across all quadrants.", "final_answer": []}',
    ]
    resp_iter = iter(responses)

    with patch("dental_agent.agent.langgraph_loop.call_llm", side_effect=lambda **kw: next(resp_iter)):
        traj, error = run_trace_gen(
            image=image,
            ground_truth=gt,
            registry=registry,
            system_prompt="",
            min_turns=3,
            max_turns=10,
        )
        assert traj is not None
        assert error is None
        assert traj["final_answer"] == []

        # Verify that the faulty turn was rejected
        statuses = [t.get("status") for t in traj.get("turns", [])]
        assert "rejected_final_answer" in statuses




