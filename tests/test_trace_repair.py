"""
Unit tests for Trace Corruption Cleaning and Verifier Self-Repair / Editor Pass.
Zero API calls executed (mock stubs only per Rule 10).
"""

import json
import tempfile
from pathlib import Path
from PIL import Image

import pytest
from dental_agent.training.trace_generation import (
    clean_unverified_traces,
    repair_and_clean_trace,
    repair_pending,
)


def test_clean_unverified_traces_purges_corruption():
    with tempfile.TemporaryDirectory() as tmpdir:
        trace_file = Path(tmpdir) / "train_cot_traces_unverified.jsonl"

        records = [
            # 1. Valid trace
            {
                "image_id": 1,
                "dataset": "dentex",
                "status": "unverified",
                "trajectory": {
                    "turns": [{"turn": 0, "raw_output": '{"thought": "clear molar view", "action": "locate_tooth"}'}],
                    "messages": [{"role": "assistant", "content": '{"thought": "clear molar view", "action": "locate_tooth"}'}],
                },
            },
            # 2. Corrupt XML fake tool call
            {
                "image_id": 2,
                "dataset": "dentex",
                "status": "unverified",
                "trajectory": {
                    "turns": [{"turn": 0, "raw_output": '<fake_tool_call>locate_tooth</fake_tool_call>'}],
                    "messages": [{"role": "assistant", "content": '<fake_tool_call>locate_tooth</fake_tool_call>'}],
                },
            },
            # 3. Corrupt multi-blob action dump in a single turn
            {
                "image_id": 3,
                "dataset": "dentex",
                "status": "unverified",
                "trajectory": {
                    "turns": [{"turn": 0, "raw_output": '{"action": "a1"}\n{"action": "a2"}\n{"action": "a3"}\n{"action": "a4"}'}],
                    "messages": [{"role": "assistant", "content": '{"action": "a1"}\n{"action": "a2"}\n{"action": "a3"}\n{"action": "a4"}'}],
                },
            },
            # 4. Failed generation
            {
                "image_id": 4,
                "dataset": "dentex",
                "status": "generation_failed",
                "failure_reason": "Timeout",
                "trajectory": {},
            },
        ]

        with open(trace_file, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        stats = clean_unverified_traces(trace_file, backup=False, purge_failed=True)

        assert stats["kept"] == 1
        assert stats["corrupted"] == 2
        assert stats["failed"] == 1

        with open(trace_file, "r", encoding="utf-8") as f:
            remaining = [json.loads(line) for line in f if line.strip()]

        assert len(remaining) == 1
        assert remaining[0]["image_id"] == 1


def test_repair_and_clean_trace_mock_success():
    img = Image.new("RGB", (100, 100), color="gray")
    gt = [{"quadrant": 4, "tooth_position": 8, "diagnosis": "Impacted"}]
    
    # Original broken record (with wrong diagnosis and XML artifact)
    broken_record = {
        "image_id": 10,
        "dataset": "dentex",
        "status": "unverified",
        "trajectory": {
            "messages": [
                {"role": "user", "content": "Analyze radiograph"},
                {"role": "assistant", "content": "<fake_tool_call>locate_tooth</fake_tool_call>\n{\"final_answer\": [{\"quadrant\": 4, \"tooth_position\": 8, \"diagnosis\": \"Caries\"}]}"},
            ],
        },
    }

    # Mock call_llm_fn that returns repaired output for repair, and grounded=true for verify
    def mock_call_llm(provider, model, system_prompt, user_content, **kwargs):
        label = kwargs.get("label", "")
        if label == "repair_and_clean_trace":
            return json.dumps({
                "thought": "Examining lower right quadrant 4, tooth 48 is horizontally impacted with crown abutting root of 47.",
                "final_answer": [{"quadrant": 4, "tooth_position": 8, "diagnosis": "Impacted Tooth", "confidence": 0.95}],
            })
        elif label == "verify_trace":
            return json.dumps({"grounded": True, "reason": "Claims match visible impaction on 48"})
        return json.dumps({"grounded": False, "reason": "Unknown"})

    ok, repaired_traj, reason = repair_and_clean_trace(
        img,
        gt,
        broken_record,
        verifier_provider="openrouter",
        verifier_model="minimax/minimax-m3:free",
        call_llm_fn=mock_call_llm,
    )

    assert ok is True
    assert repaired_traj is not None
    assert repaired_traj["repaired"] is True
    assert len(repaired_traj["final_answer"]) == 1
    assert repaired_traj["final_answer"][0]["quadrant"] == 4
    assert repaired_traj["final_answer"][0]["tooth_position"] == 8
    assert repaired_traj["final_answer"][0]["diagnosis"] == "Impacted Tooth"


def test_repair_pending_slicing_and_max_images():
    with tempfile.TemporaryDirectory() as tmpdir:
        unverified_path = Path(tmpdir) / "unverified.jsonl"
        verified_path = Path(tmpdir) / "verified.jsonl"
        
        # Create dummy image
        img_path = Path(tmpdir) / "dummy.png"
        Image.new("RGB", (50, 50), color="white").save(img_path)

        records = [
            {
                "image_id": i,
                "dataset": "dentex",
                "image_path": str(img_path),
                "ground_truth": [{"quadrant": 1, "tooth_position": i % 8 + 1, "diagnosis": "Caries"}],
                "trajectory": {
                    "messages": [{"role": "assistant", "content": '{"final_answer": [{"quadrant": 1, "tooth_position": 1, "diagnosis": "Caries"}]}'}],
                },
            }
            for i in range(1, 11)
        ]

        with open(unverified_path, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        def mock_call_llm(provider, model, system_prompt, user_content, **kwargs):
            label = kwargs.get("label", "")
            if label == "repair_and_clean_trace":
                return json.dumps({
                    "thought": "Repaired reasoning",
                    "final_answer": [{"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}],
                })
            elif label == "verify_trace":
                return json.dumps({"grounded": True, "reason": "Verified"})
            return json.dumps({"grounded": False, "reason": "Unknown"})

        # Slice 1 of 2 with max_images=2
        res = repair_pending(
            unverified_path=unverified_path,
            verified_path=verified_path,
            total_slices=2,
            slice_index=1,
            slice_seed=42,
            pacing_delay=0.0,
            max_images=2,
            call_llm_fn=mock_call_llm,
        )

        assert res["pending_repair"] == 2
        assert res["repaired_and_promoted"] == 2
        assert res["still_unverified"] == 0
