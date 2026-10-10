"""
Regression tests for SFT prompt contamination (risk R2, TraceForge review):

Stored traces embed the generation-time ``TEACHER DIRECTIVE`` — including the
ground-truth finding list — inside the FIRST user message. ``DentalSFTDataset``
rebuilds that message for training, so it must strip the directive; otherwise
SFT trains on a prompt that contains the answer while inference never will.

Locks:
1. ``strip_teacher_directive`` unit behaviour (suffix cut, clean-instruction
   substitution, identity on clean text).
2. With-tools record (directive appended to the clean instruction in a content
   list): no directive, no ``Q<q>T<p>:<diagnosis>`` finding list survives in
   the rebuilt prompt.
3. No-tools record (directive stored as the whole user turn): the rebuilt user
   message is exactly the clean instruction.
"""

import json
import tempfile
from pathlib import Path

from PIL import Image

from dental_agent.training.sft import (
    CLEAN_FIRST_USER_PROMPT,
    TEACHER_DIRECTIVE_MARKER,
    DentalSFTDataset,
    strip_teacher_directive,
)

# Verbatim shape of the with-tools directive from langgraph_loop.run_trace_gen.
WITH_TOOLS_DIRECTIVE = (
    "TEACHER DIRECTIVE: You are generating an expert demonstration trace for SFT.\n"
    "You MUST eventually reach a diagnosis covering these 2 finding(s): "
    "Q3T6:Periapical Lesion; Q1T1:Caries\n\n"
    "Use locate_tooth to find each tooth's position — do not guess or assert coordinates "
    "yourself. Never mention in your reasoning that this list, a hint, or a directive was "
    "given to you — write your thought as genuine first-look clinical analysis."
)

# Verbatim shape of the no-tools directive from trace_generation.generate_no_tools_trajectory.
NO_TOOLS_DIRECTIVE = (
    "TEACHER DIRECTIVE: You are generating an expert demonstration trace for SFT.\n"
    "This image has 1 finding(s): Q4T6:Deep Caries\n\n"
    "Write the clinical reasoning a radiologist would give for noticing these on "
    "direct visual inspection, then give your final answer covering all of them."
)


def test_strip_teacher_directive_cuts_suffix_and_keeps_clean_prefix():
    text = "Analyze this panoramic X-ray.\n\n" + WITH_TOOLS_DIRECTIVE
    assert strip_teacher_directive(text) == "Analyze this panoramic X-ray."
    # Directive-only messages (no-tools shape) get the clean instruction.
    assert strip_teacher_directive(NO_TOOLS_DIRECTIVE) == CLEAN_FIRST_USER_PROMPT
    # Clean text passes through untouched.
    clean = "Analyze this panoramic X-ray. Identify any abnormal teeth."
    assert strip_teacher_directive(clean) == clean
    assert strip_teacher_directive("") == ""
    # The marker constant is what we cut on.
    assert TEACHER_DIRECTIVE_MARKER in WITH_TOOLS_DIRECTIVE


class DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def encode(self, text, add_special_tokens=False):
        return [10] * max(len(text) // 4, 1)

    def decode(self, token_ids):
        return "dummy text"


class CapturingProcessor:
    """Records the exact messages the chat template renders for training."""

    def __init__(self):
        self.tokenizer = DummyTokenizer()
        self.image_token_id = 99999
        self.rendered_text = ""
        self.last_messages = None

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        self.last_messages = messages
        parts = []
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
            elif isinstance(content, str):
                parts.append(content)
        self.rendered_text = "\n".join(parts)
        return self.rendered_text

    def __call__(self, text, images=None, videos=None, padding=False, return_tensors=None, **kwargs):
        import torch

        seq_len = max(len(text[0]) // 4, 10)
        input_ids = torch.full((1, seq_len), 10, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones((1, seq_len), dtype=torch.long),
            "labels": input_ids.clone(),
        }


def _first_user_text(processor: CapturingProcessor) -> str:
    assert processor.last_messages is not None
    for m in processor.last_messages:
        if m.get("role") == "user":
            content = m.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    i.get("text", "") for i in content if isinstance(i, dict) and i.get("type") == "text"
                )
    raise AssertionError("no user message was rendered")


def _write_record(tmp_path: Path, record: dict) -> Path:
    trace_file = tmp_path / "traces.jsonl"
    with open(trace_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return trace_file


def test_with_tools_rebuilt_prompt_has_no_directive_or_finding_list():
    """Full dataset path: directive appended to the clean instruction must not survive."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_file = tmp_path / "sample.png"
        Image.new("RGB", (1200, 600), color=(128, 128, 128)).save(img_file)

        record = {
            "image_id": 101,
            "image_path": str(img_file),
            "ground_truth": [
                {"quadrant": 3, "tooth_position": 6, "diagnosis": "Periapical Lesion"},
                {"quadrant": 1, "tooth_position": 1, "diagnosis": "Caries"},
            ],
            "messages": [
                {"role": "system", "content": "You are a dental AI."},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": "<Image>"},
                        {
                            "type": "text",
                            "text": (
                                "Analyze this panoramic X-ray. Identify any abnormal teeth "
                                "and determine the diagnosis.\n\n" + WITH_TOOLS_DIRECTIVE
                            ),
                        },
                    ],
                },
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "thought": "The apex of 36 shows radiolucency.",
                            "final_answer": [
                                {"quadrant": 3, "tooth_position": 6, "diagnosis": "Periapical Lesion", "confidence": 0.9}
                            ],
                        }
                    ),
                },
            ],
            "turns": [],
            "final_answer": [
                {"quadrant": 3, "tooth_position": 6, "diagnosis": "Periapical Lesion", "confidence": 0.9}
            ],
        }
        processor = CapturingProcessor()
        dataset = DentalSFTDataset(
            _write_record(tmp_path, record), processor=processor, track="with_tools", data_dir=tmp_path
        )
        item = dataset[0]
        assert "input_ids" in item

        first_user = _first_user_text(processor)
        # The clean instruction survives, everything from the marker onward is gone.
        assert first_user.startswith("Analyze this panoramic X-ray.")
        assert TEACHER_DIRECTIVE_MARKER not in processor.rendered_text
        assert "TEACHER DIRECTIVE" not in first_user
        # No ground-truth finding list (Q<q>T<p>:<diagnosis> hint format) survives.
        assert "Q3T6:Periapical Lesion" not in processor.rendered_text
        assert "Q1T1:Caries" not in processor.rendered_text
        assert "finding(s):" not in processor.rendered_text
        # Assistant training target is untouched.
        assert "Periapical Lesion" in json.dumps(processor.last_messages[2]["content"])


def test_no_tools_directive_replaced_with_clean_instruction():
    """No-tools records store the directive as the WHOLE user turn."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        img_file = tmp_path / "sample.png"
        Image.new("RGB", (1200, 600), color=(128, 128, 128)).save(img_file)

        record = {
            "image_id": 202,
            "image_path": str(img_file),
            "ground_truth": [{"quadrant": 4, "tooth_position": 6, "diagnosis": "Deep Caries"}],
            "messages": [
                {"role": "system", "content": "You are an expert dental radiologist AI."},
                {"role": "user", "content": NO_TOOLS_DIRECTIVE},
                {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "thought": "Radiolucency at the apex of 46.",
                            "final_answer": [
                                {"quadrant": 4, "tooth_position": 6, "diagnosis": "Deep Caries", "confidence": 0.8}
                            ],
                        }
                    ),
                },
            ],
            "turns": [{"turn": 0, "raw_output": "{}", "parsed": {}}],
            "final_answer": [{"quadrant": 4, "tooth_position": 6, "diagnosis": "Deep Caries", "confidence": 0.8}],
        }
        processor = CapturingProcessor()
        dataset = DentalSFTDataset(
            _write_record(tmp_path, record), processor=processor, track="no_tools", data_dir=tmp_path
        )
        item = dataset[0]
        assert "input_ids" in item

        assert _first_user_text(processor) == CLEAN_FIRST_USER_PROMPT
        assert TEACHER_DIRECTIVE_MARKER not in processor.rendered_text
        # The no-tools finding list must not survive anywhere in the prompt.
        assert "Q4T6:Deep Caries" not in processor.rendered_text
        assert "finding(s):" not in processor.rendered_text
        assert "healthy scan" not in processor.rendered_text
