"""
Unit test for Conversational Assistant-Only Loss Masking in Stage 1 SFT.

Verifies:
1. System prompt tokens are masked with labels = -100.
2. User query and panoramic image tokens are masked with labels = -100.
3. Assistant clinical reasoning and tool call JSON retain labels == input_ids.
4. Assistant closing tag <|im_end|> retains labels == input_ids.
5. Environment tool return observations are masked with labels = -100.
"""

import pytest
import torch
from unittest.mock import MagicMock

from dental_agent.training.sft import build_conversational_labels


class MockTokenizer:
    def __init__(self):
        self.im_start_id = 1001
        self.im_end_id = 1002
        self.assistant_token_ids = [2001]
        self.newline_id = 3001

    def encode(self, text, add_special_tokens=False):
        if text == "<|im_start|>":
            return [self.im_start_id]
        elif text == "<|im_end|>":
            return [self.im_end_id]
        elif text == "assistant":
            return self.assistant_token_ids
        elif text == "\n":
            return [self.newline_id]
        return [5000]


def test_build_conversational_labels_masking():
    tokenizer = MockTokenizer()

    # Construct synthetic multi-turn token stream:
    # 1. System turn: <|im_start|> system \n You are dental AI <|im_end|>
    # 2. User turn: <|im_start|> user \n Analyze X-ray <|im_end|>
    # 3. Assistant turn: <|im_start|> assistant \n {"tool_calls": [...]} <|im_end|>
    # 4. Tool observation turn: <|im_start|> user \n Result of locate_tooth <|im_end|>
    # 5. Assistant final turn: <|im_start|> assistant \n {"final_answer": [...]} <|im_end|>

    system_tokens = [1001, 2002, 3001, 4001, 4002, 1002]
    user_tokens = [1001, 2003, 3001, 4003, 4004, 1002]
    assistant_1_tokens = [1001, 2001, 3001, 7001, 7002, 7003, 1002]
    tool_obs_tokens = [1001, 2003, 3001, 8001, 8002, 1002]
    assistant_2_tokens = [1001, 2001, 3001, 9001, 9002, 1002]

    all_tokens = system_tokens + user_tokens + assistant_1_tokens + tool_obs_tokens + assistant_2_tokens
    input_ids = torch.tensor([all_tokens], dtype=torch.long)

    labels = build_conversational_labels(input_ids, tokenizer)

    # Calculate exact assistant offsets
    len_sys = len(system_tokens)
    len_user = len(user_tokens)
    len_asst1 = len(assistant_1_tokens)
    len_tool = len(tool_obs_tokens)

    # 1. System tokens MUST be -100
    assert (labels[0, :len_sys] == -100).all(), "System tokens must be masked with -100"

    # 2. User prompt tokens MUST be -100
    assert (labels[0, len_sys : len_sys + len_user] == -100).all(), "User prompt tokens must be masked with -100"

    # 3. Assistant 1 tokens MUST be unmasked (reasoning + <|im_end|>)
    asst1_start = len_sys + len_user + 3  # skip <|im_start|> assistant \n
    asst1_end = len_sys + len_user + len_asst1
    assert (labels[0, asst1_start:asst1_end] == input_ids[0, asst1_start:asst1_end]).all(), "Assistant 1 tokens must match input_ids"

    # 4. Tool observation tokens MUST be -100
    tool_start = len_sys + len_user + len_asst1
    tool_end = tool_start + len_tool
    assert (labels[0, tool_start:tool_end] == -100).all(), "Tool observation tokens must be masked with -100"

    # 5. Assistant 2 final answer tokens MUST be unmasked
    asst2_start = tool_end + 3
    asst2_end = tool_end + len(assistant_2_tokens)
    assert (labels[0, asst2_start:asst2_end] == input_ids[0, asst2_start:asst2_end]).all(), "Assistant 2 tokens must match input_ids"
