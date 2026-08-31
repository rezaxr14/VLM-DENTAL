"""
Robust JSON output parsing for agent multi-turn responses.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def parse_agent_json(text: str) -> Optional[dict[str, Any]]:
    """Robustly extract and parse the primary actionable JSON dictionary from model output text,
    then normalize it to the multi-tool-call schema: a result with a single "tool"/"args" key
    pair (the older single-tool-per-turn format) gets rewritten into a one-element "tool_calls"
    list, so every downstream consumer (the reasoning/tool nodes) only ever has to handle the
    list form. Old single-tool traces and any provider still producing that format keep working
    unchanged.
    """
    result = _parse_agent_json_raw(text)
    if result is None:
        return None
    if "findings" in result and "final_answer" not in result:
        result = dict(result)
        result["final_answer"] = result.pop("findings")
    if "tool" in result and "tool_calls" not in result:
        result = dict(result)
        result["tool_calls"] = [{"tool": result.pop("tool"), "args": result.pop("args", {}) or {}}]
    return result


def _extract_candidates(text: str) -> list[str]:
    """Extract all top-level balanced JSON objects found in a string."""
    if not text or not isinstance(text, str):
        return []
    cleaned = text.strip()
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL).strip()
    
    candidates = []
    stack = []
    start = -1
    in_string = False
    escape = False
    
    for i, char in enumerate(cleaned):
        if escape:
            escape = False
            continue
        if char == '\\':
            escape = True
            continue
        if char == '"':
            in_string = not in_string
            continue
            
        if not in_string:
            if char == '{':
                if not stack:
                    start = i
                stack.append('{')
            elif char == '}':
                if stack:
                    stack.pop()
                    if not stack and start != -1:
                        candidates.append(cleaned[start:i+1])
                        start = -1
    return candidates


def count_action_blobs(text: str) -> int:
    """Count how many separate action-bearing JSON objects exist in raw model output."""
    candidates = _extract_candidates(text)
    count = 0
    for cand in candidates:
        res = _try_parse_json(cand)
        if res and ("final_answer" in res or "tool" in res or "tool_calls" in res):
            count += 1
    return count


def _parse_agent_json_raw(text: str) -> Optional[dict[str, Any]]:
    """Robustly extract and parse the primary actionable JSON dictionary from model output text.

    Intelligently handles:
    1. Mixed XML and JSON: It ignores XML tags like `<fake_tool_call>...</fake_tool_call>` and searches for the standalone `{"final_answer": ...}` or `{"tool": ...}`.
    2. Markdown code fences, including nested ones.
    3. Trailing/leading text around the JSON object.
    4. Truncated JSON repair (closing unbalanced braces/brackets/strings).
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()
    
    # Explicitly strip out <think> blocks (used by models like Qwen3-VL-Thinking)
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip()

    # 1. Look for explicit XML tool calls, but we only care about the REAL final_answer or dynamic tool.
    # The real answer is usually at the very end. Let's try to find a standalone JSON blob that contains "tool" or "final_answer".
    
    # Strip markdown code fences if they exist
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if code_blocks:
        # If there are code blocks, try the last one first (usually the final answer)
        for block in reversed(code_blocks):
            result = _try_parse_json(block.strip())
            if result is not None and ("final_answer" in result or "findings" in result or "tool" in result or "tool_calls" in result):
                return result

    # 2. Extract ALL independent {} blocks and pick the best one.
    candidates = _extract_candidates(cleaned)
                        
    # --- Multi-blob merge (handles models like minimax that dump several JSON objects per turn) ---
    # Collect every parseable candidate that carries tool_calls, final_answer, or findings.
    parsed_candidates = []
    for cand in candidates:
        res = _try_parse_json(cand)
        if res and ("final_answer" in res or "findings" in res or "tool" in res or "tool_calls" in res):
            parsed_candidates.append(res)

    if len(parsed_candidates) > 1:
        # There are multiple action-bearing blobs in this single model response.
        # Merge: accumulate all tool_calls, and surface final_answer if any blob has it.
        merged_tool_calls: list = []
        merged_final_answer = None
        merged_thought_parts: list = []

        for pc in parsed_candidates:
            if pc.get("thought"):
                merged_thought_parts.append(pc["thought"])
            # Normalise legacy single-tool format into tool_calls list
            if "tool" in pc and "tool_calls" not in pc:
                merged_tool_calls.append({"tool": pc["tool"], "args": pc.get("args", {}) or {}})
            elif "tool_calls" in pc and isinstance(pc["tool_calls"], list):
                merged_tool_calls.extend(pc["tool_calls"])
            if "final_answer" in pc and merged_final_answer is None:
                merged_final_answer = pc["final_answer"]
            elif "findings" in pc and merged_final_answer is None:
                merged_final_answer = pc["findings"]

        merged: dict = {}
        if merged_thought_parts:
            merged["thought"] = " | ".join(merged_thought_parts)
        if merged_tool_calls:
            merged["tool_calls"] = merged_tool_calls
        if merged_final_answer is not None:
            merged["final_answer"] = merged_final_answer
        if merged:
            return merged

    # Single-blob path: pick the last (most recent) actionable blob
    if parsed_candidates:
        return parsed_candidates[-1]
            
    # 3. If no balanced candidates worked (truncation), try repairing the outermost block that looks like an action
    action_idx = max(cleaned.rfind('{"final_answer"'), cleaned.rfind('{"findings"'), cleaned.rfind('{"tool"'))
    if action_idx != -1:
        fragment = cleaned[action_idx:]
        result = _repair_truncated_json(fragment)
        if result is not None and ("final_answer" in result or "findings" in result or "tool" in result or "tool_calls" in result):
            return result

    # 4. Fallback: Try repairing the absolute largest block we can find just in case
    first_brace = cleaned.find("{")
    if first_brace != -1:
        result = _repair_truncated_json(cleaned[first_brace:])
        if result is not None and ("final_answer" in result or "findings" in result or "tool" in result or "tool_calls" in result):
            return result

    return None


def _try_parse_json(text: str) -> Optional[dict[str, Any]]:
    """Try to parse text as JSON dict, with trailing comma cleanup."""
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Trailing comma cleanup
    cleaned_sub = re.sub(r",\s*([\]}])", r"\1", text)
    try:
        data = json.loads(cleaned_sub)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return None


def _repair_truncated_json(fragment: str) -> Optional[dict[str, Any]]:
    """Attempt to repair a truncated JSON fragment by closing unbalanced delimiters.

    Handles cases where the API response was cut off mid-JSON, e.g.:
      {"thought": "...", "final_answer": {"quadrant": 3, "diagnosis": "Caries", "confidence": 0.
    """
    # Strip trailing partial numeric values, incomplete keys, etc.
    repaired = fragment.rstrip()

    # Remove dangling partial values at the end (e.g. "0.", incomplete strings without closing quote)
    # Strip trailing comma or colon that precedes nothing
    repaired = re.sub(r"[,:]\s*$", "", repaired)
    # Strip trailing partial number (e.g. "0." or "12")
    repaired = re.sub(r':\s*[\d.]+\s*$', ': 0', repaired)

    # Close any open strings (odd number of unescaped quotes)
    in_string = False
    escaped = False
    for ch in repaired:
        if escaped:
            escaped = False
            continue
        if ch == '\\':
            escaped = True
            continue
        if ch == '"':
            in_string = not in_string
    if in_string:
        repaired += '"'

    # Count unbalanced openers
    open_braces = repaired.count('{') - repaired.count('}')
    open_brackets = repaired.count('[') - repaired.count(']')

    # Remove trailing comma before we close
    repaired = re.sub(r",\s*$", "", repaired)

    # Close brackets then braces
    repaired += ']' * max(0, open_brackets)
    repaired += '}' * max(0, open_braces)

    try:
        data = json.loads(repaired)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    return None

