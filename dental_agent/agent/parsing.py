"""
Robust JSON output parsing for agent multi-turn responses.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional


def parse_agent_json(text: str) -> Optional[dict[str, Any]]:
    """Robustly extract and parse a JSON dictionary from model output text.

    Handles:
    1. Direct JSON strings.
    2. Markdown code fences (```json ... ``` or ``` ... ```), including nested fences.
    3. Trailing/leading text around the outermost JSON object.
    4. Minor syntax fixes (e.g. trailing commas before closing braces).
    5. Truncated JSON repair (closing unbalanced braces/brackets/strings).
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()

    # 1. Strip markdown code fences — try all ```json blocks, pick the best one
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if code_blocks:
        # Try each code block, return the first that parses as a dict
        for block in code_blocks:
            result = _try_parse_json(block.strip())
            if result is not None:
                return result
        # If none parsed cleanly, use the longest block for repair attempts below
        cleaned = max(code_blocks, key=len).strip()

    # 2. Try direct json.loads
    result = _try_parse_json(cleaned)
    if result is not None:
        return result

    # 3. Locate outermost balanced curly braces
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        substring = cleaned[first_brace : last_brace + 1]
        result = _try_parse_json(substring)
        if result is not None:
            return result

    # 4. Attempt truncation repair: close unbalanced braces/brackets
    if first_brace != -1:
        fragment = cleaned[first_brace:]
        result = _repair_truncated_json(fragment)
        if result is not None:
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

