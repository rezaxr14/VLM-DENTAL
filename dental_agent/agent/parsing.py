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
    2. Markdown code fences (```json ... ``` or ``` ... ```).
    3. Trailing/leading text around the outermost JSON object.
    4. Minor syntax fixes (e.g. trailing commas before closing braces).
    """
    if not text or not isinstance(text, str):
        return None

    cleaned = text.strip()

    # 1. Check for markdown code blocks
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", cleaned, re.IGNORECASE)
    if code_block_match:
        cleaned = code_block_match.group(1).strip()

    # 2. Try direct json.loads
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # 3. Locate outermost balanced curly braces
    first_brace = cleaned.find("{")
    last_brace = cleaned.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        substring = cleaned[first_brace : last_brace + 1]
        try:
            data = json.loads(substring)
            if isinstance(data, dict):
                return data
        except Exception:
            # 4. Attempt simple trailing comma cleanup
            cleaned_sub = re.sub(r",\s*([\]}])", r"\1", substring)
            try:
                data = json.loads(cleaned_sub)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass

    return None
