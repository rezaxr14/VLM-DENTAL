import json
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from dental_agent.agent.prompts import NO_TOOLS_COT_TEACHER_PROMPT, ZERO_SHOT_PROMPT, build_agent_system_prompt
from dental_agent.training.trace_generation import VERIFIER_SYSTEM_PROMPT
from dental_agent.tools.registry import ToolRegistry

registry = ToolRegistry.create_default()
agent_prompt = build_agent_system_prompt(registry.format_tool_descriptions())

with open("scratch_prompts.md", "w") as f:
    f.write("# AGENT_SYSTEM_PROMPT\n```text\n" + agent_prompt + "\n```\n\n")
    f.write("# NO_TOOLS_COT_TEACHER_PROMPT\n```text\n" + NO_TOOLS_COT_TEACHER_PROMPT + "\n```\n\n")
    f.write("# ZERO_SHOT_PROMPT\n```text\n" + ZERO_SHOT_PROMPT + "\n```\n\n")
    f.write("# VERIFIER_SYSTEM_PROMPT\n```text\n" + VERIFIER_SYSTEM_PROMPT + "\n```\n\n")
