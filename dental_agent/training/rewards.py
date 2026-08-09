import json
import re
from typing import Dict, Any, List

def accuracy_reward(prediction: Dict[str, Any], ground_truth: Dict[str, Any]) -> float:
    """
    Grades the final prediction against the ground truth.
    - 1.0 (Full Credit): Correct quadrant, correct FDI tooth number, AND correct disease.
    - 0.5 (Partial Credit): Correct quadrant and FDI tooth number, but wrong disease diagnosis.
    - 0.25 (Small Credit): Correct quadrant, but wrong tooth and wrong disease.
    - 0.0 (No Credit): Total miss.
    """
    if not isinstance(prediction, dict) or not isinstance(ground_truth, dict):
        return 0.0
        
    pred_q = prediction.get("quadrant")
    pred_t = prediction.get("tooth_position")
    pred_d = str(prediction.get("diagnosis", "")).lower().strip()
    
    gt_q = ground_truth.get("quadrant")
    gt_t = ground_truth.get("tooth_position")
    gt_d = str(ground_truth.get("diagnosis", "")).lower().strip()
    
    if pred_q == gt_q and pred_t == gt_t and pred_d == gt_d:
        return 1.0
    elif pred_q == gt_q and pred_t == gt_t:
        return 0.5
    elif pred_q == gt_q:
        return 0.25
        
    return 0.0

def format_reward(completion: str) -> float:
    """
    Ensures the agent's output is parsable.
    - +1.0: Properly uses <think>...</think> tags and outputs a valid JSON block.
    - -1.0: Missing tags or malformed JSON.
    """
    # Check for <think> tags
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    think_match = think_pattern.search(completion)
    
    if not think_match:
        return -1.0
        
    # The rest of the string after </think> should be a parseable JSON
    after_think = completion[think_match.end():].strip()
    if not after_think:
        return -1.0
        
    try:
        json.loads(after_think)
        return 1.0
    except json.JSONDecodeError:
        return -1.0

def efficiency_reward(num_tool_calls: int) -> float:
    """
    Budget approach for tool usage to encourage exploration but penalize spam.
    - 0 Tool Calls: 0.0 reward (or slight penalty to discourage zero-shot)
    - 1 to 4 Tool Calls: +0.1 reward per tool call (Encourages exploration)
    - 5+ Tool Calls: -0.5 penalty for every tool call beyond the budget of 4.
    """
    if num_tool_calls == 0:
        return -0.2 # Slight penalty for guessing without looking
        
    if 1 <= num_tool_calls <= 4:
        return 0.1 * num_tool_calls
        
    # If num_tool_calls > 4
    excess_calls = num_tool_calls - 4
    return (0.1 * 4) - (0.5 * excess_calls)

def tool_validity_reward(tool_call_json: Dict[str, Any], registry_tool_names: List[str]) -> float:
    """
    - +0.2: Called a valid tool.
    - -0.5: Hallucinated a tool name that doesn't exist or malformed call.
    """
    if not isinstance(tool_call_json, dict):
        return -0.5
        
    tool_name = tool_call_json.get("tool")
    if not tool_name:
        return -0.5
        
    if tool_name in registry_tool_names:
        return 0.2
        
    return -0.5
