import pytest
from dental_agent.training.rewards import (
    accuracy_reward,
    format_reward,
    efficiency_reward,
    tool_validity_reward
)

def test_accuracy_reward():
    gt = {"quadrant": 3, "tooth_position": 8, "diagnosis": "caries"}
    
    # 1.0: Perfect match
    assert accuracy_reward({"quadrant": 3, "tooth_position": 8, "diagnosis": "Caries"}, gt) == 1.0
    
    # 0.5: Correct tooth, wrong disease
    assert accuracy_reward({"quadrant": 3, "tooth_position": 8, "diagnosis": "impacted"}, gt) == 0.5
    
    # 0.25: Correct quadrant, wrong tooth
    assert accuracy_reward({"quadrant": 3, "tooth_position": 7, "diagnosis": "caries"}, gt) == 0.25
    
    # 0.0: Total miss
    assert accuracy_reward({"quadrant": 1, "tooth_position": 8, "diagnosis": "caries"}, gt) == 0.0
    
def test_format_reward():
    # Perfect format
    good_completion = "<think>\nLet's zoom in.\n</think>\n{\"tool\": \"zoom_crop\"}"
    assert format_reward(good_completion) == 1.0
    
    # Missing tags
    bad_completion1 = "Let's zoom in. {\"tool\": \"zoom_crop\"}"
    assert format_reward(bad_completion1) == -1.0
    
    # Malformed JSON
    bad_completion2 = "<think>\nLet's zoom in.\n</think>\n{\"tool\": \"zoom_crop\""
    assert format_reward(bad_completion2) == -1.0

def test_efficiency_reward():
    # 0 calls: -0.2
    assert efficiency_reward(0) == -0.2
    
    # 1 to 4 calls: +0.1 per call
    assert efficiency_reward(1) == pytest.approx(0.1)
    assert efficiency_reward(4) == pytest.approx(0.4)
    
    # 5+ calls: penalty
    # 5 calls = (4 * 0.1) - (1 * 0.5) = 0.4 - 0.5 = -0.1
    assert efficiency_reward(5) == pytest.approx(-0.1)
    # 10 calls = (4 * 0.1) - (6 * 0.5) = 0.4 - 3.0 = -2.6
    assert efficiency_reward(10) == pytest.approx(-2.6)

def test_tool_validity_reward():
    registry = ["zoom_crop", "window_level"]
    
    # Valid
    assert tool_validity_reward({"tool": "zoom_crop"}, registry) == 0.2
    
    # Invalid tool
    assert tool_validity_reward({"tool": "fake_tool"}, registry) == -0.5
    
    # Invalid format
    assert tool_validity_reward("not a dict", registry) == -0.5
