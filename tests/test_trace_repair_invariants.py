import unittest
from unittest.mock import patch
from PIL import Image
from dental_agent.training.trace_generation import verify_trace

class TestTraceRepairInvariants(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 100), color="white")

    def test_healthy_scan_with_hallucinated_findings_deterministically_rejected(self):
        """If ground_truth is [] but candidate reported findings, it MUST be rejected immediately."""
        traj = {
            "messages": [{"role": "assistant", "content": "I see impacted tooth 18."}],
            "final_answer": [{"quadrant": 1, "tooth_position": 8, "diagnosis": "Impacted Tooth"}],
            "tool_calls": 5,
            "turns": [{"turn": 0}, {"turn": 1}]
        }
        res = verify_trace(
            image=self.image,
            ground_truth=[],
            trajectory=traj,
            provider="openrouter",
            model="minimax/minimax-m3:free",
        )
        self.assertFalse(res["grounded"])
        self.assertIn("Ground truth is empty", res["reason"])

    def test_tool_based_trace_never_triggers_fake_single_shot_repair(self):
        """Tool-based traces must not attempt single-shot text repair when rejected."""
        traj = {
            "messages": [{"role": "assistant", "content": "I see caries at 16."}],
            "final_answer": [{"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}],
            "tool_calls": 8,
            "turns": [{"turn": 0}, {"turn": 1}, {"turn": 2}]
        }
        called_labels = []
        def mock_call_llm(provider, model, sys_prompt, user_content, **kwargs):
            called_labels.append(kwargs.get("label"))
            return '{"grounded": false, "reason": "Caries at 16 is unsupported."}'

        res = verify_trace(
            image=self.image,
            ground_truth=[{"quadrant": 1, "tooth_position": 6, "diagnosis": "Periapical Lesion"}],
            trajectory=traj,
            provider="openrouter",
            model="minimax/minimax-m3:free",
            call_llm_fn=mock_call_llm,
            max_repairs=1,
        )
        self.assertFalse(res["grounded"])
        self.assertNotIn("repair_trace", called_labels)

    def test_no_tools_trace_allows_single_shot_repair(self):
        """No-tools traces (single-turn) are allowed to perform text repair."""
        traj = {
            "messages": [{"role": "assistant", "content": "Thought about caries at 16."}],
            "final_answer": [{"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}],
            "tool_calls": 0,
            "turns": [{"turn": 0}]
        }
        called_labels = []
        def mock_call_llm(provider, model, sys_prompt, user_content, **kwargs):
            lbl = kwargs.get("label")
            called_labels.append(lbl)
            if lbl == "verify_trace":
                if len(called_labels) == 1:
                    return '{"grounded": false, "reason": "Should be Periapical Lesion."}'
                else:
                    return '{"grounded": true, "reason": "Fixed and verified."}'
            elif lbl == "repair_trace":
                return '{"thought": "Corrected to Periapical Lesion", "final_answer": [{"quadrant": 1, "tooth_position": 6, "diagnosis": "Periapical Lesion"}]}'
            return '{}'

        res = verify_trace(
            image=self.image,
            ground_truth=[{"quadrant": 1, "tooth_position": 6, "diagnosis": "Periapical Lesion"}],
            trajectory=traj,
            provider="openrouter",
            model="minimax/minimax-m3:free",
            call_llm_fn=mock_call_llm,
            max_repairs=1,
        )
        self.assertTrue(res["grounded"])
        self.assertIn("repair_trace", called_labels)

if __name__ == "__main__":
    unittest.main()
