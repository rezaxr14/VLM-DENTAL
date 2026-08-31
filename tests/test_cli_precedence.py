import unittest
from unittest.mock import patch, MagicMock
from PIL import Image
import os
import argparse
from dental_agent.training.trace_generation import verify_trace, repair_and_clean_trace, _resolve_generator, _resolve_verifier
from scripts.run_trace_gen import parse_args

class TestCLIPrecedenceAndRepair(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 100), color="white")

    def test_parse_args_overrides(self):
        """CLI arguments must be parsed correctly."""
        test_cli = [
            "--mode", "verify",
            "--verifier-provider", "openrouter",
            "--verifier-model", "minimax/minimax-m3:free",
            "--generator-provider", "groq",
            "--generator-model", "llama-3.3-70b-versatile",
            "--healthy-only",
            "--no-tools"
        ]
        with patch("sys.argv", ["run_trace_gen.py"] + test_cli):
            args = parse_args()
            self.assertEqual(args.verifier_provider, "openrouter")
            self.assertEqual(args.verifier_model, "minimax/minimax-m3:free")
            self.assertEqual(args.generator_provider, "groq")
            self.assertEqual(args.generator_model, "llama-3.3-70b-versatile")
            self.assertTrue(args.healthy_only)
            self.assertTrue(args.no_tools)

    def test_verify_trace_repair_inherits_active_verifier_when_local_offline(self):
        """When GENERATOR_PROVIDER is 'local' but offline, repair must inherit active verifier provider."""
        traj = {
            "messages": [
                {"role": "assistant", "content": "I see impacted wisdom tooth 18."}
            ],
            "final_answer": [{"quadrant": 1, "tooth_position": 8, "diagnosis": "Impacted Tooth"}],
            "turns": []
        }
        gt = [] # healthy scan

        call_records = []
        def mock_call_llm(provider, model, sys_prompt, user_content, **kwargs):
            call_records.append((provider, model, kwargs.get("label")))
            if kwargs.get("label") == "verify_trace":
                if len(call_records) == 1:
                    # First check: reject
                    return '{"grounded": false, "reason": "Tooth 18 is not impacted, scan is normal."}'
                else:
                    # After repair: accept
                    return '{"grounded": true, "reason": "Repaired and confirmed healthy."}'
            elif kwargs.get("label") == "repair_trace":
                return '{"thought": "Corrected: scan is normal.", "final_answer": []}'
            return '{}'

        with patch.dict(os.environ, {"GENERATOR_PROVIDER": "local", "VERIFIER_PROVIDER": "local"}):
            with patch("dental_agent.training.trace_generation.verify_local_server_health", return_value=False):
                res = verify_trace(
                    image=self.image,
                    ground_truth=gt,
                    trajectory=traj,
                    provider="openrouter",
                    model="minimax/minimax-m3:free",
                    call_llm_fn=mock_call_llm,
                    max_repairs=1,
                )
                self.assertTrue(res.get("grounded"))
                # Verify that mock_call_llm never received ("local", ...)
                providers_called = [c[0] for c in call_records]
                self.assertNotIn("local", providers_called)
                self.assertIn("openrouter", providers_called)

    def test_repair_and_clean_trace_healthy_scan(self):
        """repair_and_clean_trace must accept final_answer: [] for healthy scans."""
        record = {
            "image_id": 3,
            "trajectory": {
                "messages": [{"role": "assistant", "content": "Hallucinated caries at 16."}],
                "final_answer": [{"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}]
            }
        }
        mock_repaired = '{"thought": "Re-evaluated all quadrants, no pathology found.", "final_answer": []}'
        with patch("dental_agent.training.trace_generation.call_llm", return_value=mock_repaired):
            with patch("dental_agent.training.trace_generation.verify_trace", return_value={"grounded": True, "reason": "Confirmed normal"}):
                ok, rep_traj, msg = repair_and_clean_trace(
                    image=self.image,
                    ground_truth=[],
                    record=record,
                    verifier_provider="openrouter",
                    verifier_model="minimax/minimax-m3:free"
                )
                self.assertTrue(ok)
                self.assertIsNotNone(rep_traj)
                self.assertEqual(rep_traj["final_answer"], [])

if __name__ == "__main__":
    unittest.main()
