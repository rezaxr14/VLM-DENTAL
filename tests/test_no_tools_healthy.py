import unittest
from unittest.mock import patch
from PIL import Image
from dental_agent.training.trace_generation import generate_no_tools_trajectory

class TestNoToolsTrajectory(unittest.TestCase):
    def setUp(self):
        self.image = Image.new("RGB", (100, 100), color="gray")

    def test_healthy_scan_final_answer_empty_list(self):
        """Healthy scan where model outputs final_answer: []"""
        mock_raw = '{"thought": "All quadrants healthy and intact", "final_answer": []}'
        with patch("dental_agent.training.trace_generation.call_llm", return_value=mock_raw):
            traj, err = generate_no_tools_trajectory(self.image, ground_truth=[])
            self.assertIsNone(err)
            self.assertIsNotNone(traj)
            self.assertTrue(traj["format_ok"])
            self.assertEqual(traj["final_answer"], [])

    def test_healthy_scan_findings_empty_list(self):
        """Healthy scan where model outputs findings: []"""
        mock_raw = '{"thought": "All quadrants healthy and intact", "findings": []}'
        with patch("dental_agent.training.trace_generation.call_llm", return_value=mock_raw):
            traj, err = generate_no_tools_trajectory(self.image, ground_truth=[])
            self.assertIsNone(err)
            self.assertIsNotNone(traj)
            self.assertTrue(traj["format_ok"])
            self.assertEqual(traj["final_answer"], [])

    def test_abnormal_scan_with_findings(self):
        """Abnormal scan with 1 caries finding"""
        mock_raw = '{"thought": "Caries detected at tooth 16", "final_answer": [{"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries", "confidence": 0.95}]}'
        gt = [{"quadrant": 1, "tooth_position": 6, "diagnosis": "Caries"}]
        with patch("dental_agent.training.trace_generation.call_llm", return_value=mock_raw):
            traj, err = generate_no_tools_trajectory(self.image, ground_truth=gt)
            self.assertIsNone(err)
            self.assertIsNotNone(traj)
            self.assertTrue(traj["format_ok"])
            self.assertEqual(len(traj["final_answer"]), 1)
            self.assertEqual(traj["final_answer"][0]["diagnosis"], "Caries")

if __name__ == "__main__":
    unittest.main()
