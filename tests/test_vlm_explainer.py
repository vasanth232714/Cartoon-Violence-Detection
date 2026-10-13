"""Unit tests for models/vlm_explainer.py."""
import unittest
import importlib.util
from unittest.mock import MagicMock, patch
from models.vlm_explainer import (
    DEFAULT_VLM_PROMPT,
    MODEL_ID,
    build_event_prompt,
    build_offline_rule_explanation,
    parse_structured_explanation,
    explain_clip_qwen,
)


class TestVLMExplainer(unittest.TestCase):
    def test_default_prompt_contains_all_seven_points(self):
        self.assertIn("1. SHORT SUMMARY", DEFAULT_VLM_PROMPT)
        self.assertIn("2. OBSERVED ACTIONS", DEFAULT_VLM_PROMPT)
        self.assertIn("3. POSSIBLE CATEGORY", DEFAULT_VLM_PROMPT)
        self.assertIn("4. TIMESTAMPED EVIDENCE", DEFAULT_VLM_PROMPT)
        self.assertIn("5. WHY IT MAY HAVE BEEN FLAGGED", DEFAULT_VLM_PROMPT)
        self.assertIn("6. UNCERTAINTY", DEFAULT_VLM_PROMPT)
        self.assertIn("7. HUMAN REVIEW", DEFAULT_VLM_PROMPT)

    def test_parse_structured_explanation(self):
        sample_output = """1. SHORT SUMMARY
Two cartoon characters are engaged in a heated sword fight.

2. OBSERVED ACTIONS
Swinging swords, dodging, and jumping between platforms.

3. POSSIBLE CATEGORY
weapon use, fight/assault

4. TIMESTAMPED EVIDENCE
VISUAL: 00:02 sword clash, 00:04 jumping kick.

5. WHY IT MAY HAVE BEEN FLAGGED
Rapid movement and possible weapon use were selected for review.

6. UNCERTAINTY
Stylized cartoon lighting effects, but sword impact is visible.

7. HUMAN REVIEW
Recommended to categorize violence severity level.
"""
        parsed = parse_structured_explanation(sample_output)
        self.assertIn("sword fight", parsed["summary"])
        self.assertIn("Swinging swords", parsed["observed_actions"])
        self.assertIn("weapon use", parsed["category"])
        self.assertIn("Rapid movement", parsed["why_flagged"])
        self.assertIn("sword clash", parsed["timestamped_evidence"])
        self.assertIn("Stylized cartoon", parsed["uncertainty"])
        self.assertIn("severity level", parsed["human_review"])

    def test_build_offline_rule_explanation(self):
        event = {
            "start_s": 5.0,
            "end_s": 10.0,
            "peak_time_s": 7.5,
            "peak_prob": 0.85,
            "severity_label": "High",
            "n_windows": 3,
        }
        res = build_offline_rule_explanation(event)
        self.assertIn("1. SHORT SUMMARY", res)
        self.assertIn("does not identify action categories", res)
        self.assertIn("not assessed", res.lower())
        parsed = parse_structured_explanation(res)
        self.assertTrue(len(parsed["summary"]) > 0)
        self.assertTrue(len(parsed["category"]) > 0)

    def test_event_prompt_keeps_modalities_separate(self):
        prompt = build_event_prompt(
            {"start_s": 10, "end_s": 12, "peak_prob": 0.9, "severity_label": "Mild"},
            [{"start_s": 10.5, "end_s": 11, "language": "es", "categories": ["Toxic language candidate"], "transcript": "texto"}],
            [{"start_s": 10.2, "end_s": 10.7, "category": "Gunshot-like sound", "confidence": 0.8}],
            clip_offset_s=9.5,
        )
        self.assertIn("do not treat it as a visual observation", prompt)
        self.assertIn("language=es", prompt)
        self.assertIn("Gunshot-like sound", prompt)
        self.assertIn("original video time 9.50s", prompt)

    @unittest.skipIf(importlib.util.find_spec("torch") is None, "requires torch")
    def test_explain_clip_qwen_with_mock(self):
        mock_model = MagicMock()
        mock_processor = MagicMock()
        import torch

        # Mock processor behavior
        mock_processor.apply_chat_template.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
        mock_processor.batch_decode.return_value = ["1. SHORT SUMMARY\nMock explanation result."]

        with patch("os.path.isfile", return_value=True):
            res = explain_clip_qwen(
                "dummy.mp4",
                fps=0.5,
                max_tokens=100,
                model=mock_model,
                processor=mock_processor,
            )
            self.assertIn("Mock explanation result", res)


if __name__ == "__main__":
    unittest.main()
