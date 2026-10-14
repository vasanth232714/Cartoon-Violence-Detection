"""Unit tests for the Verbal Abuse, Bad Words, and Profanity Detection Engine."""
from __future__ import annotations

import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from models.verbal_abuse_detector import (
    CAT_HATE,
    CAT_INSULT,
    CAT_PROFANITY,
    CAT_THREAT,
    VerbalAbuseDetector,
    VerbalAbuseSpan,
)
from pipeline.event_segmenter import CandidateEvent


class TestVerbalAbuseDetector(unittest.TestCase):
    def setUp(self):
        self.detector = VerbalAbuseDetector()

    def test_clean_text_returns_none(self):
        clean_samples = [
            "Hello there, how are you today?",
            "Look at the funny cat chasing the mouse across the yard!",
            "Good morning everybody, let us have fun!",
            "1 2 3 4 5 testing the audio microphone.",
        ]
        for s in clean_samples:
            result = self.detector.analyze_text(s)
            self.assertIsNone(result, f"Clean text was falsely flagged: {s}")

    def test_threat_detection(self):
        text = "Stop right there or I will kill you, you will pay for this!"
        span = self.detector.analyze_text(text, start_s=2.0, end_s=5.0)
        self.assertIsNotNone(span)
        self.assertIn(CAT_THREAT, span.categories)
        self.assertEqual(span.severity_level, "High")
        self.assertTrue(any("kill you" in w.lower() for w in span.flagged_words))
        self.assertNotIn("kill you", span.censored_transcript.lower())
        self.assertIn("****", span.censored_transcript)

    def test_profanity_detection(self):
        text = "What the fuck is this bullshit happening right now?"
        span = self.detector.analyze_text(text, start_s=10.0, end_s=13.0)
        self.assertIsNotNone(span)
        self.assertIn(CAT_PROFANITY, span.categories)
        self.assertIn(span.severity_level, ("Moderate", "High"))
        self.assertNotIn("fuck", span.censored_transcript.lower())

    def test_insult_bullying_detection(self):
        text = "You are a pathetic worthless loser and a complete idiot!"
        span = self.detector.analyze_text(text, start_s=1.0, end_s=4.0)
        self.assertIsNotNone(span)
        self.assertIn(CAT_INSULT, span.categories)
        self.assertTrue(any(w in ["loser", "pathetic", "worthless", "idiot"] for w in span.flagged_words))
        self.assertEqual(span.start_s, 1.0)
        self.assertEqual(span.end_s, 4.0)

    def test_shut_up_is_not_classified_as_a_threat(self):
        self.assertIsNone(self.detector.analyze_text("Shut up!", language="en"))

    def test_non_english_text_does_not_use_english_phrase_lexicons(self):
        self.assertIsNone(self.detector.analyze_text("mierda", language="es"))

    def test_toxicity_only_is_a_candidate_not_a_severity_rating(self):
        span = self.detector.analyze_text("texto de ejemplo", language="es", toxicity_score=0.96)
        self.assertIsNotNone(span)
        self.assertEqual(span.categories, ["Toxic language candidate"])
        self.assertEqual(span.severity_level, "Not assessed")
        self.assertEqual(span.flagged_words, [])

    def test_tamil_uses_tamil_abuse_model_and_verified_abusive_label(self):
        class FakeTokenizer:
            @classmethod
            def from_pretrained(cls, model_id):
                return cls()

            def __call__(self, text, **kwargs):
                return {"input_ids": torch.tensor([[1, 2]])}

        class FakeModel:
            config = types.SimpleNamespace(
                id2label={0: "LABEL_0", 1: "LABEL_1"}, num_labels=2,
            )

            @classmethod
            def from_pretrained(cls, model_id):
                return cls()

            def eval(self):
                return self

            def __call__(self, **inputs):
                return types.SimpleNamespace(logits=torch.tensor([[0.0, 3.0]]))

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoTokenizer = FakeTokenizer
        fake_transformers.AutoModelForSequenceClassification = FakeModel
        with patch.dict(sys.modules, {"transformers": fake_transformers}):
            detector = VerbalAbuseDetector(use_multilingual_classifier=True)
            score = detector._score_multilingual_toxicity("தமிழ் உரை", "ta")

        self.assertAlmostEqual(score, 0.9526, places=3)
        self.assertEqual(detector._last_toxicity_status, "scored")
        self.assertEqual(detector._last_toxicity_model_id,
                         "Hate-speech-CNERG/tamil-codemixed-abusive-MuRIL")

    def test_unsupported_language_reports_reason_instead_of_loading_wrong_model(self):
        detector = VerbalAbuseDetector(use_multilingual_classifier=True)
        self.assertIsNone(detector._score_multilingual_toxicity("తెలుగు వాక్యం", "te"))
        self.assertEqual(detector._last_toxicity_status, "unsupported_language")

    def test_censor_text(self):
        censored = self.detector.censor_text("Hey you idiot, shut up!")
        self.assertIn("****", censored)
        self.assertNotIn("idiot", censored.lower())

    def test_subtitle_parser_vtt(self):
        vtt_content = """WEBVTT

00:00:01.500 --> 00:00:04.200
Hey friend, welcome to the show!

00:00:05.100 --> 00:00:08.400
Shut up you idiot, I will destroy you!
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".vtt", delete=False, encoding="utf-8") as f:
            f.write(vtt_content)
            temp_path = Path(f.name)

        try:
            cues = self.detector._parse_subtitle_file(temp_path)
            self.assertEqual(len(cues), 2)
            self.assertAlmostEqual(cues[0][0], 1.5, places=2)
            self.assertAlmostEqual(cues[0][1], 4.2, places=2)
            self.assertEqual(cues[0][2], "Hey friend, welcome to the show!")

            self.assertAlmostEqual(cues[1][0], 5.1, places=2)
            self.assertAlmostEqual(cues[1][1], 8.4, places=2)
            self.assertIn("Shut up you idiot", cues[1][2])

            # Analyze cues directly
            span = self.detector.analyze_text(cues[1][2], start_s=cues[1][0], end_s=cues[1][1], detection_source="subtitles")
            self.assertIsNotNone(span)
            self.assertIn(CAT_THREAT, span.categories)  # explicit "destroy you", not "shut up"
            self.assertIn(CAT_INSULT, span.categories)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_subtitle_parser_srt(self):
        srt_content = """1
00:00:02,000 --> 00:00:04,500
You piece of trash!

2
00:00:06,100 --> 00:00:09,200
Just walk away peacefully.
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".srt", delete=False, encoding="utf-8") as f:
            f.write(srt_content)
            temp_path = Path(f.name)

        try:
            cues = self.detector._parse_subtitle_file(temp_path)
            self.assertEqual(len(cues), 2)
            self.assertAlmostEqual(cues[0][0], 2.0, places=2)
            self.assertAlmostEqual(cues[0][1], 4.5, places=2)
            self.assertIn("piece of trash", cues[0][2])
        finally:
            temp_path.unlink(missing_ok=True)

    def test_correlate_with_physical_events(self):
        # Physical Event 1: 0.0s to 5.0s
        # Physical Event 2: 12.0s to 16.0s
        ev1 = CandidateEvent(1, 0.0, 5.0, [0, 1], 0.9, 0.8, 0, 1, 2.5)
        ev2 = CandidateEvent(2, 12.0, 16.0, [5, 6], 0.85, 0.75, 5, 1, 14.0)
        events = [ev1, ev2]

        # Abuse span 1 overlaps with Event 1 (at 2.0s - 4.0s)
        span1 = VerbalAbuseSpan(2.0, 4.0, "I will kill you!", "****", ["kill you"], [CAT_THREAT], "High", 0.95, "subtitles")
        # Abuse span 2 is standalone (at 25.0s - 28.0s) with no physical event
        span2 = VerbalAbuseSpan(25.0, 28.0, "You idiot!", "****", ["idiot"], [CAT_INSULT], "Moderate", 0.85, "subtitles")

        matches, standalone = self.detector.correlate_with_physical_events(events, [span1, span2])

        self.assertEqual(len(matches[1]), 1)
        self.assertEqual(matches[1][0].transcript, "I will kill you!")
        self.assertEqual(len(matches[2]), 0)

        self.assertEqual(len(standalone), 1)
        self.assertEqual(standalone[0].transcript, "You idiot!")

    def test_merge_spans(self):
        s1 = VerbalAbuseSpan(2.0, 4.0, "Shut up!", "****", ["shut up"], [CAT_THREAT], "Mild", 0.75, "speech_recognition")
        s2 = VerbalAbuseSpan(4.5, 6.0, "You idiot!", "****", ["idiot"], [CAT_INSULT], "Moderate", 0.85, "speech_recognition")

        merged = self.detector._merge_spans([s1, s2], gap_tol_s=2.0)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].start_s, 2.0)
        self.assertEqual(merged[0].end_s, 6.0)
        self.assertIn("Shut up! You idiot!", merged[0].transcript)
        self.assertEqual(merged[0].severity_level, "Moderate")  # Moderate > Mild
        self.assertIn(CAT_THREAT, merged[0].categories)
        self.assertIn(CAT_INSULT, merged[0].categories)


if __name__ == "__main__":
    unittest.main()
