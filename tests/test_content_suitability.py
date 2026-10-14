import unittest
from types import SimpleNamespace
from unittest.mock import patch

from pipeline.content_suitability import (
    analyze_text_suitability,
    build_suitability_assessment,
    classify_title,
    scan_transcript_segments,
    x3d_threshold_from_text_score,
)


class TestContentSuitability(unittest.TestCase):
    def test_known_rhyme_titles_match_after_normalization(self):
        for title in ("JOHNNY, Johnny! Yes Papa", "Wheels   on the Bus!!!"):
            with self.subTest(title=title):
                result = classify_title(title, "youtube")
                self.assertTrue(result["strong_match"])
                self.assertIn("Positive kids-content indicator", result["preliminary_result"])
                self.assertNotIn("guarantee", result["preliminary_result"].lower())

    def test_misleading_title_does_not_match_a_rhyme_phrase(self):
        result = classify_title("Wheels accident on the bus crash compilation")
        self.assertFalse(result["strong_match"])
        self.assertEqual(result["matched_phrases"], [])

    def test_soldier_filename_is_conflict_context_and_needs_video_audio_review(self):
        fake_textblob = SimpleNamespace(TextBlob=lambda text: SimpleNamespace(
            sentiment=SimpleNamespace(polarity=0.0)))
        with patch.dict("sys.modules", {"textblob": fake_textblob}):
            result = classify_title("soldiers_in_Afghanistan_R.A.S", "filename")
        self.assertEqual(result["normalized_title"], "soldiers in afghanistan r a s")
        self.assertIn("soldiers", result["conflict_phrases"])
        self.assertIn("afghanistan", result["conflict_phrases"])
        self.assertEqual(result["context_result"], "Potential military/conflict content — video/audio review needed.")
        self.assertEqual(result["title_suitability"], "Needs video/audio review")
        self.assertEqual(result["title_sentiment"]["label"], "Neutral")
        self.assertEqual(result["title_sentiment"]["polarity"], 0.0)

    def test_army_afganistan_filename_is_not_classified_as_neutral(self):
        title = classify_title("army_in_afganistan_rass", "filename")
        self.assertIn("army", title["conflict_phrases"])
        self.assertIn("afganistan", title["conflict_phrases"])
        self.assertEqual(title["title_suitability"], "Needs video/audio review")
        text = analyze_text_suitability(title_context=title, transcript_status="pending")
        self.assertEqual(text["score"], 3.8)
        self.assertEqual(text["classification"], "Higher concern / review required")
        self.assertEqual(text["contributions"]["title_context_penalty_applied"], -2.2)

    def test_war_title_scores_3_8_without_stacking_conflict_and_violence_clues(self):
        title = classify_title("War scene in Afghanistan", "youtube")
        text = analyze_text_suitability(title_context=title, transcript_status="pending")
        self.assertIn("war", title["conflict_phrases"])
        self.assertIn("war scene", title["concerning_phrases"])
        self.assertEqual(text["score"], 3.8)
        self.assertEqual(text["contributions"]["title_context_penalty_applied"], -2.2)

    def test_title_matches_ignore_case_and_hyphen_variants(self):
        result = classify_title("JOHNNY-JOHNNY-YES-PAPA", "youtube")
        self.assertTrue(result["strong_match"])
        self.assertIn("johnny johnny yes papa", result["normalized_title"])
        self.assertIn("positive kids-content indicator", result["context_result"].lower())
        self.assertEqual(result["title_suitability"], "Kids-content indicator; detailed analysis pending")

    def test_mixed_kids_and_conflict_title_requires_human_review(self):
        title = classify_title("Preschool nursery rhyme about soldiers", "youtube")
        self.assertTrue(title["strong_match"])
        self.assertTrue(title["has_conflict_context"])
        self.assertTrue(title["has_mixed_kids_conflict_context"])
        result = self._assess(title_context=title)
        self.assertEqual(result["category"], "Needs human review")

    def test_conflict_keyword_does_not_match_inside_warrior(self):
        result = classify_title("Warrior Cats Preschool Story")
        self.assertFalse(result["has_conflict_context"])
        self.assertTrue(result["strong_match"])

    def test_known_rhyme_title_score_is_preliminary_not_a_safety_result(self):
        title = classify_title("Johnny Johnny Yes Papa", "youtube")
        result = analyze_text_suitability(title_context=title, transcript_status="pending")
        self.assertEqual(result["score"], 7.2)
        self.assertEqual(result["classification"], "Neutral / needs further analysis")
        self.assertFalse(result["validated"])

    def test_violent_youtube_title_scores_in_concern_band(self):
        title = classify_title("Most_Violent_Looney_Tunes_Moment_Ever", "youtube")
        result = analyze_text_suitability(title_context=title, transcript_status="pending")
        self.assertIn("violent", title["concerning_phrases"])
        self.assertEqual(result["score"], 4.5)
        self.assertEqual(result["classification"], "Potentially unsuitable")

    def test_preliminary_text_score_maps_to_x3d_threshold(self):
        self.assertEqual(x3d_threshold_from_text_score(6.0), 0.6)
        self.assertEqual(x3d_threshold_from_text_score(4.5), 0.45)
        self.assertEqual(x3d_threshold_from_text_score(3.8), 0.38)
        with self.assertRaises(ValueError):
            x3d_threshold_from_text_score(10.1)

    def test_violent_filename_gets_weaker_but_still_concerning_score(self):
        title = classify_title("Most_Violent_Looney_Tunes_Moment_Ever", "filename")
        result = analyze_text_suitability(title_context=title, transcript_status="pending")
        self.assertEqual(result["score"], 5.25)
        self.assertEqual(result["classification"], "Potentially unsuitable")

    def test_positive_lyric_evidence_reaches_7_5_without_title_only_shortcut(self):
        fake_textblob = SimpleNamespace(TextBlob=lambda text: SimpleNamespace(
            sentiment=SimpleNamespace(polarity=0.0)))
        with patch.dict("sys.modules", {"textblob": fake_textblob}):
            result = analyze_text_suitability(
                title_context=classify_title("An unrelated upload", "youtube"),
                transcript_segments=[{"text": "The wheels on the bus go round and round.", "language": "en"}],
                transcript_status="completed",
            )
        self.assertEqual(result["score"], 7.5)
        self.assertEqual(result["classification"], "Likely child-friendly")

    def test_concerning_title_suppresses_positive_lyric_signal(self):
        fake_textblob = SimpleNamespace(TextBlob=lambda text: SimpleNamespace(
            sentiment=SimpleNamespace(polarity=0.0)))
        with patch.dict("sys.modules", {"textblob": fake_textblob}):
            result = analyze_text_suitability(
                title_context=classify_title("Violent Wheels on the Bus", "youtube"),
                transcript_segments=[{"text": "The wheels on the bus go round and round.", "language": "en"}],
                transcript_status="completed",
            )
        self.assertEqual(result["score"], 4.5)
        self.assertEqual(result["contributions"]["kids_context_combined_max_title_or_lyrics"], 0.0)

    def test_neutral_english_text_gets_bounded_context_only_sentiment(self):
        fake_textblob = SimpleNamespace(TextBlob=lambda text: SimpleNamespace(
            sentiment=SimpleNamespace(polarity=0.0)))
        with patch.dict("sys.modules", {"textblob": fake_textblob}):
            result = analyze_text_suitability(
                title_context=classify_title("A plain documentary"),
                transcript_segments=[{"text": "The weather is mild today.", "language": "en"}],
                transcript_status="completed",
            )
        self.assertEqual(result["sentiment"]["status"], "completed")
        self.assertEqual(result["sentiment"]["polarity"], 0.0)
        self.assertEqual(result["sentiment"]["contribution"], 0.0)
        self.assertIn("does not measure child safety", result["sentiment"]["note"])

    def test_abusive_phrase_lowers_score_and_suppresses_correlated_sentiment(self):
        fake_textblob = SimpleNamespace(TextBlob=lambda text: SimpleNamespace(
            sentiment=SimpleNamespace(polarity=-0.9)))
        with patch.dict("sys.modules", {"textblob": fake_textblob}):
            result = analyze_text_suitability(
                title_context=classify_title("Johnny Johnny Yes Papa", "youtube"),
                transcript_segments=[{"text": "I will kill you.", "language": "en"}],
                verbal_findings=[{"category": "threat", "phrase": "kill you"}],
                transcript_status="completed",
            )
        self.assertEqual(result["contributions"]["sentiment_context_only"], 0.0)
        self.assertLess(result["score"], 6.0)
        self.assertEqual(result["classification"], "Higher concern / review required")

    def test_missing_transcript_is_partial_and_never_likely_child_friendly_from_title(self):
        result = analyze_text_suitability(
            title_context=classify_title("Wheels on the Bus", "youtube"),
            transcript_status="failed",
        )
        self.assertEqual(result["transcript_status"], "unavailable")
        self.assertTrue(result["uncertainty"])
        self.assertNotEqual(result["classification"], "Likely child-friendly")

    def test_phrase_scanner_handles_english_negation_and_tamil(self):
        findings = scan_transcript_segments([
            {"text": "I will not kill you", "language": "en", "start_s": 1, "end_s": 2},
            {"text": "உன்னை கொன்று விடுவேன்", "language": "ta", "start_s": 3, "end_s": 4},
        ])
        self.assertEqual(len(findings), 1)
        self.assertEqual(findings[0]["language"], "ta")
        self.assertIsNone(findings[0]["score"])

    def _assess(self, **overrides):
        values = dict(
            title_context=classify_title("Johnny Johnny Yes Papa", "youtube"),
            visual_status="completed", scored_visual_windows=20,
            audio_status="completed", audio_metadata={"status": "completed"},
            speech_status="completed", speech_metadata={}, transcript_count=3,
            x3d_events=[], audio_findings=[], verbal_findings=[], toxicity_scores=[0.1],
            toxicity_status="completed", run_status="completed",
        )
        values.update(overrides)
        return build_suitability_assessment(**values)

    def test_concerning_visual_evidence_overrides_kids_title(self):
        result = self._assess(x3d_events=[{"start_s": 10, "end_s": 15}])
        self.assertEqual(result["category"], "Potential violence or verbal abuse")
        self.assertTrue(any("did not override" in reason for reason in result["reasons"]))

    def test_missing_audio_is_incomplete_not_safe(self):
        result = self._assess(audio_status="failed", audio_metadata={"status": "failed"})
        self.assertEqual(result["category"], "Incomplete analysis")
        audio = next(source for source in result["sources"] if source["source"] == "Sound events")
        self.assertEqual(audio["status"], "unavailable")

    def test_incomplete_run_is_never_labeled_child_friendly(self):
        result = self._assess(run_status="interrupted")
        self.assertEqual(result["category"], "Incomplete analysis")

    def test_missing_toxicity_scores_are_not_interpreted_as_negative(self):
        result = self._assess(toxicity_scores=[], toxicity_status="unavailable")
        self.assertEqual(result["category"], "Incomplete analysis")
        toxicity = next(source for source in result["sources"] if source["source"] == "Toxicity / abuse scoring")
        self.assertEqual(toxicity["status"], "unavailable")

    def test_conflict_title_alone_requires_review_not_unsafe_or_safe(self):
        result = self._assess(title_context=classify_title("military-operation documentary", "youtube"))
        self.assertEqual(result["category"], "Needs human review")


if __name__ == "__main__":
    unittest.main()
