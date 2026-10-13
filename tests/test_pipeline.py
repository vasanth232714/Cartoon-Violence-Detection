"""Pipeline integration with a FAKE classifier (test fixture). This verifies plumbing - timestamps,
coverage, merging, evidence, persistence - NOT model quality. Real-model tests are in
test_real_model.py and are skipped unless torch, pytorchvideo and the checkpoint are available."""
import os
import tempfile
import unittest

from config.settings import AnalysisConfig
from database.repository import Repository
from pipeline.inference_pipeline import AnalysisCancelled, AnalysisFailed, run_analysis
from preprocessing.video_loader import validate_video
from tests.helpers import FakeBrightnessClassifier, RaisingClassifier, make_video


class TestPipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.repo = Repository(os.path.join(self.tmp.name, "t.db"))
        self.ev_root = __import__("pathlib").Path(self.tmp.name) / "evidence"

    def tearDown(self):
        if hasattr(self, "repo") and self.repo:
            self.repo.close()
        self.tmp.cleanup()

    def _run(self, video, clf, **kw):
        meta = validate_video(video)
        vid = self.repo.add_video(os.path.basename(video) + str(id(clf)), "v", video, meta.to_dict())
        cfg = kw.pop("config", AnalysisConfig(enable_qwen_explanations=False,
                                               enable_verbal_abuse_detection=False,
                                               enable_audio_event_detection=False,
                                               use_temporal_transformer=False))
        return run_analysis(vid, video, meta, cfg, clf, self.repo, evidence_root=self.ev_root, **kw), meta

    def test_full_run_with_event(self):
        v = make_video(os.path.join(self.tmp.name, "a.mp4"))
        calls = []
        run_id, meta = self._run(v, FakeBrightnessClassifier(), progress=lambda s, f, m: calls.append(f))
        run = self.repo.get_run(run_id)
        self.assertEqual(run["status"], "completed")
        wins = self.repo.list_windows(run_id)
        self.assertAlmostEqual(wins[-1]["end_s"], meta.duration_s, delta=0.2)  # tail covered
        self.assertEqual(run["windows_total"], len(wins))
        self.assertTrue(all(w["p_violent"] is not None for w in wins))
        events = self.repo.list_events(run_id)
        self.assertGreaterEqual(len(events), 1)
        e = events[0]
        self.assertLessEqual(e["start_s"], 5.0 + 0.01)
        self.assertGreaterEqual(e["end_s"], 9.0 - 0.01)
        ev_frames = self.repo.list_evidence(e["id"])
        self.assertGreater(len(ev_frames), 0)
        for f in ev_frames:
            self.assertTrue(os.path.isfile(f["path"]))
            self.assertTrue(e["start_s"] - 0.2 <= f["timestamp_s"] <= e["end_s"] + 0.2)
        self.assertIn(e["severity_label"], ("Mild", "Moderate", "High", "Not assessed"))
        self.assertEqual(e["description_method"].split()[0], "template")
        self.assertEqual(sorted(calls), calls)  # progress is monotonic
        self.assertEqual(calls[-1], 1.0)

    def test_no_violence_gives_no_events_not_fake_events(self):
        v = make_video(os.path.join(self.tmp.name, "dark.mp4"), bright=None)
        run_id, _ = self._run(v, FakeBrightnessClassifier())
        self.assertEqual(self.repo.get_run(run_id)["status"], "completed")
        self.assertEqual(self.repo.list_events(run_id), [])

    def test_qwen_explanation_is_generated_during_full_video_run(self):
        from unittest.mock import patch
        v = make_video(os.path.join(self.tmp.name, "qwen.mp4"))
        cfg = AnalysisConfig(enable_qwen_explanations=True, qwen_max_events=1,
                             enable_verbal_abuse_detection=False,
                             enable_audio_event_detection=False,
                             use_temporal_transformer=False)
        with patch("pipeline.inference_pipeline._qwen_explain_interval", return_value="1. SHORT SUMMARY\nA cartoon character falls.") as explain:
            run_id, _ = self._run(v, FakeBrightnessClassifier(), config=cfg)
        events = self.repo.list_events(run_id)
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0]["qwen_status"], "generated")
        self.assertIn("cartoon character falls", events[0]["qwen_explanation"])
        explain.assert_called_once()

    def test_optional_videomae_scores_are_saved_separately_from_x3d(self):
        from unittest.mock import patch
        v = make_video(os.path.join(self.tmp.name, "videomae.mp4"))
        cfg = AnalysisConfig(enable_videomae_comparison=True,
                             enable_verbal_abuse_detection=False,
                             enable_audio_event_detection=False,
                             use_temporal_transformer=False)
        def fake_compare(video_path, meta, windows, progress=None):
            if progress:
                progress(1.0, "done")
            return ([{"window_index": i, "start_s": start, "end_s": end,
                      "p_violent": 0.8, "status": "ok", "inference_ms": 1.0, "note": ""}
                     for i, (start, end) in enumerate(windows)],
                    {"status": "completed", "model_id": "HappyGook/videomae-violence-detector",
                     "num_frames": 16, "input_size": 224, "device": "cpu"})
        with patch("models.videomae_detector.run_videomae_comparison", side_effect=fake_compare):
            run_id, _ = self._run(v, FakeBrightnessClassifier(), config=cfg)
        x3d = self.repo.list_windows(run_id)
        videomae = self.repo.list_model_window_scores(run_id, "videomae")
        self.assertEqual(len(videomae), len(x3d))
        self.assertTrue(all(row["p_violent"] == 0.8 for row in videomae))
        self.assertTrue(all(row["p_violent"] != 0.8 for row in x3d))
        import json
        info = json.loads(self.repo.get_run(run_id)["model_info_json"])
        self.assertEqual(info["videomae_comparison"]["status"], "completed")

    def test_explicit_threat_severity_and_explanation_stay_consistent(self):
        from unittest.mock import patch
        from models.verbal_abuse_detector import CAT_THREAT, VerbalAbuseSpan
        v = make_video(os.path.join(self.tmp.name, "threat.mp4"))
        span = VerbalAbuseSpan(6.0, 7.0, "I will kill you", "I will **** you", ["kill you"],
                               [CAT_THREAT], "High", 0.95, "faster_whisper", language="en")
        cfg = AnalysisConfig(enable_qwen_explanations=False, enable_verbal_abuse_detection=True,
                             enable_multilingual_toxicity=False, enable_audio_event_detection=False,
                             use_temporal_transformer=False)
        with patch("models.verbal_abuse_detector.VerbalAbuseDetector.analyze_video_full",
                   return_value=([span], [], {"status": "completed", "language": "en"}, [])):
            run_id, _ = self._run(v, FakeBrightnessClassifier(), config=cfg)
        event = self.repo.list_events(run_id)[0]
        self.assertEqual(event["severity_label"], "High")
        self.assertIn("Severity estimate:** High", event["explanation"])
        self.assertIn("explicit_verbal_threat", event["severity_method"])

    def test_model_errors_make_run_incomplete(self):
        v = make_video(os.path.join(self.tmp.name, "b.mp4"))
        run_id, _ = self._run(v, RaisingClassifier())
        run = self.repo.get_run(run_id)
        self.assertEqual(run["status"], "incomplete")
        self.assertEqual(self.repo.list_events(run_id), [])
        self.assertTrue(all(w["status"] == "error" for w in self.repo.list_windows(run_id)))

    def test_cancellation_marks_interrupted(self):
        v = make_video(os.path.join(self.tmp.name, "c.mp4"))
        n = {"i": 0}

        def cancel():
            n["i"] += 1
            return n["i"] > 2

        with self.assertRaises(AnalysisCancelled):
            self._run(v, FakeBrightnessClassifier(), should_cancel=cancel)
        self.assertEqual(self.repo.list_runs()[0]["status"], "interrupted")

    def test_unexpected_failure_marks_failed(self):
        v = make_video(os.path.join(self.tmp.name, "d.mp4"))
        meta = validate_video(v)
        vid = self.repo.add_video("zz", "v", v, meta.to_dict())
        os.remove(v)  # video disappears before analysis
        with self.assertRaises(AnalysisFailed) as cm:
            run_analysis(vid, v, meta, AnalysisConfig(), FakeBrightnessClassifier(), self.repo,
                         evidence_root=self.ev_root)
        self.assertEqual(self.repo.get_run(cm.exception.run_id)["status"], "failed")


if __name__ == "__main__":
    unittest.main()
