"""Plumbing checks for PDF export and evaluation runner (FAKE classifier fixture; not model quality)."""
import os
import pathlib
import tempfile
import unittest
import importlib.util

from config.settings import AnalysisConfig
from database.repository import Repository
from evaluation.evaluation_runner import evaluate
from pipeline.inference_pipeline import run_analysis
from pipeline.report_pdf import build_pdf
from preprocessing.video_loader import validate_video
from tests.helpers import FakeBrightnessClassifier, make_video


class TestReportAndEval(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.repo = Repository(os.path.join(self.tmp.name, "t.db"))
        v = make_video(os.path.join(self.tmp.name, "a.mp4"))
        meta = validate_video(v)
        vid = self.repo.add_video("h", "a.mp4", v, meta.to_dict())
        self.run = run_analysis(vid, v, meta, AnalysisConfig(enable_qwen_explanations=False,
                                                             enable_verbal_abuse_detection=False,
                                                             enable_audio_event_detection=False,
                                                             use_temporal_transformer=False), FakeBrightnessClassifier(), self.repo,
                                evidence_root=pathlib.Path(self.tmp.name) / "ev")

    def tearDown(self):
        if hasattr(self, "repo") and self.repo:
            self.repo.close()
        self.tmp.cleanup()

    @unittest.skipIf(importlib.util.find_spec("reportlab") is None, "requires reportlab")
    def test_pdf_built_from_stored_results(self):
        eid = self.repo.list_events(self.run)[0]["id"]
        self.repo.add_review(eid, "confirmed", "ok")
        pdf = build_pdf(self.repo, self.run)
        self.assertTrue(pdf.startswith(b"%PDF"))
        self.assertGreater(len(pdf), 1500)

    def test_evaluation_against_ground_truth(self):
        ann = {"events": [{"start": 5.0, "end": 9.0, "severity": 2}]}
        res = evaluate(self.repo, [(self.run, ann)])
        self.assertEqual(res["n_runs_used"], 1)
        self.assertIn("accuracy", res["window_level"])
        self.assertEqual(res["event_level"]["tIoU>=0.3"]["tp"] + res["event_level"]["tIoU>=0.3"]["fn"], 1)

    def test_incomplete_run_not_evaluated(self):
        rid = self.repo.create_run(1, "k", {}, "{}", "x", "0")
        res = evaluate(self.repo, [(rid, {"events": []})])
        self.assertEqual(res["n_runs_used"], 0)
        self.assertEqual(len(res["skipped"]), 1)


if __name__ == "__main__":
    unittest.main()
