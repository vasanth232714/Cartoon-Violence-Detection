import os
import tempfile
import unittest

from database.repository import Repository

META = {"file_size_bytes": 10, "duration_s": 12.0, "fps": 10.0, "width": 160, "height": 120,
        "frame_count": 120, "container": "mp4", "fourcc": "mp4v"}
EV = {"event_no": 1, "start_s": 2.0, "end_s": 8.0, "duration_s": 6.0, "label": "Violence", "peak_prob": 0.9,
      "mean_prob": 0.8, "n_windows": 2, "n_positive_runs": 1, "peak_time_s": 5.0, "window_indices": [1, 2],
      "severity_score": 2, "severity_label": "Moderate", "severity_method": "rule_based_v1",
      "severity_json": {}, "motion_json": {}, "explanation": "x", "description_method": "template"}


class TestDB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.path = os.path.join(self.tmp.name, "t.db")
        self.repo = Repository(self.path)
        self.vid = self.repo.add_video("abc", "a.mp4", "/x/a.mp4", META)
        self.run = self.repo.create_run(self.vid, "k", {"a": 1}, "{}", "h1", "0.1")

    def tearDown(self):
        if hasattr(self, "repo") and self.repo:
            self.repo.close()
        self.tmp.cleanup()

    def test_video_unique_hash(self):
        self.assertEqual(self.repo.get_video_by_hash("abc")["id"], self.vid)
        with self.assertRaises(Exception):
            self.repo.add_video("abc", "b.mp4", "/x/b.mp4", META)

    def test_events_and_reviews_persist_after_reopen(self):
        eid = self.repo.add_event(self.run, EV, [{"path": "p.jpg", "timestamp_s": 3.0, "kind": "sampled"}])
        self.repo.finish_run(self.run, "completed", 1.0, {}, 5, 5, [])
        self.repo.add_review(eid, "confirmed", "looks right", 2)
        self.repo.add_review(eid, "rejected", "on second look no")
        self.repo.close()
        repo2 = Repository(self.path)
        ev = repo2.list_events(self.run)[0]
        self.assertEqual(ev["peak_prob"], 0.9)  # model output untouched by reviews
        self.assertEqual(repo2.latest_review(eid)["decision"], "rejected")
        self.assertEqual(len(repo2.list_reviews(eid)), 2)
        self.assertEqual(len(repo2.list_evidence(eid)), 1)
        self.assertEqual(repo2.dashboard_stats()["total_events"], 1)
        repo2.close()

    def test_invalid_review_rejected(self):
        eid = self.repo.add_event(self.run, EV, [])
        with self.assertRaises(ValueError):
            self.repo.add_review(eid, "maybe")
        with self.assertRaises(ValueError):
            self.repo.add_review(eid, "confirmed", human_severity=7)

    def test_qwen_and_audio_evidence_persist_and_cascade_with_run(self):
        eid = self.repo.add_event(self.run, EV, [])
        self.repo.update_event_qwen(eid, "Qwen event explanation", "generated")
        self.repo.add_audio_segments(self.run, [{
            "kind": "transcript", "start_s": 1.0, "end_s": 2.5,
            "language": "fr", "text": "Bonjour", "source": "faster_whisper",
            "details": {"words": [{"text": "Bonjour"}]},
        }])
        event = self.repo.list_events(self.run)[0]
        segment = self.repo.list_audio_segments(self.run)[0]
        self.assertEqual((event["qwen_explanation"], event["qwen_status"]), ("Qwen event explanation", "generated"))
        self.assertEqual((segment["language"], segment["text"]), ("fr", "Bonjour"))
        self.repo.delete_run(self.run)
        self.assertEqual(self.repo.list_audio_segments(self.run), [])

    def test_optional_model_window_scores_are_separate_and_cascade_with_run(self):
        self.repo.add_model_window_scores(self.run, "videomae", [{
            "window_index": 0, "start_s": 0.0, "end_s": 4.0, "p_violent": 0.72,
            "status": "ok", "inference_ms": 55.0,
        }])
        rows = self.repo.list_model_window_scores(self.run, "videomae")
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["p_violent"], 0.72)
        self.assertEqual(rows[0]["start_s"], 0.0)
        self.repo.delete_run(self.run)
        self.assertEqual(self.repo.list_model_window_scores(self.run, "videomae"), [])

    def test_stale_running_marked_interrupted(self):
        self.assertEqual(self.repo.mark_stale_running(), 1)
        self.assertEqual(self.repo.get_run(self.run)["status"], "interrupted")

    def test_find_complete_run_ignores_failed(self):
        self.repo.finish_run(self.run, "failed", 1.0, {}, 5, 2, [], "boom")
        self.assertIsNone(self.repo.find_complete_run(self.vid, "h1"))
        r2 = self.repo.create_run(self.vid, "k", {}, "{}", "h1", "0.1")
        self.repo.finish_run(r2, "completed", 1.0, {}, 5, 5, [])
        self.assertEqual(self.repo.find_complete_run(self.vid, "h1")["id"], r2)

    def test_sql_injection_string_is_data(self):
        self.repo.add_video("x'; DROP TABLE videos;--", "n.mp4", "/n", META)
        self.assertGreaterEqual(self.repo.dashboard_stats()["total_videos"], 2)

    def test_empty_dashboard(self):
        repo = Repository(os.path.join(self.tmp.name, "empty.db"))
        s = repo.dashboard_stats()
        self.assertEqual((s["total_videos"], s["total_events"], s["severity"]), (0, 0, {}))
        repo.close()

    def test_delete_run_and_clear_history(self):
        eid = self.repo.add_event(self.run, EV, [{"path": "fake_ev.jpg", "timestamp_s": 3.0, "kind": "sampled"}])
        self.repo.add_review(eid, "confirmed", "ok", 1)
        self.assertTrue(self.repo.delete_run(self.run))
        self.assertIsNone(self.repo.get_run(self.run))
        self.assertEqual(len(self.repo.list_events(self.run)), 0)
        self.assertEqual(len(self.repo.list_reviews(eid)), 0)
        self.assertFalse(self.repo.delete_run(99999))

        # Test clear_all_history
        r2 = self.repo.create_run(self.vid, "k", {}, "{}", "h2", "0.1")
        self.assertEqual(self.repo.clear_all_history(), 1)
        self.assertEqual(len(self.repo.list_runs()), 0)


if __name__ == "__main__":
    unittest.main()
