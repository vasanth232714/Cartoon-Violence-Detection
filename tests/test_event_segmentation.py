import unittest

from models.severity_assessor import assess_severity
from pipeline.event_segmenter import WindowPrediction as W
from pipeline.event_segmenter import segment_events


def wins(probs, win=4.0, stride=2.0, statuses=None):
    return [W(i, i * stride, i * stride + win, p, (statuses or {}).get(i, "ok")) for i, p in enumerate(probs)]


class TestSegmentation(unittest.TestCase):
    def test_empty_no_events(self):
        self.assertEqual(segment_events(wins([0.1, 0.2, 0.0]), 0.4, 1.0), [])
        self.assertEqual(segment_events([], 0.4, 1.0), [])

    def test_overlapping_windows_merge_into_one(self):
        ev = segment_events(wins([0.1, 0.8, 0.9, 0.7, 0.1]), 0.4, 1.0)
        self.assertEqual(len(ev), 1)
        e = ev[0]
        self.assertEqual((e.start_s, e.end_s), (2.0, 10.0))  # windows 2-6, 4-8, 6-10
        self.assertEqual(e.peak_prob, 0.9)
        self.assertEqual(e.n_positive_runs, 1)

    def test_distant_events_stay_separate_and_ordered(self):
        ev = segment_events(wins([0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]), 0.4, 1.0)
        self.assertEqual([e.event_no for e in ev], [1, 2])
        self.assertLess(ev[0].start_s, ev[1].start_s)

    def test_gap_controls_merging_and_runs(self):
        p = [0.9, 0.0, 0.9]  # windows 0-4, 2-6, 4-8 -> gap 0 -> merged, but 2 positive runs
        ev = segment_events(wins(p), 0.4, 0.0)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0].n_positive_runs, 2)

    def test_threshold_respected(self):
        self.assertEqual(len(segment_events(wins([0.5]), 0.6, 1.0)), 0)
        self.assertEqual(len(segment_events(wins([0.5]), 0.4, 1.0)), 1)

    def test_failed_windows_never_positive(self):
        ev = segment_events(wins([0.99, 0.99], statuses={0: "error", 1: "skipped"}), 0.4, 1.0)
        self.assertEqual(ev, [])

    def test_none_probability_ignored(self):
        self.assertEqual(segment_events(wins([None, None]), 0.4, 1.0), [])

    def test_min_event_duration(self):
        self.assertEqual(segment_events(wins([0.9]), 0.4, 1.0, min_event_s=5.0), [])


class TestSeverity(unittest.TestCase):
    def test_not_assessed_without_motion(self):
        self.assertEqual(assess_severity(5, 1, None).label, "Not assessed")

    def test_scores(self):
        self.assertEqual(assess_severity(2, 1, 3).score, 1)
        self.assertEqual(assess_severity(6, 2, 10).score, 2)
        self.assertEqual(assess_severity(15, 3, 30).score, 3)

    def test_probability_not_an_input(self):
        import inspect
        self.assertNotIn("prob", " ".join(inspect.signature(assess_severity).parameters))


if __name__ == "__main__":
    unittest.main()
