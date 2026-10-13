import unittest

from preprocessing.timestamp_utils import build_windows, format_timestamp, sample_frame_indices


class TestTimestamps(unittest.TestCase):
    def test_format(self):
        self.assertEqual(format_timestamp(0), "00:00.0")
        self.assertEqual(format_timestamp(65.0), "01:05.0")
        self.assertEqual(format_timestamp(3725.5), "1:02:05.5")
        self.assertEqual(format_timestamp(-1), "--:--")

    def test_windows_cover_whole_video_including_tail(self):
        for duration in (10.0, 11.3, 12.0, 300.7, 4.0, 3.0):
            w = build_windows(duration, 4.0, 2.0)
            self.assertEqual(w[0][0], 0.0)
            self.assertAlmostEqual(w[-1][1], duration, places=3)
            for (s1, e1), (s2, e2) in zip(w, w[1:]):
                self.assertLessEqual(s2, e1 + 1e-6, "windows must overlap or touch (no uncovered gap)")
                self.assertGreater(s2, s1)

    def test_short_video_single_window(self):
        self.assertEqual(build_windows(2.5, 4.0, 2.0), [(0.0, 2.5)])

    def test_empty(self):
        self.assertEqual(build_windows(0, 4, 2), [])

    def test_sample_indices_monotonic_and_clamped(self):
        idx = sample_frame_indices(8.0, 12.0, 10.0, 120, 16)
        self.assertEqual(len(idx), 16)
        self.assertEqual(idx, sorted(idx))
        self.assertLessEqual(idx[-1], 119)
        self.assertEqual(idx[0], 80)

    def test_sample_indices_tiny_clip(self):
        idx = sample_frame_indices(0, 0.2, 10.0, 3, 16)
        self.assertEqual(len(idx), 16)
        self.assertTrue(all(0 <= i <= 2 for i in idx))


if __name__ == "__main__":
    unittest.main()
