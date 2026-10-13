import os
import tempfile
import unittest

import numpy as np

from preprocessing.clip_sampler import make_windows, to_model_input
from preprocessing.frame_extractor import iter_window_frames, read_frame_at
from preprocessing.video_loader import VideoValidationError, safe_filename, validate_video
from tests.helpers import make_video


class TestPreprocessing(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = make_video(os.path.join(self.tmp.name, "v.mp4"))

    def test_metadata(self):
        m = validate_video(self.path)
        self.assertAlmostEqual(m.fps, 10.0, places=1)
        self.assertEqual((m.width, m.height), (160, 120))
        self.assertAlmostEqual(m.duration_s, 12.0, delta=0.2)
        self.assertEqual(m.container, "mp4")

    def test_avi_supported(self):
        p = make_video(os.path.join(self.tmp.name, "v.avi"), fourcc="MJPG")
        self.assertEqual(validate_video(p).container, "avi")

    def test_unsupported_extension(self):
        p = os.path.join(self.tmp.name, "x.txt")
        with open(p, "w") as f:
            f.write("hi")
        with self.assertRaises(VideoValidationError):
            validate_video(p)

    def test_corrupted_video(self):
        p = os.path.join(self.tmp.name, "bad.mp4")
        with open(p, "wb") as f:
            f.write(os.urandom(4096))
        with self.assertRaises(VideoValidationError):
            validate_video(p)

    def test_empty_file(self):
        p = os.path.join(self.tmp.name, "empty.mp4")
        open(p, "wb").close()
        with self.assertRaises(VideoValidationError):
            validate_video(p)

    def test_safe_filename(self):
        self.assertEqual(safe_filename("../../evil name!.MP4"), "evil_name.mp4")

    def test_frame_stream_covers_whole_video_with_timestamps(self):
        m = validate_video(self.path)
        wins = make_windows(m, 4.0, 2.0)
        out = list(iter_window_frames(self.path, m, wins, 16, 224))
        self.assertEqual(len(out), len(wins))
        self.assertAlmostEqual(out[-1].end_s, m.duration_s, delta=0.2)
        for wf in out:
            self.assertEqual(wf.status, "ok")
            self.assertEqual(wf.frames.shape, (16, 224, 224, 3))
            self.assertEqual(len(wf.frame_times), 16)
            self.assertTrue(all(a <= b for a, b in zip(wf.frame_times, wf.frame_times[1:])))
            self.assertGreaterEqual(wf.frame_times[0], wf.start_s - 1e-6)
        # last window must contain frames from the final seconds
        self.assertGreater(out[-1].frame_times[-1], m.duration_s - 0.5)

    def test_bright_segment_visible_in_stream(self):
        m = validate_video(self.path)
        wins = make_windows(m, 4.0, 2.0)
        means = [wf.frames.mean() for wf in iter_window_frames(self.path, m, wins, 16, 224)]
        self.assertLess(means[0], 30)       # 0-4s dark
        self.assertGreater(means[3], 100)   # 6-10s overlaps bright 5-9s... window idx 3 = 6-10s

    def test_to_model_input(self):
        x = to_model_input(np.full((16, 224, 224, 3), 255, np.uint8), (0.45,) * 3, (0.225,) * 3)
        self.assertEqual(x.shape, (3, 16, 224, 224))
        self.assertEqual(x.dtype, np.float32)
        self.assertAlmostEqual(float(x.mean()), (1 - 0.45) / 0.225, places=4)
        with self.assertRaises(ValueError):
            to_model_input(np.zeros((16, 224, 224), np.uint8), (0.45,) * 3, (0.225,) * 3)

    def test_read_frame_at(self):
        m = validate_video(self.path)
        f = read_frame_at(self.path, m, 6.0)
        self.assertEqual(f.shape[2], 3)
        self.assertGreater(f.mean(), 100)

    def test_link_downloader_validation(self):
        from preprocessing.link_downloader import LinkDownloadError, download_video_from_url, is_valid_url
        self.assertTrue(is_valid_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ"))
        self.assertTrue(is_valid_url("http://example.com/stream.mp4"))
        self.assertFalse(is_valid_url("not_a_valid_url"))
        self.assertFalse(is_valid_url(""))
        self.assertFalse(is_valid_url("http://127.0.0.1/video.mp4"))
        self.assertFalse(is_valid_url("http://169.254.169.254/latest/meta-data/"))
        self.assertFalse(is_valid_url("https://user:pass@www.youtube.com/watch?v=dQw4w9WgXcQ"))
        with self.assertRaises(LinkDownloadError):
            download_video_from_url("invalid_url", output_dir=self.tmp.name)


if __name__ == "__main__":
    unittest.main()

