"""Tests for the accuracy tooling. Synthetic data only: these check the MECHANICS (leakage-free CV, tuning,
hysteresis, adapter, cache building). They say nothing about accuracy on real cartoons."""
import json
import os
import tempfile
import unittest

import numpy as np

from models.adapter import CartoonAdapter
from pipeline.event_segmenter import WindowPrediction as W
from pipeline.event_segmenter import segment_events
from pipeline.postprocess import PostParams, run_postprocess
from tests.helpers import make_video
from training.build_cache import cache_video
from training.dataset import load_labels, window_labels
from training.fit import (HAND_SET_PRESET, RAW_BASELINE, VideoCache, evaluate_params, load_cache, oof_probabilities,
                          run_fit)


def synth_video(rng, name, n=40, events=((12, 20),), dim=24, raw_noise=0.35):
    """Windows every 2 s (4 s long). A weak 'domain-shifted' raw score + informative features."""
    starts = np.arange(n) * 2.0
    ends = starts + 4.0
    labels = window_labels(starts, ends, list(events))
    truth = (labels == 1).astype(float)
    raw = np.clip(0.3 + 0.25 * truth + rng.normal(0, raw_noise, n), 0.01, 0.99)   # barely informative
    feats = rng.normal(0, 1, (n, dim)).astype(np.float32)
    feats[:, :4] += (truth[:, None] * 1.6).astype(np.float32)                      # informative
    return VideoCache(name, starts, ends, raw, feats, np.zeros(n, bool), labels, list(events), 4.0, 2.0)


class TestWindowLabels(unittest.TestCase):
    def test_labels(self):
        y = window_labels([0, 2, 4, 6, 8], [4, 6, 8, 10, 12], [(5.0, 7.0)])
        # window 4-8 overlaps 2s -> pos; 2-6 overlaps 1s -> pos; 6-10 overlaps 1s -> pos; 0-4 and 8-12 none
        self.assertEqual(list(y), [0, 1, 1, 1, 0])

    def test_ambiguous_edge(self):
        y = window_labels([0], [4], [(3.8, 6.0)], min_overlap=1.0)
        self.assertEqual(y[0], -1)


class TestHysteresis(unittest.TestCase):
    def _w(self, ps):
        return [W(i, i * 2.0, i * 2.0 + 4.0, p) for i, p in enumerate(ps)]

    def test_extends_weak_edges_but_not_new_events(self):
        ps = [0.1, 0.45, 0.9, 0.45, 0.1, 0.45, 0.1]
        no_h = segment_events(self._w(ps), 0.6, 0.0)
        h = segment_events(self._w(ps), 0.6, 0.0, low_threshold=0.4)
        self.assertEqual((no_h[0].start_s, no_h[0].end_s), (4.0, 8.0))
        self.assertEqual((h[0].start_s, h[0].end_s), (2.0, 10.0))   # grown over the 0.45 neighbours
        self.assertEqual(len(h), 1)                                  # the lone 0.45 never starts an event

    def test_none_is_unchanged(self):
        ps = [0.1, 0.45, 0.9, 0.45]
        self.assertEqual([(e.start_s, e.end_s) for e in segment_events(self._w(ps), 0.6, 0.0)],
                         [(e.start_s, e.end_s) for e in segment_events(self._w(ps), 0.6, 0.0, low_threshold=None)])


class TestPostprocessDeterministic(unittest.TestCase):
    def test_same_input_same_output(self):
        s, e = np.arange(10) * 2.0, np.arange(10) * 2.0 + 4
        raw = [0.1, 0.5, 0.8, 0.9, 0.7, 0.2, 0.1, 0.1, 0.6, 0.1]
        a = run_postprocess(s, e, raw, None, PostParams())
        b = run_postprocess(s, e, raw, None, PostParams())
        self.assertEqual(a[0], b[0])
        self.assertEqual([(x.start_s, x.end_s) for x in a[1]], [(x.start_s, x.end_s) for x in b[1]])


class TestAdapter(unittest.TestCase):
    def test_roundtrip_and_learns(self):
        rng = np.random.default_rng(1)
        y = (rng.random(300) < 0.4).astype(int)
        X = rng.normal(size=(300, 20)); X[:, 0] += 2 * y
        ad = CartoonAdapter.fit(X[:200], y[:200], C=0.1)
        p = ad.predict_proba(X[200:])
        self.assertGreater(p[y[200:] == 1].mean(), p[y[200:] == 0].mean() + 0.3)
        with tempfile.TemporaryDirectory() as d:
            ad.save(os.path.join(d, "a.npz"))
            self.assertTrue(np.allclose(CartoonAdapter.load(os.path.join(d, "a.npz")).predict_proba(X[200:]), p))

    def test_one_class_refused(self):
        with self.assertRaises(ValueError):
            CartoonAdapter.fit(np.zeros((5, 3)), np.zeros(5))

    def test_wrong_dim_refused(self):
        ad = CartoonAdapter.fit(np.random.default_rng(0).normal(size=(20, 3)), np.array([0, 1] * 10))
        with self.assertRaises(ValueError):
            ad.predict_proba(np.zeros((1, 4)))


class TestFit(unittest.TestCase):
    def test_oof_is_leakage_free(self):
        """A video's own labels must not influence its out-of-fold scores."""
        rng = np.random.default_rng(2)
        vids = [synth_video(rng, f"v{i}.mp4") for i in range(4)]
        oof, _ = oof_probabilities(vids, C=0.1)
        flipped = [VideoCache(**{**vids[0].__dict__, "labels": 1 - np.clip(vids[0].labels, 0, 1)})] + vids[1:]
        oof2, _ = oof_probabilities(flipped, C=0.1)
        self.assertTrue(np.allclose(oof[0], oof2[0]), "held-out video's labels leaked into its own predictions")

    def test_pipeline_runs_and_adapter_helps_on_synthetic_domain_shift(self):
        rng = np.random.default_rng(3)
        ev_sets = [((12, 20),), ((6, 12), (40, 48)), ((30, 38),), ((20, 26),), ((8, 16),), ((50, 60),), ((14, 22),)]
        vids = [synth_video(rng, f"v{i}.mp4", n=45, events=e) for i, e in enumerate(ev_sets)]
        with tempfile.TemporaryDirectory() as d:
            rep = run_fit(vids, {"v5.mp4", "v6.mp4"}, os.path.join(d, "ad.npz"), os.path.join(d, "t.json"),
                          log=lambda *_: None)
            self.assertTrue(rep["adapter_used"])
            self.assertGreater(rep["adapter_auc_oof"], rep["raw_auc_oof"])
            h = rep["holdout"]
            self.assertGreater(h["tuned"]["window"]["f1"], h["raw_baseline"]["window"]["f1"])
            t = json.load(open(os.path.join(d, "t.json")))
            self.assertIn("post", t)
            self.assertTrue(os.path.isfile(os.path.join(d, "ad.npz")))

    def test_adapter_not_used_when_it_does_not_help(self):
        rng = np.random.default_rng(4)
        vids = []
        for i in range(5):
            v = synth_video(rng, f"v{i}.mp4", n=40, events=((10 + i, 18 + i),), raw_noise=0.05)
            v.feats = rng.normal(size=v.feats.shape).astype(np.float32)   # features carry no signal
            v.raw = np.clip(0.2 + 0.6 * (v.labels == 1) + rng.normal(0, 0.05, len(v.raw)), 0.01, 0.99)  # raw is good
            vids.append(v)
        with tempfile.TemporaryDirectory() as d:
            rep = run_fit(vids, set(), os.path.join(d, "ad.npz"), None, log=lambda *_: None)
            self.assertFalse(rep["adapter_used"])
            self.assertFalse(os.path.exists(os.path.join(d, "ad.npz")))

    def test_needs_both_classes(self):
        rng = np.random.default_rng(5)
        vids = [synth_video(rng, f"v{i}.mp4", events=()) for i in range(3)]
        with self.assertRaises(ValueError):
            run_fit(vids, set(), None, None, log=lambda *_: None)

    def test_baseline_params_defined(self):
        self.assertEqual(RAW_BASELINE.threshold, 0.40)
        self.assertEqual(HAND_SET_PRESET.threshold, 0.55)
        rng = np.random.default_rng(6)
        v = [synth_video(rng, "a.mp4")]
        r = evaluate_params(v, [v[0].raw], RAW_BASELINE)
        self.assertIn("score", r)


class FakeExtractor:
    """TEST FIXTURE with the same `extract` contract as X3DViolenceClassifier."""
    def extract(self, clip):
        m = float(clip.mean())
        return (0.9 if m > 0 else 0.1), np.full(5, m, np.float32), m


class TestBuildCache(unittest.TestCase):
    def test_cache_roundtrip_with_real_video_decode(self):
        with tempfile.TemporaryDirectory() as d:
            v = make_video(os.path.join(d, "a.mp4"))
            lab = os.path.join(d, "labels.json")
            json.dump({"videos": [{"file": "a.mp4", "events": [{"start": 5, "end": 9}]}]}, open(lab, "w"))
            entry = load_labels(lab)[0]
            out = cache_video(entry, FakeExtractor(), __import__("pathlib").Path(d) / "cache", 4.0, 2.0,
                              "x3d_m_realtime", log=lambda *_: None)
            vc = load_cache(os.path.join(d, "cache"))[0]
            self.assertEqual(vc.name, "a.mp4")
            self.assertEqual(vc.feats.shape[1], 5)
            self.assertGreater(int((vc.labels == 1).sum()), 0)
            self.assertGreater(int((vc.labels == 0).sum()), 0)
            # window fully inside the event (4-8s) is positive & bright; window fully outside (0-4s) negative & dark
            i_in, i_out = list(vc.starts).index(4.0), list(vc.starts).index(0.0)
            self.assertEqual((vc.labels[i_in], vc.labels[i_out]), (1, 0))
            self.assertGreater(vc.raw[i_in], 0.5)
            self.assertLess(vc.raw[i_out], 0.5)
            # second call is a cache hit (no recompute)
            self.assertEqual(cache_video(entry, None, __import__("pathlib").Path(d) / "cache", 4.0, 2.0,
                                         "x3d_m_realtime", log=lambda *_: None), out)

    def test_bad_label_entries_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            lab = os.path.join(d, "l.json")
            json.dump({"videos": [{"file": "a.mp4"}]}, open(lab, "w"))
            with self.assertRaises(ValueError):
                load_labels(lab)


if __name__ == "__main__":
    unittest.main()


class TestConfigAndTuned(unittest.TestCase):
    def test_tuned_settings_roundtrip_into_config(self):
        import config.settings as S
        post = PostParams(threshold=0.65, threshold_low=0.5, merge_gap_s=2.0, min_event_s=2.0, refine=True,
                          suppress_cuts=False, filter_spikes=False)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.json")
            with open(path, "w") as f:
                json.dump({"post": post.to_dict(), "window_seconds": 4.0, "stride_seconds": 2.0}, f)
            old = S.TUNED_SETTINGS_PATH
            S.TUNED_SETTINGS_PATH = __import__("pathlib").Path(path)
            try:
                cfg = S.AnalysisConfig.from_tuned()
            finally:
                S.TUNED_SETTINGS_PATH = old
        cfg.validate()
        self.assertEqual((cfg.threshold, cfg.threshold_low, cfg.suppress_shot_cuts), (0.65, 0.5, False))
        back = cfg.post_params()
        self.assertEqual((back.threshold, back.threshold_low, back.merge_gap_s), (0.65, 0.5, 2.0))

    def test_missing_tuned_file_is_none(self):
        import config.settings as S
        old = S.TUNED_SETTINGS_PATH
        S.TUNED_SETTINGS_PATH = __import__("pathlib").Path("/nonexistent/x.json")
        try:
            self.assertIsNone(S.AnalysisConfig.from_tuned())
        finally:
            S.TUNED_SETTINGS_PATH = old

    def test_invalid_hysteresis_rejected(self):
        from config.settings import AnalysisConfig
        with self.assertRaises(ValueError):
            AnalysisConfig(threshold=0.5, threshold_low=0.7).validate()

    def test_old_positional_construction_still_works(self):
        from config.settings import AnalysisConfig
        AnalysisConfig("x3d_m_realtime", 4.0, 2.0, 0.5, 1.0, 0.0, 6, enable_temporal_refinement=False).validate()
