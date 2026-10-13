"""Shape and bounded-sequence checks for the integrated window Transformer."""
import importlib.util
import unittest

import numpy as np

from models.temporal_transformer import build_window_temporal_transformer, predict_window_sequence


@unittest.skipIf(importlib.util.find_spec("torch") is None, "requires torch")
class TestWindowTemporalTransformer(unittest.TestCase):
    def test_scores_full_sequence_across_overlapping_chunks(self):
        model = build_window_temporal_transformer(
            input_dim=2050, d_model=16, n_layers=1, n_heads=4, max_len=8,
        )
        features = np.random.default_rng(7).normal(size=(13, 2050)).astype(np.float32)
        scores = predict_window_sequence(model, features)
        self.assertEqual(len(scores), 13)
        self.assertTrue(all(np.isfinite(score) and 0.0 <= score <= 1.0 for score in scores))


if __name__ == "__main__":
    unittest.main()
