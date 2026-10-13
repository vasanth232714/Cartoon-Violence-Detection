"""Integration tests that need REAL external assets. Skipped automatically when unavailable."""
import unittest

from models.model_loader import expected_checkpoint_path

try:
    import torch  # noqa: F401
    import pytorchvideo  # noqa: F401
    HAVE_TORCH = True
except Exception:
    HAVE_TORCH = False

CKPT = expected_checkpoint_path("x3d_m_realtime")


@unittest.skipUnless(HAVE_TORCH and CKPT.is_file(), "requires torch, pytorchvideo and the downloaded checkpoint")
class TestRealModel(unittest.TestCase):
    def test_load_and_infer_shapes(self):
        import numpy as np
        from models.model_loader import load_classifier
        clf = load_classifier("x3d_m_realtime")
        p = clf.predict_proba(np.zeros((3, 16, 224, 224), np.float32))
        self.assertTrue(0.0 <= p <= 1.0)

    def test_wrong_shape_rejected(self):
        import numpy as np
        from models.model_loader import load_classifier
        from models.violence_classifier import InvalidModelOutputError
        clf = load_classifier("x3d_m_realtime")
        with self.assertRaises(InvalidModelOutputError):
            clf.predict_proba(np.zeros((3, 8, 224, 224), np.float32))


@unittest.skipUnless(HAVE_TORCH, "requires torch")
class TestArchitecturesDefined(unittest.TestCase):
    def test_cnn_transformer_forward_shape(self):
        import torch
        from models.temporal_transformer import build_cnn_transformer
        m = build_cnn_transformer(severity_classes=4).eval()
        with torch.inference_mode():
            v, s = m(torch.zeros(1, 4, 3, 64, 64))
        self.assertEqual(tuple(v.shape), (1, 2))
        self.assertEqual(tuple(s.shape), (1, 4))

    def test_untrained_transformer_refused(self):
        from models.temporal_transformer import NotTrainedError, load_trained
        with self.assertRaises(NotTrainedError):
            load_trained(None)


class TestStrictLoadLogic(unittest.TestCase):
    @unittest.skipUnless(HAVE_TORCH, "requires torch")
    def test_mismatched_checkpoint_fails_loudly(self):
        import torch
        from models.model_loader import ModelLoadError, load_state_strict
        with self.assertRaises(ModelLoadError):
            load_state_strict(torch.nn.Linear(2, 2), {"bogus.weight": torch.zeros(1)})


if __name__ == "__main__":
    unittest.main()
