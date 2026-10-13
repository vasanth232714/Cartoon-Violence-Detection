import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from models.videomae_detector import _resolve_labels, run_videomae_comparison


class TestVideoMAEModelCardMapping(unittest.TestCase):
    def test_model_card_binary_mapping_resolves_violent_index_one(self):
        self.assertEqual(_resolve_labels({0: "Non-Violent", 1: "Violent"}), (1, 0))

    def test_rejects_unverified_or_ambiguous_label_mapping(self):
        with self.assertRaises(ValueError):
            _resolve_labels({0: "LABEL_0", 1: "LABEL_1"})
        with self.assertRaises(ValueError):
            _resolve_labels({0: "Violent", 1: "Violent"})

    def test_inference_uses_checkpoint_processor_and_cpu_device(self):
        import torch

        captured = {}

        class FakeProcessor:
            image_processor = SimpleNamespace(
                crop_size={"height": 224, "width": 224}, size={"shortest_edge": 224},
                do_resize=True, do_center_crop=True, do_rescale=True, do_normalize=True,
                image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225],
            )

            def __call__(self, frames, return_tensors):
                captured["frame_count"] = len(frames)
                return {"pixel_values": torch.zeros((1, 16, 3, 224, 224))}

        class FakeModel:
            config = SimpleNamespace(num_frames=16, id2label={0: "Non-Violent", 1: "Violent"})

            def to(self, device):
                captured["device"] = str(device)
                return self

            def eval(self):
                return self

            def __call__(self, **inputs):
                return SimpleNamespace(logits=torch.tensor([[0.0, 2.0]]))

        fake_window = SimpleNamespace(window_index=0, start_s=1.0, end_s=5.0,
                                      frames=np.zeros((16, 224, 224, 3), dtype=np.uint8),
                                      status="ok", note="")
        with patch("transformers.AutoProcessor.from_pretrained", return_value=FakeProcessor()), \
             patch("transformers.AutoModelForVideoClassification.from_pretrained", return_value=FakeModel()), \
             patch("models.videomae_detector.iter_window_frames", return_value=iter([fake_window])):
            rows, metadata = run_videomae_comparison("unused.mp4", None, [(1.0, 5.0)])
        self.assertEqual(captured["frame_count"], 16)
        self.assertEqual(captured["device"], "cpu")
        self.assertAlmostEqual(rows[0]["p_violent"], float(torch.softmax(torch.tensor([0.0, 2.0]), dim=0)[1]))
        self.assertEqual(metadata["num_frames"], 16)
        self.assertEqual(metadata["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
