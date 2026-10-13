"""Test fixtures ONLY. Synthetic videos + a fake classifier. Nothing here is used in production."""
from __future__ import annotations

import cv2
import numpy as np

from models.violence_classifier import InvalidModelOutputError


def make_video(path: str, seconds: float = 12.0, fps: int = 10, size=(160, 120), bright=(5.0, 9.0),
               fourcc: str = "mp4v") -> str:
    """Dark frames, with a bright noisy segment between bright[0]..bright[1] seconds (or none)."""
    w, h = size
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"VideoWriter could not open {path} with {fourcc}")
    rng = np.random.default_rng(0)
    for i in range(int(seconds * fps)):
        t = i / fps
        if bright and bright[0] <= t < bright[1]:
            frame = rng.integers(150, 255, (h, w, 3), dtype=np.uint8)
        else:
            frame = np.full((h, w, 3), 10, dtype=np.uint8)
        vw.write(frame)
    vw.release()
    return path


class FakeBrightnessClassifier:
    """TEST FIXTURE: returns 0.9 when the clip is bright. Not a violence model."""
    info = {"model_key": "test_fixture", "display_name": "FAKE test fixture", "training_domain": "none"}

    def predict_proba(self, clip: np.ndarray) -> float:
        return 0.9 if float(clip.mean()) > 0.0 else 0.05  # normalised: dark ~ -2, bright > 0


class RaisingClassifier:
    info = {"model_key": "test_fixture", "display_name": "FAKE raising", "training_domain": "none"}

    def predict_proba(self, clip):
        raise InvalidModelOutputError("test: invalid output")
