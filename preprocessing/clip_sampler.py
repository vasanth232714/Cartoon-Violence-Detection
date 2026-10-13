"""Window construction and model-input normalisation."""
from __future__ import annotations

import numpy as np

from preprocessing.timestamp_utils import build_windows
from preprocessing.video_loader import VideoMetadata


def make_windows(meta: VideoMetadata, window_s: float, stride_s: float) -> list[tuple[float, float]]:
    return build_windows(meta.duration_s, window_s, stride_s)


def to_model_input(frames_rgb: np.ndarray, mean: tuple, std: tuple) -> np.ndarray:
    """(T,H,W,3) uint8 -> (3,T,H,W) float32, scaled to [0,1] then (x-mean)/std."""
    if frames_rgb.ndim != 4 or frames_rgb.shape[-1] != 3:
        raise ValueError(f"Expected (T,H,W,3) frames, got {frames_rgb.shape}")
    x = frames_rgb.astype(np.float32) / 255.0
    x = (x - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(x.transpose(3, 0, 1, 2))
