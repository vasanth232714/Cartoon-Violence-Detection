"""Timestamp / window helpers. Pure Python + numpy (no torch, no cv2)."""
from __future__ import annotations

import numpy as np


def format_timestamp(seconds: float) -> str:
    """0 -> '00:00.0', 3725.5 -> '1:02:05.5'."""
    if seconds is None or seconds != seconds or seconds < 0:
        return "--:--"
    total_tenths = int(round(seconds * 10))
    tenths = total_tenths % 10
    secs_total = total_tenths // 10
    h, rem = divmod(secs_total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}.{tenths}"
    return f"{m:02d}:{s:02d}.{tenths}"


def frame_to_time(frame_index: int, fps: float) -> float:
    return float(frame_index) / float(fps)


def time_to_frame(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def build_windows(duration: float, window: float, stride: float) -> list[tuple[float, float]]:
    """Overlapping windows covering [0, duration]. The final window is anchored to the end of
    the video so the tail is always analysed. If duration <= window, one window covers all."""
    if duration <= 0:
        return []
    if duration <= window:
        return [(0.0, float(duration))]
    wins: list[tuple[float, float]] = []
    i = 0
    while True:
        s = i * stride
        if s + window > duration + 1e-9:
            break
        wins.append((round(s, 6), round(s + window, 6)))
        i += 1
    if wins[-1][1] < duration - 1e-6:
        wins.append((round(duration - window, 6), round(float(duration), 6)))
    return wins


def sample_frame_indices(start_s: float, end_s: float, fps: float, frame_count: int, n: int) -> list[int]:
    """n frame indices spread uniformly over [start_s, end_s), clamped to the valid range."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    first = min(max(time_to_frame(start_s, fps), 0), frame_count - 1)
    last = min(time_to_frame(end_s, fps) - 1, frame_count - 1)
    last = max(last, first)
    idx = np.linspace(first, last, n).round().astype(int)
    return [int(i) for i in idx]
