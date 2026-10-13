"""Sequential, bounded-memory frame streaming.

The video is decoded ONCE from start to end with grab()/retrieve(); only the frames needed by
upcoming windows are retained (resized), and frames older than the next window are evicted.
Timestamps are idx / fps (OpenCV exposes no reliable per-frame PTS; variable-frame-rate files
may drift slightly - documented limitation).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from preprocessing.timestamp_utils import sample_frame_indices
from preprocessing.video_loader import VideoMetadata


@dataclass
class WindowFrames:
    window_index: int
    start_s: float
    end_s: float
    frame_indices: list[int]
    frame_times: list[float]
    frames: np.ndarray | None  # (T, H, W, 3) uint8 RGB, None if status == 'skipped'
    status: str  # 'ok' | 'padded' | 'skipped'
    note: str = ""


def iter_window_frames(
    video_path: str, meta: VideoMetadata, windows: list[tuple[float, float]], num_frames: int, size: int
) -> Iterator[WindowFrames]:
    plans = [sample_frame_indices(s, e, meta.fps, meta.frame_count, num_frames) for s, e in windows]
    needed = set(i for p in plans for i in p)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Could not reopen video for analysis.")
    cache: dict[int, np.ndarray] = {}
    pos = 0  # index of the next frame grab() will read
    eof = False
    try:
        for w, plan in enumerate(plans):
            target = plan[-1]
            while pos <= target and not eof:
                if not cap.grab():
                    eof = True
                    break
                if pos in needed:
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        small = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
                        cache[pos] = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                pos += 1

            keys = sorted(cache)
            frames, missing = [], 0
            for idx in plan:
                if idx in cache:
                    frames.append(cache[idx])
                else:
                    missing += 1
                    prev = [k for k in keys if k <= idx]
                    nxt = [k for k in keys if k > idx]
                    sub = prev[-1] if prev else (nxt[0] if nxt else None)
                    frames.append(cache[sub] if sub is not None else None)
            times = [i / meta.fps for i in plan]
            s, e = windows[w]
            if all(f is None for f in frames):
                yield WindowFrames(w, s, e, plan, times, None, "skipped", "no frames could be decoded")
            elif missing:
                fallback = next(f for f in frames if f is not None)
                frames = [f if f is not None else fallback for f in frames]
                yield WindowFrames(w, s, e, plan, times, np.stack(frames), "padded",
                                   f"{missing}/{len(plan)} frames missing; nearest decoded frame reused")
            else:
                yield WindowFrames(w, s, e, plan, times, np.stack(frames), "ok")

            if w + 1 < len(plans):
                lo = plans[w + 1][0]
                for k in [k for k in cache if k < lo]:
                    del cache[k]
        cache.clear()
    finally:
        cap.release()


def read_frame_at(video_path: str, meta: VideoMetadata, seconds: float, max_width: int = 1280) -> np.ndarray | None:
    """Return one RGB frame near `seconds` (seek-based, for the UI frame viewer)."""
    idx = min(max(int(round(seconds * meta.fps)), 0), meta.frame_count - 1)
    cap = cv2.VideoCapture(video_path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
    finally:
        cap.release()
    if not ok or frame is None:
        return None
    if frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, (max_width, int(frame.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
