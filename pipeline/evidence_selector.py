"""Evidence frames (from the ORIGINAL video) and motion-intensity measurement for each event."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pipeline.event_segmenter import CandidateEvent
from preprocessing.video_loader import VideoMetadata

SHOT_CUT_DIFF = 45.0  # mean abs gray diff above this is treated as a shot cut (heuristic)


@dataclass
class EvidenceFrame:
    path: str
    timestamp_s: float
    kind: str  # 'peak_window' | 'sampled'


@dataclass
class MotionStats:
    mean: float | None
    p90: float | None
    n_pairs: int
    n_shot_cuts_excluded: int


def evidence_timestamps(ev: CandidateEvent, n: int, meta: VideoMetadata) -> list[tuple[float, str]]:
    last_t = max(0.0, (meta.frame_count - 1) / meta.fps)
    lo, hi = max(0.0, ev.start_s), min(ev.end_s, last_t)
    if hi < lo:
        hi = lo
    times = list(np.linspace(lo, hi, max(n - 1, 1))) if n > 1 else []
    out = [(float(t), "sampled") for t in times]
    out.append((min(max(ev.peak_time_s, lo), hi), "peak_window"))
    out.sort(key=lambda x: x[0])
    dedup: list[tuple[float, str]] = []
    for t, k in out:
        if dedup and abs(t - dedup[-1][0]) < 0.05:
            if k == "peak_window":
                dedup[-1] = (t, k)
            continue
        dedup.append((t, k))
    return dedup


def save_evidence_frames(video_path: str, meta: VideoMetadata, ev: CandidateEvent, n: int, out_dir: Path,
                         max_width: int = 1280) -> list[EvidenceFrame]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[EvidenceFrame] = []
    cap = cv2.VideoCapture(video_path)
    try:
        for k, (t, kind) in enumerate(evidence_timestamps(ev, n, meta)):
            idx = min(max(int(round(t * meta.fps)), 0), meta.frame_count - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            if frame.shape[1] > max_width:
                s = max_width / frame.shape[1]
                frame = cv2.resize(frame, (max_width, int(frame.shape[0] * s)), interpolation=cv2.INTER_AREA)
            path = out_dir / f"event{ev.event_no:03d}_{k:02d}_{idx}.jpg"
            if cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                frames.append(EvidenceFrame(str(path), idx / meta.fps, kind))
    finally:
        cap.release()
    return frames


def measure_motion(video_path: str, meta: VideoMetadata, start_s: float, end_s: float,
                   target_fps: float = 6.0, max_samples: int = 60) -> MotionStats:
    """Mean absolute grayscale difference between successive sampled frames (0-255 scale)."""
    first = min(max(int(round(start_s * meta.fps)), 0), meta.frame_count - 1)
    last = min(int(round(end_s * meta.fps)), meta.frame_count - 1)
    span = max(last - first + 1, 1)
    step = max(1, int(round(meta.fps / target_fps)), int(np.ceil(span / max_samples)))
    cap = cv2.VideoCapture(video_path)
    diffs: list[float] = []
    cuts = 0
    prev = None
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        for i in range(first, last + 1):
            if not cap.grab():
                break
            if (i - first) % step:
                continue
            ok, frame = cap.retrieve()
            if not ok or frame is None:
                continue
            g = cv2.cvtColor(cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            g = g.astype(np.float32)
            if prev is not None:
                d = float(np.abs(g - prev).mean())
                if d > SHOT_CUT_DIFF:
                    cuts += 1
                else:
                    diffs.append(d)
            prev = g
    finally:
        cap.release()
    if not diffs:
        return MotionStats(None, None, 0, cuts)
    return MotionStats(float(np.mean(diffs)), float(np.percentile(diffs, 90)), len(diffs), cuts)
