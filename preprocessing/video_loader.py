"""Video validation and metadata extraction (OpenCV)."""
from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2

from config.settings import MAX_UPLOAD_MB, SUPPORTED_EXTENSIONS


class VideoValidationError(Exception):
    """Raised with a user-friendly message when a video cannot be analysed."""


@dataclass
class VideoMetadata:
    duration_s: float
    fps: float
    width: int
    height: int
    frame_count: int
    file_size_bytes: int
    container: str
    fourcc: str

    def to_dict(self) -> dict:
        return asdict(self)


def safe_filename(name: str) -> str:
    base = Path(name).name
    stem, ext = Path(base).stem, Path(base).suffix.lower()
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._") or "video"
    return f"{stem[:80]}{ext}"


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def validate_video(path: str | Path, max_mb: int = MAX_UPLOAD_MB) -> VideoMetadata:
    p = Path(path)
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise VideoValidationError(
            f"Unsupported format '{p.suffix}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}."
        )
    if not p.is_file():
        raise VideoValidationError("The video file does not exist.")
    size = p.stat().st_size
    if size == 0:
        raise VideoValidationError("The video file is empty.")
    if size > max_mb * 1024 * 1024:
        raise VideoValidationError(f"File is {size / 1e6:.0f} MB; the limit is {max_mb} MB.")

    cap = cv2.VideoCapture(str(p))
    try:
        if not cap.isOpened():
            raise VideoValidationError("The file could not be opened as a video (corrupted or unsupported codec).")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fourcc_i = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
        fourcc = "".join(chr((fourcc_i >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") if fourcc_i else ""
        ok, frame = cap.read()
        if not ok or frame is None:
            raise VideoValidationError("The video opened but its first frame could not be decoded.")
        if not (fps > 0 and fps < 1000):
            raise VideoValidationError(f"Invalid frame rate reported by the file ({fps}).")
        if frame_count <= 0:
            raise VideoValidationError("The file does not report a usable frame count.")
        if width <= 0 or height <= 0:
            height, width = frame.shape[:2]
    finally:
        cap.release()

    return VideoMetadata(
        duration_s=frame_count / fps, fps=fps, width=width, height=height,
        frame_count=frame_count, file_size_bytes=size, container=p.suffix.lower().lstrip("."), fourcc=fourcc,
    )
