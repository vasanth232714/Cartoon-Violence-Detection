"""Optional AudioSet sound-event candidates using a pretrained AST classifier.

Audio labels are candidate sound classes, not proof that a depicted event occurred.
The model is run on overlapping 10-second excerpts from the normalized mono track.
"""
from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
import wave
from pathlib import Path
from typing import Callable

import numpy as np

from config.settings import ensure_ffmpeg

logger = logging.getLogger(__name__)
MODEL_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
SAMPLE_RATE = 16000
TARGET_LABELS = {
    "gunshot, gunfire": "Gunshot-like sound",
    "screaming": "Scream-like sound",
    "shout": "Shouting-like sound",
    "explosion": "Explosion-like sound",
    "crash": "Crash-like sound",
    "bang": "Impact-like sound",
}


def _start_audio_heartbeat(task: str) -> tuple[threading.Event, threading.Thread]:
    """Log while a local AudioSet model is being downloaded or initialized."""
    stopped = threading.Event()
    started_at = time.monotonic()

    def _heartbeat() -> None:
        while not stopped.wait(20.0):
            elapsed = time.monotonic() - started_at
            message = f"[audio-analysis] {task} is still active after {elapsed:.0f}s"
            print(message, flush=True)

    thread = threading.Thread(target=_heartbeat, name="audio-analysis-heartbeat", daemon=True)
    thread.start()
    return stopped, thread


def extract_audio_wav(video_path: str | Path) -> str | None:
    """Create a temporary 16 kHz mono PCM WAV for ASR/audio classification."""
    ffmpeg = ensure_ffmpeg() or "ffmpeg"
    fd, out_path = tempfile.mkstemp(prefix="cartoon_audio_", suffix=".wav")
    os.close(fd)
    try:
        import subprocess
        result = subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(video_path), "-vn", "-acodec", "pcm_s16le",
             "-ar", str(SAMPLE_RATE), "-ac", "1", out_path],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60,
        )
        if result.returncode == 0 and Path(out_path).is_file() and Path(out_path).stat().st_size > 1024:
            return out_path
    except Exception as exc:
        logger.info("Audio extraction failed: %s", exc)
    Path(out_path).unlink(missing_ok=True)
    return None


def _read_wav(path: str) -> np.ndarray:
    with wave.open(path, "rb") as wf:
        if wf.getframerate() != SAMPLE_RATE or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
            raise ValueError("Expected normalized 16 kHz mono PCM audio")
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32)
    return audio / 32768.0


def detect_audio_events(
    video_path: str | Path,
    threshold: float = 0.35,
    window_s: float = 10.0,
    stride_s: float = 5.0,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[list[dict], dict, list[str]]:
    """Return timestamped AudioSet sound candidates, model metadata, and warnings."""
    progress_callback = progress_callback or (lambda fraction, message: None)
    warnings: list[str] = []
    progress_callback(0.02, "Extracting a normalized audio track for sound analysis")
    logger.info("Extracting normalized audio for AudioSet sound-event analysis")
    wav_path = extract_audio_wav(video_path)
    if not wav_path:
        progress_callback(1.0, "No decodable audio track; sound analysis skipped")
        return [], {"model": MODEL_ID, "status": "audio unavailable"}, ["Audio-event analysis skipped: audio could not be extracted."]
    stop_heartbeat, heartbeat_thread = _start_audio_heartbeat("AudioSet model loading/scoring")
    try:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        progress_callback(0.10, "Loading AudioSet feature extractor and AST model")
        logger.info("Loading AudioSet AST model %s on CPU", MODEL_ID)
        feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
        model = AutoModelForAudioClassification.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
        model.eval()
        progress_callback(0.20, "AudioSet model loaded; preparing sound windows")
        audio = _read_wav(wav_path)
        duration = len(audio) / SAMPLE_RATE
        labels = getattr(model.config, "id2label", {}) or {}
        targets = {str(label).strip().lower(): TARGET_LABELS[str(label).strip().lower()]
                   for label in labels.values() if str(label).strip().lower() in TARGET_LABELS}
        if not targets:
            progress_callback(1.0, "No configured AudioSet sound labels are present in this checkpoint")
            return [], {"model": MODEL_ID, "status": "target labels absent", "available_labels": list(labels.values())}, [
                "Audio-event model loaded but none of the configured sound labels were present."
            ]
        events: list[dict] = []
        starts = list(np.arange(0.0, max(duration - window_s, 0.0) + 0.001, stride_s))
        if not starts or starts[-1] + window_s < duration:
            starts.append(max(0.0, duration - window_s))
        ordered_starts = sorted(set(float(s) for s in starts))
        total_windows = len(ordered_starts)
        logger.info("AudioSet analysis: %.1f seconds, %d windows of %.1f seconds", duration,
                    total_windows, window_s)
        with torch.inference_mode():
            for window_index, start_s in enumerate(ordered_starts, start=1):
                end_s = min(duration, start_s + window_s)
                progress_callback(
                    0.22 + 0.75 * ((window_index - 1) / max(total_windows, 1)),
                    f"Scoring sound window {window_index}/{total_windows} ({start_s:.0f}–{end_s:.0f}s)",
                )
                i0, i1 = int(start_s * SAMPLE_RATE), int(end_s * SAMPLE_RATE)
                clip = audio[i0:i1]
                if not len(clip):
                    continue
                inputs = feature_extractor(clip, sampling_rate=SAMPLE_RATE, return_tensors="pt")
                logits = model(**inputs).logits[0]
                # AudioSet is a multi-label task: several sound classes can co-occur.
                probabilities = torch.sigmoid(logits).cpu().numpy()
                for index, score in enumerate(probabilities):
                    raw_label = str(labels.get(index, labels.get(str(index), ""))).strip().lower()
                    category = targets.get(raw_label)
                    if category and float(score) >= threshold:
                        events.append({"start_s": round(start_s, 2), "end_s": round(end_s, 2),
                                       "category": category, "confidence": float(score),
                                       "source": "AudioSet AST", "raw_label": raw_label})
                if window_index == 1 or window_index % 4 == 0 or window_index == total_windows:
                    logger.info("AudioSet scored sound window %d/%d", window_index, total_windows)
        # Merge same-label overlapping windows while retaining the strongest model score.
        merged: list[dict] = []
        for item in sorted(events, key=lambda x: (x["category"], x["start_s"])):
            prev = merged[-1] if merged else None
            if prev and prev["category"] == item["category"] and item["start_s"] <= prev["end_s"] + 0.5:
                prev["end_s"] = max(prev["end_s"], item["end_s"])
                prev["confidence"] = max(prev["confidence"], item["confidence"])
            else:
                merged.append(dict(item))
        metadata = {"model": MODEL_ID, "status": "completed", "threshold": threshold,
                    "window_seconds": window_s, "stride_seconds": stride_s,
                    "model_labels": list(targets.values()), "audio_duration_s": duration}
        del model
        progress_callback(1.0, f"Sound analysis complete; {len(merged)} candidate(s) found")
        return sorted(merged, key=lambda x: x["start_s"]), metadata, warnings
    except Exception as exc:
        logger.exception("Audio-event classifier failed")
        warnings.append(f"Audio-event analysis failed: {type(exc).__name__}: {exc}")
        progress_callback(1.0, f"Sound analysis failed: {type(exc).__name__}")
        return [], {"model": MODEL_ID, "status": "failed"}, warnings
    finally:
        stop_heartbeat.set()
        heartbeat_thread.join(timeout=1.0)
        Path(wav_path).unlink(missing_ok=True)
