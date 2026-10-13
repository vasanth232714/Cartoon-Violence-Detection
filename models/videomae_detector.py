"""Optional VideoMAE comparison model; outputs remain separate from X3D."""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from preprocessing.frame_extractor import iter_window_frames
from preprocessing.video_loader import VideoMetadata

MODEL_ID = "HappyGook/videomae-violence-detector"
MODEL_CARD_URL = f"https://huggingface.co/{MODEL_ID}"
EXPECTED_FRAMES = 16
EXPECTED_SIZE = 224


def _resolve_labels(id2label: dict) -> tuple[int, int]:
    """Resolve violent/non-violent indices from the downloaded model config, never assume silently."""
    normalized = {int(index): str(label).strip().lower().replace("_", "-")
                  for index, label in id2label.items()}
    violent = [index for index, label in normalized.items()
              if "violent" in label and "non-violent" not in label and "nonviolent" not in label]
    nonviolent = [index for index, label in normalized.items()
                  if "non-violent" in label or "nonviolent" in label]
    if len(normalized) != 2 or len(violent) != 1 or len(nonviolent) != 1:
        raise ValueError(f"Unexpected VideoMAE binary label mapping: {id2label!r}")
    return violent[0], nonviolent[0]


def run_videomae_comparison(
    video_path: str,
    meta: VideoMetadata,
    windows: list[tuple[float, float]],
    progress: Callable[[float, str], None] | None = None,
) -> tuple[list[dict], dict]:
    """Run VideoMAE over the shared timestamp windows, with no GPU use and no score fusion."""
    import torch
    from transformers import AutoModelForVideoClassification, AutoProcessor

    progress = progress or (lambda fraction, message: None)
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForVideoClassification.from_pretrained(MODEL_ID)
    config = model.config
    if int(getattr(config, "num_frames", -1)) != EXPECTED_FRAMES:
        raise ValueError(f"VideoMAE checkpoint expects {getattr(config, 'num_frames', None)} frames; "
                         f"model card documents {EXPECTED_FRAMES}.")
    id2label = getattr(config, "id2label", {}) or {}
    violent_index, nonviolent_index = _resolve_labels(id2label)
    model.to(torch.device("cpu"))
    model.eval()

    # AutoProcessor applies the checkpoint's saved resize/crop, rescale, and normalization.
    # CPU execution guarantees this optional pass cannot compete with Qwen for GPU memory.
    preprocessor = getattr(processor, "image_processor", processor)
    crop_size = getattr(preprocessor, "crop_size", {}) or {}
    size = getattr(preprocessor, "size", {}) or {}
    reported_size = crop_size.get("height") or size.get("height") or size.get("shortest_edge")
    if reported_size is not None and int(reported_size) != EXPECTED_SIZE:
        raise ValueError(f"VideoMAE processor expects {reported_size}px, but model card documents {EXPECTED_SIZE}px.")

    metadata = {
        "status": "running",
        "model_id": MODEL_ID,
        "model_card_url": MODEL_CARD_URL,
        "architecture": "VideoMAE video transformer; binary classifier",
        "training_domain": "UCF-Crime and Bus Violence Dataset (surveillance footage); cartoon performance not evaluated",
        "label_mapping": {str(k): str(v) for k, v in id2label.items()},
        "violent_class_index": violent_index,
        "nonviolent_class_index": nonviolent_index,
        "num_frames": EXPECTED_FRAMES,
        "input_size": EXPECTED_SIZE,
        "preprocessing": "RGB sampled frames passed to checkpoint AutoProcessor; checkpoint resize/crop, rescale and normalization",
        "processor_config": {
            "do_resize": getattr(preprocessor, "do_resize", None),
            "do_center_crop": getattr(preprocessor, "do_center_crop", None),
            "do_rescale": getattr(preprocessor, "do_rescale", None),
            "do_normalize": getattr(preprocessor, "do_normalize", None),
            "size": size,
            "crop_size": crop_size,
            "image_mean": list(getattr(preprocessor, "image_mean", []) or []),
            "image_std": list(getattr(preprocessor, "image_std", []) or []),
        },
        "device": "cpu",
        "scores_are_calibrated": False,
        "cartoon_accuracy_claimed": False,
    }

    results: list[dict] = []
    try:
        for wf in iter_window_frames(video_path, meta, windows, EXPECTED_FRAMES, EXPECTED_SIZE):
            started = time.perf_counter()
            score = None
            status = wf.status
            note = wf.note
            if wf.frames is not None:
                if wf.frames.shape[0] != EXPECTED_FRAMES:
                    status, note = "error", f"Expected {EXPECTED_FRAMES} sampled frames, got {wf.frames.shape[0]}"
                else:
                    try:
                        # Model-card example supplies the 16 RGB frames to the checkpoint processor.
                        inputs = processor([frame for frame in wf.frames], return_tensors="pt")
                        with torch.inference_mode():
                            logits = model(**inputs).logits
                            if logits.ndim != 2 or logits.shape != (1, 2):
                                raise ValueError(f"Unexpected VideoMAE logits shape: {tuple(logits.shape)}")
                            score = float(torch.softmax(logits.float(), dim=-1)[0, violent_index].item())
                        if not np.isfinite(score) or not 0.0 <= score <= 1.0:
                            raise ValueError(f"Invalid VideoMAE violence score: {score}")
                    except Exception as exc:
                        status, note = "error", f"{type(exc).__name__}: {exc}"
            results.append({
                "window_index": wf.window_index,
                "start_s": wf.start_s,
                "end_s": wf.end_s,
                "p_violent": score,
                "status": status,
                "inference_ms": (time.perf_counter() - started) * 1000.0,
                "note": note,
            })
            progress((wf.window_index + 1) / max(len(windows), 1),
                     f"Window {wf.window_index + 1}/{len(windows)} ({wf.start_s:.1f}s–{wf.end_s:.1f}s)")
        failures = sum(row["status"] in ("error", "skipped") for row in results)
        metadata["status"] = "completed" if not failures else "completed_with_window_errors"
        metadata["windows_completed"] = sum(row["p_violent"] is not None for row in results)
        metadata["windows_total"] = len(windows)
    finally:
        del model
        del processor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return results, metadata
