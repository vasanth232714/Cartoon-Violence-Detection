"""Run the pretrained X3D-M ONCE over labelled videos and cache per-window scores + features.

  python -m training.build_cache --labels labelled/labels.json --root labelled --out training_cache

CPU cost = the same as analysing each video once. Re-running skips videos already cached with the same window settings.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from config.settings import MODEL_REGISTRY
from models.cartoon_filter import detect_shot_cut
from preprocessing.clip_sampler import make_windows, to_model_input
from preprocessing.frame_extractor import iter_window_frames
from preprocessing.video_loader import validate_video
from training.dataset import load_labels, window_labels


def cache_video(entry: dict, classifier, out_dir: Path, window_s: float, stride_s: float, model_key: str,
                min_overlap: float = 1.0, force: bool = False, log=print) -> Path | None:
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / (Path(entry["name"]).stem + ".npz")
    meta = validate_video(entry["path"])
    key = {"window_s": window_s, "stride_s": stride_s, "model_key": model_key}
    if out.is_file() and not force:
        try:
            if json.loads(str(np.load(out, allow_pickle=False)["meta"])).get("key") == key:
                log(f"  cached: {out.name}")
                return out
        except Exception:
            pass
    spec = MODEL_REGISTRY[model_key]
    wins = make_windows(meta, window_s, stride_s)
    starts, ends, raw, feats, cuts = [], [], [], [], []
    t0 = time.perf_counter()
    for wf in iter_window_frames(entry["path"], meta, wins, spec["num_frames"], spec["input_size"]):
        if wf.frames is None or wf.status == "skipped":
            continue
        try:
            p, vec, _ = classifier.extract(to_model_input(wf.frames, spec["mean"], spec["std"]))
        except Exception as e:  # skip the window, never invent a value
            log(f"  window {wf.window_index} skipped: {e}")
            continue
        starts.append(wf.start_s); ends.append(wf.end_s); raw.append(p); feats.append(vec)
        cuts.append(bool(detect_shot_cut(wf.frames)[0]))
    if not starts:
        log(f"  no usable windows in {entry['name']}")
        return None
    events = [(0.0, meta.duration_s)] if entry["whole"] == "violent" else (
        [] if entry["whole"] == "non_violent" else entry["events"])
    labels = window_labels(starts, ends, events, min_overlap)
    np.savez_compressed(
        out, starts=np.array(starts), ends=np.array(ends), raw=np.array(raw, dtype=np.float32),
        feats=np.array(feats, dtype=np.float16), cut=np.array(cuts), labels=labels,
        events=np.array(events, dtype=np.float64).reshape(-1, 2),
        meta=np.array(json.dumps({"key": key, "name": entry["name"], "duration_s": meta.duration_s,
                                  "min_overlap": min_overlap, "seconds": time.perf_counter() - t0})))
    log(f"  {entry['name']}: {len(starts)} windows, {int((labels == 1).sum())} pos / {int((labels == 0).sum())} neg "
        f"/ {int((labels == -1).sum())} ambiguous, {time.perf_counter() - t0:.0f}s")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True)
    ap.add_argument("--root", default=None)
    ap.add_argument("--out", default="training_cache")
    ap.add_argument("--window", type=float, default=4.0)
    ap.add_argument("--stride", type=float, default=2.0)
    ap.add_argument("--min-overlap", type=float, default=1.0)
    ap.add_argument("--model-key", default="x3d_m_realtime")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    from models.model_loader import load_classifier
    clf = load_classifier(a.model_key)
    for entry in load_labels(a.labels, a.root):
        print(f"{entry['name']}")
        try:
            cache_video(entry, clf, Path(a.out), a.window, a.stride, a.model_key, a.min_overlap, a.force)
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    main()
