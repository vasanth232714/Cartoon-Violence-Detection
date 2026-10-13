"""Label handling shared by build_cache.py and fit.py.

labels.json:
{"videos": [
   {"file": "ep1.mp4", "events": [{"start": 31.0, "end": 39.0}, {"start": 70, "end": 75}]},
   {"file": "calm_clip.mp4", "label": "non_violent"},          # whole video negative
   {"file": "fight_clip.mp4", "label": "violent"}              # whole video positive
]}
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_labels(labels_path: str | Path, root: str | Path | None = None) -> list[dict]:
    labels_path = Path(labels_path)
    root = Path(root) if root else labels_path.parent
    with open(labels_path, encoding="utf-8") as f:
        raw = json.load(f)
    out = []
    for v in raw["videos"]:
        entry = {"name": v["file"], "path": str(root / v["file"]), "whole": None, "events": []}
        if "events" in v:
            entry["events"] = [(float(e["start"]), float(e["end"])) for e in v["events"]]
            for s, e in entry["events"]:
                if not e > s >= 0:
                    raise ValueError(f"{v['file']}: invalid event ({s}, {e})")
        elif v.get("label") in ("violent", "non_violent"):
            entry["whole"] = v["label"]
        else:
            raise ValueError(f"{v['file']}: needs 'events' or label 'violent'/'non_violent'")
        out.append(entry)
    return out


def window_labels(starts, ends, events, min_overlap: float = 1.0) -> np.ndarray:
    """1 = positive, 0 = negative, -1 = ambiguous (touches an event by less than the required overlap).
    Ambiguous windows are excluded from training and scoring: they are label noise at event boundaries."""
    starts, ends = np.asarray(starts, float), np.asarray(ends, float)
    out = np.zeros(len(starts), dtype=np.int8)
    for i, (s, e) in enumerate(zip(starts, ends)):
        ov = max([max(0.0, min(e, ge) - max(s, gs)) for gs, ge in events] or [0.0])
        need = min(min_overlap, 0.5 * (e - s))
        if ov >= need and ov > 0:
            out[i] = 1
        elif ov > 0:
            out[i] = -1
    return out
