"""Merge window-level predictions into candidate events (boundaries are window-resolution estimates)."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class WindowPrediction:
    index: int
    start_s: float
    end_s: float
    p_violent: float | None
    status: str = "ok"  # ok | padded | skipped | error
    frame_indices: list[int] = field(default_factory=list)
    frame_times: list[float] = field(default_factory=list)
    inference_ms: float | None = None
    note: str = ""


@dataclass
class CandidateEvent:
    event_no: int
    start_s: float
    end_s: float
    window_indices: list[int]
    peak_prob: float
    mean_prob: float
    peak_window_index: int
    n_positive_runs: int
    peak_time_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    @property
    def n_windows(self) -> int:
        return len(self.window_indices)


def segment_events(windows: list[WindowPrediction], threshold: float, merge_gap_s: float,
                   min_event_s: float = 0.0, low_threshold: float | None = None) -> list[CandidateEvent]:
    """Positive window: usable status and p >= threshold. Positives are merged when they overlap or
    the gap between them is <= merge_gap_s. Events are returned in chronological order."""
    usable = {"ok", "padded"}
    ok = [w for w in windows if w.status in usable and w.p_violent is not None]
    seeds = {w.index for w in ok if w.p_violent >= threshold}
    if low_threshold is not None and low_threshold < threshold and seeds:
        # Hysteresis: a seed window (p >= threshold) is extended over ADJACENT windows with p >= low_threshold.
        # Recovers the weak start/end of real events without lowering the bar for starting a new one.
        by_index = {w.index: w for w in ok}
        grown = set(seeds)
        for direction in (-1, 1):
            for i in sorted(seeds):
                j = i + direction
                while j in by_index and by_index[j].p_violent >= low_threshold and j not in grown:
                    grown.add(j)
                    j += direction
        seeds = grown
    pos = sorted((w for w in ok if w.index in seeds), key=lambda w: (w.start_s, w.index))
    groups: list[list[WindowPrediction]] = []
    for w in pos:
        if groups and w.start_s <= max(x.end_s for x in groups[-1]) + merge_gap_s:
            groups[-1].append(w)
        else:
            groups.append([w])

    events: list[CandidateEvent] = []
    for g in groups:
        start, end = min(w.start_s for w in g), max(w.end_s for w in g)
        if end - start < min_event_s:
            continue
        idxs = sorted(w.index for w in g)
        runs = 1 + sum(1 for a, b in zip(idxs, idxs[1:]) if b - a > 1)
        peak = max(g, key=lambda w: w.p_violent)
        events.append(CandidateEvent(
            event_no=0, start_s=start, end_s=end, window_indices=idxs, peak_prob=float(peak.p_violent),
            mean_prob=float(sum(w.p_violent for w in g) / len(g)), peak_window_index=peak.index,
            n_positive_runs=runs, peak_time_s=(peak.start_s + peak.end_s) / 2.0,
        ))
    events.sort(key=lambda e: e.start_s)
    for i, e in enumerate(events, 1):
        e.event_no = i
    return events
