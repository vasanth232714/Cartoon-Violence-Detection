"""Single source of truth for turning raw window probabilities into events.

Used by BOTH the app (pipeline/inference_pipeline.py) and the tuning tool (training/fit.py), so the settings
that are tuned on labelled data behave identically at run time. Pure numpy, deterministic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Sequence

from models.cartoon_filter import CartoonPrecisionCalibrator
from models.transformer_refiner import RefinerParams
from pipeline.event_segmenter import CandidateEvent, WindowPrediction, segment_events


@dataclass
class PostParams:
    threshold: float = 0.75
    threshold_low: float | None = None      # None = no hysteresis
    merge_gap_s: float = 1.0
    min_event_s: float = 1.5
    refine: bool = True                      # temporal attention smoothing
    suppress_cuts: bool = True               # discount spikes caused by hard shot cuts
    cut_discount: float = 0.55
    filter_spikes: bool = True               # drop weak single-window events
    spike_margin: float = 0.12
    spike_floor: float = 0.55
    refiner: RefinerParams = field(default_factory=RefinerParams)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PostParams":
        d = dict(d)
        ref = RefinerParams(**{k: v for k, v in (d.pop("refiner", None) or {}).items()
                               if k in {f.name for f in fields(RefinerParams)}})
        return cls(refiner=ref, **{k: v for k, v in d.items() if k in {f.name for f in fields(cls)} - {"refiner"}})


def refine_probabilities(raw: Sequence[float | None], cut_flags: Sequence[bool] | None,
                         params: PostParams) -> list[float | None]:
    if params.refine or params.suppress_cuts:
        cal = CartoonPrecisionCalibrator(use_neural_refiner=params.refine, refiner_params=params.refiner,
                                         cut_discount=params.cut_discount)
        return cal.calibrate_window_probabilities(list(raw), shot_cut_flags=cut_flags,
                                                  suppress_cuts=params.suppress_cuts,
                                                  use_temporal_refinement=params.refine)
    return list(raw)


def detect_events(preds: list[WindowPrediction], params: PostParams) -> list[CandidateEvent]:
    """`preds[i].p_violent` must already be the refined probability."""
    events = segment_events(preds, params.threshold, params.merge_gap_s, params.min_event_s,
                            low_threshold=params.threshold_low)
    if params.filter_spikes:
        cal = CartoonPrecisionCalibrator(use_neural_refiner=False)
        events = cal.filter_candidate_events(events, preds, params.threshold, params.min_event_s,
                                             filter_isolated_spikes=True, spike_margin=params.spike_margin,
                                             spike_floor=params.spike_floor)
    return events


def run_postprocess(starts: Sequence[float], ends: Sequence[float], raw: Sequence[float | None],
                    cut_flags: Sequence[bool] | None, params: PostParams):
    """Convenience for offline tuning: returns (refined_probs, events)."""
    refined = refine_probabilities(raw, cut_flags, params)
    preds = [WindowPrediction(i, float(s), float(e), p) for i, (s, e, p) in enumerate(zip(starts, ends, refined))]
    return refined, detect_events(preds, params)
