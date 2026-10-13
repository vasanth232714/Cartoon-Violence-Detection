"""Cartoon-specific false-positive suppression and precision calibration engine.

Filters out spurious false positives in animation footage caused by:
1. Hard scene cuts and editing transitions (inter-frame optical shockwaves)
2. Isolated single-window probability spikes (flash effects, camera zooms)
3. Uniform camera pans (distinguished from turbulent localized physical combat)
4. Borderline single-window candidate event artifacts
"""
from __future__ import annotations

import math
from typing import Sequence
import numpy as np

from models.transformer_refiner import RefinerParams, TransformerRefiner

HAVE_TRANSFORMER = True  # deterministic numpy refiner; no torch needed


SHOT_CUT_THRESHOLD = 42.0  # Mean absolute inter-frame difference indicating an edit cut


def detect_shot_cut(frames: list[np.ndarray]) -> tuple[bool, float, float]:
    """Detect if a 16-frame window contains an edit cut between different camera shots.
    
    Args:
        frames: List of 16 RGB images (H, W, 3) as uint8 or float.
        
    Returns:
        (has_cut, max_cut_diff, baseline_motion)
    """
    if frames is None or len(frames) < 2:
        return False, 0.0, 0.0

    diffs: list[float] = []
    for i in range(len(frames) - 1):
        f1 = frames[i]
        f2 = frames[i + 1]
        # Fast downsampled luminance difference
        if f1.ndim == 3 and f1.shape[2] >= 3:
            # Subsample by step of 4 for near-zero CPU overhead (~0.05ms)
            g1 = (0.299 * f1[::4, ::4, 0] + 0.587 * f1[::4, ::4, 1] + 0.114 * f1[::4, ::4, 2])
            g2 = (0.299 * f2[::4, ::4, 0] + 0.587 * f2[::4, ::4, 1] + 0.114 * f2[::4, ::4, 2])
        else:
            g1 = f1[::4, ::4].astype(np.float32)
            g2 = f2[::4, ::4].astype(np.float32)

        d = float(np.abs(g1 - g2).mean())
        diffs.append(d)

    if not diffs:
        return False, 0.0, 0.0

    max_d = max(diffs)
    # Exclude the max cut to calculate baseline continuous animation motion
    other_diffs = [d for d in diffs if d != max_d]
    baseline_motion = float(np.mean(other_diffs)) if other_diffs else max_d

    has_cut = (max_d >= SHOT_CUT_THRESHOLD)
    return has_cut, max_d, baseline_motion


class CartoonPrecisionCalibrator:
    """Multi-stage Precision Calibrator designed for animation and cartoon footage."""

    def __init__(self, use_neural_refiner: bool = True, refiner_params: RefinerParams | None = None,
                 cut_discount: float = 0.55):
        self.refiner = TransformerRefiner(params=refiner_params) if use_neural_refiner else None
        self.cut_discount = cut_discount

    def calibrate_window_probabilities(
        self,
        raw_probs: Sequence[float],
        shot_cut_flags: Sequence[bool] | None = None,
        suppress_cuts: bool = True,
        use_temporal_refinement: bool = True
    ) -> list[float]:
        """Calibrate raw X3D-M probability sequence into cartoon-precision probabilities.
        
        Steps:
        1. Shot-cut discounting: If a window's spike was triggered by an edit cut
           rather than sustained physical altercation, apply transition attenuation.
        2. Temporal sequence modeling: Pass through 1D Temporal Transformer to smooth
           transient noise, suppress isolated spikes, and reinforce persistent action.
        3. Dynamic scaling: Bounds clamped to [0.0, 1.0].
        """
        if not raw_probs:
            return []

        probs = list(raw_probs)
        T = len(probs)

        # 1. Shot Cut Discounting
        if suppress_cuts and shot_cut_flags is not None and len(shot_cut_flags) == T:
            for i in range(T):
                if bool(shot_cut_flags[i]) and probs[i] is not None:
                    # Check neighbor support
                    prev_p = probs[i - 1] if i > 0 and probs[i - 1] is not None else 0.0
                    next_p = probs[i + 1] if i < T - 1 and probs[i + 1] is not None else 0.0
                    neighbor_max = max(prev_p, next_p)

                    # If this window has an edit cut and neighbors have low violence,
                    # the spike was caused by the camera shot cut!
                    if neighbor_max < 0.40:
                        # Discount cut spike by 45%
                        probs[i] = round(float(probs[i] * self.cut_discount), 4)

        # 2. Temporal Transformer Refinement
        if use_temporal_refinement and self.refiner is not None and T > 1:
            # Handle any None entries cleanly
            valid_indices = [i for i, p in enumerate(probs) if p is not None]
            if len(valid_indices) == T:
                refined = self.refiner.refine(probs)
                return [round(float(p), 4) for p in refined]
            elif len(valid_indices) > 1:
                sub_probs = [probs[i] for i in valid_indices]
                sub_refined = self.refiner.refine(sub_probs)
                res = list(probs)
                for idx, r_val in zip(valid_indices, sub_refined):
                    res[idx] = round(float(r_val), 4)
                return res

        return [round(float(p), 4) if p is not None else None for p in probs]

    def filter_candidate_events(
        self,
        events: list,
        windows: list,
        threshold: float,
        min_event_seconds: float = 0.0,
        filter_isolated_spikes: bool = True,
        spike_margin: float = 0.12,
        spike_floor: float = 0.55,
    ) -> list:
        """Filter out false positive candidate events that lack temporal backing or duration.
        
        Criteria:
        - If min_event_seconds > 0, discard events shorter than min_event_seconds.
        - If filter_isolated_spikes is True:
          Discard events that consist of only 1 window whose peak barely reached threshold
          (peak < threshold + 0.12 and no multi-window persistence).
        """
        filtered = []
        for ev in events:
            # Check duration
            if ev.duration_s < min_event_seconds:
                continue

            # Check isolated single-window spike
            if filter_isolated_spikes and ev.n_windows == 1:
                # Require single-window events to have strong confidence (e.g. >= threshold + 0.12 or >= 0.60)
                min_acceptable_peak = max(threshold + spike_margin, spike_floor)
                if ev.peak_prob < min_acceptable_peak:
                    continue

            filtered.append(ev)

        # Renumber remaining events
        for i, ev in enumerate(filtered, 1):
            ev.event_no = i

        return filtered
