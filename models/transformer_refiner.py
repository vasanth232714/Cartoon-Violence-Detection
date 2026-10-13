"""Deterministic temporal attention smoothing of window-probability sequences.

IMPORTANT (changed): the previous version blended in the output of a randomly initialised, never-trained
1-D Transformer (20% of every score). That added unreproducible noise and could not be called learning, so it
has been removed. What remains is a FIXED (non-learned) Gaussian temporal-attention kernel plus two simple
persistence rules. All constants live in `RefinerParams` so they can be tuned on labelled data
(`training/fit.py`) instead of being guessed. Pure numpy: no torch dependency, fully reproducible.

A *trained* temporal model needs cartoon-annotated sequences; see training/README.md.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

HAVE_NEURAL_REFINER = False  # no learned network is used


@dataclass
class RefinerParams:
    sigma: float = 1.25            # temporal attention width (in windows)
    w_raw: float = 0.5             # weight of the window's own score
    w_ctx: float = 0.5             # weight of the attention-smoothed context
    spike_raw_min: float = 0.45    # a window above this ...
    spike_neighbor_max: float = 0.25   # ... with both neighbours below this is an "isolated spike"
    spike_factor: float = 0.60     # isolated spikes are multiplied by this
    sustain_raw_min: float = 0.35
    sustain_neighbor_min: float = 0.35
    sustain_bonus: float = 0.08    # bonus for persistent runs


class TransformerRefiner:
    """Name kept for compatibility. Non-learned temporal self-attention smoothing."""

    def __init__(self, d_model: int = 0, sigma: float = 1.25, params: RefinerParams | None = None):
        self.params = params or RefinerParams(sigma=sigma)
        self.sigma = self.params.sigma
        self.last_attention_weights: np.ndarray | None = None

    def compute_temporal_attention(self, raw_probs: np.ndarray) -> np.ndarray:
        T = len(raw_probs)
        if T == 0:
            return np.empty((0, 0), dtype=np.float32)
        if T == 1:
            return np.ones((1, 1), dtype=np.float32)
        idx = np.arange(T, dtype=np.float32)
        dist_sq = (idx[:, None] - idx[None, :]) ** 2
        temporal = np.exp(-dist_sq / (2.0 * self.sigma ** 2))
        agree = np.exp(-np.abs(raw_probs[:, None] - raw_probs[None, :]) / 0.45)
        kernel = temporal * (0.65 + 0.35 * agree)
        return (kernel / np.maximum(kernel.sum(axis=1, keepdims=True), 1e-8)).astype(np.float32)

    def refine(self, probabilities: Sequence[float]) -> list[float]:
        return self.refine_with_attention(probabilities)[0]

    def refine_with_attention(self, probabilities: Sequence[float]) -> tuple[list[float], np.ndarray]:
        if not len(probabilities):
            self.last_attention_weights = np.empty((0, 0), dtype=np.float32)
            return [], self.last_attention_weights
        raw = np.asarray(probabilities, dtype=np.float32)
        T = len(raw)
        if T == 1:
            self.last_attention_weights = np.ones((1, 1), dtype=np.float32)
            return [round(float(raw[0]), 4)], self.last_attention_weights
        P = self.params
        attn = self.compute_temporal_attention(raw)
        self.last_attention_weights = attn
        ctx = attn @ raw
        out = np.zeros(T, dtype=np.float32)
        for i in range(T):
            prev_v = raw[i - 1] if i > 0 else 0.0
            next_v = raw[i + 1] if i < T - 1 else 0.0
            nb_max = max(prev_v, next_v)
            nb_mean = (prev_v + next_v) / (2.0 if 0 < i < T - 1 else 1.0)
            base = P.w_raw * raw[i] + P.w_ctx * ctx[i]
            if raw[i] > P.spike_raw_min and nb_max < P.spike_neighbor_max:
                val = base * P.spike_factor
            elif raw[i] > P.sustain_raw_min and nb_max > P.sustain_neighbor_min:
                val = min(1.0, base + P.sustain_bonus * min(nb_mean, 1.0))
            else:
                val = base
            out[i] = float(np.clip(val, 0.0, 1.0))
        return [round(float(x), 4) for x in out], attn

    def get_last_attention_weights(self) -> np.ndarray | None:
        return self.last_attention_weights
