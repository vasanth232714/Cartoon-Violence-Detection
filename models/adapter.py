"""Cartoon adapter: an L2-regularised logistic head trained on frozen X3D-M features.

Why this and not full fine-tuning: the pretrained backbone is run ONCE per labelled window (CPU-feasible), its
2048-d pooled feature (plus the original logit margin) is cached, and only this small head is trained. It
re-learns the decision boundary for cartoon footage, which is where the false positives/negatives come from.

Pure numpy (no torch) so it is testable and reproducible. Dimension-agnostic.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


def train_logistic(X: np.ndarray, y: np.ndarray, C: float = 0.1, iters: int = 400, lr: float = 0.03,
                   balanced: bool = True):
    """Full-batch Adam on the objective  mean_i w_i*logloss_i + ||w||^2 / (2*C*n).
    Returns (w, b, mean, std). Deterministic (zero init)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape
    if n == 0 or len(np.unique(y)) < 2:
        raise ValueError("Training needs samples from both classes.")
    mean = X.mean(0)
    std = X.std(0)
    std[std < 1e-6] = 1.0
    Z = (X - mean) / std
    pos, neg = y.sum(), n - y.sum()
    sw = np.where(y == 1, n / (2 * pos), n / (2 * neg)) if balanced else np.ones(n)
    w = np.zeros(d)
    b = 0.0
    m_w, v_w, m_b, v_b = np.zeros(d), np.zeros(d), 0.0, 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        p = _sigmoid(Z @ w + b)
        r = sw * (p - y) / n
        gw = Z.T @ r + w / (C * n)
        gb = r.sum()
        m_w = b1 * m_w + (1 - b1) * gw
        v_w = b2 * v_w + (1 - b2) * gw ** 2
        m_b = b1 * m_b + (1 - b1) * gb
        v_b = b2 * v_b + (1 - b2) * gb ** 2
        w -= lr * (m_w / (1 - b1 ** t)) / (np.sqrt(v_w / (1 - b2 ** t)) + eps)
        b -= lr * (m_b / (1 - b1 ** t)) / (np.sqrt(v_b / (1 - b2 ** t)) + eps)
    return w, b, mean, std


@dataclass
class CartoonAdapter:
    w: np.ndarray
    b: float
    mean: np.ndarray
    std: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.w.shape[0])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.shape[1] != self.dim:
            raise ValueError(f"Adapter expects {self.dim} features, got {X.shape[1]}")
        return _sigmoid(((X - self.mean) / self.std) @ self.w + self.b)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, w=self.w, b=np.array(self.b), mean=self.mean, std=self.std,
                 meta=np.array(json.dumps(self.meta)))

    @classmethod
    def load(cls, path: str | Path) -> "CartoonAdapter":
        z = np.load(path, allow_pickle=False)
        return cls(z["w"], float(z["b"]), z["mean"], z["std"], json.loads(str(z["meta"])))

    @classmethod
    def fit(cls, X, y, C: float = 0.1, meta: dict | None = None) -> "CartoonAdapter":
        w, b, mean, std = train_logistic(X, y, C=C)
        return cls(w, b, mean, std, dict(meta or {}, C=C, n_train=int(len(y)), n_pos=int(np.sum(y))))

    def describe(self) -> dict:
        keep = ("n_train", "n_pos", "n_videos", "C", "cv_auc_adapter", "cv_auc_raw", "trained_at", "note")
        return {k: self.meta[k] for k in keep if k in self.meta}
