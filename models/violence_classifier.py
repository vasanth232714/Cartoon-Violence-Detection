"""Clip-level violence classifier wrapper (real inference only; no fabricated outputs)."""
from __future__ import annotations

import math
from typing import Protocol

import numpy as np


class InvalidModelOutputError(Exception):
    """Model produced an output that cannot be interpreted as a probability."""


class ClipClassifier(Protocol):
    info: dict

    def predict_proba(self, clip: np.ndarray) -> float:  # clip: (3,T,H,W) float32
        ...


class X3DViolenceClassifier:
    """Wraps the pytorchvideo X3D-M checkpoint. Output index `violent_index` is the violent class.

    Optional accuracy features (all off unless available/requested):
      * adapter: a cartoon-trained head (models/adapter.py) applied to the pooled 2048-d feature.
      * TTA: average over the clip and its horizontal flip (violence is flip-invariant).
    """

    FEATURE_DIM = 2048

    def __init__(self, model, spec: dict, checkpoint_path: str, model_key: str, adapter=None):
        self.model = model
        self.spec = spec
        self.violent_index = spec["violent_index"]
        self.adapter = adapter
        self.use_adapter = adapter is not None
        self.use_tta = False
        self._feat = None
        # Capture the pooled feature that feeds the final projection (works for Linear or Sequential head).
        model.blocks[5].proj.register_forward_hook(self._capture)
        self._base_info = {
            "model_key": model_key,
            "display_name": spec["display_name"],
            "architecture": spec["architecture"],
            "checkpoint": str(checkpoint_path),
            "num_frames": spec["num_frames"],
            "input_size": spec["input_size"],
            "mean": list(spec["mean"]),
            "std": list(spec["std"]),
            "training_domain": spec["training_domain"],
            "license": spec["license"],
            "device": "cpu",
        }
        self.configure(use_adapter=self.use_adapter, use_tta=False)

    # -- configuration -------------------------------------------------------------------------
    def configure(self, use_adapter: bool = True, use_tta: bool = False) -> None:
        self.use_adapter = bool(use_adapter and self.adapter is not None)
        self.use_tta = bool(use_tta)
        info = dict(self._base_info)
        if self.use_adapter:
            info["adapter"] = self.adapter.describe()
            info["score_meaning"] = ("probability from a logistic head trained on this project's labelled cartoon "
                                     "windows (see adapter stats); validity depends on that data, not on RWF-2000")
        else:
            info["score_meaning"] = ("softmax probability of the 'violent' class (uncalibrated; trained on "
                                     "real-world footage, not validated on cartoons)")
        info["test_time_augmentation"] = "horizontal flip" if self.use_tta else "off"
        self.info = info

    def _capture(self, module, inputs, output):
        self._feat = inputs[0].detach()

    # -- inference -----------------------------------------------------------------------------
    def _check_shape(self, clip: np.ndarray) -> None:
        expected = (3, self.spec["num_frames"], self.spec["input_size"], self.spec["input_size"])
        if tuple(clip.shape) != expected:
            raise InvalidModelOutputError(f"Input shape {tuple(clip.shape)} != expected {expected}")

    def extract(self, clip: np.ndarray):
        """Return (raw_prob, feature_vector[2048 + 1 margin], logit_margin) for ONE clip, no adapter, no TTA.
        Used by training/build_cache.py."""
        import torch

        self._check_shape(clip)
        with torch.inference_mode():
            logits = self.model(torch.from_numpy(np.ascontiguousarray(clip)).unsqueeze(0))
            if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != 2:
                raise InvalidModelOutputError(f"Unexpected logits shape {tuple(logits.shape)}")
            f = self._feat
            if f is None or f.shape[-1] != self.FEATURE_DIM:
                raise InvalidModelOutputError(f"Unexpected feature shape {None if f is None else tuple(f.shape)}")
            feat = f.reshape(-1, self.FEATURE_DIM).mean(0).float().numpy()
            lg = logits.float()[0]
            margin = float(lg[self.violent_index] - lg[1 - self.violent_index])
            p = float(torch.softmax(lg, dim=0)[self.violent_index])
        if not math.isfinite(p) or not 0.0 <= p <= 1.0 or not np.isfinite(feat).all():
            raise InvalidModelOutputError(f"Non-finite model output (p={p})")
        return p, np.concatenate([feat, [margin]]).astype(np.float32), margin

    def _score(self, clip: np.ndarray) -> float:
        p, vec, _ = self.extract(clip)
        return float(self.adapter.predict_proba(vec)[0]) if self.use_adapter else p

    def predict_proba(self, clip: np.ndarray) -> float:
        scores = [self._score(clip)]
        if self.use_tta:
            scores.append(self._score(np.ascontiguousarray(clip[..., ::-1])))
        p = float(np.mean(scores))
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise InvalidModelOutputError(f"Non-finite or out-of-range probability: {p}")
        return p

    def predict_with_features(self, clip: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the configured X3D probability and frozen feature vector from one forward pass."""
        raw, vec, _ = self.extract(clip)
        p = float(self.adapter.predict_proba(vec[None, :])[0]) if self.use_adapter else raw
        if self.use_tta:
            raw_flip, vec_flip, _ = self.extract(np.ascontiguousarray(clip[..., ::-1]))
            p_flip = float(self.adapter.predict_proba(vec_flip[None, :])[0]) if self.use_adapter else raw_flip
            p = 0.5 * (p + p_flip)
            vec = 0.5 * (vec + vec_flip)
        if not math.isfinite(p) or not 0.0 <= p <= 1.0:
            raise InvalidModelOutputError(f"Non-finite or out-of-range probability: {p}")
        return p, vec.astype(np.float32, copy=False)
