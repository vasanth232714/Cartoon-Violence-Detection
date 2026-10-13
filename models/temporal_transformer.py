"""Temporal model architectures for a full-video X3D + sequence-Transformer pipeline.

The app can load a trained window-level Transformer checkpoint. The repository does not
ship cartoon-trained weights; until the user trains one with labeled sequences, X3D window
scores continue through deterministic post-processing. CNN-only/LSTM architectures remain research code.
"""
from __future__ import annotations

from pathlib import Path

EXPERIMENT_STATUS = [
    {"id": "E1", "name": "CNN-only baseline", "status": "Not implemented",
     "note": "Per-frame CNN classifier with score pooling; needs cartoon-labelled data."},
    {"id": "E2", "name": "CNN + LSTM", "status": "Architecture defined, untrained",
     "note": "models/temporal_transformer.py::CNNLSTMClassifier"},
    {"id": "E3", "name": "X3D features + temporal Transformer", "status": "Integrated, requires training",
     "note": "Train with training/train_temporal_transformer.py using cartoon-labeled sequence cache; no checkpoint ships."},
    {"id": "E4", "name": "CNN + Transformer + severity", "status": "Not implemented",
     "note": "Severity head defined (ordinal 4-way) but no annotated severity data exists; app uses a rule-based rubric."},
    {"id": "BASE", "name": "X3D-M (3D CNN) pretrained baseline", "status": "Integrated (inference only)",
     "note": "Trained on real-world RWF-2000, not cartoon-validated."},
]


class NotTrainedError(RuntimeError):
    pass


def build_window_temporal_transformer(input_dim: int = 2050, d_model: int = 128,
                                     n_layers: int = 2, n_heads: int = 4,
                                     max_len: int = 128):
    """Build a per-window Transformer over frozen X3D features and X3D scores.

    Input: (batch, windows, 2050) = 2048 pooled features + original X3D logit margin + one violence score.
    Output: one violence logit per window. This model is useful only after training
    on labeled cartoon sequences and loading its resulting checkpoint.
    """
    torch, nn = _torch()

    class WindowTemporalTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.max_len = max_len
            self.input_norm = nn.LayerNorm(input_dim)
            self.input_proj = nn.Linear(input_dim, d_model)
            self.position = nn.Parameter(torch.zeros(1, max_len, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                dropout=0.1, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
            self.output_norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, 1)

        def forward(self, x, padding_mask=None):
            if x.shape[1] > self.max_len:
                raise ValueError(f"Sequence length {x.shape[1]} exceeds model limit {self.max_len}")
            z = self.input_proj(self.input_norm(x)) + self.position[:, :x.shape[1]]
            z = self.encoder(z, src_key_padding_mask=padding_mask)
            return self.head(self.output_norm(z)).squeeze(-1)

    return WindowTemporalTransformer()


def load_window_temporal(checkpoint_path: str | Path):
    """Load a genuinely trained window sequence model, refusing missing checkpoints."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise NotTrainedError(f"No trained temporal Transformer checkpoint at {path}")
    torch, _ = _torch()
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint or "model_config" not in checkpoint:
        raise NotTrainedError("Temporal checkpoint format is invalid; expected model_config and state_dict")
    cfg = checkpoint["model_config"]
    model = build_window_temporal_transformer(**cfg)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    return model, checkpoint.get("metadata", {})


def predict_window_sequence(model, features):
    """Apply a trained temporal model in overlapping bounded-length chunks."""
    import numpy as np
    torch, _ = _torch()
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("Temporal features must have shape (windows, features)")
    if not len(x):
        return []
    limit = int(model.max_len)
    stride = max(1, limit - 16)
    starts = list(range(0, max(1, len(x) - limit + 1), stride))
    tail = max(0, len(x) - limit)
    if not starts or starts[-1] != tail:
        starts.append(tail)
    total = np.zeros(len(x), dtype=np.float64)
    count = np.zeros(len(x), dtype=np.float64)
    model.eval()
    with torch.inference_mode():
        for start in starts:
            end = min(start + limit, len(x))
            logits = model(torch.from_numpy(x[start:end]).unsqueeze(0))[0]
            probs = torch.sigmoid(logits).cpu().numpy()
            total[start:end] += probs
            count[start:end] += 1
    return (total / np.maximum(count, 1)).astype(np.float32).tolist()


def _torch():
    import torch
    import torch.nn as nn
    return torch, nn


def build_cnn_transformer(feature_dim: int = 512, d_model: int = 256, n_layers: int = 2, n_heads: int = 4,
                          max_len: int = 64, severity_classes: int = 0):
    torch, nn = _torch()
    import torchvision

    class CNNTransformerClassifier(nn.Module):
        """ResNet-18 per-frame features -> Transformer encoder over time -> violence (+ severity) heads."""

        def __init__(self):
            super().__init__()
            backbone = torchvision.models.resnet18(weights=None)
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.proj = nn.Linear(feature_dim, d_model)
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
            self.pos = nn.Parameter(torch.zeros(1, max_len + 1, d_model))
            layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 2, batch_first=True, dropout=0.1)
            self.encoder = nn.TransformerEncoder(layer, n_layers)
            self.violence_head = nn.Linear(d_model, 2)
            self.severity_head = nn.Linear(d_model, severity_classes) if severity_classes else None

        def forward(self, x):  # x: (B, T, 3, H, W)
            b, t = x.shape[:2]
            f = self.backbone(x.flatten(0, 1)).view(b, t, -1)
            z = torch.cat([self.cls.expand(b, -1, -1), self.proj(f)], dim=1) + self.pos[:, : t + 1]
            h = self.encoder(z)[:, 0]
            return self.violence_head(h), (self.severity_head(h) if self.severity_head else None)

    return CNNTransformerClassifier()


def build_cnn_lstm(feature_dim: int = 512, hidden: int = 256):
    torch, nn = _torch()
    import torchvision

    class CNNLSTMClassifier(nn.Module):
        def __init__(self):
            super().__init__()
            backbone = torchvision.models.resnet18(weights=None)
            backbone.fc = nn.Identity()
            self.backbone = backbone
            self.lstm = nn.LSTM(feature_dim, hidden, batch_first=True)
            self.head = nn.Linear(hidden, 2)

        def forward(self, x):
            b, t = x.shape[:2]
            f = self.backbone(x.flatten(0, 1)).view(b, t, -1)
            out, _ = self.lstm(f)
            return self.head(out[:, -1])

    return CNNLSTMClassifier()


def load_trained(checkpoint_path: str | None):
    """Load a CNN+Transformer ONLY from a real trained checkpoint."""
    if not checkpoint_path or not Path(checkpoint_path).is_file():
        raise NotTrainedError(
            "No trained CNN+Transformer checkpoint exists. Train one on cartoon-annotated data "
            "(see research/evaluation_plan.md) before enabling it in the app."
        )
    torch, _ = _torch()
    model = build_cnn_transformer()
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=True), strict=True)
    return model.eval()
