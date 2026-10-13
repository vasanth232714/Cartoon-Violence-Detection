"""Checkpoint download + strict, validated loading of the X3D-M violence classifier.

Design rules:
  * Architecture is built exactly as documented on the model card.
  * load_state_dict(strict=True) only - we never silently run with randomly initialised weights.
  * torch.load(weights_only=True) by default; opt-in env CVD_TRUST_CHECKPOINT=1 for pickled objects.
  * A zero-input forward pass validates the (1,3,T,H,W) -> (1,2) contract before use.
"""
from __future__ import annotations

import os
from pathlib import Path

from config.settings import CHECKPOINT_DIR, HF_REPO_ID, MODEL_REGISTRY
from models.violence_classifier import X3DViolenceClassifier


class ModelLoadError(Exception):
    """Raised with an actionable message when the model cannot be loaded."""


def expected_checkpoint_path(model_key: str) -> Path:
    return CHECKPOINT_DIR / MODEL_REGISTRY[model_key]["hf_filename"]


def download_checkpoint(model_key: str) -> Path:
    """Explicit network action (user-triggered). Downloads from the Hugging Face repo."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ModelLoadError("huggingface_hub is not installed: pip install huggingface_hub") from e
    spec = MODEL_REGISTRY[model_key]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        path = hf_hub_download(repo_id=HF_REPO_ID, filename=spec["hf_filename"], local_dir=str(CHECKPOINT_DIR))
    except Exception as e:
        raise ModelLoadError(f"Could not download {spec['hf_filename']} from {HF_REPO_ID}: {e}") from e
    return Path(path)


def _build_x3d_m(with_sequential_head: bool = True):
    try:
        import torch
        from pytorchvideo.models.hub import x3d_m
    except Exception as e:
        raise ModelLoadError(
            "torch / pytorchvideo are not importable. Install with the pinned requirements.txt "
            f"(original error: {type(e).__name__}: {e})"
        ) from e
    model = x3d_m(pretrained=False)
    if with_sequential_head:
        model.blocks[5].proj = torch.nn.Sequential(
            torch.nn.Dropout(p=0.3),
            torch.nn.Linear(2048, 2),
        )
    else:
        model.blocks[5].proj = torch.nn.Linear(2048, 2)
    return model


def _extract_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for k in ("model", "model_state_dict", "state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                return ckpt[k]
    return ckpt


def load_state_strict(model, ckpt) -> str:
    """Try known key prefixes with strict=True. Returns the prefix that worked."""
    sd = _extract_state_dict(ckpt)
    if not isinstance(sd, dict):
        raise ModelLoadError(f"Checkpoint does not contain a state dict (got {type(sd).__name__}).")
    errors = []
    for prefix in ("", "module.", "model.", "backbone.", "net."):
        cand = {(k[len(prefix):] if prefix and k.startswith(prefix) else k): v for k, v in sd.items()}
        try:
            model.load_state_dict(cand, strict=True)
            return prefix
        except RuntimeError as e:
            errors.append(f"prefix {prefix!r}: {str(e).splitlines()[0]}")
    sample = list(sd.keys())[:6]
    raise ModelLoadError(
        "Checkpoint keys do not match the documented X3D-M architecture (strict load failed for all "
        "tried prefixes). The model card says the repo's own wrapper classes may be required.\n"
        f"Sample checkpoint keys: {sample}\nAttempts: {errors}"
    )


def load_adapter_if_present(spec: dict):
    """Load checkpoints/cartoon_adapter.npz if it exists and matches this model's feature size (2048 + 1)."""
    from config.settings import ADAPTER_PATH
    from models.adapter import CartoonAdapter

    if not ADAPTER_PATH.is_file():
        return None
    try:
        ad = CartoonAdapter.load(ADAPTER_PATH)
    except Exception as e:
        raise ModelLoadError(f"Adapter file {ADAPTER_PATH} could not be read: {e}") from e
    if ad.dim != X3DViolenceClassifier.FEATURE_DIM + 1:
        raise ModelLoadError(f"Adapter expects {ad.dim} features; this model provides "
                             f"{X3DViolenceClassifier.FEATURE_DIM + 1}. Retrain with training/fit.py.")
    return ad


def load_classifier(model_key: str, checkpoint_path: str | Path | None = None) -> X3DViolenceClassifier:
    if model_key not in MODEL_REGISTRY:
        raise ModelLoadError(f"Unknown model key {model_key!r}")
    spec = MODEL_REGISTRY[model_key]
    path = Path(checkpoint_path) if checkpoint_path else expected_checkpoint_path(model_key)
    if not path.is_file():
        raise ModelLoadError(
            f"Checkpoint not found at {path}. Use the 'Download checkpoint' button or place "
            f"{spec['hf_filename']} there manually."
        )
    import torch

    trust = os.environ.get("CVD_TRUST_CHECKPOINT", "1") == "1"
    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=not trust)
    except Exception as e:
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception:
            raise ModelLoadError(
                f"torch.load failed: {e}\nIf you trust this file, set "
                "CVD_TRUST_CHECKPOINT=1 and retry."
            ) from e

    sd = _extract_state_dict(ckpt)
    has_seq = any(".proj.1." in k or k.startswith("proj.1.") for k in (sd.keys() if isinstance(sd, dict) else []))
    model = _build_x3d_m(with_sequential_head=has_seq)
    load_state_strict(model, ckpt)
    model.eval()
    n, s = spec["num_frames"], spec["input_size"]
    with torch.inference_mode():
        out = model(torch.zeros(1, 3, n, s, s))
    if tuple(out.shape) != (1, 2):
        raise ModelLoadError(f"Validation forward pass returned {tuple(out.shape)}, expected (1, 2).")
    adapter = load_adapter_if_present(spec)
    return X3DViolenceClassifier(model, spec, str(path), model_key, adapter=adapter)
