"""Central configuration. No secrets here; paths can be overridden by environment variables."""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env if present
env_file = BASE_DIR / ".env"
if env_file.is_file():
    with open(env_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

DATA_DIR = Path(os.environ.get("CVD_DATA_DIR", BASE_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
EVIDENCE_DIR = DATA_DIR / "evidence"
DB_PATH = Path(os.environ.get("CVD_DB_PATH", DATA_DIR / "database" / "app.db"))
CHECKPOINT_DIR = Path(os.environ.get("CVD_CHECKPOINT_DIR", BASE_DIR / "checkpoints"))

APP_VERSION = "0.1.0"
SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".webm")
MAX_UPLOAD_MB = int(os.environ.get("CVD_MAX_UPLOAD_MB", "1024"))

ADAPTER_PATH = Path(os.environ.get("CVD_ADAPTER_PATH", CHECKPOINT_DIR / "cartoon_adapter.npz"))
TEMPORAL_MODEL_PATH = Path(os.environ.get("CVD_TEMPORAL_MODEL", CHECKPOINT_DIR / "temporal_transformer.pt"))
TUNED_SETTINGS_PATH = Path(os.environ.get("CVD_TUNED_SETTINGS", BASE_DIR / "config" / "tuned_settings.json"))

HF_REPO_ID = "visionlab-ai/school-violence-detection-models"

# Facts below come from the Hugging Face model card (read on 2026-10-10).
# Anything not on the card is NOT assumed (see research/methodology.md, "Unverified").
MODEL_REGISTRY = {
    "x3d_m_realtime": {
        "display_name": "X3D-M (fine-tuned on RWF-2000, 'final_x3d_realtime')",
        "architecture": "pytorchvideo x3d_m, blocks[5].proj replaced by Linear(2048, 2)",
        "hf_filename": "final/final_x3d_realtime.pt",
        "num_frames": 16,
        "input_size": 224,
        "mean": (0.45, 0.45, 0.45),
        "std": (0.225, 0.225, 0.225),
        # Conservative default for an out-of-domain RWF-2000 checkpoint on cartoons.
        # Users can choose a lower threshold when recall matters more than precision.
        "default_threshold": 0.75,
        "violent_index": 1,  # probs[0][1] = violent per model card
        "training_domain": "RWF-2000: real-world fighting videos from YouTube (NOT cartoons)",
        "reported_val_f1": "0.93 on RWF-2000 validation (real-world footage; not cartoon)",
        "license": "MIT (per model card)",
    }
}
DEFAULT_MODEL_KEY = "x3d_m_realtime"


@dataclass
class AnalysisConfig:
    """User-configurable analysis settings; stored with every analysis run."""

    model_key: str = DEFAULT_MODEL_KEY
    window_seconds: float = 4.0
    stride_seconds: float = 2.0
    threshold: float = MODEL_REGISTRY[DEFAULT_MODEL_KEY]["default_threshold"]
    merge_gap_seconds: float = 1.0
    min_event_seconds: float = 0.0
    evidence_frames_per_event: int = 6
    enable_temporal_refinement: bool = True
    suppress_shot_cuts: bool = True
    filter_isolated_spikes: bool = True
    # --- accuracy controls (appended so older positional construction keeps working) ---
    threshold_low: float | None = None   # hysteresis: extend events over adjacent windows >= this value
    spike_margin: float = 0.12
    spike_floor: float = 0.55
    cut_discount: float = 0.55
    refiner_sigma: float = 1.25
    use_adapter: bool = True             # use the cartoon-trained head if checkpoints/cartoon_adapter.npz exists
    use_tta: bool = False                # horizontal-flip test-time augmentation (about 2x inference time)
    enable_verbal_abuse_detection: bool = True
    flag_standalone_verbal_abuse: bool = True
    # Retained for stored-config compatibility; Qwen now runs as a separate report action.
    enable_qwen_explanations: bool = False
    qwen_max_events: int = 20
    enable_multilingual_toxicity: bool = True
    enable_audio_event_detection: bool = True
    use_temporal_transformer: bool = True
    enable_videomae_comparison: bool = False
    videomae_threshold: float = 0.50
    audio_event_threshold: float = 0.35
    verbal_toxicity_threshold: float = 0.85

    def post_params(self):
        """The exact post-processing parameters used by the pipeline and by training/fit.py."""
        from models.transformer_refiner import RefinerParams
        from pipeline.postprocess import PostParams
        return PostParams(
            threshold=self.threshold, threshold_low=self.threshold_low, merge_gap_s=self.merge_gap_seconds,
            min_event_s=self.min_event_seconds, refine=self.enable_temporal_refinement,
            suppress_cuts=self.suppress_shot_cuts, cut_discount=self.cut_discount,
            filter_spikes=self.filter_isolated_spikes, spike_margin=self.spike_margin,
            spike_floor=self.spike_floor, refiner=RefinerParams(sigma=self.refiner_sigma))

    @classmethod
    def from_tuned(cls, model_key: str = DEFAULT_MODEL_KEY) -> "AnalysisConfig | None":
        """Config built from config/tuned_settings.json (written by training/fit.py), or None."""
        t = load_tuned_settings()
        if not t:
            return None
        p = t["post"]
        return cls(model_key=model_key, window_seconds=t.get("window_seconds", 4.0),
                   stride_seconds=t.get("stride_seconds", 2.0), threshold=p["threshold"],
                   merge_gap_seconds=p["merge_gap_s"], min_event_seconds=p["min_event_s"],
                   enable_temporal_refinement=p["refine"], suppress_shot_cuts=p["suppress_cuts"],
                   filter_isolated_spikes=p["filter_spikes"], threshold_low=p.get("threshold_low"),
                   spike_margin=p["spike_margin"], spike_floor=p["spike_floor"], cut_discount=p["cut_discount"],
                   refiner_sigma=(p.get("refiner") or {}).get("sigma", 1.25))

    def validate(self) -> None:
        if self.model_key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model key: {self.model_key}")
        if not 0.5 <= self.window_seconds <= 30:
            raise ValueError("window_seconds must be between 0.5 and 30")
        if not 0.25 <= self.stride_seconds <= self.window_seconds:
            raise ValueError("stride_seconds must be between 0.25 and window_seconds")
        if not 0.0 < self.threshold < 1.0:
            raise ValueError("threshold must be in (0, 1)")
        if self.merge_gap_seconds < 0 or self.min_event_seconds < 0:
            raise ValueError("merge_gap_seconds and min_event_seconds must be >= 0")
        if not 1 <= self.evidence_frames_per_event <= 24:
            raise ValueError("evidence_frames_per_event must be in [1, 24]")
        if self.threshold_low is not None and not 0.0 < self.threshold_low <= self.threshold:
            raise ValueError("threshold_low must be in (0, threshold]")
        if not 0.1 <= self.refiner_sigma <= 10:
            raise ValueError("refiner_sigma must be in [0.1, 10]")
        if not 1 <= self.qwen_max_events <= 20:
            raise ValueError("qwen_max_events must be in [1, 20]")
        if not 0.0 < self.audio_event_threshold < 1.0:
            raise ValueError("audio_event_threshold must be in (0, 1)")
        if not 0.0 < self.verbal_toxicity_threshold < 1.0:
            raise ValueError("verbal_toxicity_threshold must be in (0, 1)")
        if not 0.0 < self.videomae_threshold < 1.0:
            raise ValueError("videomae_threshold must be in (0, 1)")

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def config_hash(self) -> str:
        return hashlib.sha256(self.to_json().encode()).hexdigest()[:16]


def load_tuned_settings() -> dict | None:
    try:
        with open(TUNED_SETTINGS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def ensure_ffmpeg() -> str | None:
    """Ensure ffmpeg executable is discoverable on PATH."""
    import shutil
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.isfile(exe):
            f_dir = os.path.dirname(exe)
            target = os.path.join(f_dir, "ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            if not os.path.exists(target):
                try:
                    shutil.copy2(exe, target)
                except Exception:
                    pass
            if f_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{f_dir};{os.environ.get('PATH', '')}"
            return target if os.path.exists(target) else exe
    except Exception:
        pass
    return None


def ensure_windows_runtime_compatibility() -> None:
    """Neutralize conflicting old MSVCP140.dll bundled in pyarrow on Windows.

    PyArrow wheels on Windows ship an outdated MSVCP140.dll (v14.28) that triggers an
    access violation (0xC0000005 crash) in CTranslate2 / Faster-Whisper when loaded into
    the same process. Removing or synchronizing pyarrow's bundled dll lets Windows use the modern
    system Visual C++ runtime (v14.40+) in System32 safely.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        system_msvc = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "msvcp140.dll"
        if system_msvc.is_file():
            try:
                ctypes.CDLL(str(system_msvc))
            except Exception:
                pass

        import pyarrow
        pa_dir = Path(pyarrow.__file__).parent
        bundled_msvc = pa_dir / "msvcp140.dll"
        if bundled_msvc.is_file():
            if system_msvc.is_file():
                try:
                    import shutil
                    shutil.copy2(system_msvc, bundled_msvc)
                except Exception:
                    bundled_msvc.unlink(missing_ok=True)
            else:
                bundled_msvc.unlink(missing_ok=True)
    except Exception:
        pass


def ensure_dirs() -> None:
    for d in (UPLOAD_DIR, EVIDENCE_DIR, DB_PATH.parent, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)
    ensure_ffmpeg()
    ensure_windows_runtime_compatibility()

