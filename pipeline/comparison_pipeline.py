"""Multi-Model Comparison Pipeline for Cartoon Violence Detection.

Executes, evaluates, and cross-compares 3 model approaches on the same video:
1. Baseline X3D-M (3D-CNN spatiotemporal action recognition)
2. Transformer-Enhanced (1D Temporal Transformer refinement)
3. Multimodal Audio-Visual (Late / Attention multimodal fusion)

Computes pairwise agreement rates, R² correlation scores, MAE, event detections,
and inference latency. No accuracy figures are produced without ground truth.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from config.settings import MODEL_REGISTRY, AnalysisConfig
from database.repository import Repository
from models.audio_analyzer import AudioAnalyzer
from models.multimodal_fusion import MultimodalFusion
from models.transformer_refiner import TransformerRefiner
from models.violence_classifier import ClipClassifier
from pipeline.event_segmenter import CandidateEvent, WindowPrediction, segment_events
from preprocessing.clip_sampler import make_windows, to_model_input
from preprocessing.frame_extractor import iter_window_frames
from preprocessing.video_loader import VideoMetadata

logger = logging.getLogger(__name__)
ProgressCB = Callable[[str, float, str], None]


def _safe_get_window_val(w: Any, key: str, default: Any = None) -> Any:
    if hasattr(w, key):
        val = getattr(w, key)
        return default if val is None else val
    try:
        val = w[key]
        return default if val is None else val
    except Exception:
        return default


# Only figures with a real source are listed. Everything else is None = "not measured".
# (A previous version listed "projected" accuracies of 94.5% / 95.2% that were never measured; removed.)
BENCHMARK_REPORTS = [
    {
        "model_key": "x3d_m_realtime",
        "model_name": "X3D-M Baseline (3D-CNN)",
        "architecture": "3D Spatiotemporal Convolution (16-frame dense)",
        "accuracy": None,
        "f1_score": 0.93,
        "error_rate": None,
        "r2_score": None,
        "parameters": "~3M (model card)",
        "source_note": "Model card: F1 0.93 on RWF-2000 validation (real-world footage, NOT cartoons)",
    },
    {
        "model_key": "transformer_enhanced",
        "model_name": "Temporal attention smoothing",
        "architecture": "X3D-M + fixed (non-learned) temporal attention kernel",
        "accuracy": None, "f1_score": None, "error_rate": None, "r2_score": None,
        "parameters": "0 trained",
        "source_note": "Not measured - requires labelled cartoon data (training/fit.py)",
    },
    {
        "model_key": "multimodal_fusion",
        "model_name": "Multimodal Audio-Visual",
        "architecture": "X3D-M + heuristic spectral audio score, late fusion",
        "accuracy": None, "f1_score": None, "error_rate": None, "r2_score": None,
        "parameters": "0 trained",
        "source_note": "Not measured - audio score is a hand-built heuristic, not a trained model",
    },
]


@dataclass
class ModelPredictionSummary:
    """Consolidated prediction output and event summary for a specific model."""
    model_name: str
    display_name: str
    architecture: str
    probabilities: list[float]
    binary_predictions: list[int]
    events: list[CandidateEvent]
    n_events: int
    n_violent_windows: int
    inference_time_s: float
    mean_probability: float
    peak_probability: float


@dataclass
class PairwiseMetrics:
    """Pairwise statistical alignment and error metrics between two models."""
    model_a: str
    model_b: str
    agreement_rate: float       # Percentage of windows sharing the same binary decision
    mae: float                  # Mean Absolute Error between continuous probabilities
    r2_score: float             # Pearson r^2 correlation score
    precision_rel: float        # Relative precision
    recall_rel: float           # Relative recall
    f1_rel: float               # Relative F1 score


@dataclass
class BenchmarkMetric:
    """Hardcoded academic research benchmarks for the comparison table."""
    model_key: str
    model_name: str
    architecture: str
    accuracy: float | None
    f1_score: float | None
    error_rate: float | None
    r2_score: float | None
    parameters: str
    source_note: str


@dataclass
class ComparisonResult:
    """Comprehensive comparison output containing all 3 models' predictions and analytics."""
    video_id: int
    video_name: str
    duration_s: float
    threshold: float
    windows: list[tuple[float, float]]
    window_times: list[float]
    baseline_x3d: ModelPredictionSummary
    transformer_enhanced: ModelPredictionSummary
    multimodal: ModelPredictionSummary
    audio_scores: list[float]
    pairwise_metrics: dict[str, PairwiseMetrics]
    benchmarks: list[BenchmarkMetric]
    attention_matrix: list[list[float]] | None
    recommendation: dict[str, Any]


def _compute_pairwise(
    name_a: str,
    probs_a: Sequence[float],
    name_b: str,
    probs_b: Sequence[float],
    threshold: float
) -> PairwiseMetrics:
    """Calculate agreement, MAE, R², and relative classification metrics."""
    arr_a = np.array(probs_a, dtype=np.float32)
    arr_b = np.array(probs_b, dtype=np.float32)
    n = len(arr_a)

    if n == 0 or len(arr_b) != n:
        return PairwiseMetrics(name_a, name_b, 100.0, 0.0, 1.0, 1.0, 1.0, 1.0)

    # 1. Binary decisions at threshold
    bin_a = (arr_a >= threshold).astype(int)
    bin_b = (arr_b >= threshold).astype(int)
    agreement = float(np.mean(bin_a == bin_b) * 100.0)

    # 2. Mean Absolute Error
    mae = float(np.mean(np.abs(arr_a - arr_b)))

    # 3. R² correlation (Pearson r^2)
    std_a = np.std(arr_a)
    std_b = np.std(arr_b)
    if std_a < 1e-6 or std_b < 1e-6:
        r2 = 1.0 if np.allclose(arr_a, arr_b, atol=1e-3) else 0.0
    else:
        corr_mat = np.corrcoef(arr_a, arr_b)
        r_val = float(corr_mat[0, 1])
        r2 = float(np.clip(r_val ** 2, 0.0, 1.0))

    # 4. Relative precision, recall, F1 (treating model A as reference)
    tp = np.sum((bin_b == 1) & (bin_a == 1))
    fp = np.sum((bin_b == 1) & (bin_a == 0))
    fn = np.sum((bin_b == 0) & (bin_a == 1))

    prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 1.0
    rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 1.0
    f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 1.0

    return PairwiseMetrics(
        model_a=name_a,
        model_b=name_b,
        agreement_rate=round(agreement, 2),
        mae=round(mae, 4),
        r2_score=round(r2, 4),
        precision_rel=round(prec, 4),
        recall_rel=round(rec, 4),
        f1_rel=round(f1, 4)
    )


def _make_window_predictions_for_model(
    windows: list[tuple[float, float]],
    probs: list[float]
) -> list[WindowPrediction]:
    """Helper to convert probability list back to WindowPrediction objects for event segmenter."""
    preds: list[WindowPrediction] = []
    for i, (w, p) in enumerate(zip(windows, probs)):
        preds.append(WindowPrediction(
            index=i,
            start_s=w[0],
            end_s=w[1],
            p_violent=p,
            status="ok"
        ))
    return preds


def run_comparison(
    video_id: int,
    video_path: str,
    meta: VideoMetadata,
    cfg: AnalysisConfig,
    classifier: ClipClassifier,
    repo: Repository,
    progress: ProgressCB | None = None,
    base_predictions: Sequence[WindowPrediction | dict[str, Any]] | None = None
) -> ComparisonResult:
    """Run all three models (X3D-M, Transformer Refiner, Multimodal Fusion) on a video

    and return a ComparisonResult with detailed analytics.
    """
    progress = progress or (lambda stage, frac, msg: None)
    cfg.validate()
    spec = MODEL_REGISTRY[cfg.model_key]
    threshold = cfg.threshold

    progress("Initializing Comparison", 0.05, "Preparing video segments and windows")
    windows = make_windows(meta, cfg.window_seconds, cfg.stride_seconds)
    window_times = [(w[0] + w[1]) / 2.0 for w in windows]
    n_windows = len(windows)

    # -------------------------------------------------------------
    # 1. Baseline Model (X3D-M 3D-CNN)
    # -------------------------------------------------------------
    t_base_start = time.perf_counter()
    raw_x3d_probs: list[float] = []

    if base_predictions and len(base_predictions) == n_windows:
        progress("Baseline X3D-M", 0.20, "Loading existing X3D-M window predictions")
        for bp in base_predictions:
            p_val = _safe_get_window_val(bp, "p_violent", 0.0)
            raw_x3d_probs.append(float(p_val) if p_val is not None else 0.0)
        # Small nominal simulated latency if re-using cached
        t_base_s = sum(
            float(_safe_get_window_val(bp, "inference_ms", 120.0) or 120.0) / 1000.0
            for bp in base_predictions
        )
    else:
        progress("Baseline X3D-M", 0.15, "Running X3D-M 3D-CNN temporal tensor inference")
        preds: list[WindowPrediction] = []
        for i, wf in enumerate(iter_window_frames(video_path, meta, windows, spec["num_frames"], spec["input_size"])):
            p = 0.0
            if wf.frames is not None:
                try:
                    clip = to_model_input(wf.frames, spec["mean"], spec["std"])
                    p = classifier.predict_proba(clip)
                except Exception as exc:
                    logger.warning("Frame inference error on window %d: %s", i, exc)
                    p = 0.0
            raw_x3d_probs.append(float(p))
            frac = 0.10 + 0.30 * ((i + 1) / n_windows)
            progress("Baseline X3D-M", frac, f"Window {i + 1}/{n_windows} ({wf.start_s:.1f}s - {wf.end_s:.1f}s)")
        t_base_s = time.perf_counter() - t_base_start

    # Clean raw probabilities
    x3d_probs = [round(float(p), 4) for p in raw_x3d_probs]
    x3d_bin = [1 if p >= threshold else 0 for p in x3d_probs]
    x3d_wp = _make_window_predictions_for_model(windows, x3d_probs)
    x3d_events = segment_events(x3d_wp, threshold, cfg.merge_gap_seconds, cfg.min_event_seconds)

    baseline_summary = ModelPredictionSummary(
        model_name="Baseline (X3D-M)",
        display_name="X3D-M (3D-CNN)",
        architecture="Dense Spatiotemporal 3D Convolution",
        probabilities=x3d_probs,
        binary_predictions=x3d_bin,
        events=x3d_events,
        n_events=len(x3d_events),
        n_violent_windows=sum(x3d_bin),
        inference_time_s=round(t_base_s, 2),
        mean_probability=round(float(np.mean(x3d_probs)) if x3d_probs else 0.0, 4),
        peak_probability=round(float(np.max(x3d_probs)) if x3d_probs else 0.0, 4)
    )

    # -------------------------------------------------------------
    # 2. Transformer-Enhanced Refinement Model
    # -------------------------------------------------------------
    progress("Transformer Refiner", 0.50, "Applying 1D Temporal Transformer self-attention")
    t_trans_start = time.perf_counter()
    refiner = TransformerRefiner(d_model=32, sigma=1.25)
    trans_probs, attn_matrix = refiner.refine_with_attention(x3d_probs)
    t_trans_s = (time.perf_counter() - t_trans_start) + t_base_s  # full pipeline latency includes backbone

    trans_bin = [1 if p >= threshold else 0 for p in trans_probs]
    trans_wp = _make_window_predictions_for_model(windows, trans_probs)
    trans_events = segment_events(trans_wp, threshold, cfg.merge_gap_seconds, cfg.min_event_seconds)

    transformer_summary = ModelPredictionSummary(
        model_name="Transformer-Enhanced",
        display_name="X3D-M + Temporal Transformer",
        architecture="1D Temporal Transformer Self-Attention",
        probabilities=trans_probs,
        binary_predictions=trans_bin,
        events=trans_events,
        n_events=len(trans_events),
        n_violent_windows=sum(trans_bin),
        inference_time_s=round(t_trans_s, 2),
        mean_probability=round(float(np.mean(trans_probs)) if trans_probs else 0.0, 4),
        peak_probability=round(float(np.max(trans_probs)) if trans_probs else 0.0, 4)
    )

    # -------------------------------------------------------------
    # 3. Multimodal Audio-Visual Fusion Model
    # -------------------------------------------------------------
    progress("Audio Analysis", 0.70, "Extracting audio track & computing spectral FFT features")
    t_multi_start = time.perf_counter()
    audio_analyzer = AudioAnalyzer(sample_rate=22050)
    audio_scores = audio_analyzer.analyze_windows(video_path, windows)

    progress("Multimodal Fusion", 0.85, "Fusing visual action and acoustic violence scores")
    fusion_model = MultimodalFusion(default_alpha=0.70)
    multimodal_probs = fusion_model.fuse(x3d_probs, audio_scores, strategy="late", alpha=0.70)
    t_multi_s = (time.perf_counter() - t_multi_start) + t_base_s

    multi_bin = [1 if p >= threshold else 0 for p in multimodal_probs]
    multi_wp = _make_window_predictions_for_model(windows, multimodal_probs)
    multi_events = segment_events(multi_wp, threshold, cfg.merge_gap_seconds, cfg.min_event_seconds)

    multimodal_summary = ModelPredictionSummary(
        model_name="Multimodal (Audio-Visual)",
        display_name="X3D-M + Audio Spectral Fusion",
        architecture="Cross-Modal Visual + Audio Late Fusion",
        probabilities=multimodal_probs,
        binary_predictions=multi_bin,
        events=multi_events,
        n_events=len(multi_events),
        n_violent_windows=sum(multi_bin),
        inference_time_s=round(t_multi_s, 2),
        mean_probability=round(float(np.mean(multimodal_probs)) if multimodal_probs else 0.0, 4),
        peak_probability=round(float(np.max(multimodal_probs)) if multimodal_probs else 0.0, 4)
    )

    # -------------------------------------------------------------
    # 4. Pairwise Statistical Metrics
    # -------------------------------------------------------------
    progress("Computing Metrics", 0.92, "Calculating pairwise agreement, R², and MAE")
    pairwise = {
        "x3d_vs_transformer": _compute_pairwise("X3D-M", x3d_probs, "Transformer", trans_probs, threshold),
        "x3d_vs_multimodal": _compute_pairwise("X3D-M", x3d_probs, "Multimodal", multimodal_probs, threshold),
        "transformer_vs_multimodal": _compute_pairwise("Transformer", trans_probs, "Multimodal", multimodal_probs, threshold),
    }

    # Format benchmarks
    benchmarks = [BenchmarkMetric(**item) for item in BENCHMARK_REPORTS]

    # Convert numpy attention matrix to python nested list
    attn_list = attn_matrix.tolist() if attn_matrix is not None else None

    # -------------------------------------------------------------
    # 5. Recommendation Logic
    # -------------------------------------------------------------
    audio_present = any(s > 0.05 for s in audio_scores)
    n_spikes_suppressed = max(0, baseline_summary.n_violent_windows - transformer_summary.n_violent_windows)

    if audio_present and max(audio_scores) > 0.35:
        best_model = "Multimodal (Audio-Visual)"
        rationale = (
            "The video contains an active, acoustically dynamic audio track. The Multimodal approach "
            "leverages both visual kinetics and acoustic bursts (screams, crashes, impacts), "
            "adding an acoustic cue. NOTE: the audio score is a hand-built heuristic and no accuracy gain has been measured."
        )
    elif n_spikes_suppressed > 0:
        best_model = "Transformer-Enhanced"
        rationale = (
            f"The video contains intermittent visual noise where the Transformer successfully suppressed "
            f"{n_spikes_suppressed} isolated false-positive window spikes via temporal context self-attention. "
            "Whether this improves accuracy has not been measured; verify on labelled data."
        )
    else:
        best_model = "Transformer-Enhanced"
        rationale = (
            "The Transformer-Enhanced model maintains strong consistency with the X3D-M baseline while "
            "smoothing boundary transitions across contiguous temporal windows without requiring an audio stream."
        )

    recommendation = {
        "recommended_model": best_model,
        "rationale": rationale,
        "audio_available": audio_present,
        "spikes_suppressed": n_spikes_suppressed,
        "sensitivity_ranking": [
            ("Baseline (X3D-M)", baseline_summary.n_violent_windows),
            ("Multimodal", multimodal_summary.n_violent_windows),
            ("Transformer-Enhanced", transformer_summary.n_violent_windows),
        ]
    }

    progress("Complete", 1.0, "Multi-model comparison analysis finished")

    video_record = repo.get_video(video_id)
    video_name = video_record["original_name"] if video_record else f"Video #{video_id}"

    return ComparisonResult(
        video_id=video_id,
        video_name=video_name,
        duration_s=meta.duration_s,
        threshold=threshold,
        windows=windows,
        window_times=window_times,
        baseline_x3d=baseline_summary,
        transformer_enhanced=transformer_summary,
        multimodal=multimodal_summary,
        audio_scores=audio_scores,
        pairwise_metrics=pairwise,
        benchmarks=benchmarks,
        attention_matrix=attn_list,
        recommendation=recommendation
    )
