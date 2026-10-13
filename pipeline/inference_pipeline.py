"""End-to-end analysis: windows -> real model inference -> events -> evidence -> severity -> storage."""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Callable

from config.settings import APP_VERSION, EVIDENCE_DIR, MODEL_REGISTRY, TEMPORAL_MODEL_PATH, AnalysisConfig
from database.repository import Repository
from models.cartoon_filter import detect_shot_cut
from pipeline.postprocess import detect_events, refine_probabilities
from models.severity_assessor import assess_severity
from models.violence_classifier import ClipClassifier
from pipeline.event_segmenter import CandidateEvent, WindowPrediction, segment_events
from pipeline.evidence_selector import measure_motion, save_evidence_frames
from pipeline.explanation_generator import DESCRIPTION_METHOD, build_explanation
from preprocessing.clip_sampler import make_windows, to_model_input
from preprocessing.frame_extractor import iter_window_frames
from preprocessing.video_loader import VideoMetadata
from models.temporal_transformer import load_window_temporal, predict_window_sequence

logger = logging.getLogger(__name__)

ProgressCB = Callable[[str, float, str], None]
EVENT_LABEL = "X3D visual violence candidate (unconfirmed; binary output)"


class AnalysisCancelled(Exception):
    pass


class AnalysisFailed(Exception):
    def __init__(self, run_id: int, message: str):
        super().__init__(message)
        self.run_id = run_id


def _qwen_explain_interval(video_path: str, start_s: float, end_s: float, event_context: dict,
                          verbal_abuse: list[dict], audio_findings: list[dict], runtime: dict) -> str:
    """Load Qwen once per run and explain only a short, selected interval."""
    if runtime.get("error"):
        raise RuntimeError(runtime["error"])
    if not runtime.get("attempted"):
        runtime["attempted"] = True
        try:
            from models.vlm_explainer import load_qwen_vl
            runtime["model"], runtime["processor"] = load_qwen_vl()
        except Exception as exc:
            runtime["error"] = f"{type(exc).__name__}: {exc}"
            raise

    from models.vlm_explainer import build_event_prompt, explain_clip_qwen, extract_subclip
    duration = max(0.0, float(end_s) - float(start_s))
    clip_start = max(0.0, float(start_s) - 0.5)
    clip_end = float(end_s) + 0.5
    if duration > 30.0 or clip_end - clip_start > 30.0:
        center = float(event_context.get("peak_time_s", event_context.get("peak_s", (start_s + end_s) / 2.0)))
        clip_start = max(0.0, center - 15.0)
        clip_end = clip_start + 30.0
    sub_path = extract_subclip(video_path, clip_start, clip_end, padding_s=0.0)
    try:
        prompt = build_event_prompt(event_context, verbal_abuse, audio_findings, clip_offset_s=clip_start)
        explanation = explain_clip_qwen(sub_path, fps=0.5, max_tokens=384, prompt=prompt,
                                        model=runtime["model"], processor=runtime["processor"])
        if not explanation:
            raise RuntimeError("Qwen returned an empty explanation")
        return explanation
    finally:
        if os.path.exists(sub_path):
            os.remove(sub_path)


def run_analysis(video_id: int, video_path: str, meta: VideoMetadata, cfg: AnalysisConfig,
                 classifier: ClipClassifier, repo: Repository, progress: ProgressCB | None = None,
                 should_cancel: Callable[[], bool] | None = None, evidence_root: Path | None = None) -> int:
    cfg.validate()
    spec = MODEL_REGISTRY[cfg.model_key]
    progress = progress or (lambda stage, frac, msg: None)
    should_cancel = should_cancel or (lambda: False)
    if hasattr(classifier, "configure"):
        classifier.configure(use_adapter=cfg.use_adapter, use_tta=cfg.use_tta)
    run_id = repo.create_run(video_id, cfg.model_key, classifier.info, cfg.to_json(), cfg.config_hash(), APP_VERSION)
    t0 = time.perf_counter()
    preds: list[WindowPrediction] = []
    window_features: list = []
    warnings: list[str] = []
    timing: dict = {}
    windows: list[tuple[float, float]] = []
    try:
        progress("Preparing windows", 0.02, "Building overlapping windows over the full video")
        windows = make_windows(meta, cfg.window_seconds, cfg.stride_seconds)
        if not windows:
            raise ValueError("No analysis windows could be built for this video.")

        t_inf = 0.0
        run_model_info = dict(classifier.info)
        temporal_model = None
        temporal_metadata: dict = {}
        if cfg.use_temporal_transformer and TEMPORAL_MODEL_PATH.is_file():
            try:
                temporal_model, temporal_metadata = load_window_temporal(TEMPORAL_MODEL_PATH)
                run_model_info["temporal_transformer"] = {
                    "status": "trained checkpoint loaded",
                    "checkpoint": str(TEMPORAL_MODEL_PATH),
                    "training_metadata": temporal_metadata,
                }
            except Exception as exc:
                warnings.append(f"Trained temporal Transformer could not be loaded; deterministic temporal refinement is used ({type(exc).__name__}: {exc}).")
                run_model_info["temporal_transformer"] = {"status": "checkpoint load failed; deterministic fallback"}
        elif cfg.use_temporal_transformer:
            # A trained temporal checkpoint is an optional extension, not a requirement for
            # the pretrained X3D pipeline. Keep this as model metadata instead of a run warning.
            run_model_info["temporal_transformer"] = {
                "status": "optional checkpoint not installed; configured deterministic post-processing is active",
                "checkpoint": str(TEMPORAL_MODEL_PATH),
            }
        else:
            run_model_info["temporal_transformer"] = {"status": "disabled by configuration"}

        shot_cut_flags: list[bool] = []
        for wf in iter_window_frames(video_path, meta, windows, spec["num_frames"], spec["input_size"]):
            if should_cancel():
                raise AnalysisCancelled("Analysis cancelled by the user.")
            wp = WindowPrediction(wf.window_index, wf.start_s, wf.end_s, None, wf.status,
                                  wf.frame_indices, wf.frame_times, note=wf.note)
            has_cut = False
            if wf.frames is not None:
                has_cut, max_cut_diff, _ = detect_shot_cut(wf.frames)
                try:
                    clip = to_model_input(wf.frames, spec["mean"], spec["std"])
                    t1 = time.perf_counter()
                    if hasattr(classifier, "predict_with_features"):
                        wp.p_violent, feature = classifier.predict_with_features(clip)
                        window_features.append(feature)
                    else:
                        wp.p_violent = classifier.predict_proba(clip)
                        window_features.append(None)
                    wp.inference_ms = (time.perf_counter() - t1) * 1000
                    t_inf += wp.inference_ms / 1000
                except Exception as e:  # keep going; the run will be flagged incomplete
                    wp.status, wp.p_violent, wp.note = "error", None, f"{type(e).__name__}: {e}"
                    window_features.append(None)
            else:
                window_features.append(None)
            shot_cut_flags.append(has_cut)
            preds.append(wp)
            frac = 0.05 + 0.79 * (len(preds) / len(windows))
            progress("Model inference", frac, f"Window {len(preds)}/{len(windows)} "
                     f"({wp.start_s:.1f}s-{wp.end_s:.1f}s)")
        timing["inference_total_s"] = t_inf
        done = [p for p in preds if p.inference_ms is not None]
        timing["inference_mean_ms_per_window"] = (sum(p.inference_ms for p in done) / len(done)) if done else None

        n_err = sum(p.status == "error" for p in preds)
        n_skip = sum(p.status == "skipped" for p in preds)
        n_pad = sum(p.status == "padded" for p in preds)
        if n_err:
            warnings.append(f"{n_err} window(s) failed inference; results are INCOMPLETE.")
        if n_skip:
            warnings.append(f"{n_skip} window(s) had no decodable frames and were skipped.")
        if n_pad:
            warnings.append(f"{n_pad} window(s) had missing frames padded with the nearest decoded frame.")

        if temporal_model is not None:
            if len(window_features) != len(preds) or any(f is None for f in window_features) or any(p.p_violent is None for p in preds):
                warnings.append("Trained temporal Transformer was not applied because one or more analysis windows lack valid X3D features.")
                run_model_info["temporal_transformer"]["status"] = "not applied; incomplete X3D window sequence"
            else:
                import numpy as np
                temporal_input = np.concatenate([
                    np.stack(window_features).astype(np.float32),
                    np.asarray([p.p_violent for p in preds], dtype=np.float32)[:, None],
                ], axis=1)
                temporal_scores = predict_window_sequence(temporal_model, temporal_input)
                for wp, score in zip(preds, temporal_scores):
                    wp.note = (f"x3d={wp.p_violent:.4f} " + wp.note).strip()
                    wp.p_violent = float(score)
                run_model_info["temporal_transformer"]["status"] = "applied"

        repo.update_run_model_info(run_id, run_model_info)

        # Optional independent VideoMAE comparison. The checkpoint's own processor is used and its
        # probabilities are never fused with X3D or used to alter the baseline event list.
        if getattr(cfg, "enable_videomae_comparison", False):
            try:
                progress("VideoMAE comparison", 0.84, "Loading optional pretrained model on CPU")
                from models.videomae_detector import run_videomae_comparison
                videomae_rows, videomae_metadata = run_videomae_comparison(
                    video_path, meta, windows,
                    progress=lambda fraction, message: progress(
                        "VideoMAE comparison", 0.84 + 0.01 * min(max(float(fraction), 0.0), 1.0), message
                    ),
                )
                repo.add_model_window_scores(run_id, "videomae", videomae_rows)
                videomae_metadata["candidate_threshold"] = float(getattr(cfg, "videomae_threshold", 0.50))
                run_model_info["videomae_comparison"] = videomae_metadata
                if videomae_metadata.get("status") != "completed":
                    warnings.append(
                        f"VideoMAE comparison completed with errors in some windows "
                        f"({videomae_metadata.get('windows_completed', 0)}/{len(windows)} scored)."
                    )
            except Exception as exc:
                logger.warning("VideoMAE comparison unavailable: %s", exc)
                videomae_metadata = {
                    "status": "failed", "model_id": "HappyGook/videomae-violence-detector",
                    "model_card_url": "https://huggingface.co/HappyGook/videomae-violence-detector",
                    "error": f"{type(exc).__name__}: {exc}", "scores_are_calibrated": False,
                    "cartoon_accuracy_claimed": False,
                }
                run_model_info["videomae_comparison"] = videomae_metadata
                warnings.append(f"Optional VideoMAE comparison failed: {type(exc).__name__}: {exc}")
            repo.update_run_model_info(run_id, run_model_info)

        # Post-processing: same code path as training/fit.py, so tuned settings behave identically here.
        post = cfg.post_params()
        progress("Precision calibration", 0.85, "Temporal smoothing, shot-cut discount, hysteresis")
        raw_probs = [p.p_violent for p in preds]
        refined_probs = refine_probabilities(raw_probs, shot_cut_flags, post)
        for wp, raw_p, ref_p in zip(preds, raw_probs, refined_probs):
            if raw_p is not None and ref_p is not None and raw_p != ref_p:
                wp.note = (f"raw={raw_p:.2f} " + wp.note).strip()
                wp.p_violent = ref_p

        progress("Grouping events", 0.86, "Merging positive windows into candidate events")
        events = detect_events(preds, post)

        # Make the full visual scan durable before the slower optional speech/audio stages begin.
        repo.add_windows(run_id, preds)
        repo.update_run_progress(run_id, len(windows), len(preds) - n_err - n_skip)
        cfg_dict = {"threshold": cfg.threshold, "merge_gap_seconds": cfg.merge_gap_seconds,
                    "window_seconds": cfg.window_seconds, "stride_seconds": cfg.stride_seconds}
        out_root = (evidence_root or EVIDENCE_DIR) / f"run_{run_id}"

        # Verbal Abuse & Dialogue Profanity Analysis
        enable_verbal = getattr(cfg, "enable_verbal_abuse_detection", True)
        flag_standalone = getattr(cfg, "flag_standalone_verbal_abuse", True)
        event_abuse_map: dict[int, list] = {}
        standalone_spans: list = []
        transcript_rows: list[dict] = []
        abuse_spans: list = []
        speech_metadata: dict = {"status": "disabled"}

        if enable_verbal:
            try:
                progress("Multilingual speech analysis", 0.865, "Transcribing the full audio track and scoring speech candidates")
                logger.info("Starting full-track speech analysis for %.1f seconds of video", meta.duration_s)
                from models.verbal_abuse_detector import VerbalAbuseDetector
                v_detector = VerbalAbuseDetector(
                    use_multilingual_classifier=cfg.enable_multilingual_toxicity,
                    toxicity_threshold=cfg.verbal_toxicity_threshold,
                    whisper_model_size="base",
                )
                abuse_spans, transcripts, speech_metadata, speech_warnings = v_detector.analyze_video_full(
                    video_path, meta.duration_s,
                    progress_callback=lambda fraction, message: progress(
                        "Multilingual speech analysis",
                        0.865 + 0.04 * min(max(float(fraction), 0.0), 1.0),
                        message,
                    ),
                )
                warnings.extend(speech_warnings)
                transcript_rows.extend({
                    "kind": "transcript", "start_s": row.start_s, "end_s": row.end_s,
                    "language": row.language, "text": row.text, "source": row.source,
                    "confidence": row.confidence, "details": {"words": row.words},
                } for row in transcripts)
                event_abuse_map, standalone_spans = v_detector.correlate_with_physical_events(events, abuse_spans)
                transcript_rows.extend({
                    "kind": "verbal_abuse", "start_s": span.start_s, "end_s": span.end_s,
                    "language": span.language, "text": span.transcript,
                    "category": ", ".join(span.categories), "confidence": span.confidence,
                    "source": span.detection_source,
                    "details": {**span.details, "censored_text": span.censored_transcript,
                                "flagged_phrases": span.flagged_words, "severity": span.severity_level},
                } for span in abuse_spans)
                v_detector.release_models()
                if abuse_spans:
                    logger.info("Verbal abuse detector identified %d abuse span(s)", len(abuse_spans))
            except Exception as e:
                logger.warning("Verbal abuse analysis skipped due to error: %s", e)
                warnings.append(f"Speech and verbal-abuse analysis failed: {type(e).__name__}: {e}")
                speech_metadata = {"status": "failed", "error": f"{type(e).__name__}: {e}"}

        audio_findings: list[dict] = []
        audio_event_map: dict[int, list[dict]] = {ev.event_no: [] for ev in events}
        standalone_audio: list[dict] = []
        audio_metadata: dict = {"status": "disabled"}
        if cfg.enable_audio_event_detection:
            try:
                progress("Audio-event analysis", 0.91, "Scanning audio independently for sound-event candidates")
                from models.audio_event_detector import detect_audio_events
                audio_findings, audio_metadata, audio_warnings = detect_audio_events(
                    video_path,
                    threshold=cfg.audio_event_threshold,
                    progress_callback=lambda fraction, message: progress(
                        "Audio-event analysis",
                        0.91 + 0.06 * min(max(float(fraction), 0.0), 1.0),
                        message,
                    ),
                )
                warnings.extend(audio_warnings)
                for finding in audio_findings:
                    matches = [ev for ev in events if not (
                        finding["end_s"] < ev.start_s - 1.0 or finding["start_s"] > ev.end_s + 1.0)]
                    if matches:
                        for ev in matches:
                            audio_event_map[ev.event_no].append(finding)
                    else:
                        standalone_audio.append(finding)
                    transcript_rows.append({
                        "kind": "audio_event", "start_s": finding["start_s"], "end_s": finding["end_s"],
                        "category": finding["category"], "confidence": finding["confidence"],
                        "source": finding["source"], "details": {"raw_label": finding.get("raw_label")},
                    })
            except Exception as e:
                logger.warning("Audio-event analysis skipped due to error: %s", e)
                warnings.append(f"Audio-event analysis failed: {type(e).__name__}: {e}")
                audio_metadata = {"status": "failed", "error": f"{type(e).__name__}: {e}"}

        run_model_info["speech_analysis"] = {**speech_metadata, "enabled": enable_verbal,
                                             "toxicity_classifier_enabled": cfg.enable_multilingual_toxicity}
        run_model_info["audio_event_analysis"] = {**audio_metadata, "enabled": cfg.enable_audio_event_detection}
        repo.update_run_model_info(run_id, run_model_info)
        repo.add_audio_segments(run_id, transcript_rows)

        t_ev = time.perf_counter()
        qwen_runtime: dict = {}
        qwen_model_metadata = {"model": "Qwen/Qwen2.5-VL-3B-Instruct",
                               "status": "deferred; run from the report after candidate scan" if not getattr(cfg, "enable_qwen_explanations", False) else "active",
                               "generated_events": 0, "selected_events": 0}
        run_model_info["qwen_explanations"] = qwen_model_metadata
        repo.update_run_model_info(run_id, run_model_info)
        for i, ev in enumerate(events):
            if should_cancel():
                raise AnalysisCancelled("Analysis cancelled by the user.")
            progress("Saving event evidence", 0.975 + 0.02 * (i / max(len(events), 1)),
                     f"Event {ev.event_no}/{len(events)}: evidence frames and motion")
            frames = save_evidence_frames(video_path, meta, ev, cfg.evidence_frames_per_event, out_root)
            motion = measure_motion(video_path, meta, ev.start_s, ev.end_s)
            sev = assess_severity(ev.duration_s, ev.n_positive_runs, motion.mean)

            # Check for overlapping verbal abuse in this physical event
            v_spans = event_abuse_map.get(ev.event_no, [])
            sound_spans = audio_event_map.get(ev.event_no, [])
            sev_score = sev.score
            sev_points = dict(sev.points)
            sev_rationale = list(sev.rationale)
            sev_label = sev.label
            severity_method = sev.method

            if v_spans:
                has_high_threat = any(s.severity_level == "High" and "Verbal Aggression & Threats" in s.categories
                                      for s in v_spans)
                if has_high_threat and sev_label in ("Mild", "Moderate"):
                    sev_label = "High"
                    sev_score = 3
                    sev_points["verbal_bonus"] = 2
                    sev_rationale.append("Visual violence candidate overlaps an explicit verbal-threat candidate.")
                    severity_method = f"{sev.method}+explicit_verbal_threat"

            # Keep the serialized event and metrics-only explanation consistent.
            sev.score = sev_score
            sev.label = sev_label
            sev.points = sev_points
            sev.rationale = sev_rationale
            sev.method = severity_method

            text = build_explanation(ev, cfg_dict, run_model_info, sev, motion, len(frames), len(windows))
            if v_spans:
                all_flagged = list(dict.fromkeys([w for s in v_spans for w in s.flagged_words]))
                all_cats = sorted(set([c for s in v_spans for c in s.categories]))
                transcript_sample = " | ".join([s.transcript for s in v_spans if s.transcript])
                text += (
                    f"\n\n🗣️ **Verbal Abuse & Dialogue Detected:**\n"
                    f"- **Flagged Phrases:** {', '.join(all_flagged) or 'no specific phrase extracted'}\n"
                    f"- **Abuse Categories:** {', '.join(all_cats)}\n"
                    f"- **Transcript (original language):** *\"{transcript_sample}\"*"
                )
            if sound_spans:
                text += "\n\n🔊 **Audio-event candidates:**\n" + "\n".join(
                    f"- {s['category']} at {s['start_s']:.1f}–{s['end_s']:.1f}s (AudioSet score {s['confidence']:.2f}; sound class only)."
                    for s in sound_spans)

            sev_json = {
                "points": sev_points,
                "rationale": sev_rationale,
                "verbal_abuse": [s.to_dict() for s in v_spans],
                "audio_events": sound_spans,
                "visual_model": {"model_key": cfg.model_key, "score": ev.peak_prob},
                "severity_rubric": "review_priority_rule_v1 (unvalidated; not violence severity)",
            }

            qwen_exp = None
            qwen_status = "pending"
            if getattr(cfg, "enable_qwen_explanations", False) and i < getattr(cfg, "qwen_max_events", 20):
                try:
                    qwen_exp = _qwen_explain_interval(
                        video_path, ev.start_s, ev.end_s,
                        {"event_no": ev.event_no, "peak_prob": ev.peak_prob, "severity_label": sev_label},
                        v_spans, sound_spans, qwen_runtime
                    )
                    qwen_status = "generated"
                    qwen_model_metadata["generated_events"] = qwen_model_metadata.get("generated_events", 0) + 1
                except Exception as exc:
                    logger.warning("Qwen explanation failed for event %s: %s", ev.event_no, exc)
                    qwen_status = "failed"

            repo.add_event(run_id, {
                "event_no": ev.event_no, "start_s": ev.start_s, "end_s": ev.end_s, "duration_s": ev.duration_s,
                "label": EVENT_LABEL + (" + Verbal Abuse" if v_spans else "") + (" + Audio Event" if sound_spans else ""),
                "peak_prob": ev.peak_prob, "mean_prob": ev.mean_prob,
                "n_windows": ev.n_windows, "n_positive_runs": ev.n_positive_runs, "peak_time_s": ev.peak_time_s,
                "window_indices": ev.window_indices, "severity_score": sev_score, "severity_label": sev_label,
                "severity_method": severity_method,
                "severity_json": sev_json,
                "motion_json": {"mean": motion.mean, "p90": motion.p90, "n_pairs": motion.n_pairs,
                                "n_shot_cuts_excluded": motion.n_shot_cuts_excluded},
                "explanation": text, "description_method": DESCRIPTION_METHOD,
                "qwen_explanation": qwen_exp, "qwen_status": qwen_status,
            }, [{"path": f.path, "timestamp_s": f.timestamp_s, "kind": f.kind} for f in frames])

        # Standalone Verbal Abuse Incidents (Auditory harassment without physical motion)
        total_event_count = len(events)
        attached_audio_ids: set[int] = set()
        if enable_verbal and flag_standalone and standalone_spans:
            cur_ev_no = len(events)
            for s_idx, span in enumerate(standalone_spans):
                if should_cancel():
                    raise AnalysisCancelled("Analysis cancelled by the user.")
                cur_ev_no += 1
                total_event_count += 1
                ev_verbal = CandidateEvent(
                    event_no=cur_ev_no,
                    start_s=span.start_s,
                    end_s=span.end_s,
                    window_indices=[],
                    peak_prob=span.confidence,
                    mean_prob=span.confidence,
                    peak_window_index=0,
                    n_positive_runs=1,
                    peak_time_s=(span.start_s + span.end_s) / 2.0
                )
                frames = save_evidence_frames(video_path, meta, ev_verbal, 2, out_root)
                cat_summary = ", ".join(span.categories[:2])
                related_audio = [item for idx, item in enumerate(standalone_audio)
                                 if not (item["end_s"] < span.start_s - 1.0 or item["start_s"] > span.end_s + 1.0)]
                attached_audio_ids.update(id(item) for item in related_audio)
                exp_text = (
                    f"### 🗣️ Dialogue Incident Candidate\n\n"
                    f"**Recognized transcript (original language):** \"{span.transcript}\"\n\n"
                    f"**Flagged Language:** {', '.join(span.flagged_words)}\n\n"
                    f"**Abuse Categories:** {', '.join(span.categories)}\n\n"
                    f"**Audio Timestamp:** {span.start_s:.1f}s → {span.end_s:.1f}s ({span.end_s - span.start_s:.1f}s span)\n\n"
                    f"**Candidate score:** {span.confidence:.3f} via {span.detection_source}\n\n"
                    f"A speech signal was flagged for review. This is not a visual violence finding."
                )
                if related_audio:
                    exp_text += "\n\n**Overlapping sound candidates:** " + ", ".join(
                        f"{item['category']} ({item['confidence']:.2f})" for item in related_audio)
                repo.add_event(run_id, {
                    "event_no": cur_ev_no, "start_s": span.start_s, "end_s": span.end_s,
                    "duration_s": span.end_s - span.start_s,
                    "label": f"Verbal Speech Candidate ({cat_summary})" + (" + Audio Event" if related_audio else ""),
                    "peak_prob": span.confidence, "mean_prob": span.confidence,
                    "n_windows": 1, "n_positive_runs": 1,
                    "peak_time_s": (span.start_s + span.end_s) / 2.0,
                    "window_indices": [],
                    "severity_score": None,
                    "severity_label": "Not assessed",
                    "severity_method": "not_assessed_verbal_only",
                    "severity_json": {
                        "points": None,
                        "rationale": [f"Speech candidate category: {c}" for c in span.categories],
                        "verbal_abuse": [span.to_dict()], "audio_events": related_audio,
                    },
                    "motion_json": {"mean": 0.0, "p90": 0.0, "n_pairs": 0, "n_shot_cuts_excluded": 0},
                    "explanation": exp_text,
                    "description_method": "verbal_abuse_dialogue_engine",
                    "qwen_status": "pending",
                }, [{"path": f.path, "timestamp_s": f.timestamp_s, "kind": f.kind} for f in frames])

        # Keep audio-only sound candidates distinct and never treat loudness/sound labels as violence severity.
        cur_ev_no = max([len(events)] + [int(e["event_no"]) for e in repo.list_events(run_id)])
        for finding in standalone_audio:
            if id(finding) in attached_audio_ids:
                continue
            if should_cancel():
                raise AnalysisCancelled("Analysis cancelled by the user.")
            cur_ev_no += 1
            total_event_count += 1
            audio_event = CandidateEvent(
                event_no=cur_ev_no, start_s=finding["start_s"], end_s=finding["end_s"],
                window_indices=[], peak_prob=finding["confidence"], mean_prob=finding["confidence"],
                peak_window_index=0, n_positive_runs=1,
                peak_time_s=(finding["start_s"] + finding["end_s"]) / 2.0,
            )
            frames = save_evidence_frames(video_path, meta, audio_event, 2, out_root)
            audio_text = (
                f"Audio-event model candidate: {finding['category']} at {finding['start_s']:.2f}–"
                f"{finding['end_s']:.2f}s (score {finding['confidence']:.3f}). This is a sound class only; "
                "it does not establish a real-world event or visual action."
            )
            repo.add_event(run_id, {
                "event_no": cur_ev_no, "start_s": finding["start_s"], "end_s": finding["end_s"],
                "duration_s": finding["end_s"] - finding["start_s"],
                "label": f"Audio-only candidate: {finding['category']}",
                "peak_prob": finding["confidence"], "mean_prob": finding["confidence"],
                "n_windows": 1, "n_positive_runs": 1, "peak_time_s": audio_event.peak_time_s,
                "window_indices": [], "severity_score": None, "severity_label": "Not assessed",
                "severity_method": "not_applicable_audio_only",
                "severity_json": {"points": None, "rationale": ["Audio class is not a violence severity score."],
                                  "audio_events": [finding]},
                "motion_json": {"mean": None, "p90": None, "n_pairs": 0, "n_shot_cuts_excluded": 0},
                "explanation": audio_text, "description_method": "audio_event_classifier",
                "qwen_status": "pending",
            }, [{"path": f.path, "timestamp_s": f.timestamp_s, "kind": f.kind} for f in frames])
        run_model_info["qwen_explanations"] = qwen_model_metadata
        repo.update_run_model_info(run_id, run_model_info)

        timing["evidence_total_s"] = time.perf_counter() - t_ev

        elapsed = time.perf_counter() - t0
        status = "completed" if not warnings else ("incomplete" if (n_err or n_skip) else "completed_with_warnings")
        repo.finish_run(run_id, status, elapsed, timing, len(windows), len(preds) - n_err - n_skip, warnings)
        progress("Done", 1.0, f"Finished in {elapsed:.1f}s ({total_event_count} candidate event(s))")
        return run_id
    except AnalysisCancelled as e:
        _record_partial(repo, run_id, preds, windows)
        repo.finish_run(run_id, "interrupted", time.perf_counter() - t0, timing, len(windows), len(preds),
                        warnings, str(e))
        raise
    except Exception as e:
        _record_partial(repo, run_id, preds, windows)
        repo.finish_run(run_id, "failed", time.perf_counter() - t0, timing, len(windows), len(preds),
                        warnings, f"{type(e).__name__}: {e}")
        raise AnalysisFailed(run_id, f"{type(e).__name__}: {e}") from e


def _record_partial(repo: Repository, run_id: int, preds: list, windows: list) -> None:
    """Persist whatever window predictions exist so a failed run can be audited (never marked complete)."""
    try:
        if preds and not repo.list_windows(run_id):
            repo.add_windows(run_id, preds)
    except Exception:
        pass
