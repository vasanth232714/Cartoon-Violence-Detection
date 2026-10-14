"""Separately resumable audio and speech stages for the progressive Streamlit report."""
from __future__ import annotations

import json
import logging
from typing import Callable

from config.settings import AnalysisConfig
from database.repository import Repository
from pipeline.content_suitability import normalize_text, scan_transcript_segments

logger = logging.getLogger(__name__)
ProgressCB = Callable[[float, str], None]


def _model_info(repo: Repository, run_id: int) -> dict:
    run = repo.get_run(run_id)
    try:
        return json.loads(run["model_info_json"] or "{}") if run else {}
    except (TypeError, json.JSONDecodeError):
        return {}


def persist_stage_status(repo: Repository, run_id: int, stage: str, status: str,
                         result: dict | None = None, error: str | None = None) -> None:
    if stage not in ("visual", "audio", "speech"):
        raise ValueError(f"Unknown analysis stage: {stage}")
    if status not in ("pending", "running", "completed", "failed"):
        raise ValueError(f"Invalid stage status: {status}")
    info = _model_info(repo, run_id)
    stages = info.setdefault("analysis_stages", {})
    stage_info = dict(stages.get(stage) or {})
    stage_info.update({"status": status})
    if result is not None:
        stage_info["result"] = result
    if error is not None:
        stage_info["error"] = str(error)
    elif status != "failed":
        stage_info.pop("error", None)
    stages[stage] = stage_info
    if stage == "audio":
        section = dict(info.get("audio_event_analysis") or {})
        section.update({"status": status, **(result or {})})
        if error:
            section["error"] = str(error)
        info["audio_event_analysis"] = section
    elif stage == "speech":
        section = dict(info.get("speech_analysis") or {})
        section.update({"status": status, **(result or {})})
        if error:
            section["error"] = str(error)
        info["speech_analysis"] = section
    repo.update_run_model_info(run_id, info)
    if error:
        repo.append_run_warning(run_id, f"{stage.title()} analysis failed: {error}")


def run_audio_stage(repo: Repository, run_id: int, video_path: str, cfg: AnalysisConfig,
                    progress: ProgressCB | None = None) -> tuple[list[dict], dict, list[str]]:
    """Run only the existing AudioSet sound-event detector and persist its candidates."""
    progress = progress or (lambda fraction, message: None)
    if not cfg.enable_audio_event_detection:
        metadata = {"status": "disabled", "enabled": False, "candidate_count": 0}
        persist_stage_status(repo, run_id, "audio", "completed", metadata)
        return [], metadata, []
    from models.audio_event_detector import detect_audio_events

    findings, metadata, warnings = detect_audio_events(
        video_path, threshold=cfg.audio_event_threshold, progress_callback=progress,
    )
    if metadata.get("status") == "failed":
        raise RuntimeError(warnings[0] if warnings else "AudioSet sound-event detector failed")
    rows = [{
        "kind": "audio_event", "start_s": finding["start_s"], "end_s": finding["end_s"],
        "category": finding["category"], "confidence": finding["confidence"],
        "source": finding.get("source", metadata.get("model", "AudioSet")),
        "details": {"raw_label": finding.get("raw_label"), "evidence": finding},
    } for finding in findings]
    repo.add_audio_segments(run_id, rows)
    metadata = {**metadata, "enabled": True, "candidate_count": len(findings),
                "threshold": cfg.audio_event_threshold}
    persist_stage_status(repo, run_id, "audio", "completed", metadata)
    for warning in warnings:
        repo.append_run_warning(run_id, warning)
    return findings, metadata, warnings


def run_speech_stage(repo: Repository, run_id: int, video_path: str, duration_s: float,
                     cfg: AnalysisConfig, progress: ProgressCB | None = None
                     ) -> tuple[list[dict], list[dict], dict, list[str]]:
    """Transcribe speech and persist language/transcript plus abuse-candidate spans."""
    progress = progress or (lambda fraction, message: None)
    if not cfg.enable_verbal_abuse_detection:
        metadata = {"status": "disabled", "enabled": False, "transcript_segments": 0,
                    "abuse_candidates": 0}
        persist_stage_status(repo, run_id, "speech", "completed", metadata)
        return [], [], metadata, []
    from models.verbal_abuse_detector import VerbalAbuseDetector

    detector = VerbalAbuseDetector(
        use_multilingual_classifier=cfg.enable_multilingual_toxicity,
        toxicity_threshold=cfg.verbal_toxicity_threshold,
        whisper_model_size="base",
    )
    warnings: list[str] = []
    try:
        spans, transcripts, metadata, warnings = detector.analyze_video_full(
            video_path, duration_s, progress_callback=progress,
        )
        transcript_dicts = [row.__dict__ for row in transcripts]
        rows = [{
            "kind": "transcript", "start_s": row.start_s, "end_s": row.end_s,
            "language": row.language, "text": row.text, "source": row.source,
            "confidence": row.confidence,
            "details": {"words": row.words, "toxicity_score": row.toxicity_score,
                        "toxicity_model": row.toxicity_model,
                        "toxicity_status": row.toxicity_status},
        } for row in transcripts]
        rows.extend({
            "kind": "verbal_abuse", "start_s": span.start_s, "end_s": span.end_s,
            "language": span.language, "text": span.transcript,
            "category": ", ".join(span.categories), "confidence": span.confidence,
            "source": span.detection_source,
            "details": {**span.details, "censored_text": span.censored_transcript,
                        "flagged_phrases": span.flagged_words, "severity": span.severity_level},
        } for span in spans)
        rule_findings = scan_transcript_segments(transcript_dicts)
        rule_findings = [finding for finding in rule_findings if not any(
            float(span.start_s) <= finding["end_s"] and float(span.end_s) >= finding["start_s"]
            and normalize_text(finding["phrase"]) in normalize_text(" ".join(span.flagged_words) + " " + span.transcript)
            for span in spans
        )]
        rows.extend({
            "kind": "abuse_keyword", "start_s": finding["start_s"], "end_s": finding["end_s"],
            "language": finding["language"], "text": finding["phrase"],
            "category": finding["category"], "confidence": None,
            "source": finding["source"], "details": {"rule_version": "content_suitability_rules_v1"},
        } for finding in rule_findings)
        repo.add_audio_segments(run_id, rows)
        metadata = {**metadata, "enabled": True, "transcript_segments": len(transcripts),
                    "abuse_candidates": len(spans),
                    "phrase_rule_candidates": len(rule_findings),
                    "toxicity_classifier_enabled": cfg.enable_multilingual_toxicity}
        persist_stage_status(repo, run_id, "speech", "completed", metadata)
        for warning in warnings:
            repo.append_run_warning(run_id, warning)
        return ([span.to_dict() for span in spans] + rule_findings, transcript_dicts, metadata, warnings)
    finally:
        try:
            detector.release_models()
        except Exception:
            logger.debug("Could not release speech models", exc_info=True)


def rescore_saved_transcripts(repo: Repository, run_id: int,
                             progress: ProgressCB | None = None) -> dict:
    """Backfill language-specific toxicity/abuse scores on a saved run's transcripts."""
    progress = progress or (lambda fraction, message: None)
    from models.verbal_abuse_detector import (
        TAMIL_ABUSE_MODEL_ID,
        TOXICITY_MODEL_LANGUAGES,
        TAMIL_ABUSE_MODEL_LANGUAGES,
        VerbalAbuseDetector,
    )

    run = repo.get_run(run_id)
    cfg_data = json.loads(run["config_json"] or "{}") if run else {}
    threshold = float(cfg_data.get("verbal_toxicity_threshold", 0.85))
    transcript_rows = [row for row in repo.list_audio_segments(run_id) if row["kind"] == "transcript"]
    eligible = [row for row in transcript_rows
                if row["language"] and row["language"].split("-")[0].lower()
                in (TOXICITY_MODEL_LANGUAGES | TAMIL_ABUSE_MODEL_LANGUAGES)
                and json.loads(row["details_json"] or "{}").get("toxicity_score") is None]
    detector = VerbalAbuseDetector(use_multilingual_classifier=True, toxicity_threshold=threshold)
    scored = 0
    try:
        for index, row in enumerate(eligible, start=1):
            progress((index - 1) / max(1, len(eligible)),
                     f"Scoring saved transcript {index}/{len(eligible)} ({row['language']})")
            details = json.loads(row["details_json"] or "{}")
            score = detector._score_multilingual_toxicity(row["text"] or "", row["language"])
            details.update({
                "toxicity_score": score,
                "toxicity_model": detector._last_toxicity_model_id,
                "toxicity_status": detector._last_toxicity_status,
            })
            repo.update_audio_segment_details(row["id"], details)
            scored += int(score is not None)
        progress(1.0, f"Scoring finished: {scored}/{len(eligible)} transcript segment(s) scored")
        scores_by_model: dict[str, int] = {}
        for row in repo.list_audio_segments(run_id):
            if row["kind"] != "transcript":
                continue
            details = json.loads(row["details_json"] or "{}")
            if details.get("toxicity_score") is not None:
                model_id = str(details.get("toxicity_model") or "unknown")
                scores_by_model[model_id] = scores_by_model.get(model_id, 0) + 1
        errors = dict(detector._toxicity_model_errors)
        classifier = {
            "status": "completed" if scores_by_model else "unavailable",
            "models": sorted(scores_by_model),
            "scores_by_model": scores_by_model,
            "scored_segments": sum(scores_by_model.values()),
            "supported_languages": sorted(TOXICITY_MODEL_LANGUAGES | {"ta"}),
            "language_note": (
                "Tamil scorer targets Tamil/code-mixed abusive language and is not validated for every dialect or ASR transcript."
                if TAMIL_ABUSE_MODEL_ID in scores_by_model else None
            ),
            "errors": errors,
        }
        info = _model_info(repo, run_id)
        speech = dict(info.get("speech_analysis") or {})
        speech["toxicity_classifier"] = classifier
        info["speech_analysis"] = speech
        repo.update_run_model_info(run_id, info)
        for model_id, error in errors.items():
            repo.append_run_warning(run_id, f"Toxicity/abuse scoring failed for {model_id}: {error}")
        return classifier
    finally:
        detector.release_models()
