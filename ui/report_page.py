from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config.settings import AnalysisConfig
from database.repository import COMPLETE_STATUSES, Repository
from pipeline.report_pdf import build_pdf
from ui.common import STATUS_HELP, ft, jload, render_hero_metric, render_page_header, severity_badge, status_badge
from ui.event_investigation import render_event, review_status
from ui.timeline import build_timeline_figure
from models.vlm_explainer import (
    QWEN_MIN_AVAILABLE_RAM_BYTES,
    _available_physical_memory_bytes,
    parse_qwen_visual_review,
)
from pipeline.qwen_phase import audit_full_video_with_qwen, explain_run_candidates
from pipeline.progressive_stages import (
    persist_stage_status,
    rescore_saved_transcripts,
    run_audio_stage,
    run_speech_stage,
)
from pipeline.content_suitability import (
    analyze_text_suitability,
    build_suitability_assessment,
    classify_title,
)

logger = logging.getLogger(__name__)


def _render_qwen_phase(repo: Repository, run_id: int, video, events) -> None:
    """Offer Qwen as an explicit second step after the complete candidate list is saved."""
    st.markdown(
        "<div style='margin-top:1.5rem; margin-bottom:0.5rem; display:flex; align-items:center; gap:8px;'>"
        "<span style='font-size:1.15rem;'>🧠</span>"
        "<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin:0;'>Phase 2 · Candidate Explanations (Qwen2.5-VL)</h4>"
        "</div>",
        unsafe_allow_html=True
    )
    pending = [event for event in events
               if event["qwen_status"] != "generated" or not event["qwen_explanation"]
               or not parse_qwen_visual_review(event["qwen_explanation"])["structured"]]
    generated = len(events) - len(pending)
    if events:
        st.caption(
            f"Phase 1 saved {len(events)} candidate event(s). Phase 2 asks Qwen2.5-VL whether visible "
            f"violence is present, then explains the frames. This is a second opinion, not confirmation. "
            f"{generated} explained · {len(pending)} remaining."
        )
    else:
        st.info("Run candidate detection first. Qwen explanations run after candidate intervals are identified.")
        return

    notice_key = f"qwen_phase_notice_{run_id}"
    notice = st.session_state.pop(notice_key, None)
    if notice:
        if notice.get("error"):
            st.warning(
                f"Qwen generated {notice['generated']} explanation(s); "
                f"{notice['remaining']} candidate(s) remain. {notice['error']}"
            )
        elif notice.get("remaining"):
            st.warning(
                f"Qwen generated {notice['generated']} explanation(s); "
                f"{notice['remaining']} candidate(s) still need an explanation."
            )
        else:
            st.success(
                f"All remaining candidates now have Qwen explanations "
                f"({notice['generated']} generated in this step)."
            )

    if not pending:
        assessments = [parse_qwen_visual_review(event["qwen_explanation"])["visual_assessment"]
                       for event in events]
        no_violence = assessments.count("no_violence_observed")
        violence = assessments.count("violence_observed")
        uncertain = assessments.count("uncertain")
        st.success(
            f"Qwen reviewed all saved candidates: {violence} possible violence, "
            f"{no_violence} with no visible violence, {uncertain} uncertain."
        )
        return

    available = _available_physical_memory_bytes()
    memory_blocked = available is not None and available < QWEN_MIN_AVAILABLE_RAM_BYTES
    if memory_blocked:
        free_gib = available / 1024**3
        required_gib = QWEN_MIN_AVAILABLE_RAM_BYTES / 1024**3
        st.warning(
            f"Qwen requires ~{required_gib:.0f} GiB available RAM with its pretrained weights. "
            f"This environment currently reports {free_gib:.1f} GiB available. "
            "Candidate detection is complete; free memory to run explanations."
        )
    elif available is None:
        st.caption("Qwen uses pretrained bfloat16 weights; allow about 8 GiB of free RAM for CPU evaluation.")

    source_available = Path(video["stored_path"]).is_file()
    if not source_available:
        st.warning("The original video file is unavailable. Restore or re-upload it before running Qwen.")

    if st.button(
        f"🔎 Review and explain {len(pending)} remaining candidate(s) with Qwen2.5-VL",
        type="primary",
        use_container_width=True,
        disabled=memory_blocked or not source_available,
        key=f"qwen_phase_run_{run_id}",
    ):
        bar = st.progress(0.0)
        status = st.empty()

        def _progress(fraction: float, message: str) -> None:
            bar.progress(max(0.0, min(1.0, fraction)))
            status.caption(message)

        try:
            result = explain_run_candidates(repo, run_id, video["stored_path"], _progress)
            st.session_state[notice_key] = result
            st.rerun()
        except Exception as exc:
            st.session_state[notice_key] = {
                "generated": 0, "remaining": len(pending),
                "error": f"{type(exc).__name__}: {exc}",
            }
            st.rerun()


def _render_stage_findings(repo: Repository, run_id: int, video, cfg: dict, model_info: dict) -> None:
    """Run audio then speech once each, preserving completed output across Streamlit reruns."""
    persisted = model_info.get("analysis_stages") or {}
    if not persisted:
        return
    key = f"progressive_analysis_{run_id}"
    state = st.session_state.get(key)
    if not isinstance(state, dict):
        state = {"run_id": run_id, "video_id": int(video["id"]), "results": {}}
        for stage in ("visual", "audio", "speech"):
            value = dict(persisted.get(stage) or {})
            status = value.get("status", "pending")
            if status == "running":
                status = "pending"
            state[stage] = {**value, "status": status}
        st.session_state[key] = state
    if state.get("video_id") != int(video["id"]):
        state = {"run_id": run_id, "video_id": int(video["id"]), "results": {},
                 "visual": {"status": "completed"}, "audio": {"status": "pending"},
                 "speech": {"status": "pending"}}
        st.session_state[key] = state
    state.setdefault("results", {})
    for stage in ("audio", "speech"):
        if state[stage].get("status") == "running":
            state[stage]["status"] = "pending"
            persist_stage_status(repo, run_id, stage, "pending")

    if state["visual"].get("status") != "completed":
        st.error(f"Visual stage status: {state['visual'].get('status', 'unknown')}")
        return

    source_path = Path(video["stored_path"])
    source_available = source_path.is_file()
    duration_s = float(video["duration_s"])

    config = AnalysisConfig(**{k: v for k, v in cfg.items() if k in AnalysisConfig.__dataclass_fields__})

    def run_stage(stage: str, title: str, action):
        current = state[stage]
        if current.get("status") != "pending":
            return
        current.update(status="running", error=None)
        state[stage] = current
        st.session_state[key] = state
        persist_stage_status(repo, run_id, stage, "running")
        progress_bar = st.progress(0.0)
        message = st.empty()

        def on_progress(fraction: float, text: str):
            progress_bar.progress(max(0.0, min(float(fraction), 1.0)))
            message.caption(text)

        try:
            result = action(on_progress)
            current.update(status="completed", error=None)
            state["results"][stage] = result
        except Exception as exc:
            current.update(status="failed", error=str(exc))
        finally:
            state[stage] = current
            st.session_state[key] = state
            persist_stage_status(repo, run_id, stage, current["status"], current.get("error"))
            progress_bar.empty()
            message.empty()
            st.rerun()

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;'>"
        "<h3 style='color:#F8FAFC; font-size:1.2rem; font-weight:700; margin:0;'>Multi-Modal Audio & Speech Findings</h3>"
        "<span style='font-size:0.75rem; color:#94A3B8;'>Independent Sound & Language Channels</span>"
        "</div>",
        unsafe_allow_html=True
    )

    audio_meta = (model_info.get("audio_event_analysis") or {})
    st.caption(f"Sound-event detection · Status: {state['audio'].get('status', 'pending').title()}")
    if state["audio"].get("status") == "pending":
        run_stage("audio", "Analyzing audio and detecting sound events…",
                  lambda callback: run_audio_stage(repo, run_id, str(source_path), config, callback))
    audio_state = state["audio"]
    if audio_state.get("status") == "failed":
        st.error(f"Sound-event analysis error: {audio_state.get('error', audio_meta.get('error', 'unknown error'))}")
    elif audio_state.get("status") == "completed":
        segments = [row for row in repo.list_audio_segments(run_id) if row["kind"] == "audio_event"]
        st.markdown("#### Sound-Event Candidates (AudioSet)")
        if audio_meta.get("status") == "disabled" or not audio_meta.get("enabled", True):
            st.info("Sound-event detection was disabled for this run.")
        elif segments:
            st.dataframe(pd.DataFrame([{
                "Sound Label": row["category"] or "Sound candidate",
                "Score": f"{float(row['confidence']):.3f}" if row["confidence"] is not None else "—",
                "Time Interval": f"{ft(row['start_s'])} → {ft(row['end_s'])}",
                "Source Model": row["source"] or "AudioSet AST",
            } for row in segments]), use_container_width=True, hide_index=True)
            for row in segments[:12]:
                label = row["category"] or "Sound candidate"
                with st.expander(f"🔊 Listen to {label} ({ft(row['start_s'])}–{ft(row['end_s'])})", expanded=False):
                    if source_available:
                        st.video(str(source_path), start_time=max(int(float(row["start_s"]) - 1), 0))
                    st.caption("Source clip provided for auditory inspection.")
        else:
            st.info("No configured sound-event candidates were returned above threshold.")
        audio_result = state.get("results", {}).get("audio")
        if isinstance(audio_result, tuple) and len(audio_result) >= 3:
            for warning in audio_result[2]:
                st.warning(str(warning))

    speech_meta = model_info.get("speech_analysis") or {}
    st.caption(f"Speech transcription & verbal abuse · Status: {state['speech'].get('status', 'pending').title()}")
    if state["speech"].get("status") == "pending":
        run_stage("speech", "Transcribing speech and checking for verbal-abuse candidates…",
                  lambda callback: run_speech_stage(repo, run_id, str(source_path), duration_s, config, callback))
    speech_state = state["speech"]
    if speech_state.get("status") == "failed":
        st.error(f"Speech analysis error: {speech_state.get('error', speech_meta.get('error', 'unknown error'))}")
    elif speech_state.get("status") == "completed":
        segments = [row for row in repo.list_audio_segments(run_id) if row["kind"] == "transcript"]
        abuse = [row for row in repo.list_audio_segments(run_id)
                 if row["kind"] in ("verbal_abuse", "abuse_keyword")]
        st.markdown("#### Whisper ASR Transcripts & Language Risk")
        if speech_meta.get("status") == "disabled" or not speech_meta.get("enabled", True):
            st.info("Speech analysis was disabled for this run.")
        else:
            languages = sorted({row["language"] for row in segments if row["language"]})
            m1, m2, m3 = st.columns(3)
            m1.markdown(render_hero_metric("Transcript Segments", len(segments), "📝", "Dialogue turns", "#3B82F6"), unsafe_allow_html=True)
            m2.markdown(render_hero_metric("Languages Detected", len(languages), "🌐", ", ".join(languages) if languages else "None", "#6366F1"), unsafe_allow_html=True)
            m3.markdown(render_hero_metric("Verbal Abuse Alerts", len(abuse), "⚠️", "Flagged phrases", "#EF4444"), unsafe_allow_html=True)
            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

            if segments:
                scoreable_languages = {"en", "fr", "es", "it", "pt", "tr", "ru", "ta", "tam", "tamil"}
                missing_score_rows = [
                    row for row in segments
                    if (row["language"] or "").split("-")[0].lower() in scoreable_languages
                    and (jload(row["details_json"], {}) or {}).get("toxicity_score") is None
                ]
                st.dataframe(pd.DataFrame([{
                    "Time": f"{ft(row['start_s'])} → {ft(row['end_s'])}",
                    "Language": (row["language"] or "unknown").upper(),
                    "Transcript": row["text"] or "",
                    "Toxicity Score": (
                        f"{float((jload(row['details_json'], {}) or {}).get('toxicity_score')):.3f}"
                        if (jload(row["details_json"], {}) or {}).get("toxicity_score") is not None
                        else (
                            "Not scored"
                            if not (jload(row["details_json"], {}) or {}).get("toxicity_status")
                            else str((jload(row["details_json"], {}) or {}).get("toxicity_status")).replace("_", " ").title()
                        )),
                    "Scorer": (
                        "Tamil code-mixed abuse model"
                        if "tamil-codemixed-abusive" in str((jload(row["details_json"], {}) or {}).get("toxicity_model", ""))
                        else "Multilingual toxicity model"
                        if (jload(row["details_json"], {}) or {}).get("toxicity_model")
                        else "Not run"),
                } for row in segments]), use_container_width=True, hide_index=True)
                if missing_score_rows:
                    st.info(
                        f"{len(missing_score_rows)} transcript segment(s) lack toxicity score. "
                        "You can score them retroactively below."
                    )
                    if st.button("Score Saved Transcripts for Toxicity", key=f"rescore_toxicity_{run_id}"):
                        progress_bar = st.progress(0.0)
                        progress_text = st.empty()

                        def on_rescore_progress(fraction: float, message: str) -> None:
                            progress_bar.progress(max(0.0, min(1.0, float(fraction))))
                            progress_text.caption(message)

                        try:
                            with st.spinner("Scoring transcripts with text classifier…"):
                                result = rescore_saved_transcripts(repo, run_id, on_rescore_progress)
                            if result.get("status") == "completed":
                                st.success(f"Scored {result.get('scored_segments', 0)} transcript segment(s).")
                            else:
                                st.error("No scores were produced. " + str(result.get("errors") or ""))
                            st.rerun()
                        except Exception as exc:
                            st.error(f"Could not score transcripts: {exc}")
                        finally:
                            progress_bar.empty()
                            progress_text.empty()
            if abuse:
                st.markdown("**Flagged Lexical Abuse Candidates**")
                st.dataframe(pd.DataFrame([{
                    "Time": f"{ft(row['start_s'])} → {ft(row['end_s'])}",
                    "Language": (row["language"] or "unknown").upper(),
                    "Flagged Phrase": (jload(row["details_json"], {}) or {}).get("censored_text") or row["text"] or "",
                    "Categories": row["category"] or "—",
                    "Score": f"{float(row['confidence']):.3f}" if row["confidence"] is not None else "Lexicon Rule",
                } for row in abuse]), use_container_width=True, hide_index=True)
                for row in abuse[:12]:
                    with st.expander(f"🗣️ Review speech segment ({ft(row['start_s'])}–{ft(row['end_s'])})", expanded=False):
                        if source_available:
                            st.video(str(source_path), start_time=max(int(float(row["start_s"]) - 2), 0))
            speech_result = state.get("results", {}).get("speech")
            if isinstance(speech_result, tuple) and len(speech_result) >= 4:
                for warning in speech_result[3]:
                    st.warning(str(warning))

    stages_done = state["audio"].get("status") in ("completed", "failed") and state["speech"].get("status") in ("completed", "failed")
    if stages_done:
        statuses = [state[stage].get("status") for stage in ("audio", "speech")]
        if all(status == "completed" for status in statuses):
            st.success("All multi-modal stages completed. Combined timeline and report are ready below.")
        else:
            st.warning("Analysis completed with warnings in optional stages.")
        visual_items = []
        for event in repo.list_events(run_id):
            details = jload(event["severity_json"], {}) or {}
            if details.get("visual_model"):
                visual_items.append({"start_s": event["start_s"], "end_s": event["end_s"],
                                     "label": f"X3D Visual · Event #{event['event_no']}",
                                     "score": event["peak_prob"], "kind": "Visual"})
        audio_rows = [row for row in repo.list_audio_segments(run_id)
                      if row["kind"] in ("audio_event", "verbal_abuse", "abuse_keyword")]
        for row in audio_rows:
            visual_items.append({"start_s": row["start_s"], "end_s": row["end_s"],
                                 "label": row["category"] or ("Speech abuse" if row["kind"] in ("verbal_abuse", "abuse_keyword") else "Sound candidate"),
                                 "score": row["confidence"], "kind": "Speech" if row["kind"] in ("verbal_abuse", "abuse_keyword") else "Sound"})
        visual_items.sort(key=lambda item: (float(item["start_s"]), float(item["end_s"]), item["kind"]))
        st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin-top:1.25rem;'>Combined Multi-Modal Chronological Feed</h4>", unsafe_allow_html=True)
        if visual_items:
            fig = go.Figure()
            colors = {"Visual": "#3B82F6", "Sound": "#F59E0B", "Speech": "#A855F7"}
            for kind in ("Visual", "Sound", "Speech"):
                items = [item for item in visual_items if item["kind"] == kind]
                if items:
                    fig.add_trace(go.Bar(
                        x=[max(0.05, float(item["end_s"]) - float(item["start_s"])) for item in items],
                        base=[float(item["start_s"]) for item in items],
                        y=[item["label"] for item in items], orientation="h", name=kind,
                        marker_color=colors[kind],
                        customdata=[[item["score"]] for item in items],
                        hovertemplate="<b>%{y}</b><br>%{base:.2f}s → %{x:.2f}s<br>Score: %{customdata[0]}<extra></extra>",
                    ))
            fig.update_layout(height=max(220, min(560, 90 + 42 * len(visual_items))),
                              barmode="overlay", xaxis_title="Video Timestamp (seconds)", yaxis_title="Candidate",
                              xaxis_range=[0, duration_s], paper_bgcolor="rgba(0,0,0,0)",
                              plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#94A3B8", family="Plus Jakarta Sans, sans-serif"),
                              margin=dict(l=10, r=10, t=15, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(pd.DataFrame([{
                "Time Interval": f"{ft(item['start_s'])} → {ft(item['end_s'])}",
                "Modality": item["kind"],
                "Candidate Finding": item["label"],
                "Confidence Score": f"{float(item['score']):.3f}" if item["score"] is not None else "—",
            } for item in visual_items]), use_container_width=True, hide_index=True)
        else:
            st.info("No candidate intervals identified across visual, sound, or speech models.")

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<h3 style='color:#F8FAFC; font-size:1.2rem; font-weight:700;'>Child-Content Suitability Assessment</h3>", unsafe_allow_html=True)
        visual_events = [event for event in repo.list_events(run_id)
                         if (jload(event["severity_json"], {}) or {}).get("visual_model")]
        audio_count = sum(row["kind"] == "audio_event" for row in audio_rows)
        abuse_count = sum(row["kind"] in ("verbal_abuse", "abuse_keyword") for row in audio_rows)
        segments = [row for row in repo.list_audio_segments(run_id) if row["kind"] == "transcript"]
        transcript_count = len(segments)
        scored_windows = sum(row["p_violent"] is not None for row in repo.list_windows(run_id))

        summary_cols = st.columns(5)
        summary_cols[0].markdown(render_hero_metric("X3D Events", len(visual_events), "🎬", "Visual candidates", "#3B82F6"), unsafe_allow_html=True)
        summary_cols[1].markdown(render_hero_metric("Sound Events", audio_count, "🔊", "Audio classes", "#F59E0B"), unsafe_allow_html=True)
        summary_cols[2].markdown(render_hero_metric("Dialogue Turns", transcript_count, "📝", "ASR segments", "#6366F1"), unsafe_allow_html=True)
        summary_cols[3].markdown(render_hero_metric("Abuse Alerts", abuse_count, "⚠️", "Verbal flags", "#EF4444"), unsafe_allow_html=True)
        summary_cols[4].markdown(render_hero_metric("Scored Windows", scored_windows, "🔢", "Total 16f clips", "#10B981"), unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        fresh_run = repo.get_run(run_id)
        fresh_info = jload(fresh_run["model_info_json"], {}) or {}
        stages = fresh_info.get("analysis_stages") or {}
        title_context = classify_title(
            video["video_title"] or Path(video["original_name"]).stem,
            video["title_source"] or "filename",
        )
        speech_meta = fresh_info.get("speech_analysis") or {}
        toxicity_meta = speech_meta.get("toxicity_classifier") or {"status": "unavailable"}
        toxicity_scores = [float((jload(row["details_json"], {}) or {})["toxicity_score"])
                           for row in segments
                           if (jload(row["details_json"], {}) or {}).get("toxicity_score") is not None]
        visual_stage = stages.get("visual") or {"status": "completed" if scored_windows else "pending"}
        audio_stage = stages.get("audio") or {"status": "pending"}
        speech_stage = stages.get("speech") or {"status": "pending"}
        x3d_events = [event for event in repo.list_events(run_id)
                      if (jload(event["severity_json"], {}) or {}).get("visual_model")]
        assessment_cfg = {}
        try:
            from pipeline.content_suitability import _vocabulary
            assessment_cfg = _vocabulary().get("assessment", {})
        except Exception:
            pass
        assessment = build_suitability_assessment(
            title_context=title_context,
            visual_status=visual_stage.get("status", "pending"), scored_visual_windows=scored_windows,
            audio_status=audio_stage.get("status", "pending"),
            audio_metadata=fresh_info.get("audio_event_analysis") or {},
            speech_status=speech_stage.get("status", "pending"), speech_metadata=speech_meta,
            transcript_count=transcript_count, x3d_events=x3d_events,
            audio_findings=[row for row in audio_rows if row["kind"] == "audio_event"],
            verbal_findings=[row for row in audio_rows if row["kind"] in ("verbal_abuse", "abuse_keyword")],
            toxicity_scores=toxicity_scores,
            toxicity_threshold=float(cfg.get("verbal_toxicity_threshold", assessment_cfg.get("toxicity_threshold", 0.85))),
            toxicity_status=toxicity_meta.get("status", "unavailable"), run_status=fresh_run["status"],
        )
        abuse_rows = [row for row in audio_rows if row["kind"] in ("verbal_abuse", "abuse_keyword")]
        text_assessment = analyze_text_suitability(
            title_context=title_context,
            transcript_segments=[{
                "text": row["text"], "language": row["language"],
                "start_s": row["start_s"], "end_s": row["end_s"],
            } for row in segments],
            verbal_findings=[{
                "category": row["category"], "text": row["text"],
                "phrase": (jload(row["details_json"], {}) or {}).get("censored_text") or row["text"],
            } for row in abuse_rows],
            toxicity_scores=toxicity_scores,
            toxicity_threshold=float(cfg.get("verbal_toxicity_threshold", assessment_cfg.get("toxicity_threshold", 0.85))),
            transcript_status=speech_stage.get("status", "pending"),
            default_language=speech_meta.get("language"),
        )

        cat_color = "#10B981" if "suitable" in assessment['category'].lower() and "un" not in assessment['category'].lower() else "#EF4444"
        st.markdown(
            f"<div style='background:rgba(17,24,39,0.8); border:1px solid rgba(255,255,255,0.08); "
            f"border-left:4px solid {cat_color}; border-radius:8px; padding:1.1rem 1.3rem; margin:1rem 0;'>"
            f"<div style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em;'>Synthesized Suitability Classification</div>"
            f"<div style='font-size:1.35rem; font-weight:800; color:#F8FAFC; margin-top:2px;'>{assessment['category']}</div>"
            f"<div style='font-size:0.85rem; color:#CBD5E1; margin-top:4px;'>{title_context.get('context_result', '')}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

        st.dataframe(pd.DataFrame(assessment["sources"]), use_container_width=True, hide_index=True)
        for reason in assessment["reasons"]:
            st.markdown(f"- {reason}")

        st.markdown("#### Preliminary Text Suitability Score")
        text_score_cols = st.columns(3)
        text_score_cols[0].metric("Text Score", f"{text_assessment['score']:.2f} / 10")
        text_score_cols[1].metric("Classification", text_assessment["classification"])
        text_score_cols[2].metric(
            "English Sentiment Polarity",
            f"{text_assessment['sentiment']['polarity']:.3f}"
            if text_assessment["sentiment"]["polarity"] is not None else "Unavailable",
        )

        st.markdown("**Score Contributions (0–10 Heuristic)**")
        non_additive_signals = {
            "title_match_raw_signal", "transcript_kids_phrase_raw_signal",
            "abusive_phrase_penalty_signal_before_grouping",
            "multilingual_toxicity_penalty_signal_before_grouping",
        }
        st.dataframe(pd.DataFrame([
            {"Evidence": label.replace("_", " ").title(), "Contribution": value,
             "Treatment": "Raw signal (grouped)" if label in non_additive_signals else "Applied to score"}
            for label, value in text_assessment["contributions"].items()
            if isinstance(value, (int, float))
        ]), use_container_width=True, hide_index=True)

        st.markdown("#### Flagged Clips & Non-Event Verification Samples")
        for item in visual_items[:20]:
            with st.expander(f"{item['kind']} · {item['label']} ({ft(item['start_s'])}–{ft(item['end_s'])})", expanded=False):
                if Path(video["stored_path"]).is_file():
                    clip_start = max(0, int(float(item["start_s"]) - 2))
                    clip_end = max(clip_start + 1, int(float(item["end_s"]) + 2))
                    st.video(str(video["stored_path"]), start_time=clip_start, end_time=clip_end)
                st.caption("Timestamps are referenced to original video timeline.")

        threshold = float(cfg.get("threshold", AnalysisConfig().threshold))
        non_events = [row for row in repo.list_windows(run_id)
                      if row["p_violent"] is not None and float(row["p_violent"]) < threshold]
        if non_events:
            st.caption("Sampled windows below candidate threshold for false-negative sanity check.")
            sample_rows = sorted(non_events, key=lambda row: float(row["p_violent"]))[:5]
            for row in sample_rows:
                with st.expander(f"Non-event sample · {ft(row['start_s'])} · Score: {float(row['p_violent']):.3f}", expanded=False):
                    if Path(video["stored_path"]).is_file():
                        clip_start = max(0, int(row["start_s"]))
                        clip_end = max(clip_start + 1, int(row["end_s"]))
                        st.video(str(video["stored_path"]), start_time=clip_start, end_time=clip_end)


def _render_qwen_full_video_audit(repo: Repository, run_id: int, video, model_info: dict, events) -> None:
    """Render an optional Qwen second opinion over every video segment, including X3D misses."""
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin-top:1.5rem;'>Independent Full-Video AI Audit (Qwen2.5-VL)</h4>", unsafe_allow_html=True)
    audit = model_info.get("qwen_full_video_audit") or {}
    st.caption(
        "Qwen2.5-VL can review the source in 30-second temporal chunks to surface possible missed events. "
        "Results remain unconfirmed second opinions."
    )

    notice_key = f"qwen_audit_notice_{run_id}"
    notice = st.session_state.pop(notice_key, None)
    if notice:
        if notice.get("error"):
            st.warning(f"Qwen review status: {notice.get('status', 'failed')}. {notice['error']}")
        else:
            st.success(
                f"Qwen reviewed {notice['completed_chunks']}/{notice['total_chunks']} video segments "
                f"and reported {notice['event_count']} possible visual event(s)."
            )

    available = _available_physical_memory_bytes()
    memory_blocked = available is not None and available < QWEN_MIN_AVAILABLE_RAM_BYTES
    source_available = Path(video["stored_path"]).is_file()
    if memory_blocked:
        st.warning(
            f"Qwen needs about {QWEN_MIN_AVAILABLE_RAM_BYTES / 1024**3:.0f} GiB of free RAM for CPU review; "
            f"this machine currently has {available / 1024**3:.1f} GiB available."
        )
    if not source_available:
        st.warning("The original video is unavailable. Restore or re-upload it before running the review.")

    audit_status = audit.get("status")
    if audit_status == "completed":
        st.success("The full source video has been reviewed by Qwen.")
    else:
        button_label = "🔎 Resume full-video Qwen review" if audit_status in ("partial", "failed") else "🔎 Scan full video with Qwen for missed scenes"
        if st.button(button_label, use_container_width=True, disabled=memory_blocked or not source_available,
                     key=f"qwen_full_audit_{run_id}"):
            bar = st.progress(0.0)
            status = st.empty()

            def _progress(fraction: float, message: str) -> None:
                bar.progress(max(0.0, min(1.0, fraction)))
                status.caption(message)

            try:
                result = audit_full_video_with_qwen(
                    repo, run_id, video["stored_path"], float(video["duration_s"]), _progress,
                )
                st.session_state[notice_key] = result
                st.rerun()
            except Exception as exc:
                st.session_state[notice_key] = {
                    "status": "failed", "completed_chunks": audit.get("completed_chunks", 0),
                    "total_chunks": audit.get("total_chunks", 0), "event_count": audit.get("event_count", 0),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                st.rerun()

    if not audit:
        return

    audit_events = audit.get("events", []) or []
    audit_chunks = [chunk for chunk in (audit.get("chunks", []) or [])
                    if chunk.get("status") == "completed"]
    visual_x3d = [event for event in events
                  if "Violence (binary model output" in str(event["label"] or "")
                  or "X3D visual violence candidate" in str(event["label"] or "")]
    if audit_events:
        rows = []
        for index, q_event in enumerate(audit_events, 1):
            overlaps = [event for event in visual_x3d
                        if not (float(event["end_s"]) < float(q_event["start_s"]) - 2.0
                                or float(event["start_s"]) > float(q_event["end_s"]) + 2.0)]
            match = (", ".join(f"X3D event #{event['event_no']}" for event in overlaps)
                     if overlaps else "Qwen-only · possible X3D miss")
            rows.append({
                "Qwen candidate": f"Qwen {index}",
                "Time span": f"{ft(q_event['start_s'])} → {ft(q_event['end_s'])}",
                "Possible category": q_event.get("category", "unclear"),
                "Observed action (Qwen)": q_event.get("observed_action", ""),
                "X3D comparison": match,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def render(repo: Repository) -> None:
    render_page_header(
        title="Incident Analysis Report",
        subtitle="Forensic examination of detected visual, sound, and speech candidate intervals with multimodal attribution.",
        badge_text="Audit Mode Active",
        badge_color="#3B82F6"
    )

    runs = [r for r in repo.list_runs() if r["status"] != "running"]
    if not runs:
        st.info("No video analysis runs available yet. Analyze a video in the Analyze Video tab first.")
        return

    ids = [r["id"] for r in runs]
    default = st.session_state.get("selected_run_id", ids[0])
    if default not in ids:
        default = ids[0]

    c_sel, c_stat, c_del = st.columns([3.5, 1.3, 1.2])
    with c_sel:
        run_id = st.selectbox(
            "Select Analysis Session",
            ids,
            index=ids.index(default),
            format_func=lambda i: next(
                (f"Run #{r['id']} — {r['original_name']} ({r['started_at'][:19]})" for r in runs if r["id"] == i),
                f"Run #{i}"
            )
        )
        st.session_state["selected_run_id"] = run_id

    run = repo.get_run(run_id)
    video = repo.get_video(run["video_id"])
    title_context = classify_title(
        video["video_title"] or Path(video["original_name"]).stem,
        video["title_source"] or "filename",
    )
    cfg = jload(run["config_json"], {})
    info = jload(run["model_info_json"], {})
    timing = jload(run["timing_json"], {}) or {}
    warnings = jload(run["warnings_json"], []) or []
    display_warnings = [str(w) for w in warnings
                        if "No trained cartoon temporal Transformer checkpoint is installed" not in str(w)]
    display_warnings = [
        (warning + ". Enable the pretrained scorer in the active environment with: "
         "python -m pip install -r requirements.txt. Phrase checks still ran; multilingual toxicity did not.")
        if "Multilingual toxicity scoring was unavailable" in warning and "detoxify" in warning.lower()
        and "pip install" not in warning.lower() else warning
        for warning in display_warnings
    ]
    windows = repo.list_windows(run_id)
    events = repo.list_events(run_id)

    with c_stat:
        st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)
        shown_status = "completed" if run["status"] == "completed_with_warnings" and not display_warnings else run["status"]
        st.markdown(status_badge(shown_status), unsafe_allow_html=True)

    with c_del:
        st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Delete", key=f"del_rep_top_{run_id}", help="Permanently delete this report", use_container_width=True):
            repo.delete_run(run_id)
            st.session_state.pop("selected_run_id", None)
            st.success(f"Run #{run_id} deleted.")
            st.rerun()

    if run["status"] not in COMPLETE_STATUSES:
        st.error(f"Analysis status note: {STATUS_HELP.get(run['status'], '')} {run['error'] or ''}")
    elif run["status"] == "completed_with_warnings" and display_warnings:
        st.warning("Analysis completed with warnings: " + " ".join(display_warnings))

    # Precision & Filter Status Badge
    filter_enabled = cfg.get("enable_temporal_refinement", False) or cfg.get("suppress_shot_cuts", False)
    thr_val = cfg.get("threshold", AnalysisConfig().threshold)
    min_ev_val = cfg.get("min_event_seconds", 0.0)

    filter_color = "#10B981" if filter_enabled else "#F59E0B"
    filter_status = "Configured" if filter_enabled else "Disabled"
    st.markdown(
        f"<div style='display:inline-flex; align-items:center; gap:8px; background:rgba(17,24,39,0.7); "
        f"border:1px solid rgba(255,255,255,0.08); border-left:3px solid {filter_color}; "
        f"border-radius:6px; padding:6px 14px; margin-bottom:12px; font-size:0.82rem; color:#CBD5E1;'>"
        f"<span><b>Temporal Post-Processing: {filter_status}</b> (Threshold: {thr_val:.2f} · Min Event Duration: {min_ev_val:.1f}s)</span>"
        f"</div>",
        unsafe_allow_html=True
    )

    # Metric Cards
    m = st.columns(5)
    m[0].markdown(render_hero_metric("Video Duration", ft(video["duration_s"]), "⏱️", f"{video['duration_s']:.1f}s total", "#3B82F6"), unsafe_allow_html=True)
    m[1].markdown(render_hero_metric("Event Candidates", len(events), "⚠️", "Review intervals", "#EF4444"), unsafe_allow_html=True)
    coverage = (100.0 * run["windows_done"] / run["windows_total"]) if run["windows_total"] else 0.0
    m[2].markdown(render_hero_metric("Windows Analyzed", f"{run['windows_done']}/{run['windows_total']}", "🔢", f"{coverage:.0f}% coverage", "#6366F1"), unsafe_allow_html=True)
    proc_time = f"{(run['elapsed_s'] or 0):.1f}s"
    m[3].markdown(render_hero_metric("X3D Inference", proc_time, "⚡", "Visual latency", "#F59E0B"), unsafe_allow_html=True)
    ratio = (run["elapsed_s"] or 0) / video["duration_s"] if video["duration_s"] else 0
    m[4].markdown(render_hero_metric("Speed Ratio", f"{ratio:.2f}×", "🏎️", "Compute / real-time", "#10B981"), unsafe_allow_html=True)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Activity & Safety Distribution
    visual_events = [e for e in events
                     if ("Violence (binary model output" in str(e["label"] or "")
                         or "X3D visual violence candidate" in str(e["label"] or "")
                         or bool((jload(e["severity_json"], {}) or {}).get("visual_model")))]
    covered = _covered_seconds([(e["start_s"], e["end_s"]) for e in visual_events], video["duration_s"])
    pct_visual = 100 * covered / video["duration_s"] if video["duration_s"] else 0
    pct_unflagged = 100 - pct_visual

    col_act, col_dist = st.columns([3, 2])
    with col_act:
        st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:700;'>Visual Candidate Temporal Coverage</h4>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:rgba(17,24,39,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:10px; padding:16px;'>"
            f"<div style='display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.85rem;'>"
            f"<span style='font-weight:600; color:#10B981;'>Unflagged: {pct_unflagged:.1f}%</span>"
            f"<span style='font-weight:600; color:#EF4444;'>X3D Candidates: {pct_visual:.1f}%</span>"
            f"</div>"
            f"<div style='height:8px; border-radius:4px; background:#1E293B; overflow:hidden; display:flex;'>"
            f"<div style='width:{pct_unflagged}%; background:#10B981; height:100%;'></div>"
            f"<div style='width:{pct_visual}%; background:#EF4444; height:100%;'></div>"
            f"</div>"
            f"<div style='display:flex; justify-content:space-between; margin-top:8px; font-size:0.78rem; color:#94A3B8;'>"
            f"<span>Safe Baseline: {video['duration_s'] - covered:.1f}s</span>"
            f"<span>Candidate Windows: {covered:.1f}s</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )

    with col_dist:
        st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:700;'>Review Priority Breakdown</h4>", unsafe_allow_html=True)
        if events:
            sev_counts = pd.Series([e["severity_label"] for e in events]).value_counts()
            palette = {"High": "#EF4444", "Moderate": "#F97316", "Mild": "#F59E0B", "Not assessed": "#64748B"}
            colors = [palette.get(k, "#3B82F6") for k in sev_counts.index]
            fig_sev = go.Figure(data=[go.Pie(
                labels=sev_counts.index, values=sev_counts.values, hole=0.62,
                marker=dict(colors=colors, line=dict(color='#0A0E17', width=2)),
                textinfo='label+value'
            )])
            fig_sev.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94A3B8", family="Plus Jakarta Sans, sans-serif", size=11),
                margin=dict(l=10, r=10, t=10, b=10),
                height=130,
                showlegend=False
            )
            st.plotly_chart(fig_sev, use_container_width=True)
        else:
            st.caption("No event candidates to display.")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Interactive Timeline Section
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin-bottom:0.75rem;'>Interactive Timeline & Candidate Trajectory</h4>", unsafe_allow_html=True)
    sel_key = f"sel_event_{run_id}"
    threshold = float(cfg.get("threshold", AnalysisConfig().threshold))

    if not events:
        st.info(f"No candidate intervals crossed the detection threshold ({threshold:.2f}).")
        st.plotly_chart(build_timeline_figure(windows, [], threshold, video["duration_s"]),
                        use_container_width=True)
    else:
        c_f1, c_f2 = st.columns(2)
        all_sevs = sorted({e["severity_label"] for e in events})
        with c_f1:
            sev_sel = st.multiselect("Filter Priority Level", all_sevs, default=all_sevs)
        with c_f2:
            min_p = st.slider("Minimum Peak Confidence Score", 0.0, 1.0, 0.0, 0.05)

        shown = [e for e in events if e["severity_label"] in sev_sel and e["peak_prob"] >= min_p]
        sel_no = st.session_state.get(sel_key, shown[0]["event_no"] if shown else None)

        st.plotly_chart(build_timeline_figure(windows, shown, threshold, video["duration_s"], sel_no),
                        use_container_width=True)

        if shown:
            rows = []
            for e in shown:
                sj_raw = e["severity_json"] if "severity_json" in e.keys() else "{}"
                sj = jload(sj_raw, {}) or {}
                lbl = str(e["label"]) if "label" in e.keys() else ""
                has_verbal = bool(sj.get("verbal_abuse")) or "Verbal" in lbl
                has_audio = bool(sj.get("audio_events")) or "Audio" in lbl
                has_visual = ("Violence (binary model output" in lbl
                              or "X3D visual violence candidate" in lbl
                              or bool(sj.get("visual_model")))
                modalities = []
                if has_visual: modalities.append("Visual")
                if has_verbal: modalities.append("Speech")
                if has_audio: modalities.append("Audio")
                qwen_review = parse_qwen_visual_review(e["qwen_explanation"] or "")
                qwen_label = {
                    "violence_observed": "Possible Violence",
                    "no_violence_observed": "No Violence",
                    "uncertain": "Uncertain",
                }.get(qwen_review["visual_assessment"], "Not reviewed") if qwen_review["structured"] else "Not reviewed"
                rows.append({
                    "Event #": f"Event {e['event_no']}",
                    "Modalities": " + ".join(modalities) or "Unspecified",
                    "Time Span": f"{ft(e['start_s'])} → {ft(e['end_s'])}",
                    "Duration": f"{e['duration_s']:.1f}s",
                    "Peak Score": f"{int(round(e['peak_prob'] * 100))}%",
                    "Priority": e["severity_label"],
                    "Qwen Review": qwen_label,
                    "Audit Status": review_status(repo, e["id"]).upper()
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Event Deep Dive
        st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700;'>Forensic Event Deep Dive</h4>", unsafe_allow_html=True)
        if not shown:
            st.info("No events match current filter settings.")
        else:
            nums = [e["event_no"] for e in shown]
            if st.session_state.get(sel_key) not in nums:
                st.session_state[sel_key] = nums[0]

            def _step(d):
                i = nums.index(st.session_state[sel_key]) + d
                st.session_state[sel_key] = nums[min(max(i, 0), len(nums) - 1)]

            c_prev, c_pick, c_next = st.columns([1, 4, 1])
            with c_prev:
                st.button("◀ Previous", on_click=_step, args=(-1,), key=f"prev_{run_id}", use_container_width=True)
            with c_next:
                st.button("Next ▶", on_click=_step, args=(1,), key=f"next_{run_id}", use_container_width=True)
            with c_pick:
                def _fmt_event(n):
                    ev_match = next((x for x in shown if x["event_no"] == n), None)
                    if not ev_match:
                        return f"Event {n}"
                    sj_raw = ev_match["severity_json"] if "severity_json" in ev_match.keys() else "{}"
                    sj_match = jload(sj_raw, {}) or {}
                    lbl_match = str(ev_match["label"]) if "label" in ev_match.keys() else ""
                    v_icon = " 🗣️" if (bool(sj_match.get("verbal_abuse")) or "Verbal Abuse" in lbl_match) else ""
                    return f"Event {n}{v_icon}: {ft(ev_match['start_s'])} → {ft(ev_match['end_s'])} (Priority {ev_match['severity_label']} · Score {int(round(ev_match['peak_prob']*100))}%)"

                st.selectbox(
                    "Navigate Event",
                    nums,
                    key=sel_key,
                    label_visibility="collapsed",
                    format_func=_fmt_event
                )

            cur = next(e for e in shown if e["event_no"] == st.session_state[sel_key])
            render_event(repo, cur, video, run)

    _render_stage_findings(repo, run_id, video, cfg, info or {})
    _render_qwen_phase(repo, run_id, video, repo.list_events(run_id))
    _render_qwen_full_video_audit(repo, run_id, video, info or {}, events)

    # PDF Export Section
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700;'>Export Forensic Report</h4>", unsafe_allow_html=True)
    c_pdf1, c_pdf2 = st.columns([2, 3])
    with c_pdf1:
        if st.button("📄 Generate Formal PDF Report", use_container_width=True):
            try:
                with st.spinner("Compiling formal PDF documentation..."):
                    st.session_state[f"pdf_{run_id}"] = build_pdf(repo, run_id)
                st.success("PDF generated successfully.")
            except Exception as e:
                st.error(f"PDF generation error: {e}")

    with c_pdf2:
        if f"pdf_{run_id}" in st.session_state:
            st.download_button(
                "📥 Download Signed PDF",
                st.session_state[f"pdf_{run_id}"],
                file_name=f"violence_detection_report_run{run_id}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

    with st.expander("Technical Session Metadata & Prediction Logs", expanded=False):
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            st.markdown("**Session Parameters**")
            st.json({**cfg, "app_version": run["app_version"], "config_hash": run["config_hash"]})
        with c_t2:
            st.markdown("**Compute Latency Profile**")
            st.json({**info, "timing": timing})

        st.markdown("**Window-Level Raw Scores**")
        st.dataframe(pd.DataFrame([{
            "Index": w["window_index"],
            "Time Span": f"{ft(w['start_s'])} - {ft(w['end_s'])}",
            "Probability": f"{w['p_violent']:.4f}" if w["p_violent"] is not None else "N/A",
            "Status": w["status"],
            "Latency (ms)": f"{w['inference_ms']:.1f}" if w["inference_ms"] is not None else "—"
        } for w in windows]), use_container_width=True, hide_index=True)


def _covered_seconds(intervals, duration) -> float:
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        s, e = max(s, 0.0), min(e, duration)
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total
