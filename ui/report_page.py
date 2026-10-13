from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from collections import Counter

from database.repository import COMPLETE_STATUSES, Repository
from pipeline.report_pdf import build_pdf
from ui.common import STATUS_HELP, ft, jload, render_hero_metric, severity_badge, status_badge
from ui.event_investigation import render_event, review_status
from ui.timeline import build_timeline_figure
from models.vlm_explainer import (
    QWEN_MIN_AVAILABLE_RAM_BYTES,
    _available_physical_memory_bytes,
    parse_structured_explanation,
    parse_qwen_visual_review,
)
from pipeline.qwen_phase import audit_full_video_with_qwen, explain_run_candidates


def _render_videomae_comparison(repo: Repository, run_id: int, windows, run_info: dict, cfg: dict,
                                duration_s: float) -> None:
    """Show independent window scores, agreement, disagreements, and each model's candidates."""
    metadata = run_info.get("videomae_comparison") or {}
    if not metadata:
        return
    st.markdown("#### Optional model comparison · X3D baseline vs VideoMAE")
    if metadata.get("status") == "failed":
        st.warning(f"VideoMAE comparison failed: {metadata.get('error', 'see run warnings')}")
        return
    scores = repo.list_model_window_scores(run_id, "videomae")
    if not scores:
        st.info("VideoMAE was enabled, but no comparison window scores were saved.")
        return

    x3d_threshold = float(cfg.get("threshold", 0.75))
    videomae_threshold = float(metadata.get("candidate_threshold", cfg.get("videomae_threshold", 0.50)))
    x3d_by_index = {int(row["window_index"]): row for row in windows}
    mae_by_index = {int(row["window_index"]): row for row in scores}
    common = sorted(set(x3d_by_index) & set(mae_by_index))
    valid = [i for i in common if x3d_by_index[i]["p_violent"] is not None
             and mae_by_index[i]["p_violent"] is not None]
    if not valid:
        st.warning("There are no windows with valid scores from both models to compare.")
        return

    x3d_pos = {i: float(x3d_by_index[i]["p_violent"]) >= x3d_threshold for i in valid}
    mae_pos = {i: float(mae_by_index[i]["p_violent"]) >= videomae_threshold for i in valid}
    n_agree = sum(x3d_pos[i] == mae_pos[i] for i in valid)
    n_disagree = len(valid) - n_agree
    c1, c2, c3 = st.columns(3)
    c1.metric("Comparable windows", len(valid))
    c2.metric("Agreement", f"{n_agree / len(valid) * 100:.1f}%", f"{n_agree} windows")
    c3.metric("Disagreement", n_disagree)
    st.caption(
        f"Independent thresholds: X3D {x3d_threshold:.2f} · VideoMAE {videomae_threshold:.2f}. "
        "Scores are displayed separately; no averaging or calibration is applied. Agreement is not ground-truth accuracy."
    )
    st.caption(
        f"Model card: {metadata.get('model_id', 'HappyGook/videomae-violence-detector')} · "
        f"{metadata.get('num_frames', 16)} frames at {metadata.get('input_size', 224)}px · "
        f"device: {metadata.get('device', 'CPU')} · status: {metadata.get('status', 'unknown')}. "
        "Trained on UCF-Crime and bus-surveillance footage; cartoon performance has not been evaluated."
    )

    centers = [(float(x3d_by_index[i]["start_s"]) + float(x3d_by_index[i]["end_s"])) / 2 for i in valid]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=centers, y=[float(x3d_by_index[i]["p_violent"]) for i in valid],
                             mode="lines+markers", name="X3D baseline score", line=dict(color="#3B82F6")))
    fig.add_trace(go.Scatter(x=centers, y=[float(mae_by_index[i]["p_violent"]) for i in valid],
                             mode="lines+markers", name="VideoMAE score", line=dict(color="#A78BFA")))
    fig.add_hline(y=x3d_threshold, line_dash="dash", line_color="#3B82F6",
                  annotation_text=f"X3D threshold {x3d_threshold:.2f}")
    fig.add_hline(y=videomae_threshold, line_dash="dot", line_color="#A78BFA",
                  annotation_text=f"VideoMAE threshold {videomae_threshold:.2f}")
    fig.update_layout(height=340, xaxis_title="Video time (seconds)", yaxis_title="Model score (uncalibrated)",
                      yaxis_range=[0, 1.05], xaxis_range=[0, duration_s],
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#94A3B8"), margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(fig, use_container_width=True)

    window_score_rows = [{
        "Window": i + 1,
        "Time": f"{ft(x3d_by_index[i]['start_s'])} → {ft(x3d_by_index[i]['end_s'])}",
        "X3D score": f"{float(x3d_by_index[i]['p_violent']):.4f}",
        "VideoMAE score": f"{float(mae_by_index[i]['p_violent']):.4f}",
        "X3D decision": "candidate" if x3d_pos[i] else "below threshold",
        "VideoMAE decision": "candidate" if mae_pos[i] else "below threshold",
        "Comparison": "agree" if x3d_pos[i] == mae_pos[i] else "disagree",
    } for i in valid]
    with st.expander("All paired window scores", expanded=False):
        st.dataframe(pd.DataFrame(window_score_rows), use_container_width=True, hide_index=True)

    disagreements = []
    for i in valid:
        if x3d_pos[i] == mae_pos[i]:
            continue
        xrow, mrow = x3d_by_index[i], mae_by_index[i]
        disagreements.append({
            "Window": i + 1,
            "Time": f"{ft(xrow['start_s'])} → {ft(xrow['end_s'])}",
            "Decision": "X3D only" if x3d_pos[i] else "VideoMAE only",
            "X3D score": f"{float(xrow['p_violent']):.3f}",
            "VideoMAE score": f"{float(mrow['p_violent']):.3f}",
        })
    with st.expander(f"Disagreeing windows ({len(disagreements)})", expanded=False):
        if disagreements:
            st.dataframe(pd.DataFrame(disagreements), use_container_width=True, hide_index=True)
        else:
            st.caption("No disagreeing windows at the configured thresholds.")

    from pipeline.event_segmenter import WindowPrediction, segment_events

    def candidate_rows(model_name: str, source_rows, threshold: float):
        predictions = [WindowPrediction(
            index=int(row["window_index"]), start_s=float(row["start_s"]), end_s=float(row["end_s"]),
            p_violent=float(row["p_violent"]) if row["p_violent"] is not None else None,
            status=str(row["status"]),
        ) for row in source_rows]
        candidates = segment_events(predictions, threshold, float(cfg.get("merge_gap_seconds", 1.0)), 0.0)
        return [{
            "Model": model_name,
            "Candidate interval": f"{ft(event.start_s)} → {ft(event.end_s)}",
            "Duration": f"{event.duration_s:.1f}s",
            "Peak score": f"{event.peak_prob:.3f}",
            "Windows": event.n_windows,
        } for event in candidates]

    candidate_data = candidate_rows("X3D", windows, x3d_threshold)
    candidate_data += candidate_rows("VideoMAE", scores, videomae_threshold)
    st.markdown("**Candidate intervals by model**")
    if candidate_data:
        st.dataframe(pd.DataFrame(candidate_data), use_container_width=True, hide_index=True)
    else:
        st.caption("Neither model produced a candidate interval at its own threshold.")


def _render_qwen_phase(repo: Repository, run_id: int, video, events) -> None:
    """Offer Qwen as an explicit second step after the complete candidate list is saved."""
    st.markdown("#### Phase 2 · Check candidates and explain them with Qwen")
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
        st.info("Run the candidate scan first. Qwen runs only after candidate intervals are saved.")
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
            f"Qwen needs about {required_gib:.0f} GiB of available RAM with its lower-memory pretrained weights. "
            f"This machine currently has {free_gib:.1f} GiB available. Candidate detection is complete; "
            "free more memory or use a higher-memory machine to run explanations."
        )
    elif available is None:
        st.caption("Qwen uses the checkpoint's bfloat16 weights and reduced frame resolution; allow about 8 GiB of free RAM for CPU review.")

    source_available = Path(video["stored_path"]).is_file()
    if not source_available:
        st.warning("The original video file is unavailable. Restore or re-upload it before running Qwen.")

    if st.button(
        f"🔎 Review and explain {len(pending)} remaining candidates with Qwen",
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


def _render_qwen_full_video_audit(repo: Repository, run_id: int, video, model_info: dict, events) -> None:
    """Render an optional Qwen second opinion over every video segment, including X3D misses."""
    st.markdown("#### Independent full-video review · Qwen2.5-VL")
    audit = model_info.get("qwen_full_video_audit") or {}
    st.caption(
        "This pretrained vision-language model reviews the source in 30-second chunks, so it can report "
        "possible scenes that X3D missed and provide context for false alarms. Results remain unconfirmed; "
        "Qwen is not validated for cartoon violence and never edits or removes X3D candidates."
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
            f"this machine has {available / 1024**3:.1f} GiB available."
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
    st.caption(
        f"Status: {audit.get('status', 'unknown')} · "
        f"Segments reviewed: {audit.get('completed_chunks', 0)}/{audit.get('total_chunks', 0)} · "
        f"Possible visual events: {audit.get('event_count', 0)}"
    )
    if audit.get("last_error"):
        st.warning(audit["last_error"])

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
        for index, q_event in enumerate(audit_events, 1):
            with st.expander(
                f"Inspect Qwen candidate {index} · {ft(q_event['start_s'])}–{ft(q_event['end_s'])} · {q_event.get('category', 'unclear')}",
                expanded=False,
            ):
                if source_available:
                    st.video(video["stored_path"], start_time=max(int(float(q_event["start_s"]) - 1.0), 0))
                st.write(q_event.get("observed_action") or "Qwen did not provide an action description.")
                st.caption("Qwen's interval is a review candidate, not a confirmed finding. Verify it against the source clip.")
    else:
        st.info("Qwen did not return a timestamped visual event. This does not prove the video is violence-free.")

    if visual_x3d:
        crosscheck_rows = []
        for event in visual_x3d:
            matches = [item for item in audit_events
                       if not (float(item["end_s"]) < float(event["start_s"]) - 2.0
                               or float(item["start_s"]) > float(event["end_s"]) + 2.0)]
            if matches:
                comparison = "Qwen also reported a possible event"
            else:
                overlaps = [chunk for chunk in audit_chunks
                            if not (float(chunk["end_s"]) < float(event["start_s"])
                                    or float(chunk["start_s"]) > float(event["end_s"]))]
                assessments = {chunk.get("assessment") for chunk in overlaps}
                if "no_violence_observed" in assessments:
                    comparison = "Qwen reported no visible violence in overlapping chunk(s); X3D candidate still needs review"
                elif overlaps:
                    comparison = "Qwen did not return a matching event; review both sources"
                else:
                    comparison = "Not covered by completed Qwen chunks"
            crosscheck_rows.append({
                "X3D candidate": f"Event #{event['event_no']}",
                "Time span": f"{ft(event['start_s'])} → {ft(event['end_s'])}",
                "X3D score": f"{int(round(float(event['peak_prob']) * 100))}%",
                "Qwen cross-check": comparison,
            })
        st.markdown("**X3D candidate cross-check**")
        st.dataframe(pd.DataFrame(crosscheck_rows), use_container_width=True, hide_index=True)

    chunk_rows = []
    for chunk in audit_chunks:
        chunk_rows.append({
            "Segment": f"{ft(chunk['start_s'])} → {ft(chunk['end_s'])}",
            "Qwen assessment": chunk.get("assessment", "uncertain").replace("_", " ").title(),
            "Evidence / uncertainty": chunk.get("evidence") or "No evidence text returned.",
            "Structured answer": "Yes" if chunk.get("structured") else "No — see raw response",
        })
    if chunk_rows:
        with st.expander("Qwen segment-by-segment review", expanded=False):
            st.dataframe(pd.DataFrame(chunk_rows), use_container_width=True, hide_index=True)
        with st.expander("Raw Qwen chunk responses", expanded=False):
            for chunk in audit.get("chunks", []) or []:
                if chunk.get("raw_response"):
                    st.markdown(f"**{ft(chunk['start_s'])}–{ft(chunk['end_s'])}**")
                    st.code(chunk["raw_response"])
def render(repo: Repository) -> None:
    st.markdown(
        "<div style='margin-bottom:1.5rem;'>"
        "<h2 style='margin:0; font-size:1.75rem; font-weight:700; color:#F8FAFC; letter-spacing:-0.02em;'>"
        "Incident Analysis Report"
        "</h2>"
        "<p style='margin:4px 0 0 0; color:#94A3B8; font-size:0.9rem;'>"
        "Full-video visual, speech, and sound candidates with per-interval explanations and review."
        "</p></div>",
        unsafe_allow_html=True
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
    audio_segments = repo.list_audio_segments(run_id)

    with c_stat:
        st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)
        shown_status = "completed" if run["status"] == "completed_with_warnings" and not display_warnings else run["status"]
        st.markdown(status_badge(shown_status), unsafe_allow_html=True)

    with c_del:
        st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)
        if st.button("🗑️ Delete", key=f"del_rep_top_{run_id}", help="Permanently delete this report and its detections", use_container_width=True):
            repo.delete_run(run_id)
            st.session_state.pop("selected_run_id", None)
            st.success(f"Run #{run_id} was deleted.")
            st.rerun()

    if run["status"] not in COMPLETE_STATUSES:
        st.error(f"Analysis status note: {STATUS_HELP.get(run['status'], '')} {run['error'] or ''}")
    elif run["status"] == "completed_with_warnings" and display_warnings:
        st.warning("Analysis completed with warnings: " + " ".join(display_warnings))
    elif warnings and not display_warnings:
        st.info("The optional trained temporal Transformer checkpoint is not installed. Deterministic score smoothing was used; no training is required to run the X3D detector.")

    # Precision & Filter Status Badge
    filter_enabled = cfg.get("enable_temporal_refinement", False) or cfg.get("suppress_shot_cuts", False)
    thr_val = cfg.get("threshold", 0.75)
    min_ev_val = cfg.get("min_event_seconds", 0.0)

    if filter_enabled:
        filter_badge_html = (
            f"<div style='display:inline-flex; align-items:center; gap:8px; background:rgba(16, 185, 129, 0.08); "
            f"border:1px solid rgba(16, 185, 129, 0.25); border-radius:6px; padding:6px 14px; margin-bottom:12px; font-size:0.82rem; color:#6EE7B7;'>"
            f"<span><b>Temporal post-processing: configured</b> (Threshold: {thr_val:.2f} · Min candidate interval: {min_ev_val:.1f}s)</span>"
            f"</div>"
        )
    else:
        filter_badge_html = (
            f"<div style='display:inline-flex; align-items:center; gap:8px; background:rgba(245, 158, 11, 0.08); "
            f"border:1px solid rgba(245, 158, 11, 0.25); border-radius:6px; padding:6px 14px; margin-bottom:12px; font-size:0.82rem; color:#FCD34D;'>"
            f"<span><b>Temporal post-processing: disabled</b> (Threshold: {thr_val:.2f} · checkpoint remains unvalidated on cartoons)</span>"
            f"</div>"
        )
    st.markdown(filter_badge_html, unsafe_allow_html=True)

    # Metric Cards
    st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    m = st.columns(5)
    m[0].markdown(render_hero_metric("Duration", ft(video["duration_s"]), "", f"{video['duration_s']:.1f}s total", "#3B82F6"), unsafe_allow_html=True)
    m[1].markdown(render_hero_metric("Event Candidates", len(events), "", "Review intervals", "#EF4444"), unsafe_allow_html=True)
    coverage = (100.0 * run["windows_done"] / run["windows_total"]) if run["windows_total"] else 0.0
    m[2].markdown(render_hero_metric("Windows Analyzed", f"{run['windows_done']}/{run['windows_total']}", "", f"{coverage:.0f}% complete", "#6366F1"), unsafe_allow_html=True)
    proc_time = f"{(run['elapsed_s'] or 0):.1f}s"
    m[3].markdown(render_hero_metric("Analysis Time", proc_time, "", "All enabled stages", "#F59E0B"), unsafe_allow_html=True)
    ratio = (run["elapsed_s"] or 0) / video["duration_s"] if video["duration_s"] else 0
    m[4].markdown(render_hero_metric("Compute / Video", f"{ratio:.2f}×", "", "Elapsed time divided by duration", "#10B981"), unsafe_allow_html=True)

    # Consolidated multimodal summary and original-timestamp transcript.
    verbal_categories = Counter()
    audio_categories = Counter()
    visual_count = 0
    verbal_count = audio_count = 0
    summary_rows = []
    languages = sorted({a["language"] for a in audio_segments if a["language"]})
    for event in events:
        label = str(event["label"] or "")
        details = jload(event["severity_json"], {}) or {}
        verbal = details.get("verbal_abuse", []) or []
        sounds = details.get("audio_events", []) or []
        is_visual = "Violence (binary model output" in label or bool(details.get("visual_model"))
        visual_count += int(is_visual)
        for span in verbal:
            verbal_categories.update(span.get("categories") or [])
            verbal_count += 1
        for sound in sounds:
            audio_categories.update([sound.get("category", "Unknown audio event")])
            audio_count += 1
        qwen = parse_structured_explanation(event["qwen_explanation"] or "").get("summary", "")
        speech_text = " | ".join(dict.fromkeys(span.get("transcript", "") for span in verbal if span.get("transcript")))
        summary_rows.append({
            "Original time": f"{ft(event['start_s'])}–{ft(event['end_s'])}",
            "Visual finding": "X3D candidate · unconfirmed" if is_visual else "No X3D candidate reported",
            "Audio finding": ", ".join(dict.fromkeys(s.get("category", "") for s in sounds)) or "—",
            "Verbal abuse candidate": speech_text or ", ".join(dict.fromkeys(
                c for span in verbal for c in (span.get("categories") or []))) or "—",
            "Heuristic review priority": event["severity_label"],
            "Qwen explanation": qwen or (
                "Ready for Phase 2" if event["qwen_status"] != "generated"
                else "Generated, no summary parsed"
            ),
        })

    with st.expander("Full-video multimodal summary", expanded=True):
        st.caption(f"Source: {video['original_name']} · {video['duration_s']:.1f}s · Run status: {run['status']}")
        cols = st.columns(5)
        cols[0].metric("Visual candidates", visual_count)
        cols[1].metric("Speech candidates", verbal_count)
        cols[2].metric("Audio-event candidates", audio_count)
        cols[3].metric("Review events", len(events))
        cols[4].metric("Supported transcript languages", len(languages))
        st.caption("Transcript languages detected: " + (", ".join(languages) if languages else "none reported"))
        st.caption("X3D scores are uncalibrated class scores from a real-world-fight checkpoint, not cartoon violence confidence. Review priority uses duration, bursts, and motion only; it is not violence severity. Audio-only findings remain Not assessed.")
        if summary_rows:
            st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No event candidates were produced by the configured models.")

    with st.expander("Full transcript and timestamped audio findings", expanded=False):
        transcript_rows = []
        for segment in audio_segments:
            if segment["kind"] == "transcript":
                transcript_rows.append({
                    "Time": f"{ft(segment['start_s'])}–{ft(segment['end_s'])}",
                    "Language": segment["language"] or "unknown/subtitle",
                    "Transcript": segment["text"] or "",
                    "Source": segment["source"] or "unknown",
                })
        if transcript_rows:
            st.dataframe(pd.DataFrame(transcript_rows), use_container_width=True, hide_index=True)
        else:
            st.info("No transcript segments were stored. Check run warnings and subtitle availability.")
        audio_rows = [{
            "Time": f"{ft(a['start_s'])}–{ft(a['end_s'])}",
            "Finding": a["category"] or a["kind"],
            "Text": a["text"] or "",
            "Language": a["language"] or "—",
            "Score": f"{a['confidence']:.3f}" if a["confidence"] is not None else "—",
            "Source": a["source"] or "unknown",
        } for a in audio_segments if a["kind"] in ("verbal_abuse", "audio_event")]
        if audio_rows:
            st.markdown("**Flagged speech and sound candidates**")
            st.dataframe(pd.DataFrame(audio_rows), use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

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
        st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600;'>X3D Candidate Coverage (unconfirmed)</h4>", unsafe_allow_html=True)
        st.markdown(
            f"<div style='background:#111827; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:16px;'>"
            f"<div style='display:flex; justify-content:space-between; margin-bottom:8px; font-size:0.85rem;'>"
            f"<span style='font-weight:600; color:#10B981;'>Not flagged by X3D: {pct_unflagged:.1f}%</span>"
            f"<span style='font-weight:600; color:#EF4444;'>X3D candidate intervals: {pct_visual:.1f}%</span>"
            f"</div>"
            f"<div style='height:10px; border-radius:5px; background:#1E293B; overflow:hidden; display:flex;'>"
            f"<div style='width:{pct_unflagged}%; background:#10B981; height:100%;'></div>"
            f"<div style='width:{pct_visual}%; background:#EF4444; height:100%;'></div>"
            f"</div>"
            f"<div style='display:flex; justify-content:space-between; margin-top:10px; font-size:0.78rem; color:#64748B;'>"
            f"<span>Not flagged by X3D: {video['duration_s'] - covered:.1f}s (not verified safe)</span>"
            f"<span>Inside X3D candidates: {covered:.1f}s</span>"
            f"</div></div>",
            unsafe_allow_html=True
        )
        st.caption("This chart shows which timestamps crossed the X3D threshold. Unflagged time has not been verified as nonviolent.")

    with col_dist:
        st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600;'>Heuristic Review Priority Distribution</h4>", unsafe_allow_html=True)
        if events:
            sev_counts = pd.Series([e["severity_label"] for e in events]).value_counts()
            palette = {"High": "#EF4444", "Moderate": "#F97316", "Mild": "#F59E0B", "Not assessed": "#64748B"}
            colors = [palette.get(k, "#3B82F6") for k in sev_counts.index]
            fig_sev = go.Figure(data=[go.Pie(
                labels=sev_counts.index, values=sev_counts.values, hole=0.58,
                marker=dict(colors=colors, line=dict(color='#0B0F19', width=2)),
                textinfo='label+value'
            )])
            fig_sev.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94A3B8", family="Inter, -apple-system, sans-serif", size=11),
                margin=dict(l=10, r=10, t=10, b=10),
                height=130,
                showlegend=False
            )
            st.plotly_chart(fig_sev, use_container_width=True)
        else:
            st.caption("No event candidates to summarize.")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Interactive Timeline Section
    st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600;'>X3D class scores and multimodal candidate timeline</h4>", unsafe_allow_html=True)
    sel_key = f"sel_event_{run_id}"

    if not events:
        st.info(f"No event candidate exceeded the configured threshold ({cfg.get('threshold', 0.75)}).")
        st.plotly_chart(build_timeline_figure(windows, [], cfg.get("threshold", 0.75), video["duration_s"]),
                        use_container_width=True)
    else:
        c_f1, c_f2 = st.columns(2)
        all_sevs = sorted({e["severity_label"] for e in events})
        with c_f1:
            sev_sel = st.multiselect("Filter Review Priority", all_sevs, default=all_sevs)
        with c_f2:
            min_p = st.slider("Minimum Peak X3D Score", 0.0, 1.0, 0.0, 0.05)

        shown = [e for e in events if e["severity_label"] in sev_sel and e["peak_prob"] >= min_p]
        sel_no = st.session_state.get(sel_key, shown[0]["event_no"] if shown else None)

        st.plotly_chart(build_timeline_figure(windows, shown, cfg.get("threshold", 0.75), video["duration_s"], sel_no),
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
                if has_visual: modalities.append("Visual candidate")
                if has_verbal: modalities.append("Speech candidate")
                if has_audio: modalities.append("Audio candidate")
                qwen_review = parse_qwen_visual_review(e["qwen_explanation"] or "")
                qwen_label = {
                    "violence_observed": "Possible visible violence",
                    "no_violence_observed": "No visible violence observed",
                    "uncertain": "Uncertain",
                }.get(qwen_review["visual_assessment"], "Not reviewed") if qwen_review["structured"] else "Not reviewed"
                rows.append({
                    "Event #": f"Event {e['event_no']}",
                    "Modality": " + ".join(modalities) or "Unspecified",
                    "Time Span": f"{ft(e['start_s'])} → {ft(e['end_s'])}",
                    "Duration": f"{e['duration_s']:.1f}s",
                    "Peak X3D Score": f"{int(round(e['peak_prob'] * 100))}%",
                    "X3D Priority (pre-Qwen)": e["severity_label"],
                    "Qwen visual review": qwen_label,
                    "Review Status": review_status(repo, e["id"]).upper()
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        _render_qwen_phase(repo, run_id, video, events)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # Event Deep Dive
        st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600;'>Event Deep Dive</h4>", unsafe_allow_html=True)
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
                    return f"Event {n}{v_icon}: {ft(ev_match['start_s'])} → {ft(ev_match['end_s'])} (review priority {ev_match['severity_label']} · X3D score {int(round(ev_match['peak_prob']*100))}%)"

                st.selectbox(
                    "Navigate Event",
                    nums,
                    key=sel_key,
                    label_visibility="collapsed",
                    format_func=_fmt_event
                )

            cur = next(e for e in shown if e["event_no"] == st.session_state[sel_key])
            render_event(repo, cur, video, run)

    _render_videomae_comparison(repo, run_id, windows, info or {}, cfg, video["duration_s"])
    _render_qwen_full_video_audit(repo, run_id, video, info or {}, events)

    # Documentation & PDF Export
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600;'>Export Report</h4>", unsafe_allow_html=True)
    c_pdf1, c_pdf2 = st.columns([2, 3])
    with c_pdf1:
        if st.button("Generate PDF Report", use_container_width=True):
            try:
                with st.spinner("Generating PDF report..."):
                    st.session_state[f"pdf_{run_id}"] = build_pdf(repo, run_id)
                st.success("PDF generated successfully.")
            except Exception as e:
                st.error(f"PDF generation error: {e}")

    with c_pdf2:
        if f"pdf_{run_id}" in st.session_state:
            st.download_button(
                "📥 Download PDF Report",
                st.session_state[f"pdf_{run_id}"],
                file_name=f"violence_detection_report_run{run_id}.pdf",
                mime="application/pdf",
                type="primary",
                use_container_width=True
            )

    with st.expander("🗑️ Delete This Report", expanded=False):
        st.markdown(f"Permanently remove **Run #{run_id} ({video['original_name']})** and all its detected events and evidence from the database.")
        c_d1, c_d2 = st.columns([3, 1.5])
        with c_d1:
            confirm_del = st.checkbox(f"Confirm permanent deletion of Run #{run_id}", key=f"conf_del_bottom_{run_id}")
        with c_d2:
            if st.button("Delete Report", type="primary", disabled=not confirm_del, key=f"btn_del_bottom_{run_id}", use_container_width=True):
                repo.delete_run(run_id)
                st.session_state.pop("selected_run_id", None)
                st.success(f"Run #{run_id} was deleted.")
                st.rerun()

    with st.expander("Technical Telemetry & Prediction Logs", expanded=False):
        c_t1, c_t2 = st.columns(2)
        with c_t1:
            st.markdown("**Session Parameters**")
            st.json({**cfg, "app_version": run["app_version"], "config_hash": run["config_hash"]})
        with c_t2:
            st.markdown("**Compute Latency Profile**")
            st.json({**info, "timing": timing})

        st.markdown("**Window Level Predictions**")
        st.dataframe(pd.DataFrame([{
            "Index": w["window_index"],
            "Span": f"{ft(w['start_s'])} - {ft(w['end_s'])}",
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
