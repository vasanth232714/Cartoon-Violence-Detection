"""Single-event investigation: evidence frames, explanation, severity basis, frame viewer, review actions."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from database.repository import Repository
from models.vlm_explainer import (
    parse_structured_explanation,
    parse_qwen_visual_review,
)
from preprocessing.frame_extractor import read_frame_at
from preprocessing.video_loader import VideoMetadata
from ui.common import ft, jload, severity_badge


def review_status(repo: Repository, event_id: int) -> str:
    r = repo.latest_review(event_id)
    return r["decision"] if r else "unreviewed"


def _render_violent_clip_explainer(event, video, eid: int) -> None:
    """Show the optional second-phase Qwen explanation saved for this event."""
    st.markdown(
        "<div style='font-size:0.95rem; font-weight:700; color:#F8FAFC; margin:12px 0 6px 0; display:flex; align-items:center; gap:6px;'>"
        "<span>🎬</span> <span>Event Explanation · Qwen2.5-VL-3B</span></div>",
        unsafe_allow_html=True
    )
    saved_expl = event["qwen_explanation"]
    qwen_status = event["qwen_status"] or "not_requested"
    if not saved_expl:
        messages = {
            "disabled": "Ready for the separate Qwen explanation step in the report.",
            "skipped_due_to_event_limit": "Ready for the separate Qwen explanation step in the report.",
            "unavailable": "Qwen could not explain this event. Review the run warning, then retry from the report when ready.",
            "pending": "Candidate saved. Qwen explanation is ready to run from the report.",
            "not_requested": "Candidate saved. Qwen explanation is ready to run from the report.",
        }
        st.info(messages.get(qwen_status, f"Qwen explanation status: {qwen_status}."))
        return

    visual_review = parse_qwen_visual_review(saved_expl)
    if visual_review["structured"]:
        assessment = visual_review["visual_assessment"]
        if assessment == "no_violence_observed":
            st.success("Qwen visual review: No visible violence observed in this clip.")
        elif assessment == "violence_observed":
            st.warning("Qwen visual review: Possible visible violence observed. Review frames carefully.")
        else:
            st.info("Qwen visual review: Uncertain from the sampled frames.")
        if visual_review.get("false_alarm_reason"):
            st.caption(f"Possible false-alarm rationale: {visual_review['false_alarm_reason']}")
        st.caption("Note: This pretrained multimodal review is a second opinion; human verification remains authoritative.")

    parsed = parse_structured_explanation(saved_expl)
    sections = [
        ("Summary", "summary"),
        ("Observed actions", "observed_actions"),
        ("Possible category", "category"),
        ("Timestamped visual, audio, and speech evidence", "timestamped_evidence"),
        ("Reason flagged", "why_flagged"),
        ("Uncertainty", "uncertainty"),
        ("Human review", "human_review"),
    ]
    for title, key in sections:
        if parsed.get(key):
            st.markdown(f"**{title}:** {parsed[key]}")
    with st.expander("View full raw Qwen model response", expanded=False):
        st.text(saved_expl)
    st.download_button(
        "Download explanation (.txt)", saved_expl,
        file_name=f"event_{event['event_no']}_explanation.txt",
        mime="text/plain", use_container_width=True, key=f"dl_txt_btn_{eid}",
    )


def render_event(repo: Repository, event, video, run) -> None:
    eid = event["id"]
    status_str = review_status(repo, eid)
    rev_color = {"confirmed": "#10B981", "rejected": "#EF4444", "reviewed": "#F59E0B"}.get(status_str, "#64748B")
    qwen_review = parse_qwen_visual_review(event["qwen_explanation"] or "")
    qwen_badges = {
        "violence_observed": ("Qwen: Possible Violence", "#F97316", "rgba(249, 115, 22, 0.15)"),
        "no_violence_observed": ("Qwen: No Violence", "#10B981", "rgba(16, 185, 129, 0.15)"),
        "uncertain": ("Qwen: Uncertain", "#F59E0B", "rgba(245, 158, 11, 0.15)"),
    }
    qwen_badge = qwen_badges.get(qwen_review["visual_assessment"]) if qwen_review["structured"] else None
    is_visual_candidate = ("Violence (binary model output" in str(event["label"] or "")
                           or "X3D visual violence candidate" in str(event["label"] or ""))
    priority_badge = (
        f"<span style='padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; "
        f"color:{qwen_badge[1]}; background:{qwen_badge[2]}; border:1px solid {qwen_badge[1]}44;'>"
        f"{qwen_badge[0]}</span>"
        if qwen_badge else (
            "<span style='padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; "
            "color:#60A5FA; background:rgba(59, 130, 246, 0.15); border:1px solid rgba(59, 130, 246, 0.35);'>"
            "Unconfirmed X3D Candidate</span>"
            if is_visual_candidate else severity_badge(event["severity_label"])
        )
    )

    # Incident Banner
    st.markdown(
        f"<div style='background:rgba(17, 24, 39, 0.85); backdrop-filter:blur(8px); "
        f"border:1px solid rgba(255,255,255,0.08); border-left:3px solid #3B82F6; "
        f"border-radius:10px; padding:1.1rem 1.35rem; margin-bottom:1.25rem; "
        f"display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:0.75rem;'>"
        f"<div>"
        f"<div style='font-size:1.2rem; font-weight:800; color:#F8FAFC;'>Event #{event['event_no']}</div>"
        f"<div style='color:#94A3B8; font-size:0.86rem; margin-top:2px;'>"
        f"Interval: <b>{ft(event['start_s'])} → {ft(event['end_s'])}</b> ({event['duration_s']:.1f}s duration)"
        f"</div></div>"
        f"<div style='display:flex; gap:8px; align-items:center;'>"
        f"<div>{priority_badge}</div>"
        f"<div style='padding:3px 10px; border-radius:20px; font-size:0.75rem; font-weight:600; "
        f"color:{rev_color}; background:{rev_color}18; border:1px solid {rev_color}44;'>"
        f"Audit: {status_str.upper()}</div></div></div>",
        unsafe_allow_html=True
    )

    c_player, c_telemetry = st.columns([3, 2])

    with c_player:
        path = video["stored_path"]
        if video["file_available"] and Path(path).is_file():
            try:
                st.video(path, start_time=int(max(event["start_s"] - 0.5, 0)))
                st.caption(f"Playback cued to {ft(max(event['start_s'] - 0.5, 0))}.")
            except Exception as e:
                st.warning(f"Video player note: {e}")
        else:
            st.info("Source video file archived. Stored keyframes and metrics remain accessible.")

        # Violent Explainer directly below the event video player
        _render_violent_clip_explainer(event, video, eid)

        if video["file_available"] and Path(video["stored_path"]).is_file():
            with st.expander("🎞️ Frame-by-Frame Inspector", expanded=False):
                lo = max(event["start_s"] - 2.0, 0.0)
                hi = min(event["end_s"] + 2.0, video["duration_s"])
                t = st.slider("Timestamp (s)", float(lo), float(hi), float(event["start_s"]), 0.1, key=f"fv_{eid}")
                meta = VideoMetadata(video["duration_s"], video["fps"], video["width"], video["height"],
                                     video["frame_count"], video["file_size_bytes"], video["container"] or "",
                                     video["fourcc"] or "")
                img = read_frame_at(video["stored_path"], meta, t)
                if img is not None:
                    st.image(img, caption=f"Extracted Frame at {ft(t)}", use_column_width=True)

    with c_telemetry:
        with st.container(border=True):
            st.markdown(
                "<div style='font-size:0.74rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;'>"
                "Detection Telemetry</div>",
                unsafe_allow_html=True
            )
            conf_pct = int(round(event["peak_prob"] * 100))
            if ("Violence (binary model output" in str(event["label"] or "")
                or "X3D visual violence candidate" in str(event["label"] or "")):
                st.markdown(
                    f"- **Finding:** `Unconfirmed X3D visual candidate`\n"
                    f"- **Peak X3D Score:** **{conf_pct}%** (~{ft(event['peak_time_s'])})\n"
                    f"- **Mean Window Score:** `{event['mean_prob']:.2f}`\n"
                    f"- **Positive Windows:** `{event['n_windows']}` in `{event['n_positive_runs']}` burst(s)"
                )
                if qwen_review["structured"]:
                    verdict = qwen_review["visual_assessment"].replace("_", " ").title()
                    st.markdown(f"- **Qwen Visual Review:** `{verdict}`")
            else:
                st.markdown(
                    f"- **Finding:** `{event['label']}`\n"
                    f"- **Candidate Confidence:** **{conf_pct}%** (~{ft(event['peak_time_s'])})\n"
                    f"- **Visual Violence Score:** Not assessed for standalone audio candidate."
                )

        with st.container(border=True):
            st.markdown(
                "<div style='font-size:0.74rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:8px;'>"
                "Review Priority Rationale</div>",
                unsafe_allow_html=True
            )
            sj = jload(event["severity_json"], {}) or {}
            priority_time = " before Qwen review" if qwen_review["structured"] else ""
            st.markdown(f"**Heuristic Priority{priority_time}:** `{event['severity_label']}`")
            st.caption("Based on duration, repeated score bursts, and frame motion dynamics.")
            for line in sj.get("rationale", []):
                st.caption(f"• {line}")

            mj = jload(event["motion_json"], {}) or {}
            if mj.get("mean") is not None:
                st.caption(f"Motion Displacement: Mean {mj['mean']:.1f} | Peak p90 {mj['p90']:.1f} ({mj['n_pairs']} frame pairs)")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Keyframe Evidence Gallery
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin-bottom:0.75rem;'>Evidence Keyframes</h4>", unsafe_allow_html=True)
    ev = repo.list_evidence(eid)
    if not ev:
        st.warning("No keyframes stored for this event.")
    else:
        cols = st.columns(min(len(ev), 6))
        for i, f in enumerate(ev):
            with cols[i % len(cols)]:
                if Path(f["path"]).is_file():
                    badge = " ★ PEAK" if f["kind"] == "peak_window" else ""
                    st.image(f["path"], caption=f"{ft(f['timestamp_s'])}{badge}", use_column_width=True)
                else:
                    st.caption(f"{ft(f['timestamp_s'])}: Missing")

        labels = [f"{ft(f['timestamp_s'])} ({f['kind']})" for f in ev]
        pick = st.radio("Enlarge Keyframe", labels, horizontal=True, key=f"big_{eid}")
        f_chosen = ev[labels.index(pick)]
        if Path(f_chosen["path"]).is_file():
            st.image(f_chosen["path"], caption=f"Keyframe Snapshot at {ft(f_chosen['timestamp_s'])}", use_column_width=True)

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Modality-separated detector evidence
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin-bottom:0.75rem;'>Candidate Evidence by Modality</h4>", unsafe_allow_html=True)
    is_visual = ("Violence (binary model output" in str(event["label"] or "")
                 or "X3D visual violence candidate" in str(event["label"] or ""))
    if is_visual:
        st.write(f"**Visual Candidate:** X3D-M binary output, peak class score {event['peak_prob']:.3f} at {ft(event['peak_time_s'])}. This is an uncalibrated candidate signal.")
    else:
        st.write("**Visual:** Not assessed for this audio-only candidate.")

    verbal_spans = sj.get("verbal_abuse", []) or []
    if verbal_spans:
        st.markdown("**Speech and Verbal Language Candidates**")
        for span in verbal_spans:
            timestamp = f"{ft(span.get('start_s', 0))}–{ft(span.get('end_s', 0))}"
            categories = ", ".join(span.get("categories") or ["language candidate"])
            language = span.get("language") or "unknown"
            st.write(f"🗣️ {timestamp} · {language} · {categories} · score {float(span.get('confidence', 0)):.2f}")
            st.caption(span.get("censored_transcript") or span.get("transcript") or "No transcript text stored.")
        with st.expander("View uncensored transcript", expanded=False):
            for span in verbal_spans:
                st.code(span.get("transcript") or "", language=None)

    sound_spans = sj.get("audio_events", []) or []
    if sound_spans:
        st.markdown("**Sound-Event Candidates**")
        for sound in sound_spans:
            st.write(f"🔊 {ft(sound.get('start_s', 0))}–{ft(sound.get('end_s', 0))} · {sound.get('category', 'Unknown sound')} · score {float(sound.get('confidence', 0)):.2f}")

    with st.expander("Technical Model Attribution & Full Forensic Log", expanded=False):
        st.markdown(event["explanation"] or "No heuristic rationale stored for this event.")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Human Review Form
    st.markdown(
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;'>"
        "<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin:0;'>Human Verification & Audit Log</h4>"
        "<span style='font-size:0.74rem; color:#94A3B8;'>Verified reviews persist permanently in the database</span>"
        "</div>",
        unsafe_allow_html=True
    )
    with st.form(f"review_{eid}", clear_on_submit=True):
        c_dec, c_sev = st.columns([3, 2])
        with c_dec:
            decision = st.radio(
                "Review Decision",
                ["confirmed", "rejected", "reviewed"],
                horizontal=True,
                help=("confirmed = visual scene is violent; rejected = false positive; reviewed = inspected, pending decision"
                      if is_visual else "confirmed = speech/sound candidate matches; rejected = false positive; reviewed = inspected, pending decision")
            )
        with c_sev:
            sev = st.selectbox(
                "Human Severity Rating",
                [None, 0, 1, 2, 3],
                format_func=lambda x: "Model Default" if x is None else f"Tier {x}"
            )
        notes = st.text_area("Reviewer Notes & Observations", placeholder="Enter notes explaining why this scene was confirmed or rejected...")
        if st.form_submit_button("Save Review Decision", type="primary", use_container_width=True):
            repo.add_review(eid, decision, notes, sev)
            st.success("Review decision recorded successfully.")
            st.rerun()

    revs = repo.list_reviews(eid)
    if revs:
        for r in revs:
            sev_txt = "" if r["human_severity"] is None else f" · Tier {r['human_severity']}"
            st.caption(f"🗓️ {r['created_at']} — **{r['decision'].upper()}**{sev_txt}" + (f" — {r['notes']}" if r["notes"] else ""))
