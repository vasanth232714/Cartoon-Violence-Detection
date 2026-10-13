from __future__ import annotations

import streamlit as st

from config.settings import MODEL_REGISTRY
from models.temporal_transformer import EXPERIMENT_STATUS


def render() -> None:
    st.markdown(
        "<div style='margin-bottom:1.5rem;'>"
        "<h2 style='margin:0; font-size:1.75rem; font-weight:700; color:#F8FAFC; letter-spacing:-0.02em;'>"
        "Model Architecture & Methodology"
        "</h2>"
        "<p style='margin:4px 0 0 0; color:#94A3B8; font-size:0.9rem;'>"
        "The models used for each analysis stage, what they return, and what still needs cartoon-labeled training."
        "</p></div>",
        unsafe_allow_html=True
    )

    # Core Architecture Card
    st.markdown(
        "<div style='background:#111827; border:1px solid rgba(255,255,255,0.08); border-top:3px solid #3B82F6; "
        "border-radius:8px; padding:1.25rem; margin-bottom:1.5rem;'>"
        "<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>"
        "<div>"
        "<span style='font-size:0.75rem; font-weight:600; color:#3B82F6; text-transform:uppercase; letter-spacing:0.04em;'>Primary Visual Candidate Scorer</span>"
        "<div style='font-size:1.15rem; font-weight:700; color:#F8FAFC; margin-top:2px;'>"
        "X3D-M Deep Spatio-Temporal 3D Convolutional Network"
        "</div></div>"
        "<div style='background:rgba(16, 185, 129, 0.12); border:1px solid rgba(16, 185, 129, 0.3); border-radius:6px; "
        "padding:4px 10px; color:#10B981; font-weight:600; font-size:0.75rem;'>"
        "● Configured Baseline Model</div></div>"
        "<p style='color:#94A3B8; font-size:0.88rem; margin:8px 0 0 0; line-height:1.5;'>"
        "X3D-M scores overlapping 16-frame windows with a binary violence class. Its checkpoint was trained on real-world "
        "fight footage and has not been validated on cartoon animation; it does not name actions such as shooting or punching."
        "</p></div>",
        unsafe_allow_html=True
    )

    # 4-Stage Pipeline
    st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600; margin-bottom:0.75rem;'>End-to-End Pipeline Workflow</h4>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            "<div style='background:#111827; border:1px solid rgba(255,255,255,0.08); border-top:2px solid #3B82F6; "
            "border-radius:6px; padding:1rem; height:100%;'>"
            "<div style='color:#60A5FA; font-weight:700; font-size:0.85rem; margin-bottom:2px;'>STAGE 01</div>"
            "<div style='color:#F8FAFC; font-weight:600; font-size:0.9rem; margin-bottom:4px;'>Temporal Resampling</div>"
            "<div style='color:#94A3B8; font-size:0.8rem; line-height:1.4;'>Extracts uniform 16-frame tensor clips with sliding window overlap.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            "<div style='background:#111827; border:1px solid rgba(255,255,255,0.08); border-top:2px solid #6366F1; "
            "border-radius:6px; padding:1rem; height:100%;'>"
            "<div style='color:#818CF8; font-weight:700; font-size:0.85rem; margin-bottom:2px;'>STAGE 02</div>"
            "<div style='color:#F8FAFC; font-weight:600; font-size:0.9rem; margin-bottom:4px;'>3D-CNN Inference</div>"
            "<div style='color:#94A3B8; font-size:0.8rem; line-height:1.4;'>Produces a binary violence candidate score for each overlapping window.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            "<div style='background:#111827; border:1px solid rgba(255,255,255,0.08); border-top:2px solid #F59E0B; "
            "border-radius:6px; padding:1rem; height:100%;'>"
            "<div style='color:#FBBF24; font-weight:700; font-size:0.85rem; margin-bottom:2px;'>STAGE 03</div>"
            "<div style='color:#F8FAFC; font-weight:600; font-size:0.9rem; margin-bottom:4px;'>Full-Video Audio Pass</div>"
            "<div style='color:#94A3B8; font-size:0.8rem; line-height:1.4;'>Transcribes dialogue and detects separate sound-event candidates across the audio track.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            "<div style='background:#111827; border:1px solid rgba(255,255,255,0.08); border-top:2px solid #10B981; "
            "border-radius:6px; padding:1rem; height:100%;'>"
            "<div style='color:#34D399; font-weight:700; font-size:0.85rem; margin-bottom:2px;'>STAGE 04</div>"
            "<div style='color:#F8FAFC; font-weight:600; font-size:0.9rem; margin-bottom:4px;'>Evidence & Reports</div>"
            "<div style='color:#94A3B8; font-size:0.8rem; line-height:1.4;'>Saves evidence and Qwen explanations for selected event clips, then compiles the report.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Model Engineering Matrix
    st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600; margin-bottom:0.75rem;'>Implementation Status</h4>", unsafe_allow_html=True)

    for item in EXPERIMENT_STATUS:
        status_text = item.get("status", "")
        if "requires training" in status_text.lower():
            badge_color = "#F59E0B"
            status_label = "Checkpoint required"
        elif "Integrated" in status_text:
            badge_color = "#10B981"
            status_label = "Active Baseline"
        elif "defined" in status_text.lower():
            badge_color = "#3B82F6"
            status_label = "Architecture Defined"
        else:
            badge_color = "#94A3B8"
            status_label = "Research Roadmap"

        st.markdown(
            f"<div style='background:#111827; border:1px solid rgba(255,255,255,0.08); border-radius:6px; "
            f"padding:1rem; margin-bottom:0.6rem;'>"
            f"<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>"
            f"<span style='font-size:0.95rem; font-weight:600; color:#F8FAFC;'>{item.get('id', '')}: {item.get('name', '')}</span>"
            f"<span style='color:{badge_color}; font-size:0.75rem; font-weight:600; padding:2px 8px; border-radius:4px; border:1px solid {badge_color}33;'>"
            f"● {status_label}</span></div>"
            f"<div style='color:#94A3B8; font-size:0.82rem; margin-top:6px; line-height:1.4;'>{item.get('note', '')}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # System Capabilities Table
    st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600; margin-bottom:0.75rem;'>Capabilities and maturity</h4>", unsafe_allow_html=True)
    caps = [
        ("X3D Binary Violence Scoring", "Real-world-trained 3D-CNN scores over overlapping 16-frame windows; cartoon accuracy is unvalidated", "Implemented"),
        ("Temporal Sequence Transformer", "Optional X3D-feature sequence head; train on labeled cartoon sequences before use", "Checkpoint required"),
        ("Event Boundary Grouping", "Window-based candidates with configurable overlap, threshold, and merge gap", "Implemented"),
        ("Multilingual Speech Transcription", "Full-audio ASR with language metadata; language abuse coverage is limited", "Optional"),
        ("Sound-Event Candidates", "Independent AudioSet sound classes; not proof of a visual event", "Optional"),
        ("Automatic Qwen Explanations", "Selected event clips with separate detector context; generated text needs review", "Optional"),
        ("Heuristic Review Priority", "Unvalidated duration, motion, and explicit-threat heuristic; not violence severity", "Unvalidated"),
        ("Automated Keyframe Evidence", "Source frames saved around event windows", "Implemented"),
        ("Human Review Console", "Persistent decisions, notes, and optional human severity", "Implemented"),
        ("PDF Report Generation", "Printable summary and event table", "Implemented"),
    ]

    for title, desc, stat in caps:
        badge_color = "#10B981" if stat == "Implemented" else ("#F59E0B" if stat in ("Checkpoint required", "Unvalidated") else "#64748B")
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; align-items:center; padding:9px 12px; "
            f"background:#111827; border-bottom:1px solid rgba(255,255,255,0.04); flex-wrap:wrap; gap:8px;'>"
            f"<div>"
            f"<span style='color:#F8FAFC; font-weight:600; font-size:0.88rem;'>{title}</span>"
            f"<span style='color:#64748B; font-size:0.82rem; margin-left:8px;'>— {desc}</span>"
            f"</div>"
            f"<span style='color:{badge_color}; font-weight:600; font-size:0.75rem;'>● {stat}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    with st.expander("Technical Model Registry Specification", expanded=False):
        st.json(MODEL_REGISTRY)
