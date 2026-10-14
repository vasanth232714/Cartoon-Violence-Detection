from __future__ import annotations

import streamlit as st

from config.settings import MODEL_REGISTRY
from models.temporal_transformer import EXPERIMENT_STATUS
from ui.common import render_page_header


def render() -> None:
    render_page_header(
        title="Model Architecture & Methodology",
        subtitle="Technical documentation of deep neural architectures, multimodal pipelines, and research maturity.",
        badge_text="Architecture Reference",
        badge_color="#3B82F6"
    )

    # Core Architecture Card
    st.markdown(
        "<div style='background:rgba(17,24,39,0.8); backdrop-filter:blur(10px); "
        "border:1px solid rgba(255,255,255,0.08); border-top:3px solid #3B82F6; "
        "border-radius:10px; padding:1.3rem; margin-bottom:1.5rem; box-shadow:0 4px 16px -2px rgba(0,0,0,0.3);'>"
        "<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>"
        "<div>"
        "<span style='font-size:0.72rem; font-weight:700; color:#3B82F6; text-transform:uppercase; letter-spacing:0.06em;'>Primary Visual Backbone</span>"
        "<div style='font-size:1.25rem; font-weight:800; color:#F8FAFC; margin-top:2px;'>"
        "X3D-M Deep Spatio-Temporal 3D Convolutional Network"
        "</div></div>"
        "<div style='background:rgba(16, 185, 129, 0.12); border:1px solid rgba(16, 185, 129, 0.35); border-radius:20px; "
        "padding:4px 12px; color:#10B981; font-weight:600; font-size:0.75rem; display:flex; align-items:center; gap:5px;'>"
        "<span style='width:6px; height:6px; border-radius:50%; background:#10B981;'></span> Active Baseline Model</div></div>"
        "<p style='color:#94A3B8; font-size:0.88rem; margin:10px 0 0 0; line-height:1.55;'>"
        "X3D-M scores overlapping 16-frame spatio-temporal video clips with a binary violence classification output. "
        "The model analyzes kinetic motion, optical gradients, and physical interactions across RGB frames."
        "</p></div>",
        unsafe_allow_html=True
    )

    # 4-Stage Pipeline
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin-bottom:0.85rem;'>End-to-End Analysis Pipeline</h4>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.markdown(
            "<div style='background:rgba(17,24,39,0.7); border:1px solid rgba(255,255,255,0.08); border-top:3px solid #3B82F6; "
            "border-radius:8px; padding:1.1rem; height:100%; box-shadow:0 2px 8px rgba(0,0,0,0.2);'>"
            "<div style='color:#60A5FA; font-weight:800; font-size:0.75rem; letter-spacing:0.05em; margin-bottom:4px;'>STAGE 01</div>"
            "<div style='color:#F8FAFC; font-weight:700; font-size:0.92rem; margin-bottom:6px;'>Temporal Resampling</div>"
            "<div style='color:#94A3B8; font-size:0.82rem; line-height:1.45;'>Extracts uniform 16-frame tensor clips with sliding window temporal stride.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            "<div style='background:rgba(17,24,39,0.7); border:1px solid rgba(255,255,255,0.08); border-top:3px solid #6366F1; "
            "border-radius:8px; padding:1.1rem; height:100%; box-shadow:0 2px 8px rgba(0,0,0,0.2);'>"
            "<div style='color:#818CF8; font-weight:800; font-size:0.75rem; letter-spacing:0.05em; margin-bottom:4px;'>STAGE 02</div>"
            "<div style='color:#F8FAFC; font-weight:700; font-size:0.92rem; margin-bottom:6px;'>3D-CNN Inference</div>"
            "<div style='color:#94A3B8; font-size:0.82rem; line-height:1.45;'>Computes binary violence candidate probabilities across sequential windows.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            "<div style='background:rgba(17,24,39,0.7); border:1px solid rgba(255,255,255,0.08); border-top:3px solid #F59E0B; "
            "border-radius:8px; padding:1.1rem; height:100%; box-shadow:0 2px 8px rgba(0,0,0,0.2);'>"
            "<div style='color:#FBBF24; font-weight:800; font-size:0.75rem; letter-spacing:0.05em; margin-bottom:4px;'>STAGE 03</div>"
            "<div style='color:#F8FAFC; font-weight:700; font-size:0.92rem; margin-bottom:6px;'>Audio & Speech Pass</div>"
            "<div style='color:#94A3B8; font-size:0.82rem; line-height:1.45;'>Transcribes dialogue via Whisper ASR and flags acoustic event candidates via AudioSet.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            "<div style='background:rgba(17,24,39,0.7); border:1px solid rgba(255,255,255,0.08); border-top:3px solid #10B981; "
            "border-radius:8px; padding:1.1rem; height:100%; box-shadow:0 2px 8px rgba(0,0,0,0.2);'>"
            "<div style='color:#34D399; font-weight:800; font-size:0.75rem; letter-spacing:0.05em; margin-bottom:4px;'>STAGE 04</div>"
            "<div style='color:#F8FAFC; font-weight:700; font-size:0.92rem; margin-bottom:6px;'>Evidence & Reports</div>"
            "<div style='color:#94A3B8; font-size:0.82rem; line-height:1.45;'>Extracts evidence keyframes, synthesizes multi-modal findings, and compiles PDF reports.</div>"
            "</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Model Engineering Matrix
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin-bottom:0.85rem;'>Architectural Experimentation & Status</h4>", unsafe_allow_html=True)

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
            f"<div style='background:rgba(17,24,39,0.7); border:1px solid rgba(255,255,255,0.08); border-radius:8px; "
            f"padding:1rem 1.15rem; margin-bottom:0.6rem;'> "
            f"<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>"
            f"<span style='font-size:0.95rem; font-weight:700; color:#F8FAFC;'>{item.get('id', '')}: {item.get('name', '')}</span>"
            f"<span style='color:{badge_color}; font-size:0.75rem; font-weight:600; padding:2px 10px; border-radius:20px; "
            f"border:1px solid {badge_color}44; background:{badge_color}14;'>"
            f"● {status_label}</span></div>"
            f"<div style='color:#94A3B8; font-size:0.83rem; margin-top:6px; line-height:1.45;'>{item.get('note', '')}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # System Capabilities Table
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin-bottom:0.85rem;'>System Capabilities & Maturity</h4>", unsafe_allow_html=True)
    caps = [
        ("X3D Binary Violence Scoring", "3D-CNN scoring over overlapping 16-frame clips", "Implemented"),
        ("Temporal Sequence Transformer", "Optional sequence transformer head; train on labeled sequences", "Checkpoint required"),
        ("Event Boundary Grouping", "Window-based candidate merging with configurable stride and gap", "Implemented"),
        ("Multilingual Speech Transcription", "Full-audio ASR with language metadata via Whisper", "Optional"),
        ("Sound-Event Candidates", "Independent acoustic classes via AudioSet AST", "Optional"),
        ("Automatic Qwen Explanations", "Vision-language second opinion via Qwen2.5-VL", "Optional"),
        ("Heuristic Review Priority", "Multi-factor rating based on duration, motion, and repetition", "Heuristic"),
        ("Automated Keyframe Evidence", "Keyframe snapshots extracted around positive intervals", "Implemented"),
        ("Human Verification Console", "Persistent audit decisions, notes, and human severity ratings", "Implemented"),
        ("PDF Report Generation", "Formal exportable report with tables and evidence summaries", "Implemented"),
    ]

    for title, desc, stat in caps:
        badge_color = "#10B981" if stat == "Implemented" else ("#F59E0B" if stat in ("Checkpoint required", "Heuristic") else "#6366F1")
        st.markdown(
            f"<div style='display:flex; justify-content:space-between; align-items:center; padding:10px 14px; "
            f"background:rgba(17,24,39,0.7); border:1px solid rgba(255,255,255,0.06); border-radius:6px; "
            f"margin-bottom:4px; flex-wrap:wrap; gap:8px;'>"
            f"<div>"
            f"<span style='color:#F8FAFC; font-weight:700; font-size:0.88rem;'>{title}</span>"
            f"<span style='color:#94A3B8; font-size:0.82rem; margin-left:8px;'>— {desc}</span>"
            f"</div>"
            f"<span style='color:{badge_color}; font-weight:600; font-size:0.75rem; background:{badge_color}14; padding:2px 8px; border-radius:12px; border:1px solid {badge_color}33;'>"
            f"● {stat}</span>"
            f"</div>",
            unsafe_allow_html=True
        )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    with st.expander("Technical Model Registry Specification", expanded=False):
        st.json(MODEL_REGISTRY)
