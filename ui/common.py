from __future__ import annotations

import json

import streamlit as st

from preprocessing.timestamp_utils import format_timestamp

SEVERITY_ORDER = ["Mild", "Moderate", "High", "Not assessed"]
STATUS_HELP = {
    "completed": "All windows analysed.",
    "completed_with_warnings": "Completed with model, audio, frame, or explanation warnings; inspect the run details.",
    "incomplete": "Some windows failed or were skipped - results are NOT a full analysis.",
    "failed": "Analysis failed - partial data only.",
    "interrupted": "Analysis was interrupted - partial data only.",
    "running": "Analysis in progress.",
}


def ft(x) -> str:
    return format_timestamp(x)


def status_badge(status: str) -> str:
    cfg = {
        "completed": ("#10B981", "rgba(16, 185, 129, 0.12)", "Completed"),
        "completed_with_warnings": ("#F59E0B", "rgba(245, 158, 11, 0.12)", "Warnings"),
        "incomplete": ("#EF4444", "rgba(239, 68, 68, 0.12)", "Incomplete"),
        "failed": ("#EF4444", "rgba(239, 68, 68, 0.12)", "Failed"),
        "interrupted": ("#94A3B8", "rgba(148, 163, 184, 0.12)", "Interrupted"),
        "running": ("#3B82F6", "rgba(59, 130, 246, 0.12)", "Running"),
    }.get(status, ("#94A3B8", "rgba(255, 255, 255, 0.06)", status.title()))
    color, bg, label = cfg
    return (
        f"<span style='display:inline-flex; align-items:center; gap:6px; padding:3px 10px; "
        f"border-radius:6px; font-size:0.75rem; font-weight:600; color:{color}; "
        f"background:{bg}; border:1px solid {color}33; letter-spacing:0.02em; white-space:nowrap;'>"
        f"<span style='display:inline-block; width:6px; height:6px; border-radius:50%; background:{color};'></span>"
        f"<span>{label}</span></span>"
    )


def severity_badge(severity: str) -> str:
    cfg = {
        "High": ("High review priority", "#EF4444", "rgba(239, 68, 68, 0.12)"),
        "Moderate": ("Moderate review priority", "#F97316", "rgba(249, 115, 22, 0.12)"),
        "Mild": ("Mild review priority", "#F59E0B", "rgba(245, 158, 11, 0.12)"),
        "Not assessed": ("Unclassified", "#64748B", "rgba(100, 116, 139, 0.12)"),
    }.get(severity, (severity, "#64748B", "rgba(100, 116, 139, 0.12)"))
    label, color, bg = cfg
    return (
        f"<span style='display:inline-flex; align-items:center; gap:5px; padding:3px 10px; "
        f"border-radius:6px; font-size:0.75rem; font-weight:600; color:{color}; "
        f"background:{bg}; border:1px solid {color}40; letter-spacing:0.02em; white-space:nowrap;'>"
        f"<span style='display:inline-block; width:6px; height:6px; border-radius:50%; background:{color};'></span>"
        f"<span>{label}</span></span>"
    )


def render_hero_metric(label: str, value: str | int | float, icon: str = "", subtext: str = "", border_color: str = "#3B82F6") -> str:
    """Render a clean, responsive metric card as a single-line HTML string with no multiline indentation."""
    sub_html = f"<div style='font-size:0.75rem; color:#64748B; margin-top:0.25rem; font-weight:500;'>{subtext}</div>" if subtext else ""
    icon_html = f"<span style='font-size:1.05rem;'>{icon}</span>" if icon else ""
    return (
        f"<div style='background:#111827; border:1px solid rgba(255,255,255,0.08); "
        f"border-top:3px solid {border_color}; border-radius:8px; padding:0.9rem 1.1rem; "
        f"display:flex; flex-direction:column; justify-content:space-between; min-height:96px;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
        f"<span style='font-size:0.75rem; text-transform:uppercase; letter-spacing:0.06em; color:#94A3B8; font-weight:600;'>{label}</span>"
        f"{icon_html}</div>"
        f"<div style='font-size:1.75rem; font-weight:700; color:#F8FAFC; margin-top:0.35rem; letter-spacing:-0.02em;'>{value}</div>"
        f"{sub_html}</div>"
    )


def goto(page: str, **state) -> None:
    """Navigate from a callback/handler: applied before the sidebar radio is created on next run."""
    st.session_state["nav_target"] = page
    for k, v in state.items():
        st.session_state[k] = v


def jload(s, default=None):
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s) if s else default
    except Exception:
        return default
