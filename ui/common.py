from __future__ import annotations

import json
import streamlit as st

from preprocessing.timestamp_utils import format_timestamp

SEVERITY_ORDER = ["Mild", "Moderate", "High", "Not assessed"]
STATUS_HELP = {
    "completed": "All windows analysed successfully across all stages.",
    "completed_with_warnings": "Completed with model, audio, frame, or explanation warnings; inspect run details.",
    "incomplete": "Some windows failed or were skipped - results are NOT a full analysis.",
    "failed": "Analysis pipeline encountered a fatal error - partial data only.",
    "interrupted": "Analysis was cancelled or interrupted - partial data only.",
    "running": "Analysis in progress.",
}


def ft(x) -> str:
    """Format seconds into HH:MM:SS or MM:SS."""
    return format_timestamp(x)


def status_badge(status: str) -> str:
    """Return a modern glassmorphic status badge with pulsing or crisp status indicator dot."""
    cfg = {
        "completed": ("#10B981", "rgba(16, 185, 129, 0.12)", "rgba(16, 185, 129, 0.35)", "Completed"),
        "completed_with_warnings": ("#F59E0B", "rgba(245, 158, 11, 0.12)", "rgba(245, 158, 11, 0.35)", "Warnings"),
        "incomplete": ("#EF4444", "rgba(239, 68, 68, 0.12)", "rgba(239, 68, 68, 0.35)", "Incomplete"),
        "failed": ("#EF4444", "rgba(239, 68, 68, 0.12)", "rgba(239, 68, 68, 0.35)", "Failed"),
        "interrupted": ("#94A3B8", "rgba(148, 163, 184, 0.12)", "rgba(148, 163, 184, 0.3)", "Interrupted"),
        "running": ("#3B82F6", "rgba(59, 130, 246, 0.15)", "rgba(59, 130, 246, 0.4)", "Running"),
    }.get(status, ("#94A3B8", "rgba(255, 255, 255, 0.06)", "rgba(255, 255, 255, 0.15)", status.replace("_", " ").title()))
    color, bg, border, label = cfg

    pulse_anim = "animation: cvd-pulse 1.8s infinite ease-in-out;" if status == "running" else ""
    return (
        f"<span style='display:inline-flex; align-items:center; gap:6px; padding:3px 10px; "
        f"border-radius:20px; font-size:0.75rem; font-weight:600; color:{color}; "
        f"background:{bg}; border:1px solid {border}; letter-spacing:0.02em; white-space:nowrap; "
        f"box-shadow:0 1px 4px rgba(0,0,0,0.2);'>"
        f"<span style='display:inline-block; width:6px; height:6px; border-radius:50%; background:{color}; {pulse_anim}'></span>"
        f"<span>{label}</span></span>"
    )


def severity_badge(severity: str) -> str:
    """Return a crisp priority/severity pill badge."""
    cfg = {
        "High": ("High Review Priority", "#EF4444", "rgba(239, 68, 68, 0.12)", "rgba(239, 68, 68, 0.35)"),
        "Moderate": ("Moderate Review Priority", "#F97316", "rgba(249, 115, 22, 0.12)", "rgba(249, 115, 22, 0.35)"),
        "Mild": ("Mild Review Priority", "#F59E0B", "rgba(245, 158, 11, 0.12)", "rgba(245, 158, 11, 0.35)"),
        "Not assessed": ("Unclassified", "#64748B", "rgba(100, 116, 139, 0.12)", "rgba(100, 116, 139, 0.25)"),
    }.get(severity, (severity, "#64748B", "rgba(100, 116, 139, 0.12)", "rgba(100, 116, 139, 0.25)"))
    label, color, bg, border = cfg
    return (
        f"<span style='display:inline-flex; align-items:center; gap:5px; padding:3px 10px; "
        f"border-radius:20px; font-size:0.75rem; font-weight:600; color:{color}; "
        f"background:{bg}; border:1px solid {border}; letter-spacing:0.02em; white-space:nowrap;'>"
        f"<span style='display:inline-block; width:6px; height:6px; border-radius:50%; background:{color};'></span>"
        f"<span>{label}</span></span>"
    )


def render_hero_metric(label: str, value: str | int | float, icon: str = "", subtext: str = "", border_color: str = "#3B82F6") -> str:
    """Render a premium executive KPI metric card with glassmorphism and gradient accents."""
    sub_html = (
        f"<div style='font-size:0.75rem; color:#94A3B8; margin-top:0.35rem; font-weight:500; "
        f"display:flex; align-items:center; gap:4px;'>"
        f"<span style='display:inline-block; width:4px; height:4px; border-radius:50%; background:{border_color};'></span>"
        f"<span>{subtext}</span></div>"
    ) if subtext else ""

    icon_html = (
        f"<div style='width:28px; height:28px; border-radius:7px; background:rgba(255,255,255,0.04); "
        f"border:1px solid rgba(255,255,255,0.08); display:flex; align-items:center; justify-content:center; "
        f"font-size:0.95rem;'>{icon}</div>"
    ) if icon else ""

    return (
        f"<div style='background:linear-gradient(180deg, rgba(17, 24, 39, 0.85) 0%, rgba(15, 23, 42, 0.75) 100%); "
        f"backdrop-filter:blur(10px); -webkit-backdrop-filter:blur(10px); "
        f"border:1px solid rgba(255,255,255,0.08); border-top:3px solid {border_color}; "
        f"border-radius:10px; padding:1rem 1.15rem; min-height:102px; display:flex; flex-direction:column; "
        f"justify-content:space-between; box-shadow:0 4px 15px -3px rgba(0,0,0,0.3); transition:transform 0.15s ease;'>"
        f"<div style='display:flex; justify-content:space-between; align-items:center;'>"
        f"<span style='font-size:0.72rem; text-transform:uppercase; letter-spacing:0.07em; color:#94A3B8; font-weight:600;'>{label}</span>"
        f"{icon_html}</div>"
        f"<div style='font-size:1.85rem; font-weight:700; color:#F8FAFC; margin-top:0.25rem; "
        f"letter-spacing:-0.03em; font-feature-settings:\"tnum\" on, \"zero\" on;'>{value}</div>"
        f"{sub_html}</div>"
    )


def render_page_header(title: str, subtitle: str, badge_text: str | None = None, badge_color: str = "#10B981") -> None:
    """Render a unified enterprise page header banner."""
    badge_html = ""
    if badge_text:
        badge_html = (
            f"<div>"
            f"<span style='display:inline-flex; align-items:center; gap:6px; background:rgba(17,24,39,0.7); "
            f"border:1px solid rgba(255,255,255,0.08); padding:5px 12px; border-radius:20px; "
            f"font-size:0.75rem; font-weight:600; color:{badge_color}; box-shadow:0 1px 3px rgba(0,0,0,0.2);'>"
            f"<span style='display:inline-block; width:6px; height:6px; border-radius:50%; background:{badge_color};'></span>"
            f"{badge_text}</span></div>"
        )
    st.markdown(
        f"<div style='display:flex; justify-content:space-between; align-items:center; "
        f"margin-bottom:1.5rem; flex-wrap:wrap; gap:1rem; padding-bottom:1rem; border-bottom:1px solid rgba(255,255,255,0.06);'>"
        f"<div>"
        f"<h2 style='margin:0; font-size:1.75rem; font-weight:700; color:#F8FAFC; letter-spacing:-0.02em; "
        f"line-height:1.2;'>"
        f"{title}</h2>"
        f"<p style='margin:6px 0 0 0; color:#94A3B8; font-size:0.88rem; line-height:1.45; max-width:800px;'>"
        f"{subtitle}</p></div>"
        f"{badge_html}</div>",
        unsafe_allow_html=True
    )


def render_card_box(content_html: str, border_top_color: str | None = None) -> str:
    """Wrap content in a dark glass card container."""
    top_style = f"border-top: 3px solid {border_top_color};" if border_top_color else ""
    return (
        f"<div style='background:rgba(17, 24, 39, 0.75); backdrop-filter:blur(10px); "
        f"border:1px solid rgba(255,255,255,0.08); {top_style} border-radius:10px; "
        f"padding:1.15rem 1.25rem; margin-bottom:1rem; box-shadow:0 4px 16px -2px rgba(0,0,0,0.3);'>"
        f"{content_html}</div>"
    )


def goto(page: str, **state) -> None:
    """Navigate from a callback/handler: applied before the sidebar radio is created on next run."""
    st.session_state["nav_target"] = page
    for k, v in state.items():
        st.session_state[k] = v


def jload(s, default=None):
    """Safely load JSON string or return default."""
    if isinstance(s, (dict, list)):
        return s
    try:
        return json.loads(s) if s else default
    except Exception:
        return default
