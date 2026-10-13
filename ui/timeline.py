"""Interactive Plotly timeline: window probabilities, threshold line, and event spans."""
from __future__ import annotations

import plotly.graph_objects as go

from ui.common import ft

SEV_COLORS = {
    "Mild": "#F59E0B",
    "Moderate": "#F97316",
    "High": "#EF4444",
    "Not assessed": "#64748B"
}


def build_timeline_figure(windows, events, threshold: float, duration_s: float, selected_event_no: int | None = None):
    fig = go.Figure()
    xs, ys, hov = [], [], []
    for w in windows:
        if w["p_violent"] is None:
            continue
        xs.append((w["start_s"] + w["end_s"]) / 2)
        ys.append(w["p_violent"])
        hov.append(f"Interval: {ft(w['start_s'])} - {ft(w['end_s'])}<br>X3D class score: {w['p_violent']:.3f} (uncalibrated)")

    # Professional clean blue line
    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="lines+markers", name="Window Probability",
        text=hov, hoverinfo="text",
        line=dict(width=2.5, color="#3B82F6"),
        marker=dict(size=5, color="#60A5FA", line=dict(width=1, color="#1E3A8A"))
    ))

    # Threshold marker
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#94A3B8", line_width=1.5,
        annotation_text=f"Threshold ({threshold:.2f})",
        annotation_font=dict(color="#94A3B8", size=11),
        annotation_position="bottom right"
    )

    from ui.common import jload
    for e in events:
        sel = selected_event_no == e["event_no"]
        color = SEV_COLORS.get(e["severity_label"], "#64748B")
        sj_raw = e["severity_json"] if "severity_json" in e.keys() else "{}"
        sj = jload(sj_raw, {}) or {}
        lbl = str(e["label"]) if "label" in e.keys() else ""
        has_verbal = bool(sj.get("verbal_abuse")) or "Verbal" in lbl
        has_audio = bool(sj.get("audio_events")) or "Audio" in lbl
        tag = f"Event {e['event_no']}" + (" 🗣️" if has_verbal else "") + (" 🔊" if has_audio else "")

        fig.add_vrect(
            x0=e["start_s"], x1=e["end_s"],
            fillcolor=color,
            opacity=0.35 if sel else 0.12,
            line_width=2 if sel else (1.5 if has_verbal else 0),
            line_color=color if sel else ("#A855F7" if has_verbal else None),
            line_dash="dot" if (has_verbal and not sel) else "solid",
            annotation_text=tag,
            annotation_position="top left",
            annotation_font=dict(color=color if sel else "#94A3B8", size=11, family="Inter, sans-serif")
        )

    bad = [w for w in windows if w["p_violent"] is None]
    if bad:
        fig.add_trace(go.Scatter(
            x=[(w["start_s"] + w["end_s"]) / 2 for w in bad],
            y=[0] * len(bad),
            mode="markers",
            marker=dict(symbol="x", size=8, color="#EF4444"),
            name="Skipped Window"
        ))

    fig.update_layout(
        height=320,
        margin=dict(l=10, r=10, t=25, b=10),
        xaxis_title="Video Timeline (seconds)",
        yaxis_title="X3D class score (not calibrated confidence)",
        yaxis_range=[0, 1.05],
        xaxis_range=[0, duration_s],
        legend=dict(orientation="h", y=-0.25, font=dict(color="#94A3B8", size=11)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#94A3B8", family="Inter, -apple-system, sans-serif"),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.12)",
            tickfont=dict(color="#94A3B8")
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            zerolinecolor="rgba(255,255,255,0.12)",
            tickfont=dict(color="#94A3B8")
        )
    )
    return fig
