from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database.repository import Repository
from ui.common import ft, goto, render_hero_metric, render_page_header, severity_badge, status_badge


def render(repo: Repository) -> None:
    render_page_header(
        title="Dashboard Overview",
        subtitle="Real-time summary of animated video scans, multimodal candidates, and human verification audit logs.",
        badge_text="Candidate Review Active",
        badge_color="#10B981"
    )

    s = repo.dashboard_stats()

    if s["total_videos"] == 0:
        st.markdown(
            "<div style='text-align:center; padding:3.5rem 2rem; background:rgba(17,24,39,0.7); "
            "backdrop-filter:blur(10px); border-radius:12px; border:1px dashed rgba(255,255,255,0.15); margin-top:1rem;'>"
            "<div style='width:56px; height:56px; border-radius:14px; background:rgba(59,130,246,0.12); "
            "border:1px solid rgba(59,130,246,0.25); display:inline-flex; align-items:center; justify-content:center; "
            "font-size:1.8rem; margin-bottom:1rem;'>🎬</div>"
            "<h3 style='color:#F8FAFC; font-weight:700; margin-bottom:0.4rem; font-size:1.25rem;'>No Videos Analyzed Yet</h3>"
            "<p style='color:#94A3B8; max-width:500px; margin:0 auto 1.5rem auto; font-size:0.9rem; line-height:1.5;'>"
            "Upload an animation file or paste a YouTube / web video link to generate visual, speech, and sound candidates across the full video."
            "</p></div>",
            unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.button("⚡ Start Analyzing Video", type="primary", use_container_width=True,
                      on_click=goto, args=("Upload & Analyze",))
        return

    # Metric Cards
    c = st.columns(5)
    c[0].markdown(render_hero_metric("Videos Scanned", s["total_videos"], "🎬", "Total catalog", "#3B82F6"), unsafe_allow_html=True)
    c[1].markdown(render_hero_metric("Analysis Runs", s["total_runs"], "⚡", f"{s['completed_runs']} completed", "#6366F1"), unsafe_allow_html=True)
    c[2].markdown(render_hero_metric("Event Candidates", s["total_events"], "⚠️", "Review intervals", "#EF4444"), unsafe_allow_html=True)
    c[3].markdown(render_hero_metric("Human Reviewed", s["reviewed_events"], "👁️", "Verified scenes", "#F59E0B"), unsafe_allow_html=True)
    completion_rate = round(100 * s["completed_runs"] / s["total_runs"]) if s["total_runs"] else 100
    c[4].markdown(render_hero_metric("Pipeline Success", f"{completion_rate}%", "🛡️", "Run completion rate", "#10B981"), unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Analytics Charts
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown(
            "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;'>"
            "<h4 style='color:#F8FAFC; font-size:1rem; font-weight:700; margin:0;'>Heuristic Review Priority</h4>"
            "<span style='font-size:0.72rem; color:#64748B; text-transform:uppercase; letter-spacing:0.04em;'>Severity Distribution</span>"
            "</div>",
            unsafe_allow_html=True
        )
        if s["severity"] and any(s["severity"].values()):
            labels = list(s["severity"].keys())
            values = list(s["severity"].values())
            palette = {"High": "#EF4444", "Moderate": "#F97316", "Mild": "#F59E0B", "Not assessed": "#64748B"}
            colors = [palette.get(k, "#3B82F6") for k in labels]

            fig_donut = go.Figure(data=[go.Pie(
                labels=labels, values=values, hole=0.65,
                marker=dict(colors=colors, line=dict(color='#0A0E17', width=2)),
                textinfo='label+percent', hoverinfo='label+value+percent'
            )])
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94A3B8", family="Plus Jakarta Sans, Inter, sans-serif", size=11),
                margin=dict(l=10, r=10, t=10, b=10),
                height=250,
                showlegend=False
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.info("No event candidates recorded in completed runs.")

    with col_right:
        st.markdown(
            "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;'>"
            "<h4 style='color:#F8FAFC; font-size:1rem; font-weight:700; margin:0;'>Candidate Modalities</h4>"
            "<span style='font-size:0.72rem; color:#64748B; text-transform:uppercase; letter-spacing:0.04em;'>Detection Spectrum</span>"
            "</div>",
            unsafe_allow_html=True
        )
        if s["categories"] and any(s["categories"].values()):
            df_cat = pd.DataFrame(list(s["categories"].items()), columns=["Category", "Count"])
            fig_bar = px.bar(
                df_cat, x="Category", y="Count", text="Count",
                color="Category", color_discrete_sequence=["#3B82F6", "#6366F1", "#10B981", "#F59E0B"]
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94A3B8", family="Plus Jakarta Sans, Inter, sans-serif", size=11),
                margin=dict(l=10, r=10, t=10, b=10),
                height=250,
                showlegend=False,
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=""),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Total Incidents")
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Awaiting video detections to populate modalities.")

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Recent Video Audits
    st.markdown(
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.85rem;'>"
        "<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin:0;'>Recent Video Analyses</h4>"
        "<span style='font-size:0.75rem; color:#64748B;'>Last 10 sessions</span>"
        "</div>",
        unsafe_allow_html=True
    )
    runs = repo.list_runs()[:10]
    if runs:
        for r in runs:
            with st.container():
                c_info, c_stats, c_btn = st.columns([5, 3, 2])
                with c_info:
                    st.markdown(
                        f"<div style='font-weight:700; color:#F8FAFC; font-size:0.95rem; line-height:1.3;'>"
                        f"{r['original_name']} "
                        f"<span style='color:#64748B; font-size:0.75rem; font-weight:500;'>#Run {r['id']}</span></div>",
                        unsafe_allow_html=True
                    )
                    st.markdown(
                        f"<div style='margin-top:4px; display:flex; align-items:center; gap:8px;'>"
                        f"{status_badge(r['status'])} "
                        f"<span style='color:#64748B; font-size:0.78rem;'>{r['started_at'][:19]}</span></div>",
                        unsafe_allow_html=True
                    )
                with c_stats:
                    proc = f"{round(r['elapsed_s'], 1)}s" if r["elapsed_s"] else "—"
                    st.markdown(
                        f"<div style='font-size:0.82rem; color:#CBD5E1; margin-top:2px;'>"
                        f"Duration: <b>{ft(r['duration_s'])}</b> &nbsp;|&nbsp; Latency: <b>{proc}</b><br>"
                        f"<span style='color:#94A3B8; font-size:0.76rem;'>Windows: {r['windows_done']} / {r['windows_total']} processed</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )
                with c_btn:
                    st.button("Open Report →", key=f"dash_open_{r['id']}", on_click=goto,
                              args=("Report",), kwargs={"selected_run_id": r["id"]}, use_container_width=True)
                st.markdown("<hr style='margin:0.6rem 0; border:0; border-top:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
    else:
        st.info("No analysis runs recorded yet.")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Chronological Incident Feed
    st.markdown(
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.85rem;'>"
        "<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin:0;'>Recent Unconfirmed Candidates</h4>"
        "<span style='font-size:0.75rem; color:#64748B;'>Pending human verification</span>"
        "</div>",
        unsafe_allow_html=True
    )
    ev = repo.recent_events(10)
    if ev:
        for e in ev:
            with st.container():
                c_ev_title, c_ev_time, c_ev_prob, c_ev_sev = st.columns([4, 3, 2.5, 2.5])
                with c_ev_title:
                    st.markdown(
                        f"<div style='font-size:0.9rem; font-weight:600; color:#F8FAFC;'>"
                        f"Event #{e['event_no']} "
                        f"<span style='font-size:0.78rem; color:#94A3B8; font-weight:400;'>({e['original_name']})</span></div>",
                        unsafe_allow_html=True
                    )
                with c_ev_time:
                    st.markdown(
                        f"<span style='font-size:0.8rem; color:#CBD5E1;'>"
                        f"{ft(e['start_s'])} → {ft(e['end_s'])} "
                        f"<span style='color:#64748B;'>({e['duration_s']:.1f}s)</span></span>",
                        unsafe_allow_html=True
                    )
                with c_ev_prob:
                    conf_pct = int(round(e['peak_prob'] * 100))
                    conf_color = "#EF4444" if conf_pct > 70 else "#F97316" if conf_pct > 40 else "#10B981"
                    st.markdown(
                        f"<span style='display:inline-flex; align-items:center; gap:5px; font-weight:600; font-size:0.82rem; color:{conf_color};'>"
                        f"<span style='width:6px; height:6px; border-radius:50%; background:{conf_color};'></span>"
                        f"Score {conf_pct}%</span>",
                        unsafe_allow_html=True
                    )
                with c_ev_sev:
                    st.markdown(severity_badge(e['severity_label']), unsafe_allow_html=True)
                st.markdown("<hr style='margin:0.4rem 0; border:0; border-top:1px solid rgba(255,255,255,0.04);'>", unsafe_allow_html=True)
    else:
        st.caption("No X3D visual candidates were recorded in the completed runs. This does not verify that those videos are violence-free.")
