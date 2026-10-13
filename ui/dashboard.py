from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from database.repository import Repository
from ui.common import ft, goto, render_hero_metric, severity_badge, status_badge


def render(repo: Repository) -> None:
    # Page Header
    st.markdown(
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:1.5rem; flex-wrap:wrap; gap:1rem;'>"
        "<div>"
        "<h2 style='margin:0; font-size:1.75rem; font-weight:700; color:#F8FAFC; letter-spacing:-0.02em;'>"
        "Dashboard Overview"
        "</h2>"
        "<p style='margin:4px 0 0 0; color:#94A3B8; font-size:0.9rem;'>"
        "Summary of full-video event candidates, multimodal evidence, and human review activity."
        "</p></div>"
        "<div>"
        "<span style='display:inline-flex; align-items:center; gap:6px; background:#111827; border:1px solid rgba(255,255,255,0.08); "
        "padding:5px 12px; border-radius:6px; font-size:0.75rem; font-weight:600; color:#10B981;'>"
        "<span style='display:inline-block; width:6px; height:6px; border-radius:50%; background:#10B981;'></span>"
        "Candidate Review Pipeline</span></div></div>",
        unsafe_allow_html=True
    )

    s = repo.dashboard_stats()

    if s["total_videos"] == 0:
        st.markdown(
            "<div style='text-align:center; padding:3.5rem 2rem; background:#111827; border-radius:8px; border:1px dashed rgba(255,255,255,0.15);'>"
            "<div style='font-size:2.5rem; margin-bottom:0.5rem; color:#64748B;'>🎬</div>"
            "<h3 style='color:#F8FAFC; font-weight:600; margin-bottom:0.4rem;'>No Videos Analyzed Yet</h3>"
            "<p style='color:#94A3B8; max-width:480px; margin:0 auto 1.5rem auto; font-size:0.88rem;'>"
            "Upload a video or paste a supported link to generate visual, speech, and sound candidates across the full video."
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
    c[0].markdown(render_hero_metric("Videos Analyzed", s["total_videos"], "", "Total catalog", "#3B82F6"), unsafe_allow_html=True)
    c[1].markdown(render_hero_metric("Analysis Runs", s["total_runs"], "", f"{s['completed_runs']} completed", "#6366F1"), unsafe_allow_html=True)
    c[2].markdown(render_hero_metric("Event Candidates", s["total_events"], "", "Review intervals", "#EF4444"), unsafe_allow_html=True)
    c[3].markdown(render_hero_metric("Reviewed Events", s["reviewed_events"], "", "Human verified", "#F59E0B"), unsafe_allow_html=True)
    completion_rate = round(100 * s["completed_runs"] / s["total_runs"]) if s["total_runs"] else 100
    c[4].markdown(render_hero_metric("Pipeline Success", f"{completion_rate}%", "", "Completion rate", "#10B981"), unsafe_allow_html=True)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Analytics Charts
    col_left, col_right = st.columns([1, 1])
    with col_left:
        st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600; margin-bottom:0.75rem;'>Heuristic Review Priority Breakdown</h4>", unsafe_allow_html=True)
        if s["severity"] and any(s["severity"].values()):
            labels = list(s["severity"].keys())
            values = list(s["severity"].values())
            palette = {"High": "#EF4444", "Moderate": "#F97316", "Mild": "#F59E0B", "Not assessed": "#64748B"}
            colors = [palette.get(k, "#3B82F6") for k in labels]

            fig_donut = go.Figure(data=[go.Pie(
                labels=labels, values=values, hole=0.62,
                marker=dict(colors=colors, line=dict(color='#0B0F19', width=2)),
                textinfo='label+percent', hoverinfo='label+value+percent'
            )])
            fig_donut.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94A3B8", family="Inter, -apple-system, sans-serif", size=11),
                margin=dict(l=10, r=10, t=10, b=10),
                height=240,
                showlegend=False
            )
            st.plotly_chart(fig_donut, use_container_width=True)
        else:
            st.caption("No event candidates recorded in completed runs.")

    with col_right:
        st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600; margin-bottom:0.75rem;'>Candidate Types</h4>", unsafe_allow_html=True)
        if s["categories"] and any(s["categories"].values()):
            df_cat = pd.DataFrame(list(s["categories"].items()), columns=["Category", "Count"])
            fig_bar = px.bar(
                df_cat, x="Category", y="Count", text="Count",
                color="Category", color_discrete_sequence=["#3B82F6", "#6366F1", "#10B981"]
            )
            fig_bar.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color="#94A3B8", family="Inter, -apple-system, sans-serif", size=11),
                margin=dict(l=10, r=10, t=10, b=10),
                height=240,
                showlegend=False,
                xaxis=dict(gridcolor="rgba(255,255,255,0.06)", title=""),
                yaxis=dict(gridcolor="rgba(255,255,255,0.06)", title="Count")
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.caption("Awaiting video detections.")

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Recent Video Audits
    st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600; margin-bottom:0.75rem;'>Recent Video Analyses</h4>", unsafe_allow_html=True)
    runs = repo.list_runs()[:10]
    if runs:
        for r in runs:
            with st.container():
                c_info, c_stats, c_btn = st.columns([5, 3, 2])
                with c_info:
                    st.markdown(f"**{r['original_name']}** · Run #{r['id']}")
                    st.markdown(
                        f"<div style='margin-top:2px;'>{status_badge(r['status'])} "
                        f"<span style='color:#64748B; font-size:0.8rem; margin-left:6px;'>{r['started_at'][:19]}</span></div>",
                        unsafe_allow_html=True
                    )
                with c_stats:
                    proc = f"{round(r['elapsed_s'], 1)}s" if r["elapsed_s"] else "—"
                    st.caption(f"Duration: {ft(r['duration_s'])} | Latency: {proc}")
                    st.caption(f"Windows: {r['windows_done']}/{r['windows_total']}")
                with c_btn:
                    st.button("Open Report", key=f"dash_open_{r['id']}", on_click=goto,
                              args=("Report",), kwargs={"selected_run_id": r["id"]}, use_container_width=True)
                st.markdown("<hr style='margin:0.5rem 0; border:0; border-top:1px solid rgba(255,255,255,0.06);'>", unsafe_allow_html=True)
    else:
        st.caption("No analysis runs recorded yet.")

    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # Chronological Incident Feed
    st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600; margin-bottom:0.75rem;'>Recent Unconfirmed Candidates</h4>", unsafe_allow_html=True)
    ev = repo.recent_events(10)
    if ev:
        for e in ev:
            with st.container():
                c_ev_title, c_ev_time, c_ev_prob, c_ev_sev = st.columns([4, 3, 2, 2])
                with c_ev_title:
                    st.markdown(f"**Event #{e['event_no']}** ({e['original_name']})")
                with c_ev_time:
                    st.caption(f"Interval: {ft(e['start_s'])} → {ft(e['end_s'])} ({e['duration_s']:.1f}s)")
                with c_ev_prob:
                    conf_pct = int(round(e['peak_prob'] * 100))
                    conf_color = "#EF4444" if conf_pct > 70 else "#F97316" if conf_pct > 40 else "#10B981"
                    st.markdown(f"<span style='color:{conf_color}; font-weight:600; font-size:0.88rem;'>X3D class score {conf_pct}%</span>", unsafe_allow_html=True)
                with c_ev_sev:
                    st.markdown(severity_badge(e['severity_label']), unsafe_allow_html=True)
                st.markdown("<hr style='margin:0.35rem 0; border:0; border-top:1px solid rgba(255,255,255,0.04);'>", unsafe_allow_html=True)
    else:
        st.caption("No X3D visual candidates were recorded in the completed runs. This does not verify that those videos are violence-free.")
