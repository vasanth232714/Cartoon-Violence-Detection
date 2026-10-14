from __future__ import annotations

from pathlib import Path

import streamlit as st

from database.repository import Repository
from ui.common import STATUS_HELP, ft, goto, render_page_header, status_badge


def render(repo: Repository) -> None:
    render_page_header(
        title="Video Analysis History",
        subtitle="Catalog of previously analyzed animation videos, detection sessions, and persistent audit reports.",
        badge_text="Audit Archive",
        badge_color="#6366F1"
    )

    repo.refresh_file_availability()
    runs = repo.list_runs()

    if not runs:
        st.markdown(
            "<div style='text-align:center; padding:3.5rem 2rem; background:rgba(17,24,39,0.7); "
            "backdrop-filter:blur(10px); border-radius:12px; border:1px dashed rgba(255,255,255,0.15); margin-top:1rem;'>"
            "<div style='width:56px; height:56px; border-radius:14px; background:rgba(99,102,241,0.12); "
            "border:1px solid rgba(99,102,241,0.25); display:inline-flex; align-items:center; justify-content:center; "
            "font-size:1.8rem; margin-bottom:1rem;'>📁</div>"
            "<h3 style='color:#F8FAFC; font-weight:700; margin-bottom:0.4rem; font-size:1.25rem;'>No Video History Yet</h3>"
            "<p style='color:#94A3B8; max-width:480px; margin:0 auto 1.5rem auto; font-size:0.9rem; line-height:1.5;'>"
            "Analyze a cartoon video to generate and record your first detection report."
            "</p></div>",
            unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns([1, 2, 1])
        with c2:
            st.button("⚡ Start Analyzing Video", type="primary", use_container_width=True,
                      on_click=goto, args=("Upload & Analyze",))
        return

    # Bulk actions expander
    with st.expander("⚙️ Manage & Clear Archives", expanded=False):
        c_m1, c_m2 = st.columns([3, 1.5])
        with c_m1:
            st.caption("Permanently delete all stored analysis reports, evidence frames, and review history from the database.")
            confirm_all = st.checkbox("Confirm delete all reports", key="conf_clear_all_runs")
        with c_m2:
            if st.button("🗑️ Clear All Reports", type="primary", disabled=not confirm_all, use_container_width=True):
                n_deleted = repo.clear_all_history()
                st.session_state.pop("selected_run_id", None)
                st.success(f"Cleared {n_deleted} reports from history.")
                st.rerun()

    # Search & Filter Bar
    c_search, c_filt = st.columns([3, 2])
    with c_search:
        search_query = st.text_input("Search Videos by Filename", placeholder="Filter by video name...")
    with c_filt:
        status_filter = st.selectbox("Filter by Status", ["All Statuses", "Completed", "Warnings", "Failed / Interrupted"])

    filtered_runs = runs
    if search_query:
        filtered_runs = [r for r in filtered_runs if search_query.lower() in r["original_name"].lower()]
    if status_filter == "Completed":
        filtered_runs = [r for r in filtered_runs if r["status"] == "completed"]
    elif status_filter == "Warnings":
        filtered_runs = [r for r in filtered_runs if "warning" in r["status"]]
    elif status_filter == "Failed / Interrupted":
        filtered_runs = [r for r in filtered_runs if r["status"] in ("failed", "interrupted", "incomplete")]

    st.markdown(
        f"<div style='color:#94A3B8; font-size:0.8rem; margin:0.5rem 0 1rem 0;'>"
        f"Showing <b>{len(filtered_runs)}</b> of {len(runs)} recorded sessions</div>",
        unsafe_allow_html=True
    )

    for r in filtered_runs:
        elapsed = "—" if r["elapsed_s"] is None else f"{r['elapsed_s']:.1f}s"
        avail = "● Video Cached" if r["file_available"] else "○ Video Removed (Data Kept)"
        avail_color = "#10B981" if r["file_available"] else "#64748B"

        with st.container():
            st.markdown(
                f"<div style='background:rgba(17,24,39,0.75); backdrop-filter:blur(8px); "
                f"border:1px solid rgba(255,255,255,0.08); border-left:3px solid #3B82F6; "
                f"border-radius:9px; padding:1.15rem 1.25rem; margin-bottom:0.75rem; box-shadow:0 2px 8px rgba(0,0,0,0.25);'>"
                f"<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;'>"
                f"<div>"
                f"<span style='font-size:1.05rem; font-weight:700; color:#F8FAFC;'>{r['original_name']}</span>"
                f"<span style='color:#64748B; font-size:0.78rem; margin-left:8px; font-weight:600;'>#Run {r['id']}</span>"
                f"</div>"
                f"<div style='font-size:0.78rem; color:#94A3B8;'>{r['started_at'][:19]}</div>"
                f"</div>"
                f"<div style='display:flex; gap:16px; align-items:center; margin-top:8px; flex-wrap:wrap; font-size:0.82rem; color:#CBD5E1;'>"
                f"<span>Duration: <b>{ft(r['duration_s'])}</b></span>"
                f"<span>Compute Latency: <b>{elapsed}</b></span>"
                f"<span>Windows: <b>{r['windows_done']}/{r['windows_total']}</b></span>"
                f"<span style='color:{avail_color}; font-weight:600;'>{avail}</span>"
                f"</div></div>",
                unsafe_allow_html=True
            )

            c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
            with c1:
                st.markdown(status_badge(r["status"]), unsafe_allow_html=True)
                if r["error"]:
                    st.caption(f"Note: {r['error']}")

            with c2:
                st.button("Open Report →", key=f"hist_open_{r['id']}", on_click=goto,
                          args=("Report",), kwargs={"selected_run_id": r["id"]}, use_container_width=True)

            with c3:
                if r["file_available"]:
                    if st.button("Purge Video", key=f"hist_del_{r['id']}",
                                 help="Deletes local video file to free up storage space while retaining report metrics",
                                 use_container_width=True):
                        v = repo.get_video(r["video_id"])
                        try:
                            Path(v["stored_path"]).unlink(missing_ok=True)
                            repo.set_video_file(v["id"], v["stored_path"], False)
                            st.rerun()
                        except OSError as e:
                            st.error(f"Could not delete file: {e}")
                else:
                    st.caption("Cached file removed")

            with c4:
                if st.button("🗑️ Delete Run", key=f"hist_del_run_{r['id']}",
                             help="Permanently delete this analysis report and associated evidence from the database",
                             use_container_width=True):
                    repo.delete_run(r["id"])
                    if st.session_state.get("selected_run_id") == r["id"]:
                        st.session_state.pop("selected_run_id", None)
                    st.success(f"Report for Run #{r['id']} was deleted.")
                    st.rerun()

            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
