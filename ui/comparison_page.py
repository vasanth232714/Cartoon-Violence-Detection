"""UI: Multi-Model Analysis and Comparison Page.

Renders architectural overviews, research benchmark comparison tables,
interactive multi-model timeline overlays, pairwise agreement & correlation matrices,
event-level detection differences, and intelligent model selection recommendations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config.settings import DEFAULT_MODEL_KEY, MODEL_REGISTRY, AnalysisConfig
from database.repository import Repository
from models.model_loader import load_classifier
from pipeline.comparison_pipeline import (BENCHMARK_REPORTS, ComparisonResult,
                                          run_comparison)
from preprocessing.timestamp_utils import format_timestamp
from preprocessing.video_loader import VideoMetadata
from ui.common import jload, render_hero_metric


def _render_model_cards() -> None:
    """Render the 3 model architecture cards with dark theme styling."""
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            """
            <div style="background:#111827; border:1px solid rgba(255,255,255,0.08); border-top:3px solid #3B82F6;
                        border-radius:8px; padding:1.2rem; height:100%; display:flex; flex-direction:column; justify-content:space-between;">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:0.75rem; font-weight:700; color:#3B82F6; text-transform:uppercase; letter-spacing:0.04em;">Model 1 · Baseline</span>
                        <span style="font-size:0.72rem; padding:2px 8px; border-radius:4px; background:rgba(59,130,246,0.15); color:#60A5FA; font-weight:600;">Active 3D-CNN</span>
                    </div>
                    <div style="font-size:1.05rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;">X3D-M Backbone</div>
                    <div style="font-size:0.83rem; color:#94A3B8; line-height:1.45; margin-bottom:12px;">
                        Spatiotemporal 3D Convolutional Network performing dense 16-frame tensor feature extraction. Captures rapid kinetic movement and body dynamics directly from RGB frames.
                    </div>
                </div>
                <div style="border-top:1px solid rgba(255,255,255,0.06); padding-top:10px; margin-top:8px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#CBD5E1; margin-bottom:4px;">
                        <span>Parameters:</span> <b>3.8M</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#CBD5E1; margin-bottom:4px;">
                        <span>Modality:</span> <b>Visual (RGB Video)</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#CBD5E1;">
                        <span>Temporal Scope:</span> <b>Local (Window-level)</b>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            """
            <div style="background:#111827; border:1px solid rgba(255,255,255,0.08); border-top:3px solid #8B5CF6;
                        border-radius:8px; padding:1.2rem; height:100%; display:flex; flex-direction:column; justify-content:space-between;">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:0.75rem; font-weight:700; color:#8B5CF6; text-transform:uppercase; letter-spacing:0.04em;">Model 2 · Refinement</span>
                        <span style="font-size:0.72rem; padding:2px 8px; border-radius:4px; background:rgba(139,92,246,0.15); color:#A78BFA; font-weight:600;">Self-Attention</span>
                    </div>
                    <div style="font-size:1.05rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;">Temporal Transformer</div>
                    <div style="font-size:0.83rem; color:#94A3B8; line-height:1.45; margin-bottom:12px;">
                        1D Temporal Transformer with sinusoidal positional encodings. Models sequence-level context, suppresses isolated single-window false alarms, and reinforces sustained combat scenes.
                    </div>
                </div>
                <div style="border-top:1px solid rgba(255,255,255,0.06); padding-top:10px; margin-top:8px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#CBD5E1; margin-bottom:4px;">
                        <span>Parameters:</span> <b>4.2M (+0.4M)</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#CBD5E1; margin-bottom:4px;">
                        <span>Modality:</span> <b>Visual + Temporal Pos</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#CBD5E1;">
                        <span>Temporal Scope:</span> <b>Global (Full Sequence)</b>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            """
            <div style="background:#111827; border:1px solid rgba(255,255,255,0.08); border-top:3px solid #10B981;
                        border-radius:8px; padding:1.2rem; height:100%; display:flex; flex-direction:column; justify-content:space-between;">
                <div>
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px;">
                        <span style="font-size:0.75rem; font-weight:700; color:#10B981; text-transform:uppercase; letter-spacing:0.04em;">Model 3 · Multimodal</span>
                        <span style="font-size:0.72rem; padding:2px 8px; border-radius:4px; background:rgba(16,185,129,0.15); color:#34D399; font-weight:600;">Cross-Modal Fusion</span>
                    </div>
                    <div style="font-size:1.05rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;">Audio-Visual Fusion</div>
                    <div style="font-size:0.83rem; color:#94A3B8; line-height:1.45; margin-bottom:12px;">
                        Fuses 3D-CNN kinetic predictions with FFT acoustic features (RMS energy spikes, spectral centroid screams, zero-crossing turbulence) for joint acoustic-visual verification.
                    </div>
                </div>
                <div style="border-top:1px solid rgba(255,255,255,0.06); padding-top:10px; margin-top:8px;">
                    <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#CBD5E1; margin-bottom:4px;">
                        <span>Parameters:</span> <b>4.5M (+0.7M)</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#CBD5E1; margin-bottom:4px;">
                        <span>Modality:</span> <b>RGB Video + Spectral Audio</b>
                    </div>
                    <div style="display:flex; justify-content:space-between; font-size:0.78rem; color:#CBD5E1;">
                        <span>Temporal Scope:</span> <b>Window + Audio Sync</b>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


def _render_benchmark_table() -> None:
    """Display benchmark comparison table (only sourced figures; unmeasured values shown as 'not measured')."""
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:600; margin:1.5rem 0 0.75rem 0;'>1. Academic Benchmark Comparison</h4>", unsafe_allow_html=True)

    def _f(v, fmt):
        return "not measured" if v is None else fmt.format(v)

    df = pd.DataFrame([
        {
            "Model Approach": item["model_name"],
            "Architecture Type": item["architecture"],
            "Val Accuracy": _f(item["accuracy"], "{:.1f}%"),
            "F1-Score": _f(item["f1_score"], "{:.3f}"),
            "Error Rate": _f(item["error_rate"], "{:.1f}%"),
            "Trained parameters": item["parameters"],
            "Source": item["source_note"],
        }
        for item in BENCHMARK_REPORTS
    ])

    st.dataframe(df, use_container_width=True, hide_index=True)


def _render_timeline_chart(result: ComparisonResult) -> None:
    """Render interactive Plotly multi-model timeline overlay."""
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:600; margin:1.75rem 0 0.75rem 0;'>2. Per-Window Prediction Sequence Comparison</h4>", unsafe_allow_html=True)

    times = result.window_times
    x3d_p = result.baseline_x3d.probabilities
    trans_p = result.transformer_enhanced.probabilities
    multi_p = result.multimodal.probabilities
    audio_s = result.audio_scores

    fig = go.Figure()

    # Audio violence score (dotted background reference)
    if audio_s and any(s > 0.05 for s in audio_s):
        fig.add_trace(go.Scatter(
            x=times,
            y=audio_s,
            mode="lines",
            name="Audio Energy Score",
            line=dict(color="#F59E0B", width=1.5, dash="dot"),
            opacity=0.65,
            hovertemplate="Time: %{x:.1f}s<br>Audio Score: %{y:.3f}<extra></extra>"
        ))

    # Baseline X3D-M line
    fig.add_trace(go.Scatter(
        x=times,
        y=x3d_p,
        mode="lines+markers",
        name="Baseline (X3D-M)",
        line=dict(color="#3B82F6", width=2.5),
        marker=dict(size=5, color="#3B82F6"),
        hovertemplate="Time: %{x:.1f}s<br>X3D-M P(v): %{y:.3f}<extra></extra>"
    ))

    # Transformer-Enhanced line
    fig.add_trace(go.Scatter(
        x=times,
        y=trans_p,
        mode="lines+markers",
        name="Transformer-Enhanced",
        line=dict(color="#8B5CF6", width=2.5),
        marker=dict(size=5, color="#8B5CF6"),
        hovertemplate="Time: %{x:.1f}s<br>Transformer P(v): %{y:.3f}<extra></extra>"
    ))

    # Multimodal Fusion line
    fig.add_trace(go.Scatter(
        x=times,
        y=multi_p,
        mode="lines+markers",
        name="Multimodal (Audio-Visual)",
        line=dict(color="#10B981", width=2.5),
        marker=dict(size=5, color="#10B981"),
        hovertemplate="Time: %{x:.1f}s<br>Multimodal P(v): %{y:.3f}<extra></extra>"
    ))

    # Decision threshold dashed horizontal line
    fig.add_hline(
        y=result.threshold,
        line_dash="dash",
        line_color="#EF4444",
        line_width=1.8,
        annotation_text=f"Decision Threshold ({result.threshold:.2f})",
        annotation_position="bottom right",
        annotation_font=dict(color="#EF4444", size=11)
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#111827",
        plot_bgcolor="#0B0F19",
        margin=dict(l=40, r=30, t=30, b=40),
        height=380,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=11, color="#F8FAFC"),
            bgcolor="rgba(17, 24, 39, 0.7)"
        ),
        xaxis=dict(
            title="Video Timeline (seconds)",
            gridcolor="rgba(255, 255, 255, 0.08)",
            color="#94A3B8",
            zerolinecolor="rgba(255, 255, 255, 0.08)"
        ),
        yaxis=dict(
            title="P(Violence)",
            range=[-0.05, 1.05],
            gridcolor="rgba(255, 255, 255, 0.08)",
            color="#94A3B8",
            zerolinecolor="rgba(255, 255, 255, 0.08)"
        )
    )

    st.plotly_chart(fig, use_container_width=True)


def _render_agreement_matrices(result: ComparisonResult) -> None:
    """Render pairwise agreement, R², and MAE statistical comparison cards."""
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:600; margin:1.75rem 0 0.75rem 0;'>3. Pairwise Agreement & Correlation Matrix</h4>", unsafe_allow_html=True)

    pairs = [
        ("x3d_vs_transformer", "X3D-M vs Temporal Transformer", "#8B5CF6", "Assesses temporal smoothing effect & spike suppression"),
        ("x3d_vs_multimodal", "X3D-M vs Multimodal Fusion", "#10B981", "Assesses acoustic reinforcement & multi-sensor synergy"),
        ("transformer_vs_multimodal", "Transformer vs Multimodal", "#F59E0B", "Cross-validates temporal context vs acoustic feature cues")
    ]

    c1, c2, c3 = st.columns(3)
    cols = [c1, c2, c3]

    for col, (pair_key, title, border_col, note) in zip(cols, pairs):
        m = result.pairwise_metrics.get(pair_key)
        if not m:
            continue
        with col:
            st.markdown(
                f"""
                <div style="background:#111827; border:1px solid rgba(255,255,255,0.08); border-top:3px solid {border_col};
                            border-radius:8px; padding:1.15rem; height:100%;">
                    <div style="font-size:0.95rem; font-weight:700; color:#F8FAFC; margin-bottom:4px;">{title}</div>
                    <div style="font-size:0.75rem; color:#94A3B8; margin-bottom:12px;">{note}</div>
                    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:8px;">
                        <div style="background:#0B0F19; border:1px solid rgba(255,255,255,0.05); border-radius:6px; padding:8px 10px;">
                            <div style="font-size:0.7rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Agreement Rate</div>
                            <div style="font-size:1.2rem; font-weight:700; color:#F8FAFC;">{m.agreement_rate:.1f}%</div>
                        </div>
                        <div style="background:#0B0F19; border:1px solid rgba(255,255,255,0.05); border-radius:6px; padding:8px 10px;">
                            <div style="font-size:0.7rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">R² Correlation</div>
                            <div style="font-size:1.2rem; font-weight:700; color:#F8FAFC;">{m.r2_score:.3f}</div>
                        </div>
                        <div style="background:#0B0F19; border:1px solid rgba(255,255,255,0.05); border-radius:6px; padding:8px 10px;">
                            <div style="font-size:0.7rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Mean Abs Error</div>
                            <div style="font-size:1.2rem; font-weight:700; color:#F8FAFC;">{m.mae:.4f}</div>
                        </div>
                        <div style="background:#0B0F19; border:1px solid rgba(255,255,255,0.05); border-radius:6px; padding:8px 10px;">
                            <div style="font-size:0.7rem; color:#94A3B8; text-transform:uppercase; font-weight:600;">Relative F1</div>
                            <div style="font-size:1.2rem; font-weight:700; color:#F8FAFC;">{m.f1_rel:.3f}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )


def _render_event_comparison(result: ComparisonResult) -> None:
    """Render comparison of segmented candidate events detected by each model."""
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:600; margin:1.75rem 0 0.75rem 0;'>4. Event-Level Detection Comparison</h4>", unsafe_allow_html=True)

    models_data = [
        ("Baseline (X3D-M)", result.baseline_x3d, "#3B82F6"),
        ("Transformer-Enhanced", result.transformer_enhanced, "#8B5CF6"),
        ("Multimodal Audio-Visual", result.multimodal, "#10B981")
    ]

    event_rows = []
    for m_name, m_sum, color in models_data:
        if not m_sum.events:
            event_rows.append({
                "Model Approach": m_name,
                "Event Count": 0,
                "Detected Time Intervals": "None (No violent scenes detected)",
                "Avg Duration": "0.0s",
                "Max Peak P(v)": f"{m_sum.peak_probability:.3f}",
                "Sensitivity Profile": "Conservative" if m_sum.n_violent_windows == 0 else "Balanced"
            })
            continue

        intervals = ", ".join(f"{format_timestamp(e.start_s)}-{format_timestamp(e.end_s)}" for e in m_sum.events)
        avg_dur = np.mean([e.duration_s for e in m_sum.events])

        event_rows.append({
            "Model Approach": m_name,
            "Event Count": len(m_sum.events),
            "Detected Time Intervals": intervals,
            "Avg Duration": f"{avg_dur:.1f}s",
            "Max Peak P(v)": f"{max(e.peak_prob for e in m_sum.events):.3f}",
            "Sensitivity Profile": "High Sensitivity" if m_sum.n_violent_windows >= result.baseline_x3d.n_violent_windows else "Selective Filtering"
        })

    st.dataframe(pd.DataFrame(event_rows), use_container_width=True, hide_index=True)

    # Sensitivity & Conservatism summary callout
    sens_sorted = sorted(
        [("X3D-M", result.baseline_x3d.n_violent_windows),
         ("Transformer", result.transformer_enhanced.n_violent_windows),
         ("Multimodal", result.multimodal.n_violent_windows)],
        key=lambda x: x[1],
        reverse=True
    )
    most_sens = sens_sorted[0][0]
    most_cons = sens_sorted[-1][0]

    st.markdown(
        f"""
        <div style="background:#0B0F19; border:1px solid rgba(255,255,255,0.08); border-radius:6px; padding:10px 14px; margin-top:8px; font-size:0.83rem; color:#CBD5E1;">
            <b>Behavioral Summary:</b> <b>{most_sens}</b> proved the most sensitive detector ({sens_sorted[0][1]} positive windows), while <b>{most_cons}</b> applied the strictest validation ({sens_sorted[-1][1]} positive windows) by filtering single-window anomalies.
        </div>
        """,
        unsafe_allow_html=True
    )


def _render_recommendation(result: ComparisonResult) -> None:
    """Render intelligent model recommendation card and trade-off matrix."""
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:600; margin:1.75rem 0 0.75rem 0;'>5. Deployment Recommendation & Trade-Off Matrix</h4>", unsafe_allow_html=True)

    rec = result.recommendation
    rec_model = rec.get("recommended_model", "Transformer-Enhanced")
    rationale = rec.get("rationale", "")

    badge_color = "#10B981" if "Multimodal" in rec_model else "#8B5CF6"

    st.markdown(
        f"""
        <div style="background:#111827; border:1px solid rgba(255,255,255,0.08); border-left:4px solid {badge_color};
                    border-radius:8px; padding:1.25rem; margin-bottom:1.25rem;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:6px;">
                <span style="font-size:0.75rem; font-weight:700; color:{badge_color}; text-transform:uppercase; letter-spacing:0.04em;">Selected Candidate</span>
                <span style="font-size:0.75rem; padding:3px 10px; border-radius:12px; background:rgba(255,255,255,0.08); color:#F8FAFC; font-weight:600;">
                    ★ {rec_model}
                </span>
            </div>
            <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;">Optimal Architecture for Current Content</div>
            <p style="color:#CBD5E1; font-size:0.88rem; line-height:1.5; margin:0;">
                {rationale}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Trade-off analysis matrix
    tradeoffs = pd.DataFrame([
        {
            "Model": "Baseline (X3D-M)",
            "Primary Advantage": "Pure video-stream inference; fastest deployment with lowest memory overhead.",
            "Key Limitation": "Susceptible to brief visual flashes and camera cuts causing single-window false positives.",
            "Best Suited For": "Real-time edge streams where audio track is unavailable or bandwidth-constrained."
        },
        {
            "Model": "Transformer-Enhanced",
            "Primary Advantage": "Global temporal attention eliminates transient noise; boosts sustained multi-window fights.",
            "Key Limitation": "Requires accumulating sequence buffer; slightly smooths instantaneous micro-bursts.",
            "Best Suited For": "Automated batch moderation and content rating pipelines requiring high precision."
        },
        {
            "Model": "Multimodal (Audio-Visual)",
            "Primary Advantage": "Dual-modality confirmation prevents false alarms when sound is calm; verifies explosive combat.",
            "Key Limitation": "Requires valid, synced audio stream and lightweight FFT compute per window.",
            "Best Suited For": "Full cartoon broadcast and streaming video archives with rich sound effect design."
        }
    ])
    st.dataframe(tradeoffs, use_container_width=True, hide_index=True)


def render(repo: Repository) -> None:
    """Entry point for the Model Comparison page."""
    st.markdown(
        """
        <div style="margin-bottom:1.5rem;">
            <h2 style="margin:0; font-size:1.75rem; font-weight:700; color:#F8FAFC; letter-spacing:-0.02em;">
                🔬 Multi-Model Analysis & Comparison
            </h2>
            <p style="margin:4px 0 0 0; color:#94A3B8; font-size:0.9rem;">
                Comparative evaluation across Baseline 3D-CNN, Temporal Transformer Self-Attention, and Multimodal Audio-Visual Fusion.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Top Section: Model Architecture Cards
    _render_model_cards()

    # Session Selection & Execution Bar
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    runs = [r for r in repo.list_runs() if r["status"] != "running"]

    if not runs:
        st.info("No video analysis runs available yet. Analyze a video in the 'Analyze Video' tab first to enable multi-model comparison.")
        _render_benchmark_table()
        return

    ids = [r["id"] for r in runs]
    default_id = st.session_state.get("selected_run_id", ids[0])
    if default_id not in ids:
        default_id = ids[0]

    with st.container():
        c_sel, c_btn = st.columns([3.8, 1.2])

        with c_sel:
            selected_run_id = st.selectbox(
                "Select Completed Video Analysis Session",
                ids,
                index=ids.index(default_id),
                format_func=lambda i: next(
                    (f"Run #{r['id']} — {r['original_name']} ({r['started_at'][:19]})" for r in runs if r["id"] == i),
                    f"Run #{i}"
                ),
                key="comparison_run_select"
            )

        run = repo.get_run(selected_run_id)
        video = repo.get_video(run["video_id"])
        video_path = video["stored_path"] if video else ""
        cfg_dict = jload(run["config_json"], {})
        cfg = AnalysisConfig(**cfg_dict) if cfg_dict else AnalysisConfig()

        # Cache key for comparison results in session state
        comp_key = f"comparison_res_{selected_run_id}"

        with c_btn:
            st.markdown("<div style='margin-top:1.6rem;'></div>", unsafe_allow_html=True)
            run_clicked = st.button("🚀 Run Comparison", key=f"btn_run_comp_{selected_run_id}", type="primary", use_container_width=True)

    # Progress bar placeholders
    progress_placeholder = st.empty()
    status_placeholder = st.empty()

    if run_clicked:
        if not video_path or not Path(video_path).is_file():
            st.error(f"Source video file is no longer available on disk: {video_path}")
        else:
            progress_bar = progress_placeholder.progress(0.0)
            status_text = status_placeholder.info("Initializing multi-model comparison pipeline...")

            def _on_progress(stage: str, frac: float, msg: str):
                progress_bar.progress(min(1.0, max(0.0, frac)))
                status_text.caption(f"**{stage}**: {msg}")

            try:
                # Load baseline predictions from database to avoid re-decoding all frames if already cached
                db_windows = repo.list_windows(selected_run_id)
                classifier = load_classifier(cfg.model_key)

                meta = VideoMetadata(
                    duration_s=float(video["duration_s"]),
                    fps=float(video["fps"]),
                    width=int(video["width"]),
                    height=int(video["height"]),
                    frame_count=int(video["frame_count"]),
                    file_size_bytes=int(video["file_size_bytes"]),
                    container=str(video["container"] or "mp4"),
                    fourcc=str(video["fourcc"] or "mp4v")
                )

                result = run_comparison(
                    video_id=video["id"],
                    video_path=video_path,
                    meta=meta,
                    cfg=cfg,
                    classifier=classifier,
                    repo=repo,
                    progress=_on_progress,
                    base_predictions=db_windows if len(db_windows) > 0 else None
                )

                st.session_state[comp_key] = result
                progress_placeholder.empty()
                status_placeholder.empty()
                st.success("Multi-model comparison analysis completed successfully!")
            except Exception as exc:
                progress_placeholder.empty()
                status_placeholder.empty()
                st.error(f"Multi-model comparison encountered an error: {type(exc).__name__}: {exc}")

    # Check if comparison result exists in session state
    result: ComparisonResult | None = st.session_state.get(comp_key)

    if result is not None:
        # Hero metrics row
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        m = st.columns(5)
        m[0].markdown(render_hero_metric("X3D-M Violent Windows", result.baseline_x3d.n_violent_windows, "", f"Out of {len(result.windows)}", "#3B82F6"), unsafe_allow_html=True)
        m[1].markdown(render_hero_metric("Transformer Windows", result.transformer_enhanced.n_violent_windows, "", f"{result.transformer_enhanced.n_events} events", "#8B5CF6"), unsafe_allow_html=True)
        m[2].markdown(render_hero_metric("Multimodal Windows", result.multimodal.n_violent_windows, "", f"{result.multimodal.n_events} events", "#10B981"), unsafe_allow_html=True)
        m[3].markdown(render_hero_metric("X3D vs Trans Agreement", f"{result.pairwise_metrics['x3d_vs_transformer'].agreement_rate:.1f}%", "", f"R² = {result.pairwise_metrics['x3d_vs_transformer'].r2_score:.3f}", "#8B5CF6"), unsafe_allow_html=True)
        m[4].markdown(render_hero_metric("X3D vs Multi Agreement", f"{result.pairwise_metrics['x3d_vs_multimodal'].agreement_rate:.1f}%", "", f"R² = {result.pairwise_metrics['x3d_vs_multimodal'].r2_score:.3f}", "#10B981"), unsafe_allow_html=True)

        # Section 1: Benchmark Table
        _render_benchmark_table()

        # Section 2: Plotly Timeline Overlay
        _render_timeline_chart(result)

        # Section 3: Agreement & Correlation Matrices
        _render_agreement_matrices(result)

        # Section 4: Event-Level Detection Comparison
        _render_event_comparison(result)

        # Section 5: Model Recommendation
        _render_recommendation(result)
    else:
        # Comparison has not been run yet: display benchmark table and prompt user
        _render_benchmark_table()

        st.markdown(
            """
            <div style="background:#111827; border:1px solid rgba(255,255,255,0.08); border-radius:8px; padding:1.5rem; text-align:center; margin-top:1.5rem;">
                <div style="font-size:2rem; margin-bottom:8px;">📊</div>
                <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC; margin-bottom:6px;">
                    Multi-Model Comparison Ready to Execute
                </div>
                <div style="font-size:0.88rem; color:#94A3B8; max-width:600px; margin:0 auto 12px auto; line-height:1.5;">
                    Click the <b>🚀 Run Comparison</b> button above to execute the 1D Temporal Transformer refiner and Multimodal Audio-Visual fusion model side-by-side on this video.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
