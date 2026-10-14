from __future__ import annotations

import json
import shutil
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import streamlit as st

from config.settings import (BASE_DIR, CHECKPOINT_DIR, DEFAULT_MODEL_KEY, MAX_UPLOAD_MB, TEMPORAL_MODEL_PATH,
                             MODEL_REGISTRY, UPLOAD_DIR, AnalysisConfig)
from database.repository import Repository
from models.model_loader import ModelLoadError, download_checkpoint, expected_checkpoint_path, load_classifier
from pipeline.inference_pipeline import AnalysisCancelled, AnalysisFailed, run_analysis
from pipeline.content_suitability import (
    analyze_text_suitability,
    classify_title,
    x3d_threshold_from_text_score,
)
from preprocessing.clip_sampler import make_windows
from preprocessing.link_downloader import LinkDownloadError, download_video_from_url, is_valid_url
from preprocessing.video_loader import VideoValidationError, safe_filename, sha256_file, validate_video
from ui.common import ft, goto, render_hero_metric, render_page_header


def _canonical_video_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = BASE_DIR / resolved
    return resolved.resolve()


def _is_youtube_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    return host == "youtube.com" or host.endswith(".youtube.com") or host == "youtu.be" or host.endswith(".youtu.be")


@st.cache_resource(show_spinner="Initializing 3D-CNN Classifier (CPU)...")
def get_classifier(model_key: str, ckpt_mtime: float):
    return load_classifier(model_key)


def _store_upload(up, repo: Repository):
    """Stream the upload to disk, hash it, de-duplicate by content, validate it."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    tmp = UPLOAD_DIR / f".tmp_{uuid.uuid4().hex}{Path(up.name).suffix.lower()}"
    try:
        with open(tmp, "wb") as f:
            while chunk := up.read(1 << 20):
                f.write(chunk)
        meta = validate_video(tmp, MAX_UPLOAD_MB)
        digest = sha256_file(tmp)
        existing = repo.get_video_by_hash(digest)
        if existing:
            final = _canonical_video_path(existing["stored_path"])
            if final.is_file():
                tmp.unlink(missing_ok=True)
                if str(final) != existing["stored_path"] or not existing["file_available"]:
                    repo.set_video_file(existing["id"], str(final), True)
                return existing["id"], meta
            final.parent.mkdir(parents=True, exist_ok=True)
            tmp.replace(final)
            repo.set_video_file(existing["id"], str(final), True)
            return existing["id"], meta
        final = UPLOAD_DIR / f"{digest[:12]}_{safe_filename(up.name)}"
        tmp.replace(final)
        vid = repo.add_video(digest, up.name, str(final), meta.to_dict())
        return vid, meta
    finally:
        tmp.unlink(missing_ok=True)


def _load_demo_video(repo: Repository):
    """Load the bundled demo action scene directly into session state for immediate testing."""
    demo_path = BASE_DIR / "sample_videos" / "demo_action_scene.mp4"
    if not demo_path.is_file():
        st.error("Bundled demo video file not found.")
        return None, None
    meta = validate_video(demo_path, MAX_UPLOAD_MB)
    digest = sha256_file(demo_path)
    existing = repo.get_video_by_hash(digest)
    if existing:
        vid = existing["id"]
    else:
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        final = UPLOAD_DIR / f"{digest[:12]}_demo_action_scene.mp4"
        shutil.copy2(demo_path, final)
        vid = repo.add_video(digest, "demo_action_scene.mp4", str(final), meta.to_dict())
    return vid, meta


def render(repo: Repository) -> None:
    render_page_header(
        title="Video Ingestion & Analysis",
        subtitle="Ingest animation videos or web links to run full-video visual, sound, and speech detection.",
        badge_text="Inference Engine Ready",
        badge_color="#3B82F6"
    )

    key = DEFAULT_MODEL_KEY
    spec = MODEL_REGISTRY[key]
    ckpt = expected_checkpoint_path(key)

    with st.expander("⚙️ Neural Backbone Status & Architecture Specs", expanded=not ckpt.is_file()):
        c_m1, c_m2 = st.columns([3, 2])
        with c_m1:
            st.markdown(f"**Vision Backbone:** `{spec['display_name']}` (X3D-M 3D-CNN)")
            st.caption(f"Spatial: {spec['input_size']}×{spec['input_size']} RGB | Temporal: {spec['num_frames']} frames per window | Target: Binary Spatio-Temporal Violence")
        with c_m2:
            if ckpt.is_file():
                st.markdown(
                    "<span style='display:inline-flex; align-items:center; gap:6px; color:#10B981; font-weight:600; font-size:0.85rem;'>"
                    "<span style='width:6px; height:6px; border-radius:50%; background:#10B981; display:inline-block;'></span>"
                    "Pretrained Checkpoint Ready</span>",
                    unsafe_allow_html=True
                )
            else:
                st.warning("Model checkpoint not found.")
                if st.button("Download Pretrained Weights"):
                    try:
                        with st.spinner("Downloading checkpoint from Hugging Face..."):
                            p = download_checkpoint(key)
                        st.success(f"Downloaded to {p}")
                        st.rerun()
                    except ModelLoadError as e:
                        st.error(str(e))
        if TEMPORAL_MODEL_PATH.is_file():
            st.success(f"Trained temporal sequence checkpoint ready: {TEMPORAL_MODEL_PATH.name}")
        else:
            st.caption("Optional trained temporal sequence checkpoint not installed. Deterministic score smoothing is active by default; no training required to analyze.")

    # Video Source Ingestion Tabs
    st.markdown("<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin: 1rem 0 0.75rem 0;'>Select Video Source</h4>", unsafe_allow_html=True)
    tab_upload, tab_link, tab_demo = st.tabs([
        "📁 Local File Upload",
        "🔗 YouTube & Web Video Link",
        "🎬 Sample Cartoon Clip"
    ])

    with tab_upload:
        up = st.file_uploader(
            "Choose Animation Video File",
            type=["mp4", "avi", "mov", "mkv", "webm"],
            help=f"Supported formats: MP4, AVI, MOV, MKV, WebM up to {MAX_UPLOAD_MB} MB."
        )
        if up is not None:
            sig = (up.name, up.size)
            if st.session_state.get("upload_sig") != sig:
                try:
                    with st.spinner("Validating and storing uploaded video..."):
                        vid, meta = _store_upload(up, repo)
                    st.session_state.update(upload_sig=sig, upload_video_id=vid, upload_meta=meta)
                    st.rerun()
                except VideoValidationError as e:
                    st.session_state.pop("upload_sig", None)
                    st.error(f"Validation failure: {e}")
                except Exception as e:
                    st.session_state.pop("upload_sig", None)
                    st.error(f"Storage error: {e}")

    with tab_link:
        st.markdown(
            "<div style='font-size:0.85rem; color:#94A3B8; margin-bottom:12px; line-height:1.5;'>"
            "Paste a public video URL (such as YouTube watch, Shorts, share links, or direct MP4 files). "
            "The full video is downloaded, validated, and cached locally for rapid repeated analysis."
            "</div>",
            unsafe_allow_html=True
        )

        c_url, c_btn = st.columns([4, 1.2])
        with c_url:
            link_input = st.text_input(
                "Video URL",
                placeholder="https://www.youtube.com/watch?v=... or https://example.com/video.mp4",
                label_visibility="collapsed"
            )
        with c_btn:
            fetch_clicked = st.button("📥 Fetch Video", use_container_width=True)

        if fetch_clicked:
            if not link_input or not is_valid_url(link_input):
                st.error(
                    "Please enter a valid public HTTP/HTTPS video URL. "
                    "YouTube watch, Shorts, and share links plus direct video files are supported."
                )
            else:
                progress_bar = st.progress(0.0)
                status_label = st.empty()

                def _dl_prog(frac: float, msg: str):
                    progress_bar.progress(frac)
                    status_label.caption(msg)

                try:
                    with st.spinner("Downloading video stream..."):
                        target_path, clean_title = download_video_from_url(
                            link_input,
                            output_dir=UPLOAD_DIR,
                            max_mb=MAX_UPLOAD_MB,
                            progress_callback=_dl_prog
                        )
                    meta = validate_video(target_path, MAX_UPLOAD_MB)
                    digest = sha256_file(target_path)
                    downloaded_path = Path(target_path).resolve()
                    downloaded_size = downloaded_path.stat().st_size
                    existing = repo.get_video_by_hash(digest)
                    if existing:
                        vid = existing["id"]
                        stored_path = _canonical_video_path(existing["stored_path"])
                        if not stored_path.is_file():
                            stored_path = downloaded_path
                        elif stored_path != downloaded_path:
                            downloaded_path.unlink(missing_ok=True)
                        repo.set_video_file(vid, str(stored_path), True)
                    else:
                        vid = repo.add_video(digest, clean_title, str(downloaded_path), meta.to_dict())
                    title_source = "youtube" if _is_youtube_url(link_input) else "filename"
                    title_value = Path(clean_title).stem
                    repo.set_video_title_context(vid, title_value, title_source)

                    st.session_state.update(
                        upload_sig=(clean_title, downloaded_size),
                        upload_video_id=vid,
                        upload_meta=meta
                    )
                    st.success(f"Successfully downloaded: **{clean_title}**")
                    st.rerun()
                except LinkDownloadError as e:
                    st.error(str(e))
                except VideoValidationError as e:
                    st.error(f"Downloaded video failed validation: {e}")
                except Exception as e:
                    st.error(f"Failed to process video link: {e}")

    with tab_demo:
        st.markdown(
            "<div style='font-size:0.85rem; color:#94A3B8; margin-bottom:12px; line-height:1.5;'>"
            "Immediately evaluate model behavior on an action cartoon sequence using the bundled sample clip (12 seconds)."
            "</div>",
            unsafe_allow_html=True
        )
        if st.button("⚡ Load Bundled Sample Clip (demo_action_scene.mp4)", use_container_width=True):
            with st.spinner("Loading sample footage..."):
                demo_vid, demo_meta = _load_demo_video(repo)
                if demo_vid and demo_meta:
                    demo_path = BASE_DIR / "sample_videos" / "demo_action_scene.mp4"
                    st.session_state.update(
                        upload_sig=("demo_action_scene.mp4", demo_path.stat().st_size),
                        upload_video_id=demo_vid,
                        upload_meta=demo_meta
                    )
                    st.rerun()

    # Active Video Check
    if "upload_video_id" not in st.session_state:
        st.markdown(
            "<div style='text-align:center; padding:2.5rem 1.5rem; background:rgba(17,24,39,0.5); "
            "border:1px dashed rgba(255,255,255,0.12); border-radius:10px; margin-top:1.5rem;'>"
            "<span style='font-size:1.5rem; color:#64748B;'>⬆️</span>"
            "<div style='font-size:0.95rem; font-weight:600; color:#F8FAFC; margin-top:6px;'>No Active Video Selected</div>"
            "<div style='font-size:0.82rem; color:#94A3B8; margin-top:2px;'>Choose a local file, enter a YouTube link, or load the sample clip above.</div>"
            "</div>",
            unsafe_allow_html=True
        )
        return

    vid = st.session_state["upload_video_id"]
    meta = st.session_state["upload_meta"]
    if st.session_state.get("progressive_video_id") != vid:
        st.session_state["progressive_video_id"] = vid
        st.session_state.pop("progressive_active_run_id", None)
        st.session_state.pop("selected_run_id", None)
    video = repo.get_video(vid)
    source_path = _canonical_video_path(video["stored_path"])
    if source_path.is_file():
        if str(source_path) != video["stored_path"] or not video["file_available"]:
            repo.set_video_file(vid, str(source_path), True)
            video = repo.get_video(vid)
    else:
        st.error(f"The stored source video is missing: {source_path}")
        st.info("Fetch the video URL again or upload the source file to restore it, then retry analysis.")
        return

    title_value = video["video_title"] or Path(video["original_name"]).stem
    title_source = video["title_source"] or "filename"
    title_context = classify_title(title_value, title_source)
    preliminary_text = analyze_text_suitability(
        title_context=title_context, transcript_segments=[], transcript_status="pending",
    )
    st.session_state[f"content_title_context_{vid}"] = title_context

    # Active Video Telemetry Header
    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    c_title, c_clear = st.columns([5, 1.2])
    with c_title:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:8px;'>"
            f"<span style='font-size:1.15rem; font-weight:700; color:#F8FAFC;'>Active Video:</span>"
            f"<span style='font-size:1.15rem; font-weight:700; color:#60A5FA;'>{video['original_name']}</span>"
            f"</div>",
            unsafe_allow_html=True
        )
    with c_clear:
        if st.button("✕ Reset Video", help="Clear current selection", use_container_width=True):
            st.session_state.pop("upload_video_id", None)
            st.session_state.pop("upload_meta", None)
            st.session_state.pop("upload_sig", None)
            st.rerun()

    c_m = st.columns(5)
    c_m[0].markdown(render_hero_metric("Duration", ft(meta.duration_s), "⏱️", f"{meta.duration_s:.1f}s total", "#3B82F6"), unsafe_allow_html=True)
    c_m[1].markdown(render_hero_metric("Frame Rate", f"{meta.fps:.1f} fps", "🎞️", "Temporal rate", "#6366F1"), unsafe_allow_html=True)
    c_m[2].markdown(render_hero_metric("Resolution", f"{meta.width}×{meta.height}", "📐", f"{meta.container.upper()}", "#0EA5E9"), unsafe_allow_html=True)
    c_m[3].markdown(render_hero_metric("Total Frames", f"{meta.frame_count:,}", "🔢", "RGB frames", "#10B981"), unsafe_allow_html=True)
    file_mb = meta.file_size_bytes / 1e6
    c_m[4].markdown(render_hero_metric("File Size", f"{file_mb:.1f} MB", "💾", "Disk footprint", "#F59E0B"), unsafe_allow_html=True)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Preliminary Contextual Intel Card
    sentiment = title_context["title_sentiment"]
    sent_color = "#10B981" if sentiment["label"] == "Positive" else ("#EF4444" if sentiment["label"] == "Negative" else "#94A3B8")
    source_label = "YouTube Title" if title_source == "youtube" else "Uploaded Filename"

    with st.container(border=True):
        st.markdown(
            "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>"
            "<span style='font-size:0.75rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em;'>"
            "Contextual Risk Intelligence (Pre-Scan Heuristic)</span>"
            f"<span style='font-size:0.72rem; padding:2px 8px; border-radius:12px; background:rgba(255,255,255,0.06); color:#CBD5E1;'>Source: {source_label}</span>"
            "</div>",
            unsafe_allow_html=True
        )

        c_t1, c_t2, c_t3 = st.columns([3, 1.5, 1.5])
        with c_t1:
            st.markdown(f"**Title Analyzed:** `{title_value}`")
            st.caption(f"Context Assessment: **{title_context['title_suitability']}**")
            if title_context.get("conflict_phrases"):
                st.warning("Potential military / conflict indicator(s): " + ", ".join(title_context["conflict_phrases"]))
            if title_context.get("concerning_phrases"):
                st.warning("Concerning title phrase(s): " + ", ".join(title_context["concerning_phrases"]))
            elif title_context["matched_phrases"] and not title_context.get("conflict_phrases"):
                st.info("Likely kids' rhyme / nursery content indicator: " + ", ".join(title_context["matched_phrases"]))
        with c_t2:
            st.metric("Preliminary Text Score", f"{preliminary_text['score']:.2f} / 10")
            st.caption(f"Classification: {preliminary_text['classification']}")
        with c_t3:
            st.metric("Title Sentiment", sentiment['label'])
            pol = f"{sentiment['polarity']:.2f}" if sentiment['polarity'] is not None else "N/A"
            st.caption(f"Polarity: {pol} · Context clue only")

        st.caption(
            "Note: Title score is a contextual prior heuristic derived from lexical patterns; it is not a calibrated child-safety model. "
            "Full video frames, audio events, and speech transcripts will determine the final scene assessment."
        )

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Detection Parameters & Precision Controls
    with st.expander("⚙️ Detection Parameters & Multi-Modal Precision Controls", expanded=True):
        st.info(
            "X3D-M weights were pretrained on real-world action and fight footage. The candidate threshold is "
            "automatically adapted from the preliminary text score (score ÷ 10) to balance sensitivity. "
            "Audio and speech passes run automatically upon visual completion."
        )
        defaults = AnalysisConfig()
        win_def, str_def = defaults.window_seconds, defaults.stride_seconds
        thr_def = x3d_threshold_from_text_score(preliminary_text["score"])
        gap_def, min_ev_def = defaults.merge_gap_seconds, 2.0
        n_ev_def = defaults.evidence_frames_per_event
        ref_def = defaults.enable_temporal_refinement
        cut_def = defaults.suppress_shot_cuts
        spk_def = defaults.filter_isolated_spikes

        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#E2E8F0; margin-bottom:6px;'>1. Vision Backbone & Event Grouping</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        win = c1.number_input("Window Duration (s)", 1.0, 30.0, float(win_def), 0.5, help="Length of each temporal clip sampled (16 frames uniformly extracted)", key="win_seconds")
        stride = c2.number_input("Stride Step (s)", 0.5, 30.0, float(str_def), 0.5, help="Temporal stride between sliding clips (controls overlap)", key="stride_seconds")
        c3.metric(
            "X3D Candidate Threshold",
            f"{thr_def:.2f}",
            help=f"Derived automatically from text score ({preliminary_text['score']:.2f} ÷ 10).",
        )

        c4, c5, c6 = st.columns(3)
        gap = c4.number_input("Merge Gap (s)", 0.0, 30.0, float(gap_def), 0.5, help="Merge nearby positive windows into a continuous violent event", key="merge_gap_seconds")
        min_ev = c5.number_input("Minimum Event Duration (s)", 0.0, 30.0, float(min_ev_def), 0.5, help="Discard events shorter than this duration", key="minimum_event_seconds")
        n_ev = c6.number_input("Evidence Keyframes per Event", 1, 24, int(n_ev_def), help="Number of snapshot frames saved for manual verification", key="evidence_frames_per_event")

        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#E2E8F0; margin:12px 0 6px 0;'>2. Temporal Post-Processing Filters</div>", unsafe_allow_html=True)
        c_f1, c_f2, c_f3 = st.columns(3)
        enable_ref = c_f1.checkbox("Score smoothing", value=ref_def, help="Deterministic temporal smoothing of adjacent X3D window scores.", key="score_smoothing")
        suppress_cuts = c_f2.checkbox("Scene-cut adjustment", value=cut_def, help="Discounts windows where a hard cut may distort motion cues.", key="scene_cut_adjustment")
        filter_spikes = c_f3.checkbox("Isolated-spike filter", value=spk_def, help="Filters isolated X3D score spikes.", key="isolated_spike_filter")
        use_temporal = st.checkbox(
            "Use optional temporal checkpoint if installed",
            value=True,
            help="If no trained temporal sequence checkpoint is present, deterministic score smoothing runs normally."
        )

        st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#E2E8F0; margin:12px 0 6px 0;'>3. Multi-Modal Audio & Speech Modules</div>", unsafe_allow_html=True)
        c_v1, c_v2, c_v3 = st.columns(3)
        enable_verbal = c_v1.checkbox("Whisper ASR Speech Transcription", value=defaults.enable_verbal_abuse_detection, help="Runs local multilingual Whisper ASR across audio track.", key="enable_verbal_analysis")
        enable_toxicity = c_v2.checkbox("Multilingual Toxicity Scoring", value=defaults.enable_multilingual_toxicity, help="Scores speech for toxicity across English, Spanish, French, etc.", key="enable_toxicity_analysis")
        enable_audio = c_v3.checkbox("AudioSet Sound-Event Detection", value=defaults.enable_audio_event_detection, help="Detects sound candidates using AudioSet AST on 10s audio windows.", key="enable_audio_analysis")
        flag_standalone = st.checkbox("Create standalone speech events when no visual event overlaps", value=defaults.flag_standalone_verbal_abuse, key="standalone_speech_events")

    cfg = AnalysisConfig(
        key, float(win), float(stride), float(thr_def), float(gap), float(min_ev), int(n_ev),
        enable_temporal_refinement=bool(enable_ref),
        suppress_shot_cuts=bool(suppress_cuts),
        filter_isolated_spikes=bool(filter_spikes),
        enable_verbal_abuse_detection=bool(enable_verbal),
        flag_standalone_verbal_abuse=bool(flag_standalone),
        enable_multilingual_toxicity=bool(enable_toxicity),
        enable_audio_event_detection=bool(enable_audio),
        enable_qwen_explanations=False,
        use_temporal_transformer=bool(use_temporal),
    )
    try:
        cfg.validate()
    except ValueError as e:
        st.error(str(e))
        return

    n_windows = len(make_windows(meta, cfg.window_seconds, cfg.stride_seconds))

    existing = repo.find_complete_run(vid, cfg.config_hash())
    rerun = False
    if existing:
        st.markdown(
            f"<div style='background:rgba(59, 130, 246, 0.08); border:1px solid rgba(59, 130, 246, 0.25); "
            f"border-radius:8px; padding:12px 16px; margin:1.25rem 0; font-size:0.88rem; color:#93C5FD; display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;'>"
            f"<div><b>Existing Analysis Available:</b> This video has already been analyzed with identical settings (Run #{existing['id']}).</div>"
            f"</div>",
            unsafe_allow_html=True
        )
        c_ex1, c_ex2, c_ex3 = st.columns([2, 2, 1.2])
        with c_ex1:
            st.button("Open Existing Report →", type="primary", use_container_width=True,
                      on_click=goto, args=("Report",), kwargs={"selected_run_id": existing["id"]})
        with c_ex2:
            rerun = st.checkbox("Re-analyze Video (Creates new Run)")
        with c_ex3:
            if st.button("🗑️ Delete Old Run", key=f"del_exist_{existing['id']}", use_container_width=True):
                repo.delete_run(existing["id"])
                st.success(f"Run #{existing['id']} deleted.")
                st.rerun()

    if not st.button("▶ Run Multi-Modal Analysis", type="primary",
                     disabled=bool(existing) and not rerun, use_container_width=True):
        return

    if not ckpt.is_file():
        st.error("Model weights checkpoint not found. Please download it from the panel above.")
        return

    try:
        clf = get_classifier(key, ckpt.stat().st_mtime)
    except ModelLoadError as e:
        st.error(f"Model load error: {e}")
        return
    except MemoryError:
        st.error("Insufficient memory for neural inference.")
        return

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;'>"
        "<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:700; margin:0;'>Analysis Execution</h4>"
        "<span style='font-size:0.75rem; color:#10B981; font-weight:600;'>● Processing Active</span>"
        "</div>",
        unsafe_allow_html=True
    )

    bar = st.progress(0.0)
    stage_box = st.empty()
    t0 = time.perf_counter()

    def progress(stage: str, frac: float, msg: str) -> None:
        bar.progress(min(max(frac, 0.0), 1.0))
        elapsed = time.perf_counter() - t0
        stage_box.markdown(
            f"<div style='background:rgba(17, 24, 39, 0.8); border:1px solid rgba(255,255,255,0.08); "
            f"border-left:3px solid #3B82F6; border-radius:6px; padding:10px 14px; margin-top:8px; font-size:0.86rem; color:#E2E8F0;'>"
            f"<span style='color:#60A5FA; font-weight:700; text-transform:uppercase;'>STAGE: {stage}</span> &nbsp;|&nbsp; {msg} "
            f"<span style='float:right; color:#94A3B8; font-weight:600;'>{elapsed:.1f}s</span></div>",
            unsafe_allow_html=True
        )

    try:
        run_id = run_analysis(vid, str(source_path), meta, cfg, clf, repo, progress,
                              stop_after_visual=True)
    except AnalysisCancelled:
        st.warning("Analysis cancelled; partial data saved.")
        return
    except AnalysisFailed as e:
        st.error(f"Analysis pipeline error (Run #{e.run_id}): {e}")
        return

    run = repo.get_run(run_id)
    info = json.loads(run["model_info_json"] or "{}")
    info["content_title_context"] = title_context
    stages = info.setdefault("analysis_stages", {})
    stages.setdefault("visual", {"status": "completed", "windows": n_windows})
    stages.setdefault("audio", {"status": "pending" if cfg.enable_audio_event_detection else "completed"})
    stages.setdefault("speech", {"status": "pending" if cfg.enable_verbal_abuse_detection else "completed"})
    repo.update_run_model_info(run_id, info)
    st.session_state[f"progressive_analysis_{run_id}"] = {
        "run_id": run_id, "video_id": vid, "visual": stages["visual"],
        "audio": stages["audio"], "speech": stages["speech"], "results": {},
    }
    st.session_state["progressive_active_run_id"] = run_id
    goto("Report", selected_run_id=run_id)
    st.rerun()
