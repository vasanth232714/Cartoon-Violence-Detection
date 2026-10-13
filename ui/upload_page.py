from __future__ import annotations

import shutil
import time
import uuid
from pathlib import Path

import streamlit as st

from config.settings import (BASE_DIR, CHECKPOINT_DIR, DEFAULT_MODEL_KEY, MAX_UPLOAD_MB, TEMPORAL_MODEL_PATH,
                             MODEL_REGISTRY, UPLOAD_DIR, AnalysisConfig)
from database.repository import Repository
from models.model_loader import ModelLoadError, download_checkpoint, expected_checkpoint_path, load_classifier
from pipeline.inference_pipeline import AnalysisCancelled, AnalysisFailed, run_analysis
from preprocessing.clip_sampler import make_windows
from preprocessing.link_downloader import LinkDownloadError, download_video_from_url, is_valid_url
from preprocessing.video_loader import VideoValidationError, safe_filename, sha256_file, validate_video
from ui.common import ft, goto, render_hero_metric


def _canonical_video_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = BASE_DIR / resolved
    return resolved.resolve()


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
    st.markdown(
        "<div style='margin-bottom:1.5rem;'>"
        "<h2 style='margin:0; font-size:1.75rem; font-weight:700; color:#F8FAFC; letter-spacing:-0.02em;'>"
        "Video Analysis & Ingestion"
        "</h2>"
        "<p style='margin:4px 0 0 0; color:#94A3B8; font-size:0.9rem;'>"
        "Upload a video or paste a supported public video URL for full-video visual, audio, and speech analysis."
        "</p></div>",
        unsafe_allow_html=True
    )

    key = DEFAULT_MODEL_KEY
    spec = MODEL_REGISTRY[key]
    ckpt = expected_checkpoint_path(key)

    with st.expander("Model Checkpoint & Configuration", expanded=not ckpt.is_file()):
        c_m1, c_m2 = st.columns([3, 2])
        with c_m1:
            st.markdown(f"**Classifier:** `{spec['display_name']}`")
            st.caption(f"Spatial: {spec['input_size']}×{spec['input_size']} | Temporal: {spec['num_frames']} frames per window | Backbone: X3D-M 3D-CNN")
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
            st.caption("No trained temporal sequence checkpoint is installed. See training/README.md; deterministic score processing will be used.")

    # Video Source Ingestion Tabs
    st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600; margin-bottom:0.75rem;'>Select Video Source</h4>", unsafe_allow_html=True)
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
            "<div style='font-size:0.85rem; color:#94A3B8; margin-bottom:10px;'>"
            "Paste a supported public video link (for example, YouTube or Vimeo) or a direct video file URL. "
            "The full downloaded video is analyzed after validation. Private-network and localhost destinations are rejected."
            "</div>",
            unsafe_allow_html=True
        )

        c_url, c_btn = st.columns([4, 1])
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
                    "Please enter a public HTTP/HTTPS video URL. "
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
            "<div style='font-size:0.85rem; color:#94A3B8; margin-bottom:12px;'>"
            "Test the model immediately using the bundled 12-second action cartoon sample clip."
            "</div>",
            unsafe_allow_html=True
        )
        if st.button("⚡ Load Sample Action Clip", use_container_width=True):
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
        st.info("Select a video source above (upload a file, paste a YouTube link, or load the sample clip) to begin.")
        return

    vid = st.session_state["upload_video_id"]
    meta = st.session_state["upload_meta"]
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

    # Active Video Profile
    st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
    c_title, c_clear = st.columns([5, 1])
    with c_title:
        st.markdown(
            f"<h4 style='color:#F8FAFC; font-size:1.05rem; font-weight:600; margin:0;'>"
            f"Active Video: <span style='color:#3B82F6;'>{video['original_name']}</span></h4>",
            unsafe_allow_html=True
        )
    with c_clear:
        if st.button("Reset Video", help="Clear current selection"):
            st.session_state.pop("upload_video_id", None)
            st.session_state.pop("upload_meta", None)
            st.session_state.pop("upload_sig", None)
            st.rerun()

    c_m = st.columns(5)
    c_m[0].markdown(render_hero_metric("Duration", ft(meta.duration_s), "", f"{meta.duration_s:.1f}s", "#3B82F6"), unsafe_allow_html=True)
    c_m[1].markdown(render_hero_metric("Frame Rate", f"{meta.fps:.1f} fps", "", "Temporal rate", "#6366F1"), unsafe_allow_html=True)
    c_m[2].markdown(render_hero_metric("Resolution", f"{meta.width}×{meta.height}", "", f"{meta.container.upper()}", "#0EA5E9"), unsafe_allow_html=True)
    c_m[3].markdown(render_hero_metric("Frames", f"{meta.frame_count:,}", "", "Total frames", "#10B981"), unsafe_allow_html=True)
    file_mb = meta.file_size_bytes / 1e6
    c_m[4].markdown(render_hero_metric("File Size", f"{file_mb:.1f} MB", "", "Disk footprint", "#F59E0B"), unsafe_allow_html=True)

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)

    # Analysis Configuration & Presets
    with st.expander("⚙️ Detection Parameters & Precision Controls", expanded=True):
        st.info(
            "The installed X3D-M weights are pretrained on real-world fight footage, not cartoons. No training is "
            "required to run this detector, but cartoon scenes can be missed or falsely flagged. The precision-first "
            "default reduces borderline candidates and can miss real violence; use a lower threshold when recall matters "
            "more. Qwen can review saved candidates in Phase 2. An optional temporal checkpoint is not included."
        )

        preset = st.radio(
            "Configuration Preset",
            [
                "Precision-first threshold (0.75 — fewer false alarms, may miss violence)",
                "Higher threshold (0.65 — fewer candidates, may miss violence)",
                "Balanced threshold (0.50)",
                "Recall-first lower threshold (0.40 — more candidates and false alarms)",
                "⚙️ Custom Parameters"
            ],
            index=0,
            horizontal=False
        )

        if "Precision-first" in preset:
            win_def, str_def, thr_def, gap_def, min_ev_def, n_ev_def = 4.0, 2.0, 0.75, 1.0, 2.0, 6
            ref_def, cut_def, spk_def = True, True, True
        elif "Higher" in preset:
            win_def, str_def, thr_def, gap_def, min_ev_def, n_ev_def = 4.0, 2.0, 0.65, 1.0, 1.5, 6
            ref_def, cut_def, spk_def = True, True, True
        elif "Balanced" in preset:
            win_def, str_def, thr_def, gap_def, min_ev_def, n_ev_def = 4.0, 2.0, 0.50, 1.0, 1.0, 6
            ref_def, cut_def, spk_def = True, True, True
        elif "Recall-first" in preset:
            win_def, str_def, thr_def, gap_def, min_ev_def, n_ev_def = 4.0, 2.0, 0.40, 1.0, 0.0, 6
            ref_def, cut_def, spk_def = False, False, False
        else:
            win_def, str_def, thr_def, gap_def, min_ev_def, n_ev_def = 4.0, 2.0, 0.55, 1.0, 1.5, 6
            ref_def, cut_def, spk_def = True, True, True

        c1, c2, c3 = st.columns(3)
        win = c1.number_input("Window Duration (s)", 1.0, 30.0, win_def, 0.5, help="Length of each temporal clip sampled (16 frames uniformly extracted)", key=f"win_{preset}")
        stride = c2.number_input("Stride Step (s)", 0.5, 30.0, str_def, 0.5, help="Temporal stride between sliding clips (controls overlap)", key=f"stride_{preset}")
        thr = c3.slider("X3D Candidate Threshold", 0.05, 0.95, thr_def, 0.05, help="X3D class-score cutoff for creating review candidates. Scores are not calibrated for cartoons; raising the cutoff can remove false alarms but can also miss real animated violence.", key=f"thr_{preset}")

        c4, c5, c6 = st.columns(3)
        gap = c4.number_input("Merge Gap (s)", 0.0, 30.0, gap_def, 0.5, help="Merge nearby positive windows into a continuous violent event", key=f"gap_{preset}")
        min_ev = c5.number_input("Minimum Event Duration (s)", 0.0, 30.0, min_ev_def, 0.5, help="Discard events shorter than this duration (filters isolated blips)", key=f"minev_{preset}")
        n_ev = c6.number_input("Evidence Keyframes per Event", 1, 24, n_ev_def, help="Number of snapshot frames saved for manual verification", key=f"nev_{preset}")

        st.markdown("<div style='font-size:0.83rem; font-weight:600; color:#E2E8F0; margin:10px 0 6px 0;'>Temporal processing:</div>", unsafe_allow_html=True)
        c_f1, c_f2, c_f3 = st.columns(3)
        enable_ref = c_f1.checkbox("Score smoothing", value=ref_def, help="Deterministic smoothing of adjacent X3D window scores; this is not a trained model.", key=f"ref_{preset}")
        suppress_cuts = c_f2.checkbox("Scene-cut adjustment", value=cut_def, help="Discounts windows where a detected hard cut may distort motion cues.", key=f"cut_{preset}")
        filter_spikes = c_f3.checkbox("Isolated-spike filter", value=spk_def, help="Filters isolated X3D score spikes using the configured rule.", key=f"spk_{preset}")
        use_temporal = st.checkbox(
            "Use optional temporal checkpoint if installed",
            value=True,
            help="No temporal checkpoint ships with the app. If one is absent, the selected deterministic score post-processing runs normally; no training is required to analyze a video."
        )
        enable_videomae = st.checkbox(
            "Run optional VideoMAE violence comparison",
            value=False,
            help="Runs HappyGook/videomae-violence-detector on the same sampled windows as X3D. First use downloads about 345 MB. Runs on CPU to avoid occupying GPU memory with Qwen. This model is not validated on cartoons and its scores are not calibrated."
        )
        videomae_threshold = 0.50
        if enable_videomae:
            videomae_threshold = st.slider(
                "VideoMAE candidate threshold (independent of X3D)", 0.05, 0.95, 0.50, 0.05,
                help="Used only to list VideoMAE candidate intervals and compare per-window decisions. Scores are not averaged or calibrated."
            )

        st.markdown("<div style='font-size:0.83rem; font-weight:600; color:#E2E8F0; margin:10px 0 6px 0;'>Full-video audio analysis:</div>", unsafe_allow_html=True)
        c_v1, c_v2, c_v3 = st.columns(3)
        enable_verbal = c_v1.checkbox("Transcribe speech and flag verbal candidates", value=True, help="Runs local multilingual Whisper ASR across the full audio track. English lexicon matches and multilingual toxicity candidates are shown separately.", key=f"verbal_{preset}")
        enable_toxicity = c_v2.checkbox("Multilingual toxicity scoring", value=True, help="Optional broad toxicity signal for English, French, Spanish, Italian, Portuguese, Turkish, and Russian. It is not a threat classifier.", key=f"toxicity_{preset}")
        enable_audio = c_v3.checkbox("Detect sound-event candidates", value=True, help="Runs AudioSet AST on overlapping 10-second audio windows. First use may load/download model weights; CPU progress is shown per window.", key=f"audio_{preset}")
        st.info("Two steps: this scan finds and saves all visual, speech, and sound candidates first. After reviewing the event list, run Qwen explanations from the Report tab.")
        flag_standalone = st.checkbox("Create standalone speech events when no visual event overlaps", value=True, key=f"stand_{preset}")

    cfg = AnalysisConfig(
        key, float(win), float(stride), float(thr), float(gap), float(min_ev), int(n_ev),
        enable_temporal_refinement=bool(enable_ref),
        suppress_shot_cuts=bool(suppress_cuts),
        filter_isolated_spikes=bool(filter_spikes),
        enable_verbal_abuse_detection=bool(enable_verbal),
        flag_standalone_verbal_abuse=bool(flag_standalone),
        enable_multilingual_toxicity=bool(enable_toxicity),
        enable_audio_event_detection=bool(enable_audio),
        enable_qwen_explanations=False,
        use_temporal_transformer=bool(use_temporal),
        enable_videomae_comparison=bool(enable_videomae),
        videomae_threshold=float(videomae_threshold),
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
            f"border-radius:6px; padding:10px 14px; margin:1rem 0; font-size:0.88rem; color:#93C5FD;'>"
            f"<b>Existing Analysis Found:</b> This video has already been analyzed with identical settings (Run #{existing['id']})."
            f"</div>",
            unsafe_allow_html=True
        )
        c_ex1, c_ex2, c_ex3 = st.columns([1.5, 1.5, 1])
        with c_ex1:
            st.button("Open Existing Report", type="primary", use_container_width=True,
                      on_click=goto, args=("Report",), kwargs={"selected_run_id": existing["id"]})
        with c_ex2:
            rerun = st.checkbox("Re-analyze Video (Generates new Run)")
        with c_ex3:
            if st.button("🗑️ Delete Old Run", key=f"del_exist_{existing['id']}", use_container_width=True):
                repo.delete_run(existing["id"])
                st.success(f"Run #{existing['id']} deleted. You can now analyze again.")
                st.rerun()

    if not st.button("▶ Start Violence Detection", type="primary",
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

    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    st.markdown("<h4 style='color:#F8FAFC; font-size:1rem; font-weight:600;'>Analysis Progress</h4>", unsafe_allow_html=True)

    bar = st.progress(0.0)
    stage_box = st.empty()
    t0 = time.perf_counter()

    def progress(stage: str, frac: float, msg: str) -> None:
        bar.progress(min(max(frac, 0.0), 1.0))
        elapsed = time.perf_counter() - t0
        stage_box.markdown(
            f"<div style='background:#111827; border-left:3px solid #3B82F6; border-radius:4px; "
            f"padding:8px 12px; margin-top:6px; font-size:0.85rem; color:#E2E8F0;'>"
            f"<span style='color:#60A5FA; font-weight:600;'>{stage.upper()}</span>: {msg} "
            f"<span style='float:right; color:#64748B;'>{elapsed:.1f}s</span></div>",
            unsafe_allow_html=True
        )

    try:
        run_id = run_analysis(vid, str(source_path), meta, cfg, clf, repo, progress)
    except AnalysisCancelled:
        st.warning("Analysis cancelled; partial data saved.")
        return
    except AnalysisFailed as e:
        st.error(f"Analysis pipeline error (Run #{e.run_id}): {e}")
        return

    run = repo.get_run(run_id)
    if run["status"] == "completed":
        st.markdown(
            f"<div style='background:rgba(16, 185, 129, 0.08); border:1px solid rgba(16, 185, 129, 0.3); "
            f"border-radius:6px; padding:14px; margin:1.25rem 0; text-align:center;'>"
            f"<div style='color:#10B981; font-weight:600; font-size:0.95rem;'>Analysis Complete</div>"
            f"<div style='color:#94A3B8; font-size:0.85rem; margin-top:2px;'>"
            f"Analyzed {n_windows} temporal clips in {run['elapsed_s']:.1f}s ({run['elapsed_s'] / meta.duration_s:.2f}× real-time speed)."
            f"</div></div>",
            unsafe_allow_html=True
        )
    else:
        st.warning(f"Analysis finished with status: {run['status']}")

    st.button("📋 View Incident Report", type="primary",
              on_click=goto, args=("Report",), kwargs={"selected_run_id": run_id}, use_container_width=True)
