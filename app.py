"""Streamlit entry point: streamlit run app.py"""
from __future__ import annotations

import streamlit as st

from config.settings import (DB_PATH, DEFAULT_MODEL_KEY, MODEL_REGISTRY, TEMPORAL_MODEL_PATH,
                             CHECKPOINT_DIR, ensure_dirs)
from database.repository import Repository
from ui import dashboard, history_page, report_page, research_page, upload_page

st.set_page_config(
    page_title="Cartoon Violence Detection",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)
ensure_dirs()

# Enterprise Design System CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    color-scheme: dark;
    --bg-primary: #0A0E17;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.75);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --accent-blue: #3B82F6;
    --accent-indigo: #6366F1;
    --accent-emerald: #10B981;
    --text-primary: #F8FAFC;
    --text-secondary: #94A3B8;
}

html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background-color: #0A0E17;
    color: #F8FAFC;
    -webkit-font-smoothing: antialiased;
}

/* Subtle background illumination */
.stApp {
    background-image: 
        radial-gradient(circle at 10% 10%, rgba(37, 99, 235, 0.04) 0%, transparent 45%),
        radial-gradient(circle at 90% 90%, rgba(99, 102, 241, 0.03) 0%, transparent 45%);
}

/* Hide default streamlit clutter */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* Modern subtle scrollbar */
::-webkit-scrollbar {
    width: 6px;
    height: 6px;
}
::-webkit-scrollbar-track {
    background: #0A0E17;
}
::-webkit-scrollbar-thumb {
    background: #1E293B;
    border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
    background: #3B82F6;
}

/* Pulse animation for live and running indicators */
@keyframes cvd-pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.35; transform: scale(1.3); }
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #0E1322 !important;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
    padding-top: 1rem;
}

[data-testid="stSidebar"] > div:first-child {
    background-color: #0E1322;
}

/* Container styling for bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(17, 24, 39, 0.7);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 10px;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 1.1rem 1.25rem;
    box-shadow: 0 4px 18px -3px rgba(0, 0, 0, 0.35);
}

/* Metric styling */
div[data-testid="stMetric"] {
    background: linear-gradient(180deg, rgba(17, 24, 39, 0.85) 0%, rgba(15, 23, 42, 0.75) 100%);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 0.95rem 1.2rem;
    box-shadow: 0 4px 16px -2px rgba(0, 0, 0, 0.3);
    border-top: 3px solid #3B82F6;
    transition: transform 0.15s ease, border-color 0.15s ease;
}
div[data-testid="stMetric"]:hover {
    border-top-color: #60A5FA;
    transform: translateY(-1px);
}
div[data-testid="stMetricValue"] > div {
    color: #F8FAFC;
    font-size: 1.85rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    font-feature-settings: "tnum" on, "zero" on;
}
div[data-testid="stMetricLabel"] > div {
    color: #94A3B8;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

/* Professional buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.01em;
    padding: 0.55rem 1.25rem;
    transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1);
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: #1E293B;
    color: #F1F5F9;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
.stButton > button:hover {
    background: #2D3D54;
    border-color: rgba(255, 255, 255, 0.25);
    color: #FFFFFF;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
}
.stButton > button:active {
    transform: translateY(0) scale(0.99);
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2563EB 0%, #1D4ED8 100%);
    border: 1px solid #3B82F6;
    color: #FFFFFF;
    box-shadow: 0 2px 10px rgba(37, 99, 235, 0.35);
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #3B82F6 0%, #2563EB 100%);
    border-color: #60A5FA;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.45);
}

/* Sidebar navigation styling */
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label {
    padding: 9px 14px;
    border-radius: 8px;
    margin-bottom: 4px;
    transition: all 0.15s ease;
    cursor: pointer;
    font-weight: 500;
    font-size: 0.88rem;
    color: #94A3B8;
    background: transparent;
    border: 1px solid transparent;
    display: flex;
    align-items: center;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.05);
    color: #F8FAFC;
    border-color: rgba(255, 255, 255, 0.08);
    transform: translateX(2px);
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label[data-checked="true"],
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:has(input:checked) {
    background: linear-gradient(90deg, rgba(37, 99, 235, 0.16) 0%, rgba(37, 99, 235, 0.04) 100%) !important;
    border: 1px solid rgba(59, 130, 246, 0.35) !important;
    color: #60A5FA !important;
    font-weight: 600 !important;
}

/* Tabs styling */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.88rem;
    color: #94A3B8;
    padding: 10px 18px;
    border-radius: 6px 6px 0 0;
    transition: color 0.15s ease;
}
button[data-baseweb="tab"]:hover {
    color: #F8FAFC;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #60A5FA !important;
    border-bottom: 2px solid #3B82F6 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(17, 24, 39, 0.8) !important;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #E2E8F0 !important;
    padding: 0.75rem 1rem !important;
}
.streamlit-expanderHeader:hover {
    border-color: rgba(255, 255, 255, 0.16) !important;
}
div[data-testid="stExpanderDetails"] {
    background: rgba(15, 23, 42, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-top: none;
    border-radius: 0 0 8px 8px;
    padding: 1rem 1.25rem;
}

/* Form inputs */
input, textarea, select {
    background-color: #111827 !important;
    color: #F8FAFC !important;
    border: 1px solid rgba(255, 255, 255, 0.12) !important;
    border-radius: 7px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}
input:focus, textarea:focus, select:focus {
    border-color: #3B82F6 !important;
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.25) !important;
}

/* Dataframe containers */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    overflow: hidden;
    background: #0F172A;
}

/* Alert notifications */
div[data-testid="stAlert"] {
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    backdrop-filter: blur(8px);
}

/* File uploader */
div[data-testid="stFileUploader"] {
    border: 1px dashed rgba(255, 255, 255, 0.18);
    border-radius: 10px;
    padding: 12px;
    background: rgba(17, 24, 39, 0.5);
    transition: border-color 0.2s ease, background 0.2s ease;
}
div[data-testid="stFileUploader"]:hover {
    border-color: #3B82F6;
    background: rgba(59, 130, 246, 0.04);
}

/* Progress bar */
div[data-testid="stProgress"] > div > div > div > div {
    background: linear-gradient(90deg, #3B82F6 0%, #6366F1 50%, #10B981 100%) !important;
    border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_repo() -> Repository:
    return Repository(DB_PATH)


repo = get_repo()
repo.mark_stale_running()

PAGES = [
    "📊 Dashboard",
    "⚡ Analyze Video",
    "📋 Incident Report",
    "📁 Video History",
    "🔬 Model & Architecture",
]

_LOOKUP = {
    "📊 Dashboard": "Dashboard",
    "Dashboard": "Dashboard",
    "⚡ Analyze Video": "Upload & Analyze",
    "⚡ Video Studio": "Upload & Analyze",
    "Upload & Analyze": "Upload & Analyze",
    "📋 Incident Report": "Report",
    "🔍 Incident Report": "Report",
    "Report": "Report",
    "📁 Video History": "History",
    "📁 Audit Archives": "History",
    "History": "History",
    "🔬 Model & Architecture": "Research Status",
    "🧠 Model Architecture": "Research Status",
    "Research Status": "Research Status",
    "🔬 Research": "Research Status",
    "Research": "Research Status",
}

_CANONICAL_TO_DISPLAY = {
    "Dashboard": "📊 Dashboard",
    "Upload & Analyze": "⚡ Analyze Video",
    "Report": "📋 Incident Report",
    "History": "📁 Video History",
    "Research Status": "🔬 Model & Architecture",
}

# Synchronize programmatic navigation (e.g. goto("Report")) with sidebar radio widget state
if "nav_target" in st.session_state:
    target = st.session_state.pop("nav_target")
    st.session_state["app_page"] = target
    st.session_state["nav_radio_selection"] = _CANONICAL_TO_DISPLAY.get(target, PAGES[0])

if "nav_radio_selection" not in st.session_state:
    cur = st.session_state.get("app_page", "Dashboard")
    st.session_state["nav_radio_selection"] = _CANONICAL_TO_DISPLAY.get(cur, PAGES[0])

# Modern Enterprise Sidebar Brand Header
st.sidebar.markdown("""
<div style="padding: 0.6rem 0 1.25rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1.1rem;">
    <div style="display:flex; align-items:center; gap:12px;">
        <div style="background:linear-gradient(135deg, #2563EB 0%, #4F46E5 50%, #7C3AED 100%); width:38px; height:38px; border-radius:10px; display:flex; align-items:center; justify-content:center; font-size:1.2rem; box-shadow:0 4px 12px rgba(37,99,235,0.4); border:1px solid rgba(255,255,255,0.15);">
            🎬
        </div>
        <div>
            <div style="font-size:1.02rem; font-weight:800; color:#F8FAFC; letter-spacing:-0.02em; line-height:1.2;">
                Cartoon Violence Detection
            </div>
            <div style="display:flex; align-items:center; gap:6px; margin-top:3px;">
                <span style="font-size:0.7rem; color:#94A3B8; font-weight:500; letter-spacing:0.02em;">
                    Multi-Modal Safety AI
                </span>
                <span style="font-size:0.65rem; background:rgba(37,99,235,0.15); border:1px solid rgba(37,99,235,0.3); color:#60A5FA; padding:1px 6px; border-radius:10px; font-weight:600;">v3.4</span>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

page_sel = st.sidebar.radio(
    "Navigation",
    PAGES,
    key="nav_radio_selection",
    label_visibility="collapsed"
)
page = _LOOKUP.get(page_sel, "Dashboard")
st.session_state["app_page"] = page

# Sidebar status reflects the files currently installed; optional models load when enabled.
x3d_path = CHECKPOINT_DIR / MODEL_REGISTRY[DEFAULT_MODEL_KEY]["hf_filename"]
x3d_state, x3d_color = (("Weights Ready", "#10B981") if x3d_path.is_file()
                        else ("Checkpoint Missing", "#F59E0B"))
temporal_state, temporal_color = (("Trained Weights Ready", "#10B981") if TEMPORAL_MODEL_PATH.is_file()
                                  else ("Deterministic Smoothing (No Trained Weights)", "#F59E0B"))

st.sidebar.markdown(f"""
<div style="margin-top: 1.75rem; background: rgba(17, 24, 39, 0.7); backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 9px; padding: 12px 14px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.25);">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
        <span style="font-size:0.72rem; font-weight:700; color:#94A3B8; text-transform:uppercase; letter-spacing:0.06em;">Pipeline Telemetry</span>
        <span style="display:inline-flex; align-items:center; gap:4px; font-size:0.68rem; color:#10B981; font-weight:600;">
            <span style="width:5px; height:5px; border-radius:50%; background:#10B981;"></span> Online
        </span>
    </div>
    <div style="font-size:0.78rem; color:#CBD5E1; line-height:1.65;">
        <div style="display:flex; justify-content:space-between; align-items:center;">
            <span>X3D-M 3D-CNN:</span>
            <b style="color:{x3d_color}; font-size:0.74rem;">{x3d_state}</b>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2px;">
            <span>Temporal Head:</span>
            <b style="color:{temporal_color}; font-size:0.74rem;">{temporal_state}</b>
        </div>
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2px; font-size:0.73rem; color:#94A3B8;">
            <span>Audio & VLM:</span>
            <span>On-Demand CPU</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
with st.sidebar.expander("ℹ️ Research System Context", expanded=False):
    st.caption("Final Year Project: Automated Spatio-Temporal Violence Scene Detection in Animated Content.")
    st.caption("Architecture: X3D-M 3D-CNN · Multilingual Whisper ASR · AudioSet AST · Qwen2.5-VL")
    st.caption("Execution Engine: Optimized CPU Inference")

if page == "Dashboard":
    dashboard.render(repo)
elif page == "Upload & Analyze":
    upload_page.render(repo)
elif page == "Report":
    report_page.render(repo)
elif page == "History":
    history_page.render(repo)
else:
    research_page.render()
