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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
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
    background: #0B0F19;
}
::-webkit-scrollbar-thumb {
    background: #1E293B;
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover {
    background: #3B82F6;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background: #0E131F;
    border-right: 1px solid rgba(255, 255, 255, 0.07);
}

/* Container styling for bordered containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #111827;
    border-radius: 8px;
    border: 1px solid rgba(255, 255, 255, 0.08) !important;
    padding: 1rem 1.25rem;
}

/* Metric styling */
div[data-testid="stMetric"] {
    background: #111827;
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 8px;
    padding: 0.9rem 1.15rem;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
    border-top: 3px solid #3B82F6;
}
div[data-testid="stMetricValue"] > div {
    color: #F8FAFC;
    font-size: 1.75rem;
    font-weight: 700;
    letter-spacing: -0.02em;
}
div[data-testid="stMetricLabel"] > div {
    color: #94A3B8;
    font-weight: 600;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Professional buttons */
.stButton > button {
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.88rem;
    letter-spacing: 0.01em;
    padding: 0.5rem 1.15rem;
    transition: all 0.15s ease;
    border: 1px solid rgba(255, 255, 255, 0.12);
    background: #1E293B;
    color: #F1F5F9;
}
.stButton > button:hover {
    background: #334155;
    border-color: rgba(255, 255, 255, 0.2);
    color: #FFFFFF;
}
.stButton > button[kind="primary"] {
    background: #2563EB;
    border: 1px solid #1D4ED8;
    color: #FFFFFF;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.1);
}
.stButton > button[kind="primary"]:hover {
    background: #1D4ED8;
    border-color: #1E40AF;
}

/* Sidebar navigation styling */
[data-testid="stSidebar"] .stRadio label {
    padding: 8px 12px;
    border-radius: 6px;
    margin-bottom: 2px;
    transition: all 0.15s ease;
    cursor: pointer;
    font-weight: 500;
    font-size: 0.9rem;
    color: #94A3B8;
    background: transparent;
    border: 1px solid transparent;
}
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] > label:hover {
    background: rgba(255, 255, 255, 0.04);
    color: #F8FAFC;
}

/* Tabs styling */
button[data-baseweb="tab"] {
    font-weight: 600;
    font-size: 0.88rem;
    color: #94A3B8;
    padding: 8px 16px;
}
button[data-baseweb="tab"][aria-selected="true"] {
    color: #60A5FA !important;
    border-bottom-color: #3B82F6 !important;
}

/* Expanders */
.streamlit-expanderHeader {
    background: #111827 !important;
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    color: #E2E8F0 !important;
}

/* Form inputs */
input, textarea, select {
    background-color: #111827 !important;
    color: #F8FAFC !important;
    border-color: rgba(255, 255, 255, 0.12) !important;
    border-radius: 6px !important;
}

/* Dataframe containers */
div[data-testid="stDataFrame"] {
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 6px;
    overflow: hidden;
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

# Clean Enterprise Sidebar Brand Header
st.sidebar.markdown("""
<div style="padding: 0.5rem 0 1.25rem 0; border-bottom: 1px solid rgba(255,255,255,0.08); margin-bottom: 1rem;">
    <div style="display:flex; align-items:center; gap:10px;">
        <div style="background:linear-gradient(135deg, #2563EB, #1D4ED8); width:32px; height:32px; border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:1.05rem; box-shadow:0 2px 4px rgba(37,99,235,0.3);">
            🎬
        </div>
        <div>
            <div style="font-size:1.02rem; font-weight:700; color:#F8FAFC; letter-spacing:-0.01em; line-height:1.2;">
                Cartoon Violence Detection
            </div>
            <div style="font-size:0.7rem; color:#94A3B8; font-weight:500; letter-spacing:0.02em;">
                Automated Scene Analysis
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
x3d_state, x3d_color = (("checkpoint file present", "#10B981") if x3d_path.is_file()
                        else ("checkpoint missing", "#F59E0B"))
temporal_state, temporal_color = (("trained checkpoint present", "#10B981") if TEMPORAL_MODEL_PATH.is_file()
                                  else ("training data/checkpoint required", "#F59E0B"))
st.sidebar.markdown(f"""
<div style="margin-top: 2rem; background: #111827; border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 8px; padding: 12px 14px;">
    <div style="font-size:0.75rem; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.04em;">Pipeline Model Files</div>
    <div style="font-size:0.78rem; color:#CBD5E1; margin-top:6px; line-height:1.6;">
        X3D-M: <b style="color:{x3d_color};">{x3d_state}</b><br>
        Temporal Transformer: <b style="color:{temporal_color};">{temporal_state}</b><br>
        Qwen and audio models: load when enabled
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
with st.sidebar.expander("ℹ️ Project Info", expanded=False):
    st.caption("Final Year Project: Automated Spatio-Temporal Violence Scene Detection in Animated Content.")
    st.caption("CPU pipeline: X3D-M · optional audio models · optional Qwen2.5-VL")

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
