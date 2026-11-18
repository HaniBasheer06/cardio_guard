"""
Frontend Utilities and UI Helper Functions.
Handles CSS injection, theme toggling, sleek compact headers, and safety disclaimers.
"""

from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSS_FILE = PROJECT_ROOT / "assets" / "style.css"


def apply_custom_css():
    """Injects high-contrast, professional healthcare design stylesheet with light/dark support."""
    if CSS_FILE.exists():
        with open(CSS_FILE, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

    is_dark = st.session_state.get("dark_mode", False)
    if is_dark:
        st.markdown(
            """
            <style>
            .stApp { background-color: #090e1a !important; color: #f8fafc !important; }
            h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }
            p, span, li { color: #cbd5e1 !important; }
            label[data-testid="stWidgetLabel"] p, label p { color: #f1f5f9 !important; }
            .metric-card { background-color: #111c35 !important; border-color: #1e293b !important; border-top-color: #38bdf8 !important; }
            .metric-card-value { color: #38bdf8 !important; }
            .metric-card-title { color: #94a3b8 !important; }
            .metric-card-subtitle { color: #cbd5e1 !important; }
            .info-card { background-color: #111c35 !important; border-color: #1e293b !important; border-left-color: #38bdf8 !important; color: #e2e8f0 !important; }
            .info-card h4 { color: #38bdf8 !important; }
            [data-testid="stSidebar"] { background-color: #0b1326 !important; border-right: 1px solid #1e293b !important; }
            [data-testid="stSidebar"] * { color: #f1f5f9 !important; }
            div[data-baseweb="input"], div[data-baseweb="select"] { background-color: #131e38 !important; border-color: #334155 !important; }
            input { color: #ffffff !important; }
            div[data-baseweb="tab-list"] { background-color: #111c35 !important; }
            button[data-baseweb="tab"] { color: #94a3b8 !important; }
            button[data-baseweb="tab"][aria-selected="true"] { background-color: #1e293b !important; color: #38bdf8 !important; }
            div[data-testid="stExpander"] { background-color: #111c35 !important; border-color: #1e293b !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            .stApp { background-color: #f8fafc !important; color: #0f172a !important; }
            h1, h2, h3, h4, h5, h6 { color: #0f172a !important; font-weight: 700 !important; }
            p, span, li { color: #334155 !important; }
            label[data-testid="stWidgetLabel"] p, label p { color: #0f172a !important; font-weight: 700 !important; font-size: 0.95rem !important; }
            .metric-card-title { color: #475569 !important; font-weight: 700 !important; }
            .metric-card-subtitle { color: #334155 !important; }
            .metric-card-value { color: #0284c7 !important; }
            </style>
            """,
            unsafe_allow_html=True,
        )


def render_sidebar():
    """Renders sleek sidebar with quick navigation and theme switcher."""
    with st.sidebar:
        st.markdown("### 🫀 CardioGuard AI")
        st.caption("Clinical Decision Support Platform")
        st.markdown("---")

        st.markdown("#### 🎨 Theme")
        is_dark = st.toggle("🌙 Dark Mode", value=st.session_state.get("dark_mode", False))
        if is_dark != st.session_state.get("dark_mode", False):
            st.session_state["dark_mode"] = is_dark
            st.rerun()

        st.markdown("---")
        st.caption("Active Pipeline: **Random Forest (100 Trees)**")
        st.caption("Cohort: **UCI Cleveland (ID: 45)**")


def render_header(title: str = "", subtitle: str = "", badge: str = "Clinical ML Dashboard", *args, **kwargs):
    """Renders a sleek, modern, compact header without unnecessary bulk."""
    header_title = title or kwargs.get("title", "")
    header_sub = subtitle or kwargs.get("subtitle", "")
    header_badge = badge or kwargs.get("badge", "Clinical ML Dashboard")

    sub_html = f"<p style='margin: 0.35rem 0 0 0; font-size: 0.95rem; opacity: 0.92;'>{header_sub}</p>" if header_sub else ""

    st.markdown(
        f"""
        <div class="medical-header" style="padding: 1.5rem 2rem; margin-bottom: 1.25rem; border-radius: 14px;">
            <div style="display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
                <div>
                    <h1 style="font-size: 1.85rem !important; margin: 0 !important; color: #ffffff !important;">{header_title}</h1>
                    {sub_html}
                </div>
                <span class="badge-tag" style="background: rgba(255,255,255,0.22); color: #ffffff; padding: 0.3rem 0.8rem;">
                    <span style="display: inline-block; width: 6px; height: 6px; border-radius: 50%; background: #34d399; margin-right: 5px;"></span>
                    {header_badge}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_safety_disclaimer(*args, **kwargs):
    """Renders a clean, collapsible disclaimer footer so it doesn't clutter the view."""
    with st.expander("🛡️ Medical & Academic Safety Notice", expanded=False):
        st.caption(
            "This application is an educational and technical demonstration developed for research purposes "
            "using the benchmark 1988 UCI Cleveland Heart Disease dataset. "
            "This application is NOT a clinically certified medical device and must never replace licensed medical diagnosis."
        )
