"""
Page 3: Interactive Data Analysis & Cohort Visualizations
Plotly-powered interactive exploratory data analysis based on the UCI Cleveland Heart Disease cohort.
"""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
try:
    import src.utils
    importlib.reload(src.utils)
    from src.utils import apply_custom_css, render_header, render_safety_disclaimer, render_sidebar
except Exception:
    try:
        import utils.utils
        importlib.reload(utils.utils)
        from utils.utils import apply_custom_css, render_header, render_safety_disclaimer, render_sidebar
    except Exception:
        def apply_custom_css(*args, **kwargs): pass
        def render_sidebar(*args, **kwargs): pass
        def render_header(title="", subtitle="", badge="Clinical ML Dashboard", *args, **kwargs):
            st.title(title or kwargs.get("title", ""))
            sub = subtitle or kwargs.get("subtitle", "")
            if sub: st.caption(sub)
        def render_safety_disclaimer(*args, **kwargs): pass
from src.data_loader import load_dataset
from src.model_training import load_evaluation_metrics
from utils.visualization import (
    create_target_distribution_chart,
    create_age_distribution_chart,
    create_cholesterol_distribution_chart,
    create_blood_pressure_chart,
    create_max_heart_rate_chart,
    create_correlation_heatmap,
    create_feature_importance_chart,
)

st.set_page_config(page_title="Data Analysis | CardioGuard", page_icon="📈", layout="wide")
apply_custom_css()
render_sidebar()

render_header(
    title="Cohort Exploratory Visualizations",
    subtitle="Interactive Physiological Distributions & Correlation Matrix",
    badge="Visual Analytics",
)

# Load real dataset & metrics
try:
    X, y = load_dataset()
    df = X.copy()
    df["num"] = y.values
    metrics = load_evaluation_metrics()
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Row 1: Target Donut + Age Histogram
col1, col2 = st.columns(2)
with col1:
    fig_target = create_target_distribution_chart(df)
    st.plotly_chart(fig_target, use_container_width=True)

with col2:
    fig_age = create_age_distribution_chart(df)
    st.plotly_chart(fig_age, use_container_width=True)

st.markdown("---")

# Row 2: Cholesterol + Blood Pressure
col3, col4 = st.columns(2)
with col3:
    fig_chol = create_cholesterol_distribution_chart(df)
    st.plotly_chart(fig_chol, use_container_width=True)

with col4:
    fig_bp = create_blood_pressure_chart(df)
    st.plotly_chart(fig_bp, use_container_width=True)

st.markdown("---")

# Row 3: Max Heart Rate + Feature Importance
col5, col6 = st.columns(2)
with col5:
    fig_hr = create_max_heart_rate_chart(df)
    st.plotly_chart(fig_hr, use_container_width=True)

with col6:
    fig_fi = create_feature_importance_chart(metrics["feature_importances"])
    st.plotly_chart(fig_fi, use_container_width=True)

st.markdown("---")

# Row 4: Full Feature Correlation Heatmap
st.markdown("### 🧬 Feature Correlation Matrix")
fig_corr = create_correlation_heatmap(df)
st.plotly_chart(fig_corr, use_container_width=True)

render_safety_disclaimer()
