"""
Page 1: Executive Dashboard & System Topology
Concise overview of dataset characteristics, model benchmarks, and pipeline topology.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import importlib
try:
    import src.utils
    importlib.reload(src.utils)
    from src.utils import apply_custom_css, render_header, render_safety_disclaimer, render_sidebar
except Exception:
    def apply_custom_css(*args, **kwargs): pass
    def render_sidebar(*args, **kwargs): pass
    def render_header(title="", subtitle="", badge="Dashboard", *args, **kwargs): st.title(title)
    def render_safety_disclaimer(*args, **kwargs): pass

from src.data_loader import get_dataset_summary
from src.model_training import load_evaluation_metrics

st.set_page_config(page_title="Dashboard | CardioGuard", page_icon="📈", layout="wide")
apply_custom_css()
render_sidebar()

render_header(
    title="Executive Project Dashboard",
    subtitle="Dataset Characteristics, Benchmark Topology, and Pipeline Specifications",
    badge="CardioGuard Core",
)

metrics = load_evaluation_metrics()
summary = get_dataset_summary()

# 4 Stat Metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Patient Records", f"{summary['total_instances']}", "UCI Cleveland Dataset")
m2.metric("Clinical Features", f"{summary['total_features']}", "Continuous & Categorical")
m3.metric("Ensemble Accuracy", f"{metrics['accuracy']*100:.1f}%", "Held-out Test Cohort")
m4.metric("Models Evaluated", "3 Classifiers", "RF, LogReg, Decision Tree")

st.markdown("---")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 🗄️ Dataset Feature Specifications")
    spec_data = {
        "Feature Domain": ["Demographics", "Baseline Chemistry & Vitals", "Stress Electrocardiography", "Fluoroscopy & Scintigraphy"],
        "Features Included": ["Age, Sex", "Resting BP, Cholesterol, Fasting Blood Sugar", "Rest ECG, Max HR, Angina, ST Oldpeak, Slope", "Major Vessels (CA), Thalassemia (Thal)"],
        "Feature Count": ["2", "3", "5", "3"],
    }
    st.dataframe(pd.DataFrame(spec_data), hide_index=True, use_container_width=True)

with col_b:
    st.markdown("### ⚙️ Scikit-learn Pipeline Steps")
    pipeline_steps = [
        {"Stage": "1. Ingestion", "Method": "UCI Cleveland Repository (id=45)"},
        {"Stage": "2. Imputation", "Method": "SimpleImputer (Median continuous, Mode categorical)"},
        {"Stage": "3. Scaling", "Method": "StandardScaler (Zero-mean, unit-variance)"},
        {"Stage": "4. Encoding", "Method": "OneHotEncoder (handle_unknown='ignore')"},
        {"Stage": "5. Modeling", "Method": "Random Forest (100 Trees), Logistic Reg, Decision Tree"},
        {"Stage": "6. Attribution", "Method": "SHAP TreeExplainer Local Contributions"},
    ]
    st.dataframe(pd.DataFrame(pipeline_steps), hide_index=True, use_container_width=True)

render_safety_disclaimer()
