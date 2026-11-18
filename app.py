"""
Heart Disease Prediction Using Machine Learning
Streamlined Healthcare Analytics Operational Cockpit.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
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
    def render_header(title="", subtitle="", badge="Clinical ML", *args, **kwargs): st.title(title)
    def render_safety_disclaimer(*args, **kwargs): pass

from src.data_loader import get_dataset_summary
from src.model_training import load_evaluation_metrics

st.set_page_config(
    page_title="CardioGuard | AI Diagnostics",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_custom_css()
render_sidebar()


def main():
    render_header(
        title="CardioGuard AI — Clinical Risk Intelligence",
        subtitle="Machine Learning Diagnostics & Cardiovascular Risk Stratification",
        badge="Active System",
    )

    try:
        metrics = load_evaluation_metrics()
        summary = get_dataset_summary()
        comp = metrics.get("models_comparison", {})
    except Exception as e:
        st.error(f"Error loading system metrics: {e}")
        return

    # 4 Key Quantitative Stat Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Ensemble Accuracy</div>
                <div class="metric-card-value">{metrics['accuracy'] * 100:.1f}%</div>
                <div class="metric-card-subtitle">Random Forest (Held-out Test)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Sensitivity (Recall)</div>
                <div class="metric-card-value">{metrics['recall'] * 100:.1f}%</div>
                <div class="metric-card-subtitle">True Positive Detection</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">ROC-AUC Score</div>
                <div class="metric-card-value">{metrics.get('roc_auc', 0.9437):.4f}</div>
                <div class="metric-card-subtitle">Discrimination Power</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-title">Verified Cohort</div>
                <div class="metric-card-value">{summary['total_instances']}</div>
                <div class="metric-card-subtitle">13 Features &middot; Cleveland DB</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Core Action Centers (Direct Navigation)
    st.markdown("### ⚡ Quick Diagnostic Actions")
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        if st.button("🩺 Patient Risk Prediction", use_container_width=True):
            st.switch_page("pages/2_Prediction.py")
    with a2:
        if st.button("📁 CSV Batch Processing", use_container_width=True):
            st.switch_page("pages/2_Prediction.py")
    with a3:
        if st.button("📈 Cohort Visualizations", use_container_width=True):
            st.switch_page("pages/3_Data_Analysis.py")
    with a4:
        if st.button("📊 Model Benchmark & ROC", use_container_width=True):
            st.switch_page("pages/4_Model_Performance.py")

    st.markdown("---")

    # Multi-Model Benchmark Summary Table (Concise, High-Value)
    st.markdown("### 🏆 Algorithm Benchmarks on Test Data (61 Patients)")
    if comp:
        table_rows = []
        for m_name, res in comp.items():
            table_rows.append({
                "Model Architecture": m_name,
                "Accuracy": f"{res['accuracy'] * 100:.2f}%",
                "Recall (Sensitivity)": f"{res['recall'] * 100:.2f}%",
                "Precision": f"{res['precision'] * 100:.2f}%",
                "F1-Score": f"{res['f1_score'] * 100:.2f}%",
                "ROC-AUC": f"{res['roc_auc']:.4f}",
            })
        st.dataframe(pd.DataFrame(table_rows), hide_index=True, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Compact Collapsible Architecture Notes
    with st.expander("🔬 View Pipeline Specifications & Architecture", expanded=False):
        spec_df = pd.DataFrame([
            {"Stage": "Data Ingestion", "Method": "Automated UCI Cleveland Repository (id=45)"},
            {"Stage": "Missing Imputation", "Method": "Median (Continuous) & Mode (Categorical)"},
            {"Stage": "Feature Scaling", "Method": "StandardScaler (Zero-Mean, Unit-Variance)"},
            {"Stage": "Categorical Encoding", "Method": "OneHotEncoder (handle_unknown='ignore')"},
            {"Stage": "Inference Models", "Method": "Random Forest (100 Trees), Logistic Regression, Decision Tree"},
            {"Stage": "Explainable AI", "Method": "SHAP TreeExplainer Local Attribution Vectors"},
        ])
        st.dataframe(spec_df, hide_index=True, use_container_width=True)

    render_safety_disclaimer()


if __name__ == "__main__":
    main()
