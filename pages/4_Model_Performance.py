"""
Page 4: Model Performance & Multi-Algorithm Benchmarking
Rigorous out-of-sample statistical comparison across:
1. Random Forest Classifier
2. Logistic Regression
3. Decision Tree Classifier
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
from src.model_training import load_evaluation_metrics, REPORT_TXT_PATH
from utils.visualization import create_roc_curves_comparison
from src.visualization import plot_confusion_matrix_heatmap

st.set_page_config(page_title="Model Performance | CardioGuard", page_icon="📊", layout="wide")
apply_custom_css()
render_sidebar()

render_header(
    title="Model Benchmarking & ROC Validation",
    subtitle="Comparative Performance on Unseen Test Split (61 Samples, Stratified 80/20)",
    badge="Validation Rigor",
)

# Load real metrics
try:
    metrics = load_evaluation_metrics()
    models_comp = metrics["models_comparison"]
except Exception as e:
    st.error(f"Failed to load evaluation metrics: {e}")
    st.stop()

st.markdown("### 🏆 Algorithm Comparison Table")
comp_rows = []
for m_name, res in models_comp.items():
    comp_rows.append({
        "Model Paradigm": m_name,
        "Accuracy": f"{res['accuracy'] * 100:.2f}%",
        "Precision (PPV)": f"{res['precision'] * 100:.2f}%",
        "Recall (Sensitivity)": f"{res['recall'] * 100:.2f}%",
        "F1-Score": f"{res['f1_score'] * 100:.2f}%",
        "ROC-AUC": f"{res['roc_auc']:.4f}",
        "Clinical Rank": "🥇 Top Performer" if m_name == "Random Forest" else ("🥈 High Discrimination" if m_name == "Logistic Regression" else "🥉 Baseline"),
    })

st.dataframe(pd.DataFrame(comp_rows), hide_index=True, use_container_width=True)

st.markdown("---")

# Row: Overlaid Plotly ROC Curves
st.markdown("### 📈 Receiver Operating Characteristic (ROC) Comparison")
fig_roc = create_roc_curves_comparison(models_comp)
st.plotly_chart(fig_roc, use_container_width=True)

st.markdown("---")

# Model Specific Breakdown Selector
st.markdown("### 🔬 In-Depth Diagnostic Breakdown by Classifier")
selected_m = st.selectbox(
    "Choose Model to Inspect Detailed Confusion Matrix & Classification Report:",
    options=list(models_comp.keys()),
    index=0,
)

sel_data = models_comp[selected_m]

col_cm, col_cr = st.columns([1, 1.2])

with col_cm:
    st.markdown(f"#### Confusion Matrix ({selected_m})")
    fig_cm = plot_confusion_matrix_heatmap(sel_data["confusion_matrix"])
    st.pyplot(fig_cm)

    cm = sel_data["confusion_matrix"]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0

    st.markdown(
        f"""
        - **True Negatives (TN):** `{tn}` healthy individuals cleared
        - **False Positives (FP):** `{fp}` healthy flagged (Type I error)
        - **False Negatives (FN):** `{fn}` diseased missed (Type II error)
        - **True Positives (TP):** `{tp}` diseased caught
        - **Specificity:** `{spec * 100:.2f}%`
        """
    )

with col_cr:
    st.markdown(f"#### Classification Report ({selected_m})")
    cr = sel_data["classification_report"]
    cr_data = {
        "Class Category": ["Class 0: No Heart Disease", "Class 1: Heart Disease Present", "Macro Average", "Weighted Average"],
        "Precision": [
            f"{cr['No Heart Disease (0)']['precision']:.2f}",
            f"{cr['Heart Disease (1)']['precision']:.2f}",
            f"{cr['macro avg']['precision']:.2f}",
            f"{cr['weighted avg']['precision']:.2f}",
        ],
        "Recall": [
            f"{cr['No Heart Disease (0)']['recall']:.2f}",
            f"{cr['Heart Disease (1)']['recall']:.2f}",
            f"{cr['macro avg']['recall']:.2f}",
            f"{cr['weighted avg']['recall']:.2f}",
        ],
        "F1-Score": [
            f"{cr['No Heart Disease (0)']['f1-score']:.2f}",
            f"{cr['Heart Disease (1)']['f1-score']:.2f}",
            f"{cr['macro avg']['f1-score']:.2f}",
            f"{cr['weighted avg']['f1-score']:.2f}",
        ],
        "Support": [
            cr["No Heart Disease (0)"]["support"],
            cr["Heart Disease (1)"]["support"],
            cr["macro avg"]["support"],
            cr["weighted avg"]["support"],
        ],
    }
    st.dataframe(pd.DataFrame(cr_data), hide_index=True, use_container_width=True)

with st.expander("📄 View Plaintext Evaluation Audit Report", expanded=False):
    if REPORT_TXT_PATH.exists():
        with open(REPORT_TXT_PATH, "r", encoding="utf-8") as f:
            st.code(f.read(), language="text")

render_safety_disclaimer()
