"""
Page 5: Prediction History & Audit Logging
Local audit tracking of demonstration runs, session predictions, and risk classifications.
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
from utils.prediction import HeartDiseasePredictor

st.set_page_config(page_title="Prediction History | CardioGuard", page_icon="🕒", layout="wide")
apply_custom_css()
render_sidebar()

render_header(
    title="Prediction History & Audit Trail",
    subtitle="Review of Demonstration Runs, Logged Inference Timestamps, and Risk Classifications",
    badge="Session Audit",
)

render_safety_disclaimer()

# Load history records
history_records = HeartDiseasePredictor.load_history()

# Controls bar
c1, c2, c3 = st.columns([1, 1, 2])
with c1:
    st.metric("Total Logged Runs", len(history_records))

with c2:
    if st.button("🗑️ Clear History", use_container_width=True):
        HeartDiseasePredictor.clear_history()
        st.success("History log successfully cleared!")
        st.rerun()

with c3:
    if history_records:
        export_df = pd.DataFrame(history_records)
        csv_data = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Export History to CSV",
            data=csv_data,
            file_name="CardioGuard_Prediction_History.csv",
            mime="text/csv",
            use_container_width=True,
        )

st.markdown("---")

if not history_records:
    st.info("ℹ️ No predictions logged yet. Run predictions from the **Prediction** page to populate this audit table.")
else:
    # Format table for display
    display_rows = []
    for r in history_records:
        display_rows.append({
            "Run ID": r.get("id"),
            "Timestamp": r.get("timestamp"),
            "Model Used": r.get("model", "Random Forest"),
            "Age": r.get("age"),
            "Sex": r.get("sex"),
            "Prediction": r.get("prediction"),
            "Disease Risk Probability": f"{r.get('probability', 0.0)}%",
            "Risk Tier": r.get("risk_level"),
        })

    st.markdown("### 📋 Recent Diagnostic Runs (Newest First)")
    st.dataframe(pd.DataFrame(display_rows), hide_index=True, use_container_width=True)

    st.caption("🔒 **Data Privacy Notice:** All session history is stored purely locally in `reports/prediction_history.json`. No personal identifiable information (PII) is captured or transmitted.")
