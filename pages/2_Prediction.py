"""
Page 2: Heart Disease Prediction & Batch Analytics
Interactive single-patient diagnostic assessment with SHAP Explainable AI,
downloadable PDF clinical reports, and CSV batch processing.
"""

import sys
import io
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
from utils.visualization import create_shap_local_waterfall
from utils.report_generator import generate_pdf_report
from utils.preprocessing import FEATURE_NAMES

st.set_page_config(page_title="Prediction | CardioGuard", page_icon="⚡", layout="wide")
apply_custom_css()
render_sidebar()

render_header(
    title="Clinical Risk Prediction & Diagnostics",
    subtitle="Interactive Patient Risk Stratification, SHAP Local Attribution, and Batch Analytics",
    badge="Inference & XAI",
)

# Create tabs for Single Patient vs Batch CSV
tab_single, tab_batch = st.tabs(["🩺 Single Patient Assessment", "📁 Batch CSV Prediction"])

# ==============================================================================
# TAB 1: SINGLE PATIENT ASSESSMENT
# ==============================================================================
with tab_single:
    st.markdown("### 📋 Patient Clinical Profile Form")
    st.write("Input patient diagnostic markers across all 13 standard clinical parameters:")

    # Model Selection & Demo Presets
    c_m1, c_m2 = st.columns([1, 2])
    with c_m1:
        selected_model = st.selectbox(
            "Select Inference Model",
            options=["Random Forest", "Logistic Regression", "Decision Tree"],
            index=0,
            help="Choose the underlying classifier used for inference.",
        )
    with c_m2:
        st.write("**Quick Demonstration Presets:**")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            if st.button("Load Typical Low-Risk Case", use_container_width=True):
                st.session_state["p_age"] = 45
                st.session_state["p_sex"] = 0
                st.session_state["p_cp"] = 2
                st.session_state["p_trestbps"] = 118
                st.session_state["p_chol"] = 185
                st.session_state["p_fbs"] = 0
                st.session_state["p_restecg"] = 0
                st.session_state["p_thalach"] = 172
                st.session_state["p_exang"] = 0
                st.session_state["p_oldpeak"] = 0.2
                st.session_state["p_slope"] = 1
                st.session_state["p_ca"] = 0
                st.session_state["p_thal"] = 3
                st.rerun()
        with p_c2:
            if st.button("Load Typical High-Risk Case", use_container_width=True):
                st.session_state["p_age"] = 62
                st.session_state["p_sex"] = 1
                st.session_state["p_cp"] = 4
                st.session_state["p_trestbps"] = 160
                st.session_state["p_chol"] = 286
                st.session_state["p_fbs"] = 1
                st.session_state["p_restecg"] = 2
                st.session_state["p_thalach"] = 108
                st.session_state["p_exang"] = 1
                st.session_state["p_oldpeak"] = 2.8
                st.session_state["p_slope"] = 2
                st.session_state["p_ca"] = 3
                st.session_state["p_thal"] = 7
                st.rerun()

    # Form with 13 features organized into 3 panels
    with st.form("single_prediction_form"):
        p_tab1, p_tab2, p_tab3 = st.tabs([
            "1. Demographics & Baseline Vitals",
            "2. Symptoms & Exercise Stress Test",
            "3. Fluoroscopy & Nuclear Perfusion",
        ])

        with p_tab1:
            col1, col2 = st.columns(2)
            with col1:
                age = st.number_input(
                    "1. Age (years)",
                    min_value=18,
                    max_value=100,
                    value=st.session_state.get("p_age", 55),
                    help="Patient age in years.",
                )
                sex = st.selectbox(
                    "2. Biological Sex",
                    options=[1, 0],
                    format_func=lambda x: "Male (1)" if x == 1 else "Female (0)",
                    index=0 if st.session_state.get("p_sex", 1) == 1 else 1,
                )
                trestbps = st.number_input(
                    "4. Resting Blood Pressure (mm Hg)",
                    min_value=80,
                    max_value=220,
                    value=st.session_state.get("p_trestbps", 130),
                )
            with col2:
                chol = st.number_input(
                    "5. Serum Cholesterol (mg/dl)",
                    min_value=100,
                    max_value=600,
                    value=st.session_state.get("p_chol", 240),
                )
                fbs = st.selectbox(
                    "6. Fasting Blood Sugar > 120 mg/dl",
                    options=[0, 1],
                    format_func=lambda x: "False: <= 120 mg/dl (0)" if x == 0 else "True: > 120 mg/dl (1)",
                    index=0 if st.session_state.get("p_fbs", 0) == 0 else 1,
                )

        with p_tab2:
            col3, col4 = st.columns(2)
            with col3:
                cp = st.selectbox(
                    "3. Chest Pain Type (cp)",
                    options=[1, 2, 3, 4],
                    format_func=lambda x: {
                        1: "Typical Angina (1)",
                        2: "Atypical Angina (2)",
                        3: "Non-anginal Pain (3)",
                        4: "Asymptomatic (4)",
                    }[x],
                    index=[1, 2, 3, 4].index(st.session_state.get("p_cp", 2)),
                )
                restecg = st.selectbox(
                    "7. Resting ECG Results (restecg)",
                    options=[0, 1, 2],
                    format_func=lambda x: {
                        0: "Normal (0)",
                        1: "ST-T Wave Abnormality (1)",
                        2: "Left Ventricular Hypertrophy (2)",
                    }[x],
                    index=[0, 1, 2].index(st.session_state.get("p_restecg", 0)),
                )
                thalach = st.number_input(
                    "8. Maximum Heart Rate Achieved (thalach)",
                    min_value=60,
                    max_value=220,
                    value=st.session_state.get("p_thalach", 150),
                )
            with col4:
                exang = st.selectbox(
                    "9. Exercise-Induced Angina (exang)",
                    options=[0, 1],
                    format_func=lambda x: "No: Absent (0)" if x == 0 else "Yes: Present (1)",
                    index=0 if st.session_state.get("p_exang", 0) == 0 else 1,
                )
                oldpeak = st.number_input(
                    "10. ST Depression Induced by Exercise (oldpeak)",
                    min_value=0.0,
                    max_value=7.0,
                    value=float(st.session_state.get("p_oldpeak", 1.0)),
                    step=0.1,
                )
                slope = st.selectbox(
                    "11. Slope of Peak Exercise ST Segment (slope)",
                    options=[1, 2, 3],
                    format_func=lambda x: {
                        1: "Upsloping (1)",
                        2: "Flat (2)",
                        3: "Downsloping (3)",
                    }[x],
                    index=[1, 2, 3].index(st.session_state.get("p_slope", 1)),
                )

        with p_tab3:
            col5, col6 = st.columns(2)
            with col5:
                ca = st.selectbox(
                    "12. Number of Major Vessels Colored by Fluoroscopy (ca)",
                    options=[0, 1, 2, 3],
                    format_func=lambda x: f"{x} Major Vessel(s)",
                    index=[0, 1, 2, 3].index(st.session_state.get("p_ca", 0)),
                )
            with col6:
                thal = st.selectbox(
                    "13. Thalassemia Status (thal)",
                    options=[3, 6, 7],
                    format_func=lambda x: {
                        3: "Normal (3)",
                        6: "Fixed Defect (6)",
                        7: "Reversible Defect (7)",
                    }[x],
                    index=[3, 6, 7].index(st.session_state.get("p_thal", 3)),
                )

        st.markdown("<br>", unsafe_allow_html=True)
        submit_btn = st.form_submit_button("🩺 Run Risk Prediction", use_container_width=True)

    if submit_btn:
        payload = {
            "age": age,
            "sex": sex,
            "cp": cp,
            "trestbps": trestbps,
            "chol": chol,
            "fbs": fbs,
            "restecg": restecg,
            "thalach": thalach,
            "exang": exang,
            "oldpeak": oldpeak,
            "slope": slope,
            "ca": ca,
            "thal": thal,
        }

        try:
            predictor = HeartDiseasePredictor(model_name=selected_model)
            result = predictor.predict(payload, save_history=True)

            is_positive = result["prediction"] == 1
            prob_dis = result["probability_disease"]
            prob_safe = result["probability_no_disease"]
            risk_tier = result["risk_level"]

            st.markdown("---")
            st.markdown("### 📊 Diagnostic Evaluation Result")

            if is_positive:
                st.markdown(
                    f"""
                    <div class="result-card-positive">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h2 style="color: #be123c; margin: 0;">⚠️ Heart Disease Risk Detected (Class 1)</h2>
                            <span class="badge-tag badge-high">{risk_tier}</span>
                        </div>
                        <p style="color: #4c0519; margin-top: 0.75rem; font-size: 1.05rem;">
                            Model classifies this profile as having an estimated risk probability of 
                            <strong>{prob_dis * 100:.2f}%</strong>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="result-card-negative">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h2 style="color: #15803d; margin: 0;">✅ No Heart Disease Detected (Class 0)</h2>
                            <span class="badge-tag badge-low">{risk_tier}</span>
                        </div>
                        <p style="color: #14532d; margin-top: 0.75rem; font-size: 1.05rem;">
                            Model classifies this profile as negative for coronary stenosis with 
                            a healthy confidence score of <strong>{prob_safe * 100:.2f}%</strong>.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # Probabilities
            c_p1, c_p2 = st.columns(2)
            with c_p1:
                st.write(f"**Healthy Class Probability (0):** {prob_safe * 100:.2f}%")
                st.progress(prob_safe)
            with c_p2:
                st.write(f"**Heart Disease Probability (1):** {prob_dis * 100:.2f}%")
                st.progress(prob_dis)

            # SHAP Explainability Section
            shap_info = result.get("shap_analysis", {})
            if shap_info.get("available", False):
                st.markdown("### 🔍 Explainable AI (SHAP Local Attribution)")
                st.caption(
                    "SHAP (SHapley Additive exPlanations) decomposes the prediction into exact feature-level "
                    "contributions for this patient. Red bars elevate risk, green bars protect/reduce risk."
                )

                fig_shap = create_shap_local_waterfall(shap_info)
                st.plotly_chart(fig_shap, use_container_width=True)

                st.info(
                    f"💡 **Clinical Decision Narrative:** {shap_info.get('summary', '')} "
                    f"*{shap_info.get('note', '')}*"
                )

            # PDF Download Section
            st.markdown("### 📑 Generate Clinical PDF Report")
            try:
                pdf_bytes = generate_pdf_report(result)
                st.download_button(
                    label="📥 Download Clinical PDF Report",
                    data=pdf_bytes,
                    file_name=f"CardioGuard_Report_{result['timestamp'].replace(':', '-').replace(' ', '_')}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as pdf_err:
                st.warning(f"PDF generation note: {pdf_err}")

            st.caption(f"🛡️ **Academic Governance:** {result['disclaimer']}")

        except Exception as err:
            st.error(f"Inference error: {err}")

# ==============================================================================
# TAB 2: CSV BATCH PREDICTION
# ==============================================================================
with tab_batch:
    st.markdown("### 📁 Batch Multi-Patient CSV Prediction")
    st.write(
        "Upload a CSV file containing patient records with the 13 required clinical features. "
        "The system will validate columns, impute any missing entries safely, and generate batch predictions."
    )

    # Download Template CSV
    template_df = pd.DataFrame([
        {
            "age": 55, "sex": 1, "cp": 2, "trestbps": 130, "chol": 240,
            "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0, "oldpeak": 1.0,
            "slope": 1, "ca": 0, "thal": 3
        },
        {
            "age": 62, "sex": 1, "cp": 4, "trestbps": 160, "chol": 286,
            "fbs": 1, "restecg": 2, "thalach": 108, "exang": 1, "oldpeak": 2.8,
            "slope": 2, "ca": 3, "thal": 7
        }
    ])
    csv_template = template_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Sample CSV Template",
        data=csv_template,
        file_name="heart_disease_batch_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Upload Patient Records (CSV format)", type=["csv"])

    if uploaded_file is not None:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write(f"**Loaded {len(input_df)} records.** Preview of uploaded data:")
            st.dataframe(input_df.head(), use_container_width=True)

            if st.button("🚀 Process Batch Predictions", use_container_width=True):
                predictor = HeartDiseasePredictor(model_name="Random Forest")
                processed_df, summary = predictor.predict_batch(input_df)

                st.success(
                    f"Successfully processed {summary['total_records']} patient records! "
                    f"Found {summary['positive_count']} predicted positive ({summary['positive_count']/summary['total_records']*100:.1f}%) "
                    f"and {summary['negative_count']} predicted negative."
                )

                if summary["warnings"]:
                    for w in summary["warnings"]:
                        st.warning(w)

                # Metric Cards for Batch
                b1, b2, b3 = st.columns(3)
                b1.metric("High Risk Patients", f"{summary['high_risk_count']}")
                b2.metric("Moderate Risk Patients", f"{summary['moderate_risk_count']}")
                b3.metric("Low Risk Patients", f"{summary['low_risk_count']}")

                st.markdown("#### Annotated Output Data:")
                st.dataframe(processed_df, use_container_width=True)

                # Download processed CSV
                out_csv = processed_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="📥 Download Annotated Predictions CSV",
                    data=out_csv,
                    file_name="CardioGuard_Batch_Predictions_Output.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        except Exception as e:
            st.error(f"Error processing uploaded CSV: {e}")

render_safety_disclaimer()
