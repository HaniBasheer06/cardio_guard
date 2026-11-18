"""
Page 6: About Project & Comprehensive Documentation
Academic documentation detailing clinical motivation, target mapping, multi-algorithm theory, and ethical limits.
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

st.set_page_config(page_title="About Project | CardioGuard", page_icon="ℹ️", layout="wide")
apply_custom_css()
render_sidebar()

render_header(
    title="About the CardioGuard System",
    subtitle="Comprehensive Academic Documentation, Mathematical Formulations, and Methodological Governance",
    badge="Documentation & Governance",
)

render_safety_disclaimer()

# Section 1: Problem Statement & Objectives
st.markdown("### 1. Problem Statement & Clinical Objectives")
st.markdown(
    """
    **Problem Statement:**  
    Cardiovascular diseases (CVDs) are the leading contributor to global mortality, claiming roughly 17.9 million 
    lives annually. While invasive diagnostic modalities like coronary angiography remain the gold standard, 
    they are invasive, cost-prohibitive, and frequently inaccessible in low-resource outpatient clinics. 
    There is a critical academic and engineering need for non-invasive machine learning triage tools that 
    can stratify patient risk from standard routine clinical markers without introducing data leakage.

    **Project Objectives:**
    1. **End-to-End Pipeline Automation:** Automatically download and cache verified records from the UCI repository.
    2. **Leakage-Free Preprocessing:** Fit all imputers, scalers, and encoders strictly on the training partition via Scikit-learn Pipelines.
    3. **Multi-Model Benchmarking:** Empirically evaluate Random Forest against Logistic Regression and Decision Tree on held-out test data.
    4. **Explainable AI (XAI):** Unpack individual patient decisions using SHAP (SHapley Additive exPlanations) values.
    5. **Clinical Usability:** Provide downloadable PDF diagnostic reports, CSV batch processing, and audit tracking.
    """
)

st.markdown("---")

# Section 2: Dataset Description & Target Mapping
st.markdown("### 2. Dataset Description & Target Binarization")
st.markdown(
    """
    The project leverages the canonical **Cleveland Heart Disease Database** hosted at the 
    **University of California, Irvine (UCI) Machine Learning Repository** (Dataset ID `45`).

    **Original Clinical Target (`num`):**
    In the raw clinical database, angiographic disease severity was recorded on an integer scale from 0 to 4:
    - **`0`**: No heart disease detected (diameter narrowing < 50% in any major coronary vessel).
    - **`1`**: Mild heart disease present (> 50% diameter narrowing).
    - **`2`**: Moderate heart disease present.
    - **`3`**: Severe heart disease present.
    - **`4`**: Critical multi-vessel disease present.

    **Binary Classification Formulation:**
    In alignment with established cardiological machine learning literature:
    - Target is binarized such that **`0` = No Heart Disease** (`num == 0`) and **`1` = Heart Disease Detected** (`num > 0`).
    - This creates an effective binary screening formulation distinguishing non-stenotic individuals from patients requiring further cardiology workup.
    """
)

st.markdown("---")

# Section 3: Machine Learning Algorithms Used
st.markdown("### 3. Machine Learning Algorithms & Pipeline Architecture")
st.markdown(
    """
    1. **Random Forest Classifier (`sklearn.ensemble.RandomForestClassifier`):**
       - Primary ensemble architecture combining 100 decorrelated decision trees.
       - Hyperparameters: `max_depth=6`, `min_samples_split=4`, `min_samples_leaf=2`, `random_state=42`.
       - Employs Bootstrap Aggregation (bagging) with random feature subspace sampling, minimizing variance.
    2. **Logistic Regression (`sklearn.linear_model.LogisticRegression`):**
       - Serves as the parametric linear baseline with L2 regularization (`C=1.0`, `max_iter=1000`).
       - Provides direct log-odds probability calibration across the decision surface.
    3. **Decision Tree Classifier (`sklearn.tree.DecisionTreeClassifier`):**
       - Serves as the single-tree baseline (`max_depth=4`, `min_samples_split=6`).
       - Illustrates hierarchical split boundaries based on Gini impurity.
    """
)

st.markdown("---")

# Section 4: Technology Stack
st.markdown("### 4. Technology Stack")
tech_df = pd.DataFrame([
    {"Component": "Frontend Dashboard", "Technology": "Streamlit 1.30+", "Role": "Reactive multi-page web UI, theme switcher, forms"},
    {"Component": "Machine Learning", "Technology": "Scikit-Learn 1.3+", "Role": "Pipeline, ColumnTransformer, RF, LogReg, DT"},
    {"Component": "Explainable AI (XAI)", "Technology": "SHAP 0.42+", "Role": "TreeExplainer local feature attribution & waterfall charts"},
    {"Component": "Interactive Charts", "Technology": "Plotly 5.15+", "Role": "ROC curves, histograms, correlation heatmaps, donuts"},
    {"Component": "PDF Generation", "Technology": "ReportLab 4.0+", "Role": "High-resolution clinical PDF report compilation"},
    {"Component": "Data Ingestion", "Technology": "ucimlrepo 0.0.7", "Role": "Automated UCI Cleveland dataset acquisition"},
    {"Component": "Data Manipulation", "Technology": "Pandas & NumPy", "Role": "DataFrames, type safety, matrix operations"},
])
st.dataframe(tech_df, hide_index=True, use_container_width=True)

st.markdown("---")

# Section 5: Limitations & Future Enhancements
col_l, col_f = st.columns(2)

with col_l:
    st.markdown("### 5. Known Limitations")
    st.markdown(
        """
        - **Historical Cohort Size:** The Cleveland dataset contains 303 records collected in 1988; modern medical software requires validation across contemporary multi-ethnic cohorts.
        - **Binary Simplification:** Heart disease is a progressive continuum rather than a strictly discrete binary state.
        - **Imputation:** 6 patient records contained missing values in fluoroscopy (`ca`) and thalassemia (`thal`), imputed via median/mode.
        """
    )

with col_f:
    st.markdown("### 6. Future Enhancements")
    st.markdown(
        """
        - **Multi-Center Federation:** Train and cross-validate across the Hungarian, Swiss, and Long Beach cohorts of the UCI repository.
        - **Deep Neural Architecture:** Implement TabNet or Multi-Layer Perceptrons (MLP) for comparison with tree ensembles.
        - **FHIR & EHR Interoperability:** Support automated data ingestion via standard Fast Healthcare Interoperability Resources (FHIR) protocols.
        """
    )

st.markdown("---")

# Section 6: Academic Disclaimer
st.markdown("### 7. Ethical Governance & Academic Declaration")
st.markdown(
    """
    This software is released strictly for academic demonstration, educational research, 
    and computer science engineering coursework. Under no circumstances should this software 
    be considered a substitute for professional medical advice, clinical diagnosis, or therapeutic regimens.
    Always consult certified cardiologists and licensed healthcare practitioners for cardiovascular assessments.
    """
)
