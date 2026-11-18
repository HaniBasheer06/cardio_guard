# 🫀 CardioGuard: AI-Powered Healthcare Analytics & Heart Disease Prediction

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-F7931E.svg)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable%20AI-success.svg)](https://shap.readthedocs.io/)
[![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-blueviolet.svg)](https://plotly.com/)
[![ReportLab](https://img.shields.io/badge/ReportLab-Clinical%20PDF-red.svg)](https://www.reportlab.com/)

> **Academic Project Title:** Heart Disease Prediction Using Machine Learning  
> **Course / Purpose:** Capstone & College Faculty Demonstration Project  
> **Primary Models:** Random Forest, Logistic Regression, Decision Tree  
> **Explainability:** TreeExplainer SHAP Local Feature Attribution  
> **Frontend:** Modern Multi-Page Streamlit Dashboard with Dark/Light Mode  

---

## 📌 1. Project Introduction

Cardiovascular diseases (CVDs) remain the leading cause of global mortality, claiming roughly 17.9 million lives each year according to the World Health Organization (WHO). Standard coronary angiography is invasive and expensive.

**CardioGuard** is an upgraded, AI-powered healthcare analytics platform designed to assess cardiovascular disease risk non-invasively using 13 clinical markers from the benchmark **UCI Cleveland Heart Disease dataset**.

### Key System Capabilities:
1. **Automated UCI Data Ingestion:** Automated loading via `ucimlrepo` with an offline fallback cache.
2. **Leakage-Free Preprocessing:** Pipelines with median continuous imputation, mode categorical imputation, standard scaling, and one-hot encoding.
3. **Multi-Model Benchmarking:** Side-by-side empirical comparison of:
   - **Random Forest Classifier** (Primary Ensemble - 90.16% Accuracy, 96.43% Recall)
   - **Logistic Regression** (Linear Baseline - 86.89% Accuracy, 92.86% Recall)
   - **Decision Tree Classifier** (Single-Tree Baseline - 78.69% Accuracy, 78.57% Recall)
4. **Explainable AI (SHAP):** TreeExplainer local attribution showing exact risk-elevating and risk-protecting features for every patient.
5. **Clinical PDF Generation:** Instant high-resolution ReportLab PDF reports.
6. **Batch CSV Processing:** Upload multi-patient CSV files for validation, missing data handling, and batch risk scoring with CSV export.
7. **Local Prediction Audit History:** Local session logging of demonstration runs with timestamps and zero PII storage.
8. **Dark Mode & Light Mode:** Toggleable medical theme styling.

---

## 💻 2. Technology Stack

| Domain | Technology | Purpose |
|---|---|---|
| **Programming Language** | Python 3.10+ | Core language for training and application backend |
| **Frontend Framework** | Streamlit 1.30+ | Multi-page clinical dashboard with custom CSS theme |
| **Machine Learning** | Scikit-learn 1.3+ | `Pipeline`, `ColumnTransformer`, `RandomForest`, `LogisticRegression`, `DecisionTree` |
| **Explainable AI** | SHAP 0.42+ | TreeExplainer feature attributions for individual predictions |
| **Interactive Visualizations** | Plotly 5.15+ | Interactive ROC curves, distributions, correlation heatmaps |
| **Static Visualizations** | Matplotlib & Seaborn | Confusion matrix heatmaps and diagnostic plots |
| **PDF Report Compilation** | ReportLab 4.0+ | Generation of clinical demonstration PDF documents |
| **Model Persistence** | Joblib 1.3+ | Disk serialization of fitted Scikit-learn pipelines |
| **Data Ingestion & Tabular** | ucimlrepo & Pandas | Direct UCI dataset downloading and DataFrame operations |

---

## 🗄️ 3. Dataset & 13 Clinical Features

- **Source:** [UCI Machine Learning Repository - Heart Disease Dataset (ID: 45)](https://archive.ics.uci.edu/dataset/45/heart+disease)
- **Instances:** 303 patient observations from Cleveland Clinic Foundation

| # | Feature | Clinical Description | Data Type | Range / Permissible Values |
|---|---|---|---|---|
| 1 | `age` | Age in years | Numerical | 29 to 77 |
| 2 | `sex` | Biological sex | Categorical | `1` = Male, `0` = Female |
| 3 | `cp` | Chest pain type | Categorical | `1`: Typical, `2`: Atypical, `3`: Non-anginal, `4`: Asymptomatic |
| 4 | `trestbps` | Resting blood pressure on admission | Numerical | 94 to 200 mm Hg |
| 5 | `chol` | Serum cholesterol | Numerical | 126 to 564 mg/dl |
| 6 | `fbs` | Fasting blood sugar > 120 mg/dl | Categorical | `1` = True, `0` = False |
| 7 | `restecg` | Resting electrocardiogram | Categorical | `0`: Normal, `1`: ST-T wave anomaly, `2`: LV hypertrophy |
| 8 | `thalach` | Maximum heart rate achieved | Numerical | 71 to 202 bpm |
| 9 | `exang` | Exercise-induced angina | Categorical | `1` = Yes, `0` = No |
| 10 | `oldpeak` | ST depression induced by exercise | Numerical | 0.0 to 6.2 mm |
| 11 | `slope` | Slope of peak exercise ST segment | Categorical | `1`: Upsloping, `2`: Flat, `3`: Downsloping |
| 12 | `ca` | Major vessels colored by fluoroscopy | Discrete | 0, 1, 2, 3 (4 missing imputed) |
| 13 | `thal` | Thalassemia status | Categorical | `3`: Normal, `6`: Fixed defect, `7`: Reversible defect |

### Target Mapping:
- Raw dataset `num` is integer `0` to `4`.
- Binarized: **`0` = No Heart Disease** (< 50% stenosis), **`1` = Heart Disease Present** (>= 50% stenosis).

---

## 📁 4. Project Folder Structure

```text
Heart_Disease_Prediction/
│
├── app.py                          # Executive application dashboard entrypoint
├── train_model.py                  # Multi-model training and evaluation script
├── requirements.txt                # Pinned dependencies
├── README.md                       # Complete academic documentation
├── Heart_Disease_Prediction_Project.zip # Standalone project archive
│
├── data/
│   ├── README.md                   # Dataset metadata & feature descriptions
│   └── heart_disease.csv           # Cached UCI Cleveland dataset
│
├── model/
│   ├── .gitkeep
│   ├── heart_disease_rf_model.joblib      # Random Forest fitted pipeline
│   ├── heart_disease_logreg_model.joblib  # Logistic Regression fitted pipeline
│   └── heart_disease_dtree_model.joblib   # Decision Tree fitted pipeline
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # Automated UCI dataset fetcher
│   ├── model_training.py           # Multi-model training and metrics engine
│   ├── preprocessing.py            # Backward compatibility re-export
│   ├── prediction.py               # Backward compatibility re-export
│   ├── visualization.py            # Matplotlib & Plotly bridge
│   └── utils.py                    # Theme toggling, header banner, CSS
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py            # Pipeline ColumnTransformer & input validation
│   ├── prediction.py               # Inference engine, SHAP calculation, history
│   ├── visualization.py            # Plotly interactive charts & ROC curves
│   └── report_generator.py         # ReportLab PDF report generation
│
├── pages/
│   ├── 1_Dashboard.py              # Executive overview & system architecture
│   ├── 2_Prediction.py             # 13-feature diagnostic form, SHAP & CSV batch
│   ├── 3_Data_Analysis.py          # Interactive Plotly cohort visualizations
│   ├── 4_Model_Performance.py      # 3-model benchmark comparison & ROC curves
│   ├── 5_Prediction_History.py     # Local audit trail & run history
│   └── 6_About_Project.py          # Problem statement, algorithms, limitations
│
├── assets/
│   └── style.css                   # Custom CSS styling with light & dark theme
│
└── reports/
    ├── evaluation_report.txt       # Plaintext multi-model evaluation report
    ├── metrics.json                # Complete benchmark metrics dictionary
    ├── prediction_history.json     # Local demonstration audit log
    └── test_data.csv               # Held-out 20% test partition with predictions
```

---

## ⚡ 5. Step-by-Step Installation & Execution Guide

### Step 1: Open Folder in VS Code
Open VS Code, press `Ctrl+K Ctrl+O` (File -> Open Folder), and select the project folder:
```text
c:\Users\LENOVO\OneDrive\Desktop\Heart_Disease_Prediction
```

### Step 2: Open Terminal and Activate Virtual Environment
Open terminal in VS Code (``Ctrl + ` ``) and activate the environment:
```powershell
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### Step 3: Train & Benchmark All Models
Run the multi-model training script:
```powershell
python train_model.py
```
This trains **Random Forest**, **Logistic Regression**, and **Decision Tree** on the identical stratified 80/20 split and outputs the benchmark table:
```text
================================================================================
 CARDIO GUARD - MULTI-MODEL TRAINING & BENCHMARKING PIPELINE
================================================================================
Model                  Accuracy     Precision    Recall       F1-Score     ROC-AUC   
--------------------------------------------------------------------------------
Random Forest           90.16%      84.38%      96.43%      90.00%     0.9437
Logistic Regression     86.89%      81.25%      92.86%      86.67%     0.9578
Decision Tree           78.69%      75.86%      78.57%      77.19%     0.8701
--------------------------------------------------------------------------------
SUCCESS: Multi-model training complete.
```

### Step 4: Launch the Streamlit Dashboard
```powershell
streamlit run app.py
```
Streamlit opens in your browser at `http://localhost:8501`.

---

## 🛡️ 6. Medical Safety Disclaimer

> **IMPORTANT CLINICAL NOTICE:**  
> This software is an **educational machine learning demonstration** created strictly for academic research and coursework.  
> **This system is NOT a clinically certified diagnostic device** and must **NEVER** be used to diagnose, treat, or replace professional medical consultations. Always consult licensed healthcare practitioners and certified cardiologists for medical decisions.
