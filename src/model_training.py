"""
Multi-Model Training and Evaluation Engine.
Trains, evaluates, and compares:
1. Random Forest Classifier (Primary Ensemble)
2. Logistic Regression (Linear Baseline)
3. Decision Tree Classifier (Single Tree Baseline)

Computes Accuracy, Precision, Recall, F1, Confusion Matrix, and ROC-AUC curves on held-out test data.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

from src.data_loader import load_dataset
from utils.preprocessing import (
    FEATURE_NAMES,
    create_preprocessor,
    binarize_target,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "heart_disease_rf_model.joblib"
LOGREG_PATH = MODEL_DIR / "heart_disease_logreg_model.joblib"
DTREE_PATH = MODEL_DIR / "heart_disease_dtree_model.joblib"
REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
REPORT_TXT_PATH = REPORTS_DIR / "evaluation_report.txt"
TEST_DATA_PATH = REPORTS_DIR / "test_data.csv"


def train_and_evaluate_all_models(
    test_size: float = 0.2, random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains and compares Random Forest, Logistic Regression, and Decision Tree.
    Uses exact same stratified train/test split and preprocessing pipeline.
    """
    # 1. Load data
    X_raw, y_raw = load_dataset()
    X = X_raw[FEATURE_NAMES].copy()
    y = binarize_target(y_raw)

    # 2. Stratified train-test split (avoids data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    models_config = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=6,
            min_samples_split=4,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            C=1.0,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            min_samples_split=6,
            min_samples_leaf=3,
            random_state=random_state,
        ),
    }

    comparison_results: Dict[str, Any] = {}
    fitted_pipelines: Dict[str, Pipeline] = {}

    for name, clf in models_config.items():
        preprocessor = create_preprocessor()
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", clf),
            ]
        )

        # Fit exclusively on training set
        pipeline.fit(X_train, y_train)
        fitted_pipelines[name] = pipeline

        # Predictions & Probabilities
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        auc = float(roc_auc_score(y_test, y_proba))
        cm = confusion_matrix(y_test, y_pred).tolist()
        clf_rep = classification_report(
            y_test,
            y_pred,
            target_names=["No Heart Disease (0)", "Heart Disease (1)"],
            output_dict=True,
        )

        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        comparison_results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "confusion_matrix": cm,
            "classification_report": clf_rep,
            "roc_curve": {
                "fpr": fpr.tolist(),
                "tpr": tpr.tolist(),
            },
        }

    # Primary model is Random Forest
    rf_pipeline = fitted_pipelines["Random Forest"]
    preprocessor_step = rf_pipeline.named_steps["preprocessor"]
    classifier_step = rf_pipeline.named_steps["classifier"]
    encoded_feature_names = preprocessor_step.get_feature_names_out()
    raw_importances = classifier_step.feature_importances_

    # Aggregate feature importances back to original 13 features
    feature_importances: Dict[str, float] = {f: 0.0 for f in FEATURE_NAMES}
    for feat_name, imp in zip(encoded_feature_names, raw_importances):
        feature_key = feat_name.split("__")[1].split("_")[0]
        feature_importances[feature_key] = feature_importances.get(feature_key, 0.0) + float(imp)

    total_imp = sum(feature_importances.values())
    if total_imp > 0:
        feature_importances = {
            k: float(v / total_imp)
            for k, v in sorted(feature_importances.items(), key=lambda item: item[1], reverse=True)
        }

    # Master metrics dictionary
    master_metrics = {
        "primary_model": "Random Forest",
        "accuracy": comparison_results["Random Forest"]["accuracy"],
        "precision": comparison_results["Random Forest"]["precision"],
        "recall": comparison_results["Random Forest"]["recall"],
        "f1_score": comparison_results["Random Forest"]["f1_score"],
        "roc_auc": comparison_results["Random Forest"]["roc_auc"],
        "confusion_matrix": comparison_results["Random Forest"]["confusion_matrix"],
        "classification_report": comparison_results["Random Forest"]["classification_report"],
        "feature_importances": feature_importances,
        "models_comparison": comparison_results,
        "test_sample_count": len(y_test),
        "train_sample_count": len(y_train),
        "random_state": random_state,
        "test_size": test_size,
    }

    # Persist artifacts
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    joblib.dump(fitted_pipelines["Random Forest"], MODEL_PATH)
    joblib.dump(fitted_pipelines["Logistic Regression"], LOGREG_PATH)
    joblib.dump(fitted_pipelines["Decision Tree"], DTREE_PATH)

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(master_metrics, f, indent=4)

    # Format comparison report
    report_text = f"""================================================================================
HEART DISEASE PREDICTION - COMPREHENSIVE MULTI-MODEL EVALUATION REPORT
Dataset: UCI Cleveland Heart Disease Dataset (id=45)
Split Strategy: Stratified 80/20 Holdout (random_state={random_state})
================================================================================

1. MODEL BENCHMARK COMPARISON TABLE
--------------------------------------------------------------------------------
Model                 Accuracy    Precision    Recall      F1-Score    ROC-AUC
--------------------------------------------------------------------------------
"""
    for m_name, res in comparison_results.items():
        report_text += (
            f"{m_name:<20}  {res['accuracy']*100:6.2f}%    "
            f"{res['precision']*100:6.2f}%    {res['recall']*100:6.2f}%    "
            f"{res['f1_score']*100:6.2f}%    {res['roc_auc']:6.4f}\n"
        )

    report_text += f"""
2. PRIMARY MODEL (RANDOM FOREST) CONFUSION MATRIX
--------------------------------------------------------------------------------
True Negatives (TN): {master_metrics['confusion_matrix'][0][0]}
False Positives (FP): {master_metrics['confusion_matrix'][0][1]}
False Negatives (FN): {master_metrics['confusion_matrix'][1][0]}
True Positives (TP): {master_metrics['confusion_matrix'][1][1]}

3. TOP PREDICTIVE FEATURES (AGGREGATED GINI IMPORTANCE):
--------------------------------------------------------------------------------
"""
    for rank, (feat, val) in enumerate(feature_importances.items(), 1):
        report_text += f"  {rank:2d}. {feat:<12}: {val * 100:6.2f}%\n"

    report_text += """
================================================================================
DISCLAIMER: Strictly for educational and technical research purposes.
================================================================================
"""

    with open(REPORT_TXT_PATH, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Save test set with predictions
    test_df = X_test.copy()
    test_df["actual"] = y_test.values
    test_df["rf_predicted"] = rf_pipeline.predict(X_test)
    test_df["rf_proba"] = rf_pipeline.predict_proba(X_test)[:, 1]
    test_df.to_csv(TEST_DATA_PATH, index=False)

    return master_metrics


def load_trained_pipeline(model_name: str = "Random Forest") -> Pipeline:
    """Loads specified pipeline from disk (defaults to Random Forest)."""
    mapping = {
        "Random Forest": MODEL_PATH,
        "Logistic Regression": LOGREG_PATH,
        "Decision Tree": DTREE_PATH,
    }
    target_path = mapping.get(model_name, MODEL_PATH)
    if not target_path.exists():
        train_and_evaluate_all_models()
    return joblib.load(target_path)


def load_evaluation_metrics() -> Dict[str, Any]:
    """Loads multi-model comparison evaluation metrics from disk."""
    if not METRICS_PATH.exists():
        return train_and_evaluate_all_models()
    with open(METRICS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def train_and_evaluate_model(test_size: float = 0.2, random_state: int = 42):
    """Backward compatibility wrapper returning (primary_pipeline, metrics)."""
    metrics = train_and_evaluate_all_models(test_size, random_state)
    return load_trained_pipeline("Random Forest"), metrics
