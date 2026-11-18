"""Reproducible, held-out comparison of the project's tabular classifiers."""

from typing import Any, Dict

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_loader import load_dataset
from src.model_training import load_evaluation_metrics, load_trained_pipeline
from src.preprocessing import FEATURE_NAMES, binarize_target, create_preprocessor


def _aggregate_random_forest_importances(pipeline: Pipeline) -> Dict[str, float]:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    encoded_names = preprocessor.get_feature_names_out()
    importances = {feature: 0.0 for feature in FEATURE_NAMES}

    for encoded_name, importance in zip(encoded_names, classifier.feature_importances_):
        raw_name = encoded_name.split("__", 1)[-1]
        feature = next(
            (
                name
                for name in FEATURE_NAMES
                if raw_name == name or raw_name.startswith(f"{name}_")
            ),
            None,
        )
        if feature is not None:
            importances[feature] += float(importance)

    total = sum(importances.values())
    if total:
        importances = {feature: value / total for feature, value in importances.items()}
    return importances


def evaluate_model_comparison() -> Dict[str, Any]:
    """Evaluate all models against the same saved stratified holdout.

    The persisted Random Forest is evaluated as-is. The comparison estimators
    are trained on the matching training partition without changing saved files.
    """
    metrics = load_evaluation_metrics()
    random_state = int(metrics.get("random_state", 42))
    test_size = float(metrics.get("test_size", 0.2))

    raw_features, raw_target = load_dataset()
    features = raw_features[FEATURE_NAMES].copy()
    target = binarize_target(raw_target)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    logistic_regression = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                LogisticRegression(max_iter=2000, solver="liblinear"),
            ),
        ]
    )
    decision_tree = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    random_state=random_state,
                ),
            ),
        ]
    )
    random_forest = load_trained_pipeline()
    estimators: Dict[str, Pipeline] = {
        "Logistic Regression": logistic_regression,
        "Decision Tree": decision_tree,
        "Random Forest": random_forest,
    }

    results: Dict[str, Dict[str, Any]] = {}
    roc_data: Dict[str, Dict[str, Any]] = {}
    confusion_matrices: Dict[str, Any] = {}
    for name, estimator in estimators.items():
        if name != "Random Forest":
            estimator.fit(X_train, y_train)

        predictions = estimator.predict(X_test).astype(int)
        classes = list(estimator.classes_)
        if 1 not in classes:
            raise RuntimeError(f"{name} did not produce the positive class.")
        positive_scores = estimator.predict_proba(X_test)[:, classes.index(1)]
        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, positive_scores)

        results[name] = {
            "accuracy": float(accuracy_score(y_test, predictions)),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1_score": float(f1_score(y_test, predictions, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, positive_scores)),
        }
        confusion_matrices[name] = matrix.tolist()
        roc_data[name] = {
            "false_positive_rate": false_positive_rate.tolist(),
            "true_positive_rate": true_positive_rate.tolist(),
            "roc_auc": results[name]["roc_auc"],
        }

    return {
        "models": results,
        "roc_curves": roc_data,
        "confusion_matrices": confusion_matrices,
        "feature_importances": _aggregate_random_forest_importances(random_forest),
        "test_sample_count": int(len(y_test)),
        "train_sample_count": int(len(y_train)),
        "test_size": test_size,
        "random_state": random_state,
    }
