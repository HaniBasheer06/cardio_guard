"""
Heart Disease Prediction - Multi-Model Training Script
Executes end-to-end dataset loading, preprocessing, and training for:
1. Random Forest Classifier
2. Logistic Regression
3. Decision Tree Classifier
Evaluates on held-out test data and serializes artifacts to model/ and reports/.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_training import train_and_evaluate_all_models, MODEL_PATH, REPORT_TXT_PATH, METRICS_PATH


def main():
    print("=" * 80)
    print(" CARDIO GUARD - MULTI-MODEL TRAINING & BENCHMARKING PIPELINE")
    print("=" * 80)
    print("1. Loading UCI Cleveland Heart Disease Dataset (id=45)...")
    print("2. Preprocessing & establishing 80/20 stratified split (seed=42)...")
    print("3. Training & comparing: Random Forest, Logistic Regression, Decision Tree...")

    metrics = train_and_evaluate_all_models(test_size=0.2, random_state=42)

    print("\n" + "=" * 80)
    print(" MODEL BENCHMARK COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Model':<22} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'ROC-AUC':<10}")
    print("-" * 80)
    for model_name, res in metrics["models_comparison"].items():
        print(
            f"{model_name:<22} "
            f"{res['accuracy'] * 100:6.2f}%     "
            f"{res['precision'] * 100:6.2f}%     "
            f"{res['recall'] * 100:6.2f}%     "
            f"{res['f1_score'] * 100:6.2f}%     "
            f"{res['roc_auc']:6.4f}"
        )

    print("-" * 80)
    print(f" PRIMARY MODEL: Random Forest (Accuracy: {metrics['accuracy'] * 100:.2f}%, Recall: {metrics['recall'] * 100:.2f}%)")
    print(" TOP 5 PREDICTIVE CLINICAL FEATURES:")
    for rank, (feat, imp) in enumerate(list(metrics["feature_importances"].items())[:5], 1):
        print(f"   {rank}. {feat:<12}: {imp * 100:5.2f}%")
    print("-" * 80)
    print(f" Saved Model Pipeline: {MODEL_PATH}")
    print(f" Saved Evaluation JSON: {METRICS_PATH}")
    print(f" Saved Full Report:    {REPORT_TXT_PATH}")
    print("=" * 80)
    print("SUCCESS: Multi-model training complete.")


if __name__ == "__main__":
    main()
