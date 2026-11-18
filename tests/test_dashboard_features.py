import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.model_comparison import evaluate_model_comparison
from src.model_training import load_evaluation_metrics, load_trained_pipeline
from src.prediction import HeartDiseasePredictor
from src.prediction_history import (
    clear_prediction_history,
    load_prediction_history,
    save_prediction,
)
from src.preprocessing import (
    FEATURE_NAMES,
    binarize_target,
    validate_feature_dataframe,
)
from src.report_generator import generate_prediction_pdf


class DashboardFeatureTests(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "age": 55,
            "sex": 1,
            "cp": 2,
            "trestbps": 130,
            "chol": 240,
            "fbs": 0,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 1.0,
            "slope": 1,
            "ca": 0,
            "thal": 3,
        }

    def test_original_target_values_map_zero_to_negative_and_positive_values_to_class_one(self):
        mapped = binarize_target(pd.Series([0, 1, 2, 3, 4]))
        self.assertEqual(mapped.tolist(), [0, 1, 1, 1, 1])

    def test_batch_validation_preserves_missing_values_and_rejects_bad_columns(self):
        batch = pd.DataFrame([self.sample])
        batch.loc[0, "ca"] = np.nan

        validated = validate_feature_dataframe(batch, allow_missing=True)
        self.assertTrue(pd.isna(validated.loc[0, "ca"]))
        self.assertEqual(list(validated.columns), FEATURE_NAMES)

        with self.assertRaisesRegex(ValueError, "unexpected columns"):
            validate_feature_dataframe(batch.assign(name="not-a-feature"))

        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            validate_feature_dataframe(pd.concat([batch, batch[["ca"]]], axis=1))

        invalid_category = pd.DataFrame([self.sample])
        invalid_category.loc[0, "cp"] = 5
        with self.assertRaisesRegex(ValueError, "must be one of"):
            validate_feature_dataframe(invalid_category)

        invalid_range = pd.DataFrame([self.sample])
        invalid_range.loc[0, "age"] = 101
        with self.assertRaisesRegex(ValueError, "must be 18 to 100"):
            validate_feature_dataframe(invalid_range)

    def test_batch_prediction_imputes_missing_values_and_marks_scores_uncalibrated(self):
        predictor = HeartDiseasePredictor(load_trained_pipeline())
        batch = pd.DataFrame([self.sample, self.sample])
        batch["thal"] = np.nan

        result = predictor.predict_batch(batch)

        self.assertEqual(len(result), 2)
        self.assertTrue(result["prediction"].isin([0, 1]).all())
        self.assertFalse(result["probability_is_calibrated"].any())

    def test_single_prediction_uses_class_labels_without_unvalidated_risk_bands(self):
        predictor = HeartDiseasePredictor(load_trained_pipeline())

        result = predictor.predict(self.sample)

        self.assertEqual(
            result["prediction_label"],
            f"Model output: class {result['prediction']}",
        )
        self.assertFalse(result["probability_is_calibrated"])
        self.assertNotIn("risk_level", result)

    def test_model_comparison_matches_saved_random_forest_holdout(self):
        comparison = evaluate_model_comparison()
        saved_metrics = load_evaluation_metrics()

        self.assertEqual(
            set(comparison["models"]),
            {"Logistic Regression", "Decision Tree", "Random Forest"},
        )
        self.assertEqual(comparison["test_sample_count"], saved_metrics["test_sample_count"])
        self.assertAlmostEqual(
            comparison["models"]["Random Forest"]["accuracy"],
            saved_metrics["accuracy"],
        )
        self.assertEqual(
            comparison["confusion_matrices"]["Random Forest"],
            saved_metrics["confusion_matrix"],
        )

    def test_history_can_be_saved_loaded_and_cleared_locally(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            history_path = Path(temp_dir) / "history.sqlite3"
            with patch("src.prediction_history.HISTORY_DB_PATH", history_path):
                result = {"prediction": 1, "probability_disease": 0.75}
                save_prediction(
                    self.sample,
                    result,
                    datetime(2026, 1, 2, tzinfo=timezone.utc),
                )
                history = load_prediction_history()
                self.assertEqual(len(history), 1)
                self.assertEqual(history.iloc[0]["prediction"], 1)

                clear_prediction_history()
                self.assertTrue(load_prediction_history().empty)

    def test_prediction_report_is_a_pdf_with_disclaimer(self):
        predictor = HeartDiseasePredictor(load_trained_pipeline())
        result = predictor.predict(self.sample)
        report = generate_prediction_pdf(
            self.sample,
            result,
            datetime(2026, 1, 2, tzinfo=timezone.utc),
        )

        self.assertTrue(report.startswith(b"%PDF-"))
        self.assertGreater(len(report), 1000)


if __name__ == "__main__":
    unittest.main()
