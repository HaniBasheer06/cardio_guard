"""Model-agnostic, local sensitivity explanations for demonstration predictions."""

from typing import Any, Dict, List

import pandas as pd

from src.preprocessing import FEATURE_METADATA, FEATURE_NAMES, validate_input_data
from src.prediction import HeartDiseasePredictor


def build_reference_profile(features: pd.DataFrame) -> Dict[str, Any]:
    """Build a typical cohort profile using medians and most-common categories."""
    reference: Dict[str, Any] = {}
    for feature in FEATURE_NAMES:
        values = pd.to_numeric(features[feature], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"No valid cohort values are available for '{feature}'.")

        if "options" in FEATURE_METADATA[feature]:
            value = values.mode().iloc[0]
            reference[feature] = int(value)
        else:
            reference[feature] = float(values.median())
    return reference


def explain_prediction(
    predictor: HeartDiseasePredictor,
    input_values: Dict[str, Any],
    reference_profile: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Estimate per-feature score sensitivity by replacing values with cohort typicals.

    This one-feature-at-a-time comparison is descriptive, not causal, and does not
    attribute interactions between features.
    """
    current = validate_input_data(input_values)
    if not hasattr(predictor.pipeline, "predict_proba"):
        raise RuntimeError("The loaded model cannot produce probability scores.")

    classes = list(predictor.pipeline.classes_)
    positive_index = classes.index(1)
    current_score = float(predictor.pipeline.predict_proba(current)[0][positive_index])
    explanations: List[Dict[str, Any]] = []

    for feature in FEATURE_NAMES:
        reference_value = reference_profile[feature]
        changed = current.copy()
        changed.loc[changed.index[0], feature] = reference_value
        reference_score = float(
            predictor.pipeline.predict_proba(changed)[0][positive_index]
        )
        score_change = current_score - reference_score
        explanations.append(
            {
                "Feature": FEATURE_METADATA[feature]["label"],
                "Input value": input_values[feature],
                "Cohort reference": reference_value,
                "Score change": score_change,
            }
        )

    return sorted(explanations, key=lambda row: abs(row["Score change"]), reverse=True)
