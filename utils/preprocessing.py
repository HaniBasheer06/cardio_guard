"""
Preprocessing Module for Heart Disease Prediction.
Handles feature typing, target binarization, missing value imputation, pipeline building,
and single-row / batch-CSV validation.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Exact 13 features in standard clinical order
FEATURE_NAMES: List[str] = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]

# Categorical vs Numerical segregation
NUMERICAL_FEATURES: List[str] = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
CATEGORICAL_FEATURES: List[str] = ["sex", "cp", "fbs", "restecg", "exang", "slope", "thal"]

# Human-readable labels and descriptions
FEATURE_METADATA: Dict[str, Dict[str, Any]] = {
    "age": {"label": "Age", "unit": "years", "min": 18, "max": 100, "default": 55},
    "sex": {
        "label": "Sex",
        "options": {1: "Male (1)", 0: "Female (0)"},
        "default": 1,
    },
    "cp": {
        "label": "Chest Pain Type",
        "options": {
            1: "Typical Angina (1)",
            2: "Atypical Angina (2)",
            3: "Non-anginal Pain (3)",
            4: "Asymptomatic (4)",
        },
        "default": 2,
    },
    "trestbps": {
        "label": "Resting Blood Pressure",
        "unit": "mm Hg",
        "min": 80,
        "max": 220,
        "default": 130,
    },
    "chol": {
        "label": "Serum Cholesterol",
        "unit": "mg/dl",
        "min": 100,
        "max": 600,
        "default": 240,
    },
    "fbs": {
        "label": "Fasting Blood Sugar > 120 mg/dl",
        "options": {1: "True (> 120 mg/dl)", 0: "False (<= 120 mg/dl)"},
        "default": 0,
    },
    "restecg": {
        "label": "Resting ECG Results",
        "options": {
            0: "Normal (0)",
            1: "ST-T Wave Abnormality (1)",
            2: "Probable / Definite LV Hypertrophy (2)",
        },
        "default": 0,
    },
    "thalach": {
        "label": "Maximum Heart Rate Achieved",
        "unit": "bpm",
        "min": 60,
        "max": 220,
        "default": 150,
    },
    "exang": {
        "label": "Exercise Induced Angina",
        "options": {1: "Yes (1)", 0: "No (0)"},
        "default": 0,
    },
    "oldpeak": {
        "label": "ST Depression (Oldpeak)",
        "unit": "mm",
        "min": 0.0,
        "max": 7.0,
        "step": 0.1,
        "default": 1.0,
    },
    "slope": {
        "label": "Slope of Peak Exercise ST Segment",
        "options": {1: "Upsloping (1)", 2: "Flat (2)", 3: "Downsloping (3)"},
        "default": 1,
    },
    "ca": {
        "label": "Major Vessels Colored by Fluoroscopy",
        "options": {0: "0 vessels", 1: "1 vessel", 2: "2 vessels", 3: "3 vessels"},
        "default": 0,
    },
    "thal": {
        "label": "Thalassemia Status",
        "options": {3: "Normal (3)", 6: "Fixed Defect (6)", 7: "Reversible Defect (7)"},
        "default": 3,
    },
}


def binarize_target(y: pd.Series) -> pd.Series:
    """
    Converts multi-class UCI target into binary:
      0 = No heart disease (<50% diameter narrowing)
      1 = Heart disease present (>50% diameter narrowing)
    """
    return (y.astype(float) > 0).astype(int)


def create_preprocessor() -> ColumnTransformer:
    """
    Constructs a Scikit-learn ColumnTransformer.
    - Numerical features: Median Imputation + Standard Scaling
    - Categorical features: Most Frequent Imputation + OneHotEncoding
    """
    numerical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, NUMERICAL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    return preprocessor


def validate_input_data(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Validates a dictionary of user inputs, ensures all 13 features are present,
    converts to correct types, and returns a 1-row DataFrame with exact feature order.
    """
    missing_fields = [feat for feat in FEATURE_NAMES if feat not in data]
    if missing_fields:
        raise ValueError(f"Missing required input features: {missing_fields}")

    cleaned: Dict[str, Any] = {}
    for feat in FEATURE_NAMES:
        val = data[feat]
        try:
            if feat in NUMERICAL_FEATURES:
                cleaned[feat] = float(val)
            else:
                cleaned[feat] = int(val)
        except (ValueError, TypeError) as err:
            raise ValueError(f"Invalid format for field '{feat}': {val}") from err

    return pd.DataFrame([cleaned])[FEATURE_NAMES]


def validate_batch_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validates a batch DataFrame uploaded by the user.
    Checks column presence, maps column names case-insensitively, handles missing values,
    and returns a clean DataFrame with the 13 required features.

    Returns:
        Tuple[pd.DataFrame, List[str]]: (clean_df, warning_messages)
    """
    warnings: List[str] = []
    
    # Map columns case-insensitively
    col_mapping = {c.strip().lower(): c for c in df.columns}
    missing_cols = [f for f in FEATURE_NAMES if f.lower() not in col_mapping]
    
    if missing_cols:
        raise ValueError(
            f"Uploaded CSV is missing the following required clinical columns: {missing_cols}. "
            f"Required columns are: {FEATURE_NAMES}"
        )

    # Rename to canonical lower-case feature names
    rename_dict = {col_mapping[f.lower()]: f for f in FEATURE_NAMES}
    cleaned_df = df[list(rename_dict.keys())].rename(columns=rename_dict).copy()

    # Check for missing values
    null_counts = cleaned_df.isnull().sum()
    if null_counts.sum() > 0:
        affected = {k: int(v) for k, v in null_counts.items() if v > 0}
        warnings.append(
            f"Note: Found missing entries across columns {affected}. "
            "These will be safely imputed using median/mode statistics by the pipeline."
        )

    # Convert types safely
    for num_col in NUMERICAL_FEATURES:
        cleaned_df[num_col] = pd.to_numeric(cleaned_df[num_col], errors="coerce")

    for cat_col in CATEGORICAL_FEATURES:
        cleaned_df[cat_col] = pd.to_numeric(cleaned_df[cat_col], errors="coerce")

    # Reorder columns strictly
    cleaned_df = cleaned_df[FEATURE_NAMES]
    return cleaned_df, warnings
