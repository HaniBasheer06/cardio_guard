"""
Data Loader Module for UCI Cleveland Heart Disease Dataset.
Handles automated fetching from the UCI repository or loading from local cache.
"""

import os
from pathlib import Path
from typing import Tuple, Optional
import pandas as pd

# Define default paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_PATH = PROJECT_ROOT / "data" / "heart_disease.csv"
UCI_DATASET_ID = 45


def load_dataset(use_cache: bool = True, save_cache: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Loads the UCI Cleveland Heart Disease dataset.
    Attempts to fetch from the UCI Machine Learning Repository via ucimlrepo.
    Falls back gracefully to the local CSV file if offline.

    Args:
        use_cache (bool): If True, checks for existing local data/heart_disease.csv before downloading.
        save_cache (bool): If True, saves newly fetched data to data/heart_disease.csv.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X_features, y_target)
    """
    # 1. Check local cache first if requested
    if use_cache and LOCAL_DATA_PATH.exists():
        df = pd.read_csv(LOCAL_DATA_PATH)
        if "num" in df.columns:
            X = df.drop(columns=["num"])
            y = df["num"]
            return X, y

    # 2. Attempt fetching via ucimlrepo
    try:
        from ucimlrepo import fetch_ucirepo
        heart_disease = fetch_ucirepo(id=UCI_DATASET_ID)
        X = heart_disease.data.features.copy()
        y = heart_disease.data.targets["num"].copy() if "num" in heart_disease.data.targets.columns else heart_disease.data.targets.iloc[:, 0].copy()

        # Cache locally
        if save_cache:
            LOCAL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            combined_df = pd.concat([X, y.rename("num")], axis=1)
            combined_df.to_csv(LOCAL_DATA_PATH, index=False)

        return X, y

    except Exception as exc:
        # 3. Fallback to local CSV if ucimlrepo fails (e.g. offline)
        if LOCAL_DATA_PATH.exists():
            df = pd.read_csv(LOCAL_DATA_PATH)
            X = df.drop(columns=["num"])
            y = df["num"]
            return X, y
        raise RuntimeError(
            f"Unable to download dataset from UCI and local file not found at {LOCAL_DATA_PATH}. "
            f"Original error: {exc}"
        ) from exc


def get_dataset_summary() -> dict:
    """
    Returns summary statistics for the dataset.
    """
    X, y = load_dataset()
    binary_y = (y > 0).astype(int)
    
    return {
        "total_instances": len(X),
        "total_features": X.shape[1],
        "feature_names": list(X.columns),
        "missing_values": X.isnull().sum().to_dict(),
        "target_counts_original": y.value_counts().to_dict(),
        "target_counts_binary": binary_y.value_counts().to_dict(),
        "positive_rate": float(binary_y.mean()),
    }
