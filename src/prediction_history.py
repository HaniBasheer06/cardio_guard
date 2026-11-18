"""Local SQLite storage for anonymous educational prediction examples."""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pandas as pd

from src.preprocessing import FEATURE_NAMES, validate_input_data

PROJECT_ROOT = Path(__file__).resolve().parent.parent
HISTORY_DB_PATH = PROJECT_ROOT / "data" / "prediction_history.sqlite3"


def _connect() -> sqlite3.Connection:
    HISTORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(HISTORY_DB_PATH, timeout=10)
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            prediction INTEGER NOT NULL,
            estimated_disease_score REAL NOT NULL,
            input_values_json TEXT NOT NULL
        )
        """
    )
    return connection


@contextmanager
def _transaction() -> Iterator[sqlite3.Connection]:
    connection = _connect()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def save_prediction(
    input_values: Dict[str, Any],
    result: Dict[str, Any],
    created_at: Optional[datetime] = None,
) -> None:
    """Persist only feature values and model output; no identifiers are collected."""
    validated = validate_input_data(input_values).iloc[0]
    normalized_inputs = {
        feature: float(validated[feature]) for feature in FEATURE_NAMES
    }
    recorded_at = created_at or datetime.now(timezone.utc)
    if recorded_at.tzinfo is None:
        recorded_at = recorded_at.replace(tzinfo=timezone.utc)
    timestamp = recorded_at.astimezone(timezone.utc).isoformat(timespec="seconds")

    with _transaction() as connection:
        connection.execute(
            """
            INSERT INTO predictions
                (created_at, prediction, estimated_disease_score, input_values_json)
            VALUES (?, ?, ?, ?)
            """,
            (
                timestamp,
                int(result["prediction"]),
                float(result["probability_disease"]),
                json.dumps(normalized_inputs),
            ),
        )


def load_prediction_history(limit: Optional[int] = None) -> pd.DataFrame:
    """Return recent local examples, newest first."""
    if limit is not None and limit < 1:
        raise ValueError("History limit must be positive.")
    if not HISTORY_DB_PATH.exists():
        return pd.DataFrame(
            columns=[
                "id",
                "created_at",
                "prediction",
                "estimated_disease_score",
                "input_values_json",
            ]
        )

    with _transaction() as connection:
        query = """
            SELECT id, created_at, prediction, estimated_disease_score, input_values_json
            FROM predictions
            ORDER BY id DESC
        """
        if limit is None:
            return pd.read_sql_query(query, connection)
        return pd.read_sql_query(f"{query} LIMIT ?", connection, params=(limit,))


def clear_prediction_history() -> None:
    """Clear only the local prediction-history table."""
    if not HISTORY_DB_PATH.exists():
        return
    with _transaction() as connection:
        connection.execute("DELETE FROM predictions")
