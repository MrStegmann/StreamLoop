"""Resilient data ingestion for Telco Customer Churn dataset."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Final

import pandas as pd
import requests

logger = logging.getLogger(__name__)

DEFAULT_REMOTE_URL: Final[str] = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
)
DEFAULT_LOCAL_FALLBACK: Final[Path] = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

EXPECTED_COLUMNS: Final[list[str]] = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]


class IngestionError(Exception):
    """Raised when data ingestion fails across both remote and local sources."""


def validate_raw_dataset(df: pd.DataFrame, min_rows: int = 1000) -> None:
    """Validate that the incoming DataFrame meets schema requirements.

    Args:
        df: DataFrame to validate.
        min_rows: Minimum required number of records.

    Raises:
        ValueError: If columns are missing or record count is insufficient.
    """
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if len(df) < min_rows:
        raise ValueError(
            f"Insufficient records in dataset: received {len(df)}, expected at least {min_rows}."
        )


def clean_churn_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean types and handle whitespace in Telco dataset.

    - TotalCharges: Converts whitespace strings to 0.0 and coerces to float.
    - Churn: Maps 'Yes' -> 1, 'No' -> 0 if string.
    """
    cleaned_df = df.copy()

    # TotalCharges coercion
    if "TotalCharges" in cleaned_df.columns:
        if cleaned_df["TotalCharges"].dtype == object:
            cleaned_df["TotalCharges"] = cleaned_df["TotalCharges"].replace(
                r"^\s*$", "0.0", regex=True
            )
        cleaned_df["TotalCharges"] = pd.to_numeric(
            cleaned_df["TotalCharges"], errors="coerce"
        ).fillna(0.0)

    # Churn target coercion: Convert "Yes"/"No" or string values to binary integer 1 / 0
    if "Churn" in cleaned_df.columns:
        if not pd.api.types.is_numeric_dtype(cleaned_df["Churn"]):
            cleaned_df["Churn"] = (
                cleaned_df["Churn"]
                .astype(str)
                .str.strip()
                .map({"Yes": 1, "No": 0})
                .fillna(0)
                .astype(int)
            )

    return cleaned_df


def load_churn_data(
    remote_url: str | None = None,
    local_fallback_path: str | Path | None = None,
    min_rows: int = 1000,
) -> pd.DataFrame:
    """Ingest Telco Churn dataset with remote-first and local-fallback strategy.

    Args:
        remote_url: Remote URL to download CSV from. Defaults to DEFAULT_REMOTE_URL.
        local_fallback_path: Local filesystem path to fallback CSV. Defaults to DEFAULT_LOCAL_FALLBACK.
        min_rows: Minimum required rows for validation.

    Returns:
        Cleaned, schema-validated pandas DataFrame.

    Raises:
        IngestionError: If both remote and local sources fail.
        ValueError: If dataset schema is invalid.
    """
    url = remote_url or DEFAULT_REMOTE_URL
    local_path = Path(local_fallback_path or DEFAULT_LOCAL_FALLBACK)
    raw_df: pd.DataFrame | None = None

    # Try remote ingestion first
    try:
        logger.info("Attempting remote data ingestion from %s", url)
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        raw_df = pd.read_csv(io.StringIO(resp.text))
        logger.info("Remote data ingestion succeeded (%d rows).", len(raw_df))
    except Exception as exc:
        logger.warning(
            "Remote ingestion failed (%s). Attempting local fallback: %s", exc, local_path
        )

    # Fallback to local if remote failed
    if raw_df is None:
        if local_path.exists():
            try:
                logger.info("Loading dataset from local fallback: %s", local_path)
                raw_df = pd.read_csv(local_path)
                logger.info("Local fallback loaded successfully (%d rows).", len(raw_df))
            except Exception as exc:
                logger.error("Failed to read local fallback file: %s", exc)
                raise IngestionError(
                    f"Both remote URL and local fallback failed to load data. Local error: {exc}"
                ) from exc
        else:
            raise IngestionError(
                f"Both remote URL and local fallback failed. Local file does not exist: {local_path}"
            )

    # Validate raw schema
    # Use min_rows or if dataset is smaller (e.g. tests with mock df), validate
    validate_raw_dataset(raw_df, min_rows=min(min_rows, len(raw_df)))

    # Clean and cast
    cleaned = clean_churn_dataset(raw_df)
    return cleaned
