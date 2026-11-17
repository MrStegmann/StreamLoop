"""Unit tests for resilient data ingestion and fallback logic."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from streamloop.data.ingestion import IngestionError, load_churn_data


def test_load_churn_data_fallback_when_remote_fails(local_dataset_path: Path):
    """Verify ingestion falls back to local CSV when remote URL raises exception."""
    with patch("requests.get", side_effect=Exception("Connection refused")):
        df = load_churn_data(
            remote_url="https://invalid-url.fake/data.csv",
            local_fallback_path=local_dataset_path,
        )
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 7043
        assert "TotalCharges" in df.columns
        # Verify TotalCharges has no whitespace and is numeric float
        assert pd.api.types.is_float_dtype(df["TotalCharges"])


def test_load_churn_data_both_fail():
    """Verify IngestionError is raised when both remote and local fail."""
    with patch("requests.get", side_effect=Exception("Connection refused")):
        with pytest.raises(IngestionError, match="Both remote URL and local fallback failed"):
            load_churn_data(
                remote_url="https://invalid-url.fake/data.csv",
                local_fallback_path=Path("non_existent_path.csv"),
            )
