"""Contract tests for Telco Customer Churn dataset schema."""

from __future__ import annotations

import pytest

from streamloop.data.ingestion import EXPECTED_COLUMNS, validate_raw_dataset


def test_expected_columns_count():
    """Verify schema contract mandates exactly 21 columns."""
    assert len(EXPECTED_COLUMNS) == 21
    assert "customerID" in EXPECTED_COLUMNS
    assert "Churn" in EXPECTED_COLUMNS
    assert "TotalCharges" in EXPECTED_COLUMNS


def test_validate_raw_dataset_success(sample_raw_telco_df):
    """Test validation passes on compliant DataFrame."""
    # sample_raw_telco_df has 100 rows; validate_raw_dataset should allow configurable min_rows or default
    validate_raw_dataset(sample_raw_telco_df, min_rows=50)


def test_validate_raw_dataset_missing_column(sample_raw_telco_df):
    """Test validation fails when mandatory column is missing."""
    invalid_df = sample_raw_telco_df.drop(columns=["Contract"])
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_raw_dataset(invalid_df, min_rows=50)


def test_validate_raw_dataset_insufficient_rows(sample_raw_telco_df):
    """Test validation fails when row count is below threshold."""
    tiny_df = sample_raw_telco_df.iloc[:5]
    with pytest.raises(ValueError, match="Insufficient records"):
        validate_raw_dataset(tiny_df, min_rows=10)
