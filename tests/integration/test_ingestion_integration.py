"""Integration tests for data ingestion, schema validation, and partitioning."""

from __future__ import annotations

from pathlib import Path

from streamloop.data.ingestion import load_churn_data
from streamloop.data.split import split_churn_dataset


def test_real_dataset_ingestion_and_split(local_dataset_path: Path):
    """Verify ingestion of the actual 7043-row dataset and stratified split."""
    df = load_churn_data(local_fallback_path=local_dataset_path)

    assert len(df) == 7043
    assert "TotalCharges" in df.columns
    # Check that whitespace rows are coerced to float NaN (preserving missingness for pipeline imputer)
    assert df["TotalCharges"].isna().sum() == 11

    split = split_churn_dataset(df, test_size=0.2, random_state=42)

    # 80/20 of 7043 -> Train 5634, Test 1409
    assert len(split.X_train) == 5634
    assert len(split.X_test) == 1409
    assert "customerID" not in split.X_train.columns
    assert "customerID" not in split.X_test.columns

    # Stratified balance check: churn rate in train and test should match within 0.5%
    train_rate = split.y_train.mean()
    test_rate = split.y_test.mean()
    assert abs(train_rate - test_rate) < 0.005
