"""Stratified dataset partitioning for customer churn data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class SplitDataset:
    """Encapsulates partitioned dataset and split metadata."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    random_state: int
    test_size: float


def split_churn_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    target_column: str = "Churn",
    id_column: str = "customerID",
) -> SplitDataset:
    """Perform leak-free stratified train-test split.

    Drops ID column and splits features/labels preserving class ratio.

    Args:
        df: Input DataFrame containing features, ID column, and target.
        test_size: Proportion of holdout test set (default: 0.20).
        random_state: Seed for reproducibility (default: 42).
        target_column: Name of binary target column.
        id_column: Name of customer identifier column to drop.

    Returns:
        SplitDataset containing train/test splits.
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    feature_df = df.copy()
    if id_column in feature_df.columns:
        feature_df = feature_df.drop(columns=[id_column])

    # Ensure target is binary int (1 for positive/Yes, 0 for negative/No)
    if not pd.api.types.is_numeric_dtype(feature_df[target_column]):
        feature_df[target_column] = (
            feature_df[target_column]
            .astype(str)
            .str.strip()
            .map({"Yes": 1, "No": 0})
            .fillna(0)
            .astype(int)
        )

    y = feature_df[target_column]
    X = feature_df.drop(columns=[target_column])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    return SplitDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        random_state=random_state,
        test_size=test_size,
    )
