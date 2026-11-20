"""Unit tests for ColumnTransformer and RandomForest pipeline builder."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from streamloop.data.ingestion import clean_churn_dataset
from streamloop.data.split import split_churn_dataset
from streamloop.pipeline.builder import build_churn_pipeline


def test_build_churn_pipeline_structure():
    """Verify built pipeline contains preprocessor and classifier stages with default params."""
    pipeline = build_churn_pipeline(random_state=42)
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    rf = pipeline.named_steps["classifier"]
    assert rf.random_state == 42
    assert rf.class_weight is None


def test_build_churn_pipeline_custom_classifier_params():
    """Verify classifier_params overrides default classifier settings."""
    pipeline = build_churn_pipeline(
        random_state=42,
        classifier_params={"class_weight": "balanced", "n_estimators": 50},
    )
    rf = pipeline.named_steps["classifier"]
    assert rf.class_weight == "balanced"
    assert rf.n_estimators == 50


def test_clean_churn_dataset_preserves_nan():
    """Verify clean_churn_dataset coerces whitespace/empty strings to NaN instead of 0.0."""
    raw_df = pd.DataFrame(
        {
            "TotalCharges": [" ", " 100.5 ", "", "NaN", "0.0"],
            "Churn": ["Yes", "No", "No", "Yes", "No"],
        }
    )
    cleaned = clean_churn_dataset(raw_df)
    assert pd.api.types.is_float_dtype(cleaned["TotalCharges"])
    assert cleaned["TotalCharges"].isna().sum() == 3
    assert np.isnan(cleaned["TotalCharges"].iloc[0])
    assert cleaned["TotalCharges"].iloc[1] == 100.5
    assert np.isnan(cleaned["TotalCharges"].iloc[2])
    assert np.isnan(cleaned["TotalCharges"].iloc[3])
    assert cleaned["TotalCharges"].iloc[4] == 0.0


def test_pipeline_imputer_handles_nan_totalcharges(sample_raw_telco_df: pd.DataFrame):
    """Verify SimpleImputer handles NaN in TotalCharges during pipeline fit and transform."""
    # Inject missing TotalCharges into dataset
    df = sample_raw_telco_df.copy()
    cleaned = clean_churn_dataset(df)
    cleaned.loc[cleaned.index[:5], "TotalCharges"] = np.nan

    split = split_churn_dataset(cleaned, test_size=0.2, random_state=42)
    pipeline = build_churn_pipeline(random_state=42)

    # Fitting should succeed without error and impute median TotalCharges
    pipeline.fit(split.X_train, split.y_train)
    preds = pipeline.predict(split.X_test)
    assert len(preds) == len(split.X_test)

    # Imputer statistics check
    preprocessor = pipeline.named_steps["preprocessor"]
    imputer = preprocessor.named_transformers_["num"].named_steps["imputer"]
    assert len(imputer.statistics_) == 3  # tenure, MonthlyCharges, TotalCharges
    assert not np.isnan(imputer.statistics_[2])


def test_pipeline_fit_and_predict(sample_raw_telco_df: pd.DataFrame):
    """Verify pipeline fits on training split and generates predictions."""
    split = split_churn_dataset(sample_raw_telco_df, test_size=0.2, random_state=42)
    pipeline = build_churn_pipeline(random_state=42)

    pipeline.fit(split.X_train, split.y_train)
    preds = pipeline.predict(split.X_test)
    probas = pipeline.predict_proba(split.X_test)

    assert len(preds) == len(split.X_test)
    assert probas.shape == (len(split.X_test), 2)


def test_train_and_evaluate_baseline(sample_raw_telco_df: pd.DataFrame):
    """Verify baseline training and Touch 1 evaluation with boundary monitoring."""
    from streamloop.evaluation.boundary import LeakageBoundaryMonitor
    from streamloop.pipeline.builder import train_and_evaluate_baseline

    split = split_churn_dataset(sample_raw_telco_df, test_size=0.2, random_state=42)
    monitor = LeakageBoundaryMonitor(max_allowed_touches=2)

    fitted_pipeline, metrics = train_and_evaluate_baseline(split, monitor, random_state=42)

    assert monitor.touch_count == 1
    assert 0.0 <= metrics.f2_score <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert len(metrics.confusion_matrix) == 2
