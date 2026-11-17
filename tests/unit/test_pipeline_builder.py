"""Unit tests for ColumnTransformer and RandomForest pipeline builder."""

from __future__ import annotations

import pandas as pd
from sklearn.pipeline import Pipeline

from streamloop.data.split import split_churn_dataset
from streamloop.pipeline.builder import build_churn_pipeline


def test_build_churn_pipeline_structure():
    """Verify built pipeline contains preprocessor and classifier stages."""
    pipeline = build_churn_pipeline(random_state=42)
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    rf = pipeline.named_steps["classifier"]
    assert rf.random_state == 42
    assert rf.class_weight == "balanced"


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
