"""Unit tests for two-stage hyperparameter tuning routines."""

from __future__ import annotations

import pandas as pd

from streamloop.data.split import split_churn_dataset
from streamloop.tuning.search import (
    build_stage2_param_grid,
    extract_candidate_evaluations,
    run_stage1_randomized_search,
)


def test_build_stage2_param_grid():
    """Verify focused grid builds sensible ranges around Stage 1 parameters."""
    stage1_best_params = {
        "classifier__n_estimators": 200,
        "classifier__max_depth": 8,
        "classifier__min_samples_split": 10,
        "classifier__min_samples_leaf": 4,
        "classifier__max_features": "sqrt",
        "classifier__criterion": "gini",
        "classifier__class_weight": "balanced",
    }

    grid = build_stage2_param_grid(stage1_best_params)
    assert isinstance(grid, dict)
    assert 200 in grid["classifier__n_estimators"]
    assert 8 in grid["classifier__max_depth"]
    assert len(grid["classifier__n_estimators"]) > 1


def test_stage1_search_execution(sample_raw_telco_df: pd.DataFrame):
    """Verify Stage 1 randomized search executes and extracts evaluations."""
    split = split_churn_dataset(sample_raw_telco_df, test_size=0.2, random_state=42)

    # Use n_iter=2 and cv_splits=2 for rapid unit test execution
    search_result = run_stage1_randomized_search(
        X_train=split.X_train,
        y_train=split.y_train,
        n_iter=2,
        cv_splits=2,
        random_state=42,
        n_jobs=1,
    )

    assert search_result is not None
    candidates = extract_candidate_evaluations(search_result.cv_results_, cv_splits=2)
    assert len(candidates) == 2
    assert all(len(c.split_scores) == 2 for c in candidates)
    assert all(c.mean_score >= 0.0 for c in candidates)
