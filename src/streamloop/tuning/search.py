"""Two-stage hyperparameter exploration with RandomizedSearchCV and GridSearchCV."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from streamloop.evaluation.metrics import f2_scorer
from streamloop.pipeline.builder import build_churn_pipeline
from streamloop.tuning.selection import CandidateEvaluation

logger = logging.getLogger(__name__)

STAGE1_PARAM_DISTRIBUTIONS: dict[str, list[Any]] = {
    "classifier__n_estimators": [100, 150, 200, 250, 300, 400],
    "classifier__max_depth": [None, 5, 8, 10, 12, 15, 20],
    "classifier__min_samples_split": [2, 5, 10, 15, 20],
    "classifier__min_samples_leaf": [1, 2, 4, 6, 8, 10],
    "classifier__max_features": ["sqrt", "log2", 0.2, 0.4, 0.6],
    "classifier__criterion": ["gini", "entropy"],
    "classifier__class_weight": ["balanced", "balanced_subsample"],
}


def build_stage2_param_grid(stage1_best_params: dict[str, Any]) -> dict[str, list[Any]]:
    """Build a focused parameter grid around the Stage 1 best configuration (~36-72 combinations).

    Args:
        stage1_best_params: Parameter dictionary of best configuration from Stage 1.

    Returns:
        Focused parameter grid dictionary for GridSearchCV.
    """
    n_est = stage1_best_params.get("classifier__n_estimators", 200)
    max_d = stage1_best_params.get("classifier__max_depth", None)
    min_split = stage1_best_params.get("classifier__min_samples_split", 10)
    min_leaf = stage1_best_params.get("classifier__min_samples_leaf", 4)
    max_feat = stage1_best_params.get("classifier__max_features", "sqrt")
    crit = stage1_best_params.get("classifier__criterion", "gini")
    weight = stage1_best_params.get("classifier__class_weight", "balanced")

    # Localized n_estimators (2 values)
    n_estimators_grid = sorted(list({max(50, n_est - 50), n_est, n_est + 50}))[:2]

    # Localized max_depth (2-3 values)
    if max_d is None:
        max_depth_grid = [None, 15, 20]
    else:
        max_depth_grid = sorted(list({max(3, max_d - 2), max_d, max_d + 2}))

    # Localized min_samples_split (2 values)
    min_samples_split_grid = sorted(list({max(2, min_split - 2), min_split, min_split + 2}))[:2]

    # Localized min_samples_leaf (2 values)
    min_samples_leaf_grid = sorted(list({max(1, min_leaf - 1), min_leaf, min_leaf + 1}))[:2]

    grid: dict[str, list[Any]] = {
        "classifier__n_estimators": n_estimators_grid,
        "classifier__max_depth": max_depth_grid,
        "classifier__min_samples_split": min_samples_split_grid,
        "classifier__min_samples_leaf": min_samples_leaf_grid,
        "classifier__max_features": [max_feat],
        "classifier__criterion": [crit],
        "classifier__class_weight": [weight],
    }
    return grid


def extract_candidate_evaluations(
    cv_results: dict[str, Any],
    cv_splits: int,
) -> list[CandidateEvaluation]:
    """Extract CandidateEvaluation instances with fold distributions from scikit-learn cv_results_.

    Args:
        cv_results: cv_results_ dictionary from RandomizedSearchCV or GridSearchCV.
        cv_splits: Number of cross-validation folds.

    Returns:
        List of CandidateEvaluation objects.
    """
    candidates: list[CandidateEvaluation] = []
    n_candidates = len(cv_results["params"])

    for i in range(n_candidates):
        params = cv_results["params"][i]
        split_scores = [float(cv_results[f"split{k}_test_score"][i]) for k in range(cv_splits)]
        mean_score = float(cv_results["mean_test_score"][i])
        std_score = float(cv_results["std_test_score"][i])
        standard_error = float(std_score / np.sqrt(cv_splits))

        candidates.append(
            CandidateEvaluation(
                candidate_id=i,
                params=params,
                split_scores=[round(s, 4) for s in split_scores],
                mean_score=round(mean_score, 4),
                std_score=round(std_score, 4),
                standard_error=round(standard_error, 4),
            )
        )
    return candidates


def run_stage1_randomized_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_iter: int = 50,
    cv_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    """Execute Stage 1 broad hyperparameter exploration using RandomizedSearchCV with F2 score.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        n_iter: Number of parameter settings sampled (default: 50).
        cv_splits: Number of stratified cross-validation folds (default: 5).
        random_state: Reproducibility seed.
        n_jobs: Parallel worker count.

    Returns:
        Fitted RandomizedSearchCV instance.
    """
    logger.info(
        "Initiating Stage 1: RandomizedSearchCV (%d iterations x %d folds, metric=F2, n_jobs=%d)...",
        n_iter,
        cv_splits,
        n_jobs,
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    pipeline = build_churn_pipeline(random_state=random_state)

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=STAGE1_PARAM_DISTRIBUTIONS,
        n_iter=n_iter,
        scoring=f2_scorer,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=0,
        refit=False,  # We handle refitting explicitly after 1-SE selection
    )
    search.fit(X_train, y_train)
    logger.info("Stage 1 complete. Best mean F2: %.4f", search.best_score_)
    return search


def run_stage2_grid_search(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    param_grid: dict[str, list[Any]],
    cv_splits: int = 5,
    random_state: int = 42,
    n_jobs: int = -1,
) -> GridSearchCV:
    """Execute Stage 2 focused hyperparameter refinement using GridSearchCV with F2 score.

    Args:
        X_train: Training feature matrix.
        y_train: Training target labels.
        param_grid: Parameter grid dictionary.
        cv_splits: Number of stratified cross-validation folds.
        random_state: Reproducibility seed.
        n_jobs: Parallel worker count.

    Returns:
        Fitted GridSearchCV instance.
    """
    total_candidates = np.prod([len(v) for v in param_grid.values()])
    logger.info(
        "Initiating Stage 2: Focused GridSearchCV (%d candidates x %d folds, metric=F2, n_jobs=%d)...",
        total_candidates,
        cv_splits,
        n_jobs,
    )
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    pipeline = build_churn_pipeline(random_state=random_state)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=f2_scorer,
        cv=cv,
        n_jobs=n_jobs,
        verbose=0,
        refit=False,
    )
    search.fit(X_train, y_train)
    logger.info("Stage 2 complete. Top mean F2: %.4f", search.best_score_)
    return search
