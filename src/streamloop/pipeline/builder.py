"""Pipeline construction and baseline evaluation for customer churn."""

from __future__ import annotations

import logging
from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from streamloop.data.split import SplitDataset
from streamloop.evaluation.boundary import LeakageBoundaryMonitor
from streamloop.evaluation.metrics import EvaluationMetrics, compute_evaluation_metrics

logger = logging.getLogger(__name__)

NUMERICAL_FEATURES: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
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
]


def build_preprocessor(
    numerical_cols: list[str] | None = None,
    categorical_cols: list[str] | None = None,
) -> ColumnTransformer:
    """Build declarative ColumnTransformer encapsulating scaling and encoding."""
    num_cols = numerical_cols or NUMERICAL_FEATURES
    cat_cols = categorical_cols or CATEGORICAL_FEATURES

    num_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    cat_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def build_churn_pipeline(
    random_state: int = 42,
    classifier_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Build full scikit-learn Pipeline encapsulating preprocessing and classifier.

    Args:
        random_state: Random state for deterministic behavior.
        classifier_params: Optional hyperparameter dictionary for RandomForestClassifier.

    Returns:
        Unified scikit-learn Pipeline.
    """
    params: dict[str, Any] = {
        "random_state": random_state,
        "class_weight": "balanced",
    }
    if classifier_params:
        params.update(classifier_params)

    preprocessor = build_preprocessor()
    classifier = RandomForestClassifier(**params)

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )
    return pipeline


def train_and_evaluate_baseline(
    split: SplitDataset,
    monitor: LeakageBoundaryMonitor,
    random_state: int = 42,
) -> tuple[Pipeline, EvaluationMetrics]:
    """Fit baseline pipeline on training split and evaluate on holdout test set (Touch 1).

    Args:
        split: Train/test dataset partitions.
        monitor: Leakage monitor to record holdout access.
        random_state: Random seed.

    Returns:
        Tuple of (fitted_pipeline, evaluation_metrics).
    """
    logger.info("Building and fitting baseline pipeline with default hyperparameters...")
    pipeline = build_churn_pipeline(random_state=random_state)
    pipeline.fit(split.X_train, split.y_train)

    # Touch 1: Holdout Test Evaluation
    monitor.record_touch("Baseline Holdout Test Evaluation")
    logger.info("Evaluating baseline model on holdout test set [Touch %d]", monitor.touch_count)

    y_pred = pipeline.predict(split.X_test)
    y_proba = pipeline.predict_proba(split.X_test)[:, 1]

    metrics = compute_evaluation_metrics(
        y_true=split.y_test.to_numpy(),
        y_pred=y_pred,
        y_proba=y_proba,
    )
    return pipeline, metrics
