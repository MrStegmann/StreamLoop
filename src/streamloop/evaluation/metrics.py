"""Evaluation metrics and custom scorers for churn classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_f2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate F2 Score (beta=2, pos_label=1).

    F2 weighs recall higher than precision: F2 = (1 + 2^2) * (P * R) / (2^2 * P + R)
    """
    return float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0))


# Scikit-learn scorer object for GridSearchCV / RandomizedSearchCV
f2_scorer = make_scorer(fbeta_score, beta=2, pos_label=1, response_method="predict")


@dataclass
class EvaluationMetrics:
    """Encapsulates evaluation metrics for classification models."""

    f2_score: float
    f1_score: float
    precision: float
    recall: float
    pr_auc: float
    roc_auc: float
    confusion_matrix: list[list[int]]

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "f2_score": self.f2_score,
            "f1_score": self.f1_score,
            "precision": self.precision,
            "recall": self.recall,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "confusion_matrix": self.confusion_matrix,
        }


def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> EvaluationMetrics:
    """Compute comprehensive classification metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred: Predicted binary labels (0 or 1).
        y_proba: Predicted probability of positive class (1).

    Returns:
        EvaluationMetrics instance with all calculated scores.
    """
    f2 = calculate_f2_score(y_true, y_pred)
    f1 = float(f1_score(y_true, y_pred, pos_label=1, zero_division=0))
    prec = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    rec = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

    try:
        pr_auc = float(average_precision_score(y_true, y_proba))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true, y_proba))
    except Exception:
        roc_auc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    return EvaluationMetrics(
        f2_score=round(f2, 4),
        f1_score=round(f1, 4),
        precision=round(prec, 4),
        recall=round(rec, 4),
        pr_auc=round(pr_auc, 4),
        roc_auc=round(roc_auc, 4),
        confusion_matrix=cm,
    )
