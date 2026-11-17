"""1-Standard-Error (1-SE) stability selection rule and refitting utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from streamloop.pipeline.builder import build_churn_pipeline

logger = logging.getLogger(__name__)


@dataclass
class CandidateEvaluation:
    """Stores evaluation distribution across CV folds for a hyperparameter candidate."""

    candidate_id: int
    params: dict[str, Any]
    split_scores: list[float]
    mean_score: float
    std_score: float
    standard_error: float
    rank: int = 1
    within_1_se: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


def select_winning_candidate_1se(
    candidates: list[CandidateEvaluation],
) -> tuple[CandidateEvaluation, CandidateEvaluation, float]:
    """Apply the 1-Standard-Error (1-SE) stability selection rule.

    1. Identify candidate with highest mean score: S_max with SE_max = std / sqrt(k).
    2. Compute threshold: T = S_max - SE_max.
    3. Filter all candidates where mean_score >= T.
    4. Select candidate among qualifying pool with lowest fold-to-fold std_score
       (breaking ties by highest mean_score).

    Args:
        candidates: List of evaluated hyperparameter configurations.

    Returns:
        Tuple of (best_mean_candidate, winning_1se_candidate, threshold).

    Raises:
        ValueError: If candidates list is empty.
    """
    if not candidates:
        raise ValueError("No candidate evaluations provided.")

    # Sort candidates by mean score descending to assign ranks
    sorted_by_mean = sorted(candidates, key=lambda c: c.mean_score, reverse=True)
    for rank, cand in enumerate(sorted_by_mean, start=1):
        cand.rank = rank

    best_mean_candidate = sorted_by_mean[0]
    threshold = best_mean_candidate.mean_score - best_mean_candidate.standard_error

    # Mark candidates within 1-SE boundary
    for cand in candidates:
        cand.within_1_se = bool(cand.mean_score >= threshold)

    qualifying_candidates = [c for c in candidates if c.within_1_se]

    # Select candidate with lowest std_score (prioritizing stability/generalization)
    winning_candidate = min(
        qualifying_candidates,
        key=lambda c: (round(c.std_score, 6), -round(c.mean_score, 6)),
    )

    logger.info(
        "1-SE Rule Selection: Best Mean=%.4f (std=%.4f, SE=%.4f, threshold=%.4f) -> "
        "Selected Candidate #%d: Mean=%.4f (std=%.4f, SE=%.4f, within_1_se=%s)",
        best_mean_candidate.mean_score,
        best_mean_candidate.std_score,
        best_mean_candidate.standard_error,
        threshold,
        winning_candidate.candidate_id,
        winning_candidate.mean_score,
        winning_candidate.std_score,
        winning_candidate.standard_error,
        winning_candidate.within_1_se,
    )

    return best_mean_candidate, winning_candidate, threshold


def refit_winning_pipeline(
    winning_candidate: CandidateEvaluation,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
) -> Pipeline:
    """Instantiate and refit the pipeline with winning hyperparameters on full training split.

    Args:
        winning_candidate: Winning candidate selected via 1-SE rule.
        X_train: Training feature matrix.
        y_train: Training target labels.
        random_state: Random state seed.

    Returns:
        Fitted scikit-learn Pipeline.
    """
    pipeline = build_churn_pipeline(random_state=random_state)
    # Apply winning hyperparameters
    pipeline.set_params(**winning_candidate.params)

    logger.info(
        "Refitting winning pipeline on full training set (%d samples) with params: %s",
        len(X_train),
        winning_candidate.params,
    )
    pipeline.fit(X_train, y_train)
    return pipeline
