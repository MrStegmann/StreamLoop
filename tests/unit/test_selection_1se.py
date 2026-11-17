"""Unit tests for 1-Standard-Error (1-SE) stability selection algorithm."""

from __future__ import annotations

import numpy as np
import pytest

from streamloop.tuning.selection import CandidateEvaluation, select_winning_candidate_1se


def test_calculate_1se_selection():
    """Verify 1-SE rule selects lower variance model within 1 SE of best mean."""
    # Candidate 0: Highest mean (0.75), high std (0.05) -> SE = 0.05 / sqrt(5) = 0.02236 -> threshold = 0.7276
    # Candidate 1: Slightly lower mean (0.74), very low std (0.01) -> within 1 SE, should WIN due to lowest std
    # Candidate 2: Low mean (0.70), low std (0.01) -> outside 1 SE
    k = 5
    c0 = CandidateEvaluation(
        candidate_id=0,
        params={"classifier__max_depth": 10},
        split_scores=[0.70, 0.80, 0.75, 0.73, 0.77],  # mean=0.75, std ~ 0.038
        mean_score=0.75,
        std_score=0.038,
        standard_error=0.038 / np.sqrt(k),
    )
    c1 = CandidateEvaluation(
        candidate_id=1,
        params={"classifier__max_depth": 6},
        split_scores=[0.74, 0.741, 0.739, 0.74, 0.74],  # mean=0.74, std ~ 0.0007
        mean_score=0.74,
        std_score=0.0007,
        standard_error=0.0007 / np.sqrt(k),
    )
    c2 = CandidateEvaluation(
        candidate_id=2,
        params={"classifier__max_depth": 3},
        split_scores=[0.70, 0.70, 0.70, 0.70, 0.70],  # mean=0.70, std=0.0
        mean_score=0.70,
        std_score=0.0,
        standard_error=0.0,
    )

    candidates = [c0, c1, c2]
    best_mean, winner_1se, threshold = select_winning_candidate_1se(candidates)

    assert best_mean.candidate_id == 0
    # Winner must be Candidate 1 because it's within 1 SE of c0 and has lower std than c0
    assert winner_1se.candidate_id == 1
    assert winner_1se.within_1_se is True
    assert threshold == pytest.approx(0.75 - (0.038 / np.sqrt(5)), rel=1e-3)


def test_select_winning_candidate_empty():
    """Verify empty candidates list raises ValueError."""
    with pytest.raises(ValueError, match="No candidate evaluations provided"):
        select_winning_candidate_1se([])
