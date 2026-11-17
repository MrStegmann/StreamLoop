"""Unit tests for Spanish markdown report generation."""

from __future__ import annotations

from pathlib import Path

from streamloop.evaluation.metrics import EvaluationMetrics
from streamloop.reporting.spanish_report import generate_spanish_tuning_report
from streamloop.tuning.selection import CandidateEvaluation


def test_spanish_report_improvements_and_tables(tmp_path: Path):
    """Verify table formatting and improvement calculations in report."""
    base = EvaluationMetrics(
        f2_score=0.5000,
        f1_score=0.5000,
        precision=0.5000,
        recall=0.5000,
        pr_auc=0.5000,
        roc_auc=0.5000,
        confusion_matrix=[[500, 100], [100, 300]],
    )
    tuned = EvaluationMetrics(
        f2_score=0.7500,
        f1_score=0.6500,
        precision=0.5500,
        recall=0.8500,
        pr_auc=0.6500,
        roc_auc=0.8500,
        confusion_matrix=[[450, 150], [30, 370]],
    )
    cand = CandidateEvaluation(
        candidate_id=0,
        params={
            "classifier__n_estimators": 100,
            "classifier__max_depth": 5,
            "classifier__min_samples_split": 2,
            "classifier__min_samples_leaf": 1,
            "classifier__max_features": "sqrt",
            "classifier__criterion": "gini",
            "classifier__class_weight": "balanced",
        },
        split_scores=[0.75, 0.75],
        mean_score=0.75,
        std_score=0.0,
        standard_error=0.0,
        within_1_se=True,
    )

    out_file = tmp_path / "test_report.md"
    generate_spanish_tuning_report(
        output_path=out_file,
        baseline_metrics=base,
        tuned_metrics=tuned,
        best_mean_candidate=cand,
        winning_candidate=cand,
        candidates=[cand],
        test_touch_count=2,
    )

    text = out_file.read_text(encoding="utf-8")
    assert "+50.0%" in text or "50.0%" in text  # (0.75 - 0.50)/0.50 = +50% F2 improvement
    assert "+70.0%" in text or "70.0%" in text  # (0.85 - 0.50)/0.50 = +70% Recall improvement
    assert "Touch" in text or "2 veces" in text
