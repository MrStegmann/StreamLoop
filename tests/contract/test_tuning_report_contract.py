"""Contract test for Spanish tuning report structure and contents."""

from __future__ import annotations

from pathlib import Path

from streamloop.evaluation.metrics import EvaluationMetrics
from streamloop.reporting.spanish_report import generate_spanish_tuning_report
from streamloop.tuning.selection import CandidateEvaluation


def test_spanish_tuning_report_contract(tmp_path: Path):
    """Verify generated Spanish report adheres to contract structure and sections."""
    baseline_metrics = EvaluationMetrics(
        f2_score=0.5123,
        f1_score=0.5512,
        precision=0.6234,
        recall=0.4947,
        pr_auc=0.5892,
        roc_auc=0.8234,
        confusion_matrix=[[930, 105], [190, 184]],
    )
    tuned_metrics = EvaluationMetrics(
        f2_score=0.7102,
        f1_score=0.6430,
        precision=0.5612,
        recall=0.9679,
        pr_auc=0.6710,
        roc_auc=0.8654,
        confusion_matrix=[[750, 285], [12, 362]],
    )

    best_mean_cand = CandidateEvaluation(
        candidate_id=1,
        params={"classifier__n_estimators": 200, "classifier__max_depth": 10},
        split_scores=[0.70, 0.72, 0.71, 0.69, 0.71],
        mean_score=0.7060,
        std_score=0.0114,
        standard_error=0.0051,
        rank=1,
        within_1_se=True,
    )

    winner_cand = CandidateEvaluation(
        candidate_id=2,
        params={"classifier__n_estimators": 150, "classifier__max_depth": 8},
        split_scores=[0.702, 0.705, 0.701, 0.704, 0.703],
        mean_score=0.7030,
        std_score=0.0016,
        standard_error=0.0007,
        rank=2,
        within_1_se=True,
    )

    report_path = tmp_path / "tuning_report.md"
    generate_spanish_tuning_report(
        output_path=report_path,
        baseline_metrics=baseline_metrics,
        tuned_metrics=tuned_metrics,
        best_mean_candidate=best_mean_cand,
        winning_candidate=winner_cand,
        candidates=[best_mean_cand, winner_cand],
        test_touch_count=2,
    )

    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")

    # Contract required top-level title and 4 sections
    assert (
        "# Informe de Ajuste de Hiperparámetros: Modelo de Predicción de Churn StreamLoop"
        in content
    )
    assert "## 1. Resumen Ejecutivo y Métricas Comparativas" in content
    assert "## 2. Justificación de la Métrica de Negocio" in content
    assert "## 3. Exploración de Hiperparámetros y Espacio de Búsqueda" in content
    assert "## 4. Análisis de Estabilidad y Criterio de Selección (Regla 1-SE)" in content

    # Metrics present in section 1
    assert "0.5123" in content
    assert "0.7102" in content
    assert "F2 Score" in content
    assert "Recall" in content
    assert "PR-AUC" in content
    assert "ROC-AUC" in content

    # Asymmetric cost & formulas in section 2
    assert "falso negativo" in content.lower()
    assert "falso positivo" in content.lower()
    assert "F_2" in content or "F2" in content

    # No placeholders
    assert "[TODO]" not in content
    assert "[TBD]" not in content
    assert "[INSERT" not in content
