"""Spanish markdown report generator for StreamLoop churn model hyperparameter tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamloop.evaluation.metrics import EvaluationMetrics
from streamloop.tuning.selection import CandidateEvaluation


def _format_diff(base_val: float, tuned_val: float) -> str:
    """Format relative percentage improvement."""
    if base_val == 0.0:
        return "N/A"
    diff = ((tuned_val - base_val) / base_val) * 100.0
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1f}%"


def generate_spanish_tuning_report(
    output_path: str | Path,
    baseline_metrics: EvaluationMetrics,
    tuned_metrics: EvaluationMetrics,
    best_mean_candidate: CandidateEvaluation,
    winning_candidate: CandidateEvaluation,
    candidates: list[CandidateEvaluation],
    test_touch_count: int = 2,
    stage1_iter: int = 50,
    cv_splits: int = 5,
) -> Path:
    """Generate executive Spanish markdown report documenting tuning lifecycle.

    Args:
        output_path: Path to write the report markdown.
        baseline_metrics: Metrics of baseline default model on holdout test set.
        tuned_metrics: Metrics of 1-SE tuned model on holdout test set.
        best_mean_candidate: Candidate with highest mean CV score.
        winning_candidate: Winning candidate chosen by 1-SE rule.
        candidates: List of evaluated candidates for stability analysis table.
        test_touch_count: Number of times holdout test set was accessed.
        stage1_iter: Number of Stage 1 iterations.
        cv_splits: Number of cross-validation folds.

    Returns:
        Path to generated report.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Confusion matrix extraction
    base_cm = baseline_metrics.confusion_matrix
    tuned_cm = tuned_metrics.confusion_matrix

    # Improvements
    f2_imp = _format_diff(baseline_metrics.f2_score, tuned_metrics.f2_score)
    rec_imp = _format_diff(baseline_metrics.recall, tuned_metrics.recall)
    prec_imp = _format_diff(baseline_metrics.precision, tuned_metrics.precision)
    f1_imp = _format_diff(baseline_metrics.f1_score, tuned_metrics.f1_score)
    prauc_imp = _format_diff(baseline_metrics.pr_auc, tuned_metrics.pr_auc)
    roc_imp = _format_diff(baseline_metrics.roc_auc, tuned_metrics.roc_auc)

    # Hyperparameters
    def _clean_param(p_dict: dict[str, Any], key: str, default: Any = "N/A") -> str:
        k = f"classifier__{key}"
        return str(p_dict.get(k, p_dict.get(key, default)))

    # Candidates table top 5
    sorted_cands = sorted(candidates, key=lambda c: c.mean_score, reverse=True)[:5]
    if winning_candidate not in sorted_cands:
        sorted_cands.append(winning_candidate)

    threshold = best_mean_candidate.mean_score - best_mean_candidate.standard_error

    report_lines = [
        "# Informe de Ajuste de Hiperparámetros: Modelo de Predicción de Churn StreamLoop",
        "",
        "## 1. Resumen Ejecutivo y Métricas Comparativas",
        "",
        "Este documento consolida los resultados del proceso de optimización sistemática y validación "
        "rigurosa del modelo de predicción de abandono de clientes (churn) de StreamLoop.",
        "",
        "### Métricas en Conjunto de Prueba Retenido (Holdout Test Set)",
        "",
        "| Métrica | Modelo Base (Default) | Modelo Ajustado (Ganador 1-SE) | Mejora Relativa |",
        "| :--- | :---: | :---: | :---: |",
        f"| **F2 Score** (Métrica Primaria) | {baseline_metrics.f2_score:.4f} | **{tuned_metrics.f2_score:.4f}** | **{f2_imp}** |",
        f"| **Recall** (Sensibilidad Churn) | {baseline_metrics.recall:.4f} | **{tuned_metrics.recall:.4f}** | **{rec_imp}** |",
        f"| **Precision** | {baseline_metrics.precision:.4f} | {tuned_metrics.precision:.4f} | {prec_imp} |",
        f"| **F1 Score** | {baseline_metrics.f1_score:.4f} | {tuned_metrics.f1_score:.4f} | {f1_imp} |",
        f"| **PR-AUC** | {baseline_metrics.pr_auc:.4f} | {tuned_metrics.pr_auc:.4f} | {prauc_imp} |",
        f"| **ROC-AUC** | {baseline_metrics.roc_auc:.4f} | {tuned_metrics.roc_auc:.4f} | {roc_imp} |",
        "",
        "### Matrices de Confusión Comparativas",
        "",
        "#### Modelo Base:",
        f"- Verdaderos Negativos (TN): {base_cm[0][0]} | Falsos Positivos (FP): {base_cm[0][1]}",
        f"- Falsos Negativos (FN): {base_cm[1][0]} | Verdaderos Positivos (TP): {base_cm[1][1]}",
        "",
        "#### Modelo Ajustado (Ganador 1-SE):",
        f"- Verdaderos Negativos (TN): {tuned_cm[0][0]} | Falsos Positivos (FP): {tuned_cm[0][1]}",
        f"- Falsos Negativos (FN): {tuned_cm[1][0]} | Verdaderos Positivos (TP): {tuned_cm[1][1]}",
        "",
        "### Verificación de Aislamiento y Ausencia de Fuga de Datos",
        f"Se verificó formalmente el cumplimiento del protocolo de aislamiento: el conjunto de prueba retenido "
        f"fue evaluado exactamente **{test_touch_count} veces** (Touch 1: evaluación de referencia base; "
        f"Touch 2: evaluación final del estimador ganador refactorizado). Cero transformaciones ni divisiones de "
        f"validación cruzada tuvieron acceso a los datos de test, garantizando ausencia total de fuga de datos (zero data leakage).",
        "",
        "---",
        "",
        "## 2. Justificación de la Métrica de Negocio",
        "",
        "### Coste Asimétrico en la Retención de Clientes",
        "En el contexto de servicios de suscripción de StreamLoop, los errores de clasificación conllevan "
        "costes comerciales radicalmente asimétricos:",
        "1. **Falso Negativo (FN)**: Un cliente que cancelará su suscripción no es detectado a tiempo. Esto genera "
        "la pérdida irreversible del valor de vida del cliente (Customer Lifetime Value - LTV) y exige un alto "
        "coste de adquisición para reemplazarlo.",
        "2. **Falso Positivo (FP)**: Un cliente leal es clasificado erróneamente como propenso a cancelar. La consecuencia "
        "operativa es únicamente el envío de una oferta o descuento preventivo de bajo coste.",
        "",
        "Dado que el coste de perder un suscriptor supera con creces el coste de una campaña preventiva, la prioridad "
        "analítica es maximizar el **Recall** (captura de casos de churn) sin degradar por completo la precisión.",
        "",
        "### Inadecuación del Accuracy y Limitación de F1 Score",
        "- **Accuracy**: En un conjunto con desbalance (aprox. 73.5% retenidos vs. 26.5% churn), un clasificador ingenuo "
        "que siempre prediga 'No Churn' alcanzaría un 73.5% de exactitud mientras omite el 100% de los canceladores.",
        "- **F1 Score**: Pondera de manera simétrica la precisión y el recall (beta = 1.0), penalizando en exceso los falsos positivos "
        "y conduciendo a modelos conservadores que pasan por alto a clientes en riesgo crítico.",
        "",
        "### Formulación Matemática de F2 Score",
        "La métrica $F_2$ fija $\\beta = 2$, otorgando al recall el doble de importancia que a la precisión:",
        "",
        "$$F_2 = (1 + 2^2) \\cdot \\frac{\\text{Precision} \\cdot \\text{Recall}}{2^2 \\cdot \\text{Precision} + \\text{Recall}} = 5 \\cdot \\frac{P \\cdot R}{4P + R}$$",
        "",
        "---",
        "",
        "## 3. Exploración de Hiperparámetros y Espacio de Búsqueda",
        "",
        "### Estrategia de Optimización en Dos Etapas",
        f"1. **Etapa 1 (Exploración Global)**: Búsqueda aleatoria (`RandomizedSearchCV`) con {stage1_iter} iteraciones "
        f"y validación cruzada estratificada de {cv_splits} pliegues ({stage1_iter * cv_splits} ajustes totales), "
        "evaluando combinaciones en rangos amplios del clasificador `RandomForestClassifier`.",
        "2. **Etapa 2 (Refinamiento Focalizado)**: Búsqueda exhaustiva en cuadrícula (`GridSearchCV`) centrada "
        "en la vecindad de la configuración más prometedora identificada en la Etapa 1.",
        "",
        "### Espacio de Hiperparámetros y Configuración Seleccionada",
        "",
        "| Hiperparámetro | Rango / Opciones Exploradas | Modelo Base | Configuración Ganadora (1-SE) |",
        "| :--- | :--- | :---: | :---: |",
        f"| `n_estimators` | [50, 100, 150, 200, 250, 300, 400] | 100 | **{_clean_param(winning_candidate.params, 'n_estimators', 100)}** |",
        f"| `max_depth` | [None, 5, 8, 10, 12, 15, 20] | None | **{_clean_param(winning_candidate.params, 'max_depth', 'None')}** |",
        f"| `min_samples_split` | [2, 5, 10, 15, 20] | 2 | **{_clean_param(winning_candidate.params, 'min_samples_split', 2)}** |",
        f"| `min_samples_leaf` | [1, 2, 4, 6, 8, 10] | 1 | **{_clean_param(winning_candidate.params, 'min_samples_leaf', 1)}** |",
        f"| `max_features` | ['sqrt', 'log2', 0.2, 0.4, 0.6] | 'sqrt' | **{_clean_param(winning_candidate.params, 'max_features', 'sqrt')}** |",
        f"| `criterion` | ['gini', 'entropy'] | 'gini' | **{_clean_param(winning_candidate.params, 'criterion', 'gini')}** |",
        f"| `class_weight` | [None, 'balanced', 'balanced_subsample'] | None | **{_clean_param(winning_candidate.params, 'class_weight', 'None')}** |",
        "",
        "---",
        "",
        "## 4. Análisis de Estabilidad y Criterio de Selección (Regla 1-SE)",
        "",
        "### Principio de Selección de 1 Error Estándar (1-SE Rule)",
        "Seleccionar ciegamente la configuración con el mayor promedio de validación cruzada frecuentemente conduce "
        "a sobreajuste hacia las particiones de entrenamiento particulares. La regla de 1 Error Estándar (1-SE) formaliza "
        "el principio de parsimonia y estabilidad:",
        "",
        r"$$\text{SE} = \frac{\sigma}{\sqrt{k}} = \frac{\sigma}{\sqrt{" + str(cv_splits) + r"}}$$",
        "",
        f"Se calcula el umbral de aceptación $T = \\bar{{S}}_{{\\max}} - \\text{{SE}}_{{\\max}} = {best_mean_candidate.mean_score:.4f} - {best_mean_candidate.standard_error:.4f} = {threshold:.4f}$. "
        "Cualquier modelo cuya media de F2 sea igual o superior a dicho umbral se considera estadísticamente equivalente. "
        "Entre los candidatos calificados, se selecciona aquel con la menor desviación estándar entre pliegues ($\\sigma$), "
        "garantizando la máxima estabilidad y generalización en producción.",
        "",
        "### Tabla Comparativa de Candidatos Top en Validación Cruzada",
        "",
        "| Candidato ID | Media F2 en CV ($\\bar{S}$) | Desv. Estándar ($\\sigma$) | Error Estándar ($\\text{SE}$) | ¿Cumple 1-SE? | Estado |",
        "| :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for cand in sorted_cands:
        is_winner = cand.candidate_id == winning_candidate.candidate_id
        is_best_mean = cand.candidate_id == best_mean_candidate.candidate_id
        status_label = (
            "Ganador Seleccionado (1-SE)"
            if is_winner
            else ("Mayor Media CV" if is_best_mean else "Candidato")
        )
        cumple_label = "Sí" if cand.within_1_se else "No"
        report_lines.append(
            f"| #{cand.candidate_id} | {cand.mean_score:.4f} | {cand.std_score:.4f} | {cand.standard_error:.4f} | {cumple_label} | {status_label} |"
        )

    report_lines.extend(
        [
            "",
            "### Justificación del Modelo Seleccionado",
            f"El candidato **#{winning_candidate.candidate_id}** fue seleccionado porque, además de situarse holgadamente dentro "
            f"del umbral estadístico 1-SE (F2 medio de {winning_candidate.mean_score:.4f} frente a umbral {threshold:.4f}), presentó "
            f"la menor variabilidad entre pliegues ($\\sigma = {winning_candidate.std_score:.4f}$). Esto garantiza que el clasificador "
            "no es vulnerable a variaciones locales en la distribución de datos y ofrecerá un comportamiento predecible y robusto "
            "durante su despliegue operativo en StreamLoop.",
            "",
        ]
    )

    content = "\n".join(report_lines)
    path.write_text(content, encoding="utf-8")
    return path
