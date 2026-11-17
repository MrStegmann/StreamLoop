# Tuning Report Contract: Structure & Analytical Content (`tuning_report.md`)

**Branch**: `001-tune-churn-model`  
**Contract Type**: Markdown Document & Governance Contract  
**Target File**: `tuning_report.md` (or `reports/tuning_report.md`)  
**Language Requirement**: Spanish (Mandatory per FR-012 and SC-005)

---

## Required Document Structure

The generated `tuning_report.md` MUST contain four mandatory top-level analytical sections:

```markdown
# Informe de Ajuste de Hiperparámetros: Modelo de Predicción de Churn StreamLoop

## 1. Resumen Ejecutivo y Métricas Comparativas
- Tabla comparativa completa entre Baseline (hiperparámetros por defecto) y Modelo Ajustado (ganador 1-SE).
- Métricas requeridas:
  - F2 Score (métrica primaria)
  - F1 Score
  - Precision
  - Recall
  - PR-AUC (Precision-Recall Area Under Curve)
  - ROC-AUC
- Matrices de confusión para ambos modelos (TN, FP, FN, TP).
- Declaración de mejora porcentual en F2 Score y Recall.
- Verificación formal de ausencia de fuga de datos (conjunto de test evaluado exactamente 2 veces).

## 2. Justificación de la Métrica de Negocio
- Justificación formal de la selección de $F_2$ Score ($\beta=2$).
- Explicación del coste asimétrico: el coste de perder un suscriptor inadvertidamente (falso negativo) supera ampliamente el coste operativo de enviar una oferta de retención innecesaria (falso positivo).
- Contraste formal con el Accuracy (inviable por el desbalance 73.5% / 26.5%) y con $F_1$ Score (ponderación simétrica insuficiente para la retención).
- Formulación matemática explícita de $F_2$.

## 3. Exploración de Hiperparámetros y Espacio de Búsqueda
- Descripción de la estrategia de optimización en 2 etapas:
  - Etapa 1: Exploración aleatoria amplia (RandomizedSearchCV con 50 iteraciones y 5-fold cross-validation estratificado, 250 ajustes).
  - Etapa 2: Refinamiento enfocado por cuadrícula (GridSearchCV alrededor de la cuenca óptima).
- Tabla detallada de hiperparámetros explorados y configuración final seleccionada para `RandomForestClassifier`:
  - `n_estimators`
  - `max_depth`
  - `min_samples_split`
  - `min_samples_leaf`
  - `max_features`
  - `class_weight` (mitigación algorítmica del desbalance)
  - `criterion`

## 4. Análisis de Estabilidad y Criterio de Selección (Regla 1-SE)
- Justificación detallada del trade-off entre media de validación cruzada y varianza entre folds.
- Explicación matemática de la regla de 1 error estándar: $\text{SE} = \frac{\sigma}{\sqrt{k}}$.
- Tabla de las mejores configuraciones candidatas mostrando:
  - Media de F2 en CV ($\bar{S}$)
  - Desviación estándar entre folds ($\sigma$)
  - Error estándar ($\text{SE}$)
  - Indicador de cumplimiento del umbral 1-SE ($\bar{S} \ge \bar{S}_{\max} - \text{SE}_{\max}$)
- Justificación de por qué el modelo final seleccionado minimiza el riesgo de sobreajuste y garantiza generalización estable en producción.
```

---

## Validation & Acceptance Rules

1. **Language Validation**: The report must be written entirely in Spanish with accurate ML and business terminology.
2. **Numeric Consistency**: The numbers in the report tables MUST match the recorded execution metrics on the holdout test set and cross-validation logs.
3. **No Unfilled Placeholders**: The document must not contain placeholders such as `[TODO]`, `[TBD]`, or `[INSERT METRIC]`.
