# Phase 0 Research: Systematic Churn Model Hyperparameter Tuning

**Branch**: `001-tune-churn-model`  
**Feature**: Systematic Churn Model Hyperparameter Tuning  
**Date**: 2026-11-17  
**Status**: Completed  

---

## 1. Data Ingestion & Offline Fallback Strategy

### Decision
Implement a resilient ingestion module that first attempts an HTTP GET request to the canonical IBM Telco Customer Churn raw CSV URL:
`https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
If an HTTP error, DNS resolution failure, connection timeout, or offline status occurs, the module logs a warning and loads the local backup copy from:
`data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
If the local file is missing when network fails, the loader raises a clear `FileNotFoundError` explaining how to provide the local data.

### Rationale
- Complies strictly with Clarification 4 and requirement FR-001 (Always Remote with Local Fallback).
- Guarantees execution works in isolated, air-gapped, or offline development environments without breaking CI/CD pipelines.
- Standardizes ingestion while keeping the raw data untouched.

### Alternatives Considered
- *Local-only ingestion*: Rejected because it requires manual manual downloading prior to execution and fails to satisfy remote ingestion requirement FR-001.
- *Remote-only without fallback*: Rejected because network hiccups or GitHub raw endpoint rate-limiting would cause build/run failures.

---

## 2. Preprocessing & Leak-Free Pipeline Architecture

### Decision
Construct a unified `sklearn.pipeline.Pipeline` with a `ColumnTransformer` that encapsulates all transformations and the classifier. Specifically:
1. **Target extraction**: Binary map `Churn`: `{'Yes': 1, 'No': 0}`.
2. **Identifier dropping**: Exclude `customerID` from feature matrix $X$.
3. **Data wrangling on `TotalCharges`**: Convert `TotalCharges` to numeric (`pd.to_numeric(errors='coerce')`), imputing the 11 known whitespace/empty entries with `0.0` (matching `tenure == 0` for brand-new subscribers) in a leak-free custom or standard imputation transformer.
4. **Numerical features**: [`tenure`, `MonthlyCharges`, `TotalCharges`] processed via `SimpleImputer(strategy='median')` and `StandardScaler()`.
5. **Categorical features**: All remaining categorical columns (16 features, e.g. `Contract`, `PaymentMethod`, `InternetService`, etc.) processed via `OneHotEncoder(handle_unknown='ignore', drop='first')`.
6. **Classifier step**: `RandomForestClassifier(random_state=42)`.

### Rationale
- Satisfies Constitution Principle I (Pipeline-First & Leak-Free Architecture) and FR-003.
- Fitting transformers (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`) only on `X_train` inside cross-validation folds ensures zero target or distribution leakage from the validation folds or the test set.

### Alternatives Considered
- *Pandas `get_dummies` outside pipeline*: Rejected as an explicit violation of Constitution Principle I (causes column mismatch during inference and data leakage).
- *Dropping rows with blank `TotalCharges`*: Rejected because it removes valid new customers ($N=11$) and alters test distribution unnecessarily.

---

## 3. Class Imbalance Handling

### Decision
Use algorithmic weighting within the classifier via `class_weight='balanced'` or `class_weight='balanced_subsample'` inside `RandomForestClassifier`.

### Rationale
- Directly fulfills Clarification 3 and FR-003.
- The Telco churn dataset exhibits ~26.5% positive churn rate (approx 1:3 ratio).
- Synthetic oversampling (e.g. SMOTE) on one-hot categorical features can synthesize unrealistic binary combinations (e.g., fractional or inconsistent category representations) and adds computational overhead.
- Tree-based class weighting adjusts the cost function directly at each tree node or bootstrap sample without distorting the underlying data manifold.

### Alternatives Considered
- *SMOTE / ADASYN via `imblearn`*: Rejected per Clarification 3 to avoid synthetic artifacts on sparse one-hot encoded categorical columns.
- *Random undersampling*: Rejected because discarding ~50% of retained subscriber data discards valuable pattern information.

---

## 4. Evaluation & Optimization Metric ($F_2$ Score)

### Decision
Standardize on the $F_2$ Score (`fbeta_score` with $\beta=2$, `pos_label=1`) as the primary cross-validation optimization metric:
$$F_2 = (1 + 2^2) \cdot \frac{\text{Precision} \cdot \text{Recall}}{2^2 \cdot \text{Precision} + \text{Recall}} = 5 \cdot \frac{\text{Precision} \cdot \text{Recall}}{4 \cdot \text{Precision} + \text{Recall}}$$
Configured via `sklearn.metrics.make_scorer(fbeta_score, beta=2, pos_label=1)`.
Comprehensive evaluation additionally tracks:
- Precision, Recall ($F_1$ and $F_2$)
- PR-AUC (`average_precision_score`)
- ROC-AUC (`roc_auc_score`)
- Confusion Matrix (TP, FP, TN, FN)

### Rationale
- Fulfills Clarification 1, FR-008, and Constitution Principle IV.
- In subscription services like StreamLoop, undetected churn incurs a severe customer lifetime value loss, whereas sending a proactive retention offer (e.g. discount email) incurs negligible unit cost. $\beta=2$ weights recall twice as heavily as precision while continuing to penalize reckless false-positive spamming.
- Accuracy is rejected per Constitution Principle IV as 73.5% accuracy can be achieved trivially by predicting no churn.

### Alternatives Considered
- *Standard $F_1$ Score ($\beta=1$)*: Treats precision and recall symmetrically, failing to capture the business asymmetry of churn retention costs.
- *Raw Recall*: Leads to degenerate solutions where the model predicts everyone churns, blowing up marketing costs.

---

## 5. Two-Stage Hyperparameter Optimization Strategy

### Decision
Execute a sequential two-stage tuning strategy on the training split using `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`:
1. **Stage 1: Broad Exploration (`RandomizedSearchCV`)**:
   - Budget: `n_iter=50`, $k=5$ folds (250 total fits), parallelized (`n_jobs=-1`).
   - Distribution space:
     - `classifier__n_estimators`: `randint(100, 500)`
     - `classifier__max_depth`: `[None, 5, 8, 12, 16, 20]`
     - `classifier__min_samples_split`: `randint(2, 20)`
     - `classifier__min_samples_leaf`: `randint(1, 10)`
     - `classifier__max_features`: `['sqrt', 'log2', 0.2, 0.4, 0.6]`
     - `classifier__class_weight`: `['balanced', 'balanced_subsample']`
     - `classifier__criterion`: `['gini', 'entropy']`
2. **Stage 2: Focused Grid Search (`GridSearchCV`)**:
   - Extract top candidate configurations from Stage 1.
   - Construct a targeted grid around the best hyperparameter ranges (e.g., 3-4 points per sensitive parameter, resulting in 36–72 candidate combinations x 5 folds = 180–360 fits).
   - Compute full fold-by-fold results and store the full search matrix.

### Rationale
- Fulfills FR-006, FR-007, and Clarification 5.
- Stage 1 rapidly searches high-dimensional space without combinatorial explosion; Stage 2 refines the vicinity of the optimal basin of attraction.

### Alternatives Considered
- *Single-stage exhaustive grid search*: Combinatorial explosion ($5 \times 6 \times 6 \times 5 \times 4 \times 2 \approx 7200$ combinations $\times 5$ folds = 36,000 fits) would be computationally intractable.
- *Bayesian Optimization (Optuna)*: While effective, standardizing on scikit-learn's native `RandomizedSearchCV` and `GridSearchCV` avoids additional heavy dependencies, perfectly aligns with Constitution Principle I, and directly produces the required `cv_results_` fold matrix for stability analysis.

---

## 6. Model Selection Rule: 1-Standard-Error (1-SE) Rule Implementation

### Decision
Implement the 1-SE selection rule as follows:
1. Identify the configuration $c^*$ with the maximum mean cross-validation score across folds:
   $$\bar{S}_{\max} = \max_c \bar{S}_c$$
2. Compute the standard error of configuration $c^*$:
   $$\text{SE}^* = \frac{\sigma_{c^*}}{\sqrt{k}} \quad (\text{with } k=5)$$
3. Identify candidate set $\mathcal{C}_{\text{1-SE}}$ consisting of all configurations $c$ satisfying:
   $$\bar{S}_c \ge \bar{S}_{\max} - \text{SE}^*$$
4. From $\mathcal{C}_{\text{1-SE}}$, select the winning model $c_{\text{winner}}$ that minimizes the fold standard deviation:
   $$c_{\text{winner}} = \arg\min_{c \in \mathcal{C}_{\text{1-SE}}} \sigma_c$$
5. Refit the pipeline with $c_{\text{winner}}$ parameters on the entire training set.

### Rationale
- Fulfills Clarification 2, FR-010, and SC-003.
- Prevents selecting an unstable configuration that merely overfitted one or two CV folds, favoring robust models that perform consistently across splits.

### Alternatives Considered
- *Strict argmax of mean score*: Vulnerable to high variance across folds; the peak may be an artifact of split noise.
- *Smallest model / max regularization within 1-SE*: Standard in Lasso/Ridge, but in Random Forest, tree depth/leaf size variance trade-offs are best captured by fold stability as specified in Clarification 2.

---

## 7. Holdout Test Split Isolation & Double-Touch Protocol

### Decision
Isolate the test set (`test_size=0.2, stratify=y, random_state=42`) completely from the tuning loop:
- **Touch 1**: Immediately after fitting baseline pipeline on `(X_train, y_train)`, evaluate on `(X_test, y_test)` to record baseline benchmark.
- **Tuning Quarantine**: Randomized search and grid search execute strictly on `X_train` with internal cross-validation.
- **Touch 2**: The refitted winning pipeline is evaluated once on `(X_test, y_test)`.
- Verification check logs timestamps/invocation count ensuring no leakage occurred.

### Rationale
- Fulfills FR-004, FR-005, FR-011, SC-001, and Constitution Principle I.

---

## 8. Spanish Tuning Report Specification (`tuning_report.md`)

### Decision
Generate `tuning_report.md` in Spanish adhering to the 4 mandatory sections:
1. **Resumen Ejecutivo y Métricas Comparativas**: Tabla comparativa (Baseline vs. Modelo Ajustado) mostrando $F_2$, Precision, Recall, PR-AUC, ROC-AUC y Matriz de Confusión sobre el conjunto de test.
2. **Justificación de la Métrica de Negocio**: Explicación detallada de por qué se prioriza el $F_2$ ($\beta=2$) sobre el $F_1$ y el Accuracy en el contexto de la retención de clientes de StreamLoop.
3. **Exploración de Hiperparámetros y Espacio de Búsqueda**: Configuración de Fase 1 (aleatoria) y Fase 2 (grid), hiperparámetros finales seleccionados y su interpretación técnica.
4. **Análisis de Estabilidad y Criterio 1-SE**: Justificación de la configuración ganadora demostrando la media frente a la variabilidad entre folds, aplicando la regla de 1 error estándar.

### Rationale
- Direct requirement from User Story 3, FR-012, and SC-005.

---

## 9. Modular Code Architecture & Packaging

### Decision
Create a clean, modular Python package under `src/` managed via `uv`:
```text
src/
├── streamloop/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── ingestion.py       # Download with fallback & schema parsing
│   │   └── split.py           # Stratified train/test split
│   ├── pipeline/
│   │   ├── __init__.py
│   │   └── builder.py         # ColumnTransformer + Pipeline builder
│   ├── tuning/
│   │   ├── __init__.py
│   │   ├── search.py          # Two-stage RandomizedSearchCV & GridSearchCV
│   │   └── selection.py       # 1-SE rule calculation & candidate filter
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         # F2 scorer, PR-AUC, ROC-AUC, classification reports
│   ├── reporting/
│   │   ├── __init__.py
│   │   └── spanish_report.py  # Markdown generator for tuning_report.md
│   └── cli.py                 # Command line runner (uv run python -m streamloop.cli)
```

### Rationale
- Satisfies Constitution Principle II, Principle III, Principle V.
- Clean separation of concerns allows dedicated unit and integration testing via `pytest`.

---

## Conclusion
All technical unknowns are resolved. The design is fully aligned with StreamLoop Constitution v1.0.0 and ready for Phase 1 (Data Model, Contracts, and Quickstart).
