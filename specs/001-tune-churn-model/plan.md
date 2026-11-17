# Implementation Plan: Systematic Churn Model Hyperparameter Tuning

**Branch**: `001-tune-churn-model` | **Date**: 2026-11-17 | **Spec**: [spec.md](file:///j:/GitHub/StreamLoop/specs/001-tune-churn-model/spec.md)

**Input**: Feature specification from `specs/001-tune-churn-model/spec.md`

---

## Summary

StreamLoop requires a leak-free, production-grade machine learning pipeline to systematically tune a customer churn classifier. The solution ingests customer data (remote IBM Telco URL with local CSV fallback), enforces an isolated 80/20 train/test split, and encapsulates all preprocessing and estimation inside a unified scikit-learn `Pipeline`. Optimization proceeds through a two-stage search over `RandomForestClassifier` hyperparameters: a broad exploration with `RandomizedSearchCV` ($n=50$ iterations, $k=5$ stratified folds) followed by a focused `GridSearchCV` around promising parameter regions. Candidates are scored using the $F_2$ Score ($\beta=2$) to prioritize churn recall over precision. Final model selection applies the 1-Standard-Error (1-SE) rule, selecting the lowest-variance configuration within 1 SE of the best mean CV score. The winning model is evaluated once on the holdout test set (accessed strictly twice throughout the lifecycle) and documented via an executive Spanish markdown tuning report (`tuning_report.md`).

---

## Technical Context

**Language/Version**: Python >= 3.11 managed via `uv`

**Primary Dependencies**: `scikit-learn` (>=1.4), `pandas` (>=2.2), `numpy` (>=1.26), `joblib` (>=1.3), `requests` (>=2.31)

**Storage**: Local filesystem:
- Raw data: `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`
- Serialized model: `models/churn_model_tuned.joblib`
- Analytical reports: `reports/tuning_report.md`

**Testing**: `pytest` (>=8.0) executed via `uv run pytest`

**Target Platform**: Cross-platform (Windows / Linux / macOS) CLI execution

**Project Type**: Modular ML library and command-line pipeline

**Performance Goals**:
- Stage 1 Randomized Search: 50 iterations x 5 folds (250 fits) in < 180 seconds using `n_jobs=-1`
- Stage 2 Focused Grid Search: 36–72 candidates x 5 folds (180–360 fits) in < 180 seconds using `n_jobs=-1`
- End-to-end execution of full pipeline in < 6 minutes on standard developer hardware

**Constraints**:
- Strict zero data leakage: holdout test set is evaluated exactly twice (baseline and final model) and never touched during preprocessing fit or cross-validation
- Class imbalance handled via algorithmic weighting (`class_weight='balanced'` / `'balanced_subsample'`) in `RandomForestClassifier` without synthetic resampling
- Optimization metric strictly locked to $F_2$ Score (`fbeta_score` with $\beta=2$, `pos_label=1`)
- Final candidate selection governed strictly by the 1-Standard-Error rule ($\text{SE} = \frac{\sigma}{\sqrt{k}}$)
- Ingestion must query remote URL first with automatic fallback to local CSV upon failure
- Report generation delivered in professional Spanish (`tuning_report.md`)

**Scale/Scope**:
- Dataset: 7,043 rows, 21 columns
- Cross-validation: 5-fold stratified cross-validation ($k=5$)
- Codebase modularized under `src/streamloop/` with unit and contract tests in `tests/`

---

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Constitutional Principle | Requirement Summary | Architectural Compliance Assessment | Status |
|---|---|---|---|
| **I. Pipeline-First & Leak-Free Architecture** | Encapsulate all transformations, encoding, scaling, and estimator in declarative `Pipeline`. No ad-hoc transformations on full dataset. | All data transformations (`ColumnTransformer`, `OneHotEncoder`, `StandardScaler`, `SimpleImputer`) are encapsulated inside a scikit-learn `Pipeline` alongside `RandomForestClassifier`. `fit` executes strictly on `X_train`. | **PASS** |
| **II. Modern Environment & Dependency Governance** | Python >= 3.11 managed strictly via `uv` with `pyproject.toml` and `uv.lock`. All execution via `uv run`. | Runtimes and dependencies are managed exclusively with `uv`. All CLI and test executions run via `uv run`. | **PASS** |
| **III. Deterministic Data Processing & Vectorization** | Vectorized operations in `pandas`/`numpy`. Standardized, explicit `random_state` across all splits, searches, and models. | Random seeds (`seed=42`) parameterized across train/test splits, `StratifiedKFold`, `RandomizedSearchCV`, and `RandomForestClassifier`. Vectorized operations utilized throughout. | **PASS** |
| **IV. Imbalanced Classification & Rigorous Evaluation** | Never rely on raw accuracy. Use algorithmic weighting (`class_weight`) or calibrated resampling. Benchmark with PR-AUC, ROC-AUC, F-scores, Confusion Matrix. | Imbalance handled algorithmically via `class_weight='balanced'` in `RandomForestClassifier`. Optimization driven by $F_2$ Score. Comprehensive evaluation tracks PR-AUC, ROC-AUC, Precision, Recall, and Confusion Matrices. | **PASS** |
| **V. Dual-Track EDA to Modular Production Pipelines** | Production logic and transformers MUST reside in modular Python packages in `src/`, backed by tests. | All code is organized as a modular Python package in `src/streamloop/`, with complete CLI runner and `pytest` suite in `tests/`. No production logic in notebooks. | **PASS** |

*Post-Design Evaluation*: All 5 constitutional principles are fully satisfied with zero violations.

---

## Project Structure

### Documentation (this feature)

```text
specs/001-tune-churn-model/
├── spec.md                     # Feature specification
├── plan.md                     # Implementation plan (/speckit-plan output)
├── research.md                 # Phase 0 research findings (/speckit-plan output)
├── data-model.md               # Phase 1 data model & state lifecycle (/speckit-plan output)
├── quickstart.md               # Phase 1 quickstart validation guide (/speckit-plan output)
├── contracts/                  # Phase 1 interface contracts (/speckit-plan output)
│   ├── cli-contract.md
│   ├── dataset-schema-contract.md
│   └── tuning-report-contract.md
├── checklists/
│   └── requirements.md         # Specification quality checklist
└── tasks.md                    # Phase 2 implementation task list (/speckit-tasks output)
```

### Source Code (repository root)

```text
src/
└── streamloop/
    ├── __init__.py
    ├── data/
    │   ├── __init__.py
    │   ├── ingestion.py        # Remote URL fetch with local CSV fallback
    │   └── split.py            # Stratified train/test splitting with seed control
    ├── pipeline/
    │   ├── __init__.py
    │   └── builder.py          # ColumnTransformer & unified Pipeline construction
    ├── tuning/
    │   ├── __init__.py
    │   ├── search.py           # Stage 1 RandomizedSearchCV & Stage 2 GridSearchCV
    │   └── selection.py        # 1-Standard-Error (1-SE) stability selection rule
    ├── evaluation/
    │   ├── __init__.py
    │   ├── metrics.py          # F2 scorer, PR-AUC, ROC-AUC, Confusion Matrix calculation
    │   └── boundary.py        # Test set touch monitor to verify zero leakage
    ├── reporting/
    │   ├── __init__.py
    │   └── spanish_report.py   # Markdown generator for Spanish tuning report
    └── cli.py                  # Main CLI entrypoint (uv run python -m streamloop.cli)

data/
└── raw/
    └── WA_Fn-UseC_-Telco-Customer-Churn.csv  # Local fallback dataset

models/
└── .gitkeep                    # Directory for serialized joblib model artifacts

reports/
└── .gitkeep                    # Directory for generated tuning_report.md

tests/
├── conftest.py                 # Shared test fixtures (sample DataFrames, synthetic splits)
├── contract/
│   ├── test_cli_contract.py
│   ├── test_schema_contract.py
│   └── test_tuning_report_contract.py
├── integration/
│   ├── test_end_to_end_pipeline.py
│   └── test_ingestion_integration.py
└── unit/
    ├── test_ingestion_fallback.py
    ├── test_leakage_boundary.py
    ├── test_pipeline_builder.py
    ├── test_search_stages.py
    ├── test_selection_1se.py
    └── test_spanish_report.py

pyproject.toml                  # Project dependencies and tool configurations
README.md                       # Project overview and run instructions
```

**Structure Decision**: Selected a modular single-project Python architecture with standard `src/` layout. Core domain concerns (data ingestion, pipeline construction, hyperparameter tuning, model evaluation, and reporting) are isolated into dedicated subpackages within `src/streamloop/` to enable rigorous unit testing and independent contract verification under `tests/`.

---

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
|---|---|---|
| *None* | N/A | Full compliance with StreamLoop Constitution v1.0.0 achieved without exceptions. |
