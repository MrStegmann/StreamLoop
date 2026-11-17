<!--
Sync Impact Report
- Version change: 0.0.0 (template) → 1.0.0
- List of modified principles:
  - [PRINCIPLE_1_NAME] → I. Pipeline-First & Leak-Free Architecture (scikit-learn & imbalanced-learn)
  - [PRINCIPLE_2_NAME] → II. Modern Environment & Dependency Governance (uv & Python)
  - [PRINCIPLE_3_NAME] → III. Deterministic Data Processing & Vectorization (pandas & numpy)
  - [PRINCIPLE_4_NAME] → IV. Imbalanced Classification & Rigorous Evaluation (RandomForestClassifier & imbalanced-learn)
  - [PRINCIPLE_5_NAME] → V. Dual-Track EDA to Modular Production Pipelines (Jupyter, Matplotlib, Seaborn)
- Added sections:
  - Technology Stack & Tooling Constraints
  - Development Workflow & Quality Gates
- Removed sections:
  - None (template placeholders converted to concrete project principles)
- Follow-up TODOs:
  - None
-->

# StreamLoop Constitution

## Core Principles

### I. Pipeline-First & Leak-Free Architecture (scikit-learn & imbalanced-learn)
All data transformations, feature engineering steps, categorical encodings, scaling, resampling, and model inference MUST be encapsulated inside declarative scikit-learn `Pipeline` or `imbalanced-learn` (`imblearn.pipeline.Pipeline`) objects. Ad-hoc, manual feature transformations applied across whole datasets outside cross-validation folds are strictly forbidden. Every custom transformer MUST inherit from `BaseEstimator` and `TransformerMixin`, implement stateless `fit` or stateful parameters strictly learned from training data, and provide deterministic `transform` methods.
*Rationale: Encapsulating preprocessing and resampling in formal pipelines guarantees zero data leakage between training and testing splits and ensures end-to-end reproducibility during cross-validation, hyperparameter tuning, and production inference.*

### II. Modern Environment & Dependency Governance (uv & Python)
All Python environments, runtimes (Python >= 3.11), and dependencies MUST be strictly managed using `uv`. Every project dependency MUST be declared in `pyproject.toml` with reproducible, locked resolution preserved in `uv.lock`. Team members and automated CI/CD runners MUST execute commands through `uv run` (e.g., `uv run pytest`, `uv run python -m ...`). Direct, untracked installations via global `pip` are prohibited.
*Rationale: `uv` provides deterministic, high-speed virtual environment and package management, eliminating environment drift, transitive dependency breakage, and platform-specific synchronization discrepancies.*

### III. Deterministic Data Processing & Vectorization (pandas & numpy)
Data ingestion, cleaning, and tabular wrangling MUST utilize typed, schema-validated `pandas` DataFrames. Numerical operations and array transformations MUST leverage vectorized `numpy` computations without Python-level iteration loops over rows. Every data operation involving randomness—including dataset partitioning, train-test splitting, cross-validation folds, and data subsampling—MUST explicitly set and parameterize a standardized `random_state` seed.
*Rationale: High-throughput tabular workflows require vectorized performance, explicit column schema validation, and strict random seed control to ensure byte-for-byte reproducibility across runs.*

### IV. Imbalanced Classification & Rigorous Evaluation (RandomForestClassifier & imbalanced-learn)
When tackling classification tasks with class distribution skew, model training MUST account for imbalance using algorithmic weighting (e.g., `class_weight='balanced'` or `BalancedRandomForestClassifier`) or calibrated resampling strategies (e.g., SMOTE, RandomUnderSampler) integrated strictly through `imblearn.pipeline.Pipeline`. Evaluation MUST NEVER rely on raw accuracy. Models MUST be evaluated and benchmarked using Precision-Recall AUC (PR-AUC), ROC-AUC, Macro/Balanced F1-Score, and detailed Confusion Matrices. Baseline and tuned `RandomForestClassifier` models MUST document hyperparameter search spaces (`n_estimators`, `max_depth`, `min_samples_split`, `criterion`) and feature importance metrics.
*Rationale: High overall accuracy on imbalanced datasets frequently masks catastrophic minority-class failure. Enforcing threshold-independent evaluation metrics and leak-free resampling guarantees robust real-world predictive validity.*

### V. Dual-Track EDA to Modular Production Pipelines (Jupyter, Matplotlib, Seaborn)
Jupyter Notebooks are designated exclusively for exploratory data analysis (EDA), interactive hypothesis testing, and diagnostic data visualization using `matplotlib` and `seaborn`. Notebooks MUST be executed linearly from top to bottom without hidden states, cleared of heavy binary outputs or sensitive data before committing, and paired with exported visual artifacts (saved in structured directories such as `reports/figures/`). Production logic, reusable transformers, and training routines MUST NOT reside in notebooks; they MUST be refactored into modular, testable Python packages within `src/` or equivalent modules.
*Rationale: Separating rapid visual exploration from production codebase engineering prevents untested notebook logic from contaminating deployment pipelines while maximizing research agility.*

## Technology Stack & Tooling Constraints

The project standardizes on the following core ecosystem:
- **Language & Runtime**: Python >= 3.11 managed via `uv`.
- **Core ML Frameworks**: `scikit-learn` and `imbalanced-learn`.
- **Baseline Classifier**: `RandomForestClassifier` (and ensemble variants) with documented hyperparameter grids and feature importance tracking.
- **Data & Computation**: `pandas` for tabular data structures and `numpy` for vectorized numerical operations.
- **Exploratory & Diagnostic Visualization**: `jupyter notebook`, `matplotlib`, and `seaborn`.
- **Code Quality & Testing**: Automated linting/formatting via `ruff` and unit/contract testing via `pytest`, executed under `uv run`.

## Development Workflow & Quality Gates

Every feature, model modification, and pipeline improvement MUST pass the following verification gates prior to merge:
1. **Pipeline Integrity**: Any new transformer or preprocessing step MUST be tested against simulated data splits to verify `fit`/`transform` contract compliance and absence of target leakage.
2. **Deterministic Reproduction**: Random seeds (`random_state`) MUST be injected consistently across data splits, resampling routines, and ensemble models.
3. **Evaluation Thresholds**: Pull requests introducing model changes MUST provide comparative performance tables against established baselines across precision, recall, PR-AUC, and F1-score.
4. **Notebook Hygiene & Modularization**: Exploratory notebooks MUST execute top-to-bottom without error. Any logic intended for automated pipelines MUST be migrated to modular Python files backed by unit tests.
5. **Quality & Dependency Check**: `uv lock --check` and test suites MUST pass cleanly in CI.

## Governance

1. **Constitutional Authority**: This constitution defines the non-negotiable architectural and engineering rules for StreamLoop. All designs, specifications, tasks, code reviews, and pull requests MUST comply with these principles.
2. **Amendment Procedure**: Amendments to this constitution require a formal proposal detailing the rationale, impact assessment on existing pipelines and models, and consensus among core contributors.
3. **Semantic Versioning**:
   - **MAJOR (X.0.0)**: Removal or fundamental redefinition of core architectural principles, model governance constraints, or tooling lifecycles.
   - **MINOR (1.X.0)**: Addition of new principles, new supported libraries, or materially expanded workflow gates.
   - **PATCH (1.0.X)**: Wording refinements, documentation clarifications, and non-semantic corrections.
4. **Compliance Review**: All Pull Requests and automated reviews MUST explicitly verify compliance against the Core Principles outlined in this document.

**Version**: 1.0.0 | **Ratified**: 2026-11-17 | **Last Amended**: 2026-11-17
