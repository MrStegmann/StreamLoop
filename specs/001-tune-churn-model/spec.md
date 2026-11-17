# Feature Specification: Systematic Churn Model Hyperparameter Tuning

**Feature Branch**: `001-tune-churn-model`

**Created**: 2026-11-17

**Status**: Draft

**Input**: User description: "StreamLoop — Tuning the Churn Model: Take a classifier trained on StreamLoop's customer data, and systematically tune it — first with a broad, cheap search, then with a focused, precise one — while avoiding mistakes that make tuning results meaningless (leaking test set into search, judging models on the wrong metric, or trusting a single number without checking how much it moves across folds). Produce a baseline, two-stage hyperparameter search, fold stability analysis, final single-pass holdout evaluation, and a tuning report in Spanish."

## Clarifications

### Session 2026-11-17

- Q: Which primary cross-validation scoring metric should be used to optimize the hyperparameter searches? (FR-008) → A: $F_2$ Score (`fbeta` with $\beta=2$), prioritizing recall twice as heavily as precision to penalize missed churners while discouraging excessive false positives.
- Q: What formal decision rule should determine the winning configuration when trading off mean cross-validation score against fold-to-fold variance? (FR-010) → A: 1-Standard-Error (1-SE) Rule, selecting the candidate with the lowest fold standard deviation among those whose mean cross-validation score falls within one standard error ($\text{SE} = \frac{\sigma}{\sqrt{k}}$) of the top mean score.
- Q: How should the pipeline address class imbalance during model training and hyperparameter search? (FR-003) → A: Algorithmic Weighting (`class_weight`), relying on `class_weight='balanced'` or `'balanced_subsample'` inside `RandomForestClassifier` to prevent synthetic artifacts on one-hot categorical features while counteracting class skew efficiently.
- Q: How should dataset ingestion handle caching and offline availability? (FR-001) → A: Always Remote with Local Fallback (Option B), attempting to query the IBM Telco URL first on each execution and falling back to `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` only if network or HTTP errors occur.
- Q: What cross-validation fold count and randomized search iteration budget should be standardized for the stage 1 exploration? (FR-006) → A: Stratified 5-Fold cross-validation with 50 iterations (`n_iter=50`, $k=5$), evaluating 250 candidate fits in stage 1 with parallel workers (`n_jobs=-1`).

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Establishing a Leak-Free Baseline Benchmark (Priority: P1)

As a machine learning engineer and tech lead, I need an end-to-end, leak-free pipeline trained with default hyperparameters and benchmarked against a holdout test split so that StreamLoop has an objective, reproducible reference point before tuning.

**Why this priority**: Without an established baseline evaluated under strict leak-free conditions, tuning improvements cannot be verified or trusted.

**Independent Test**: Ingest the raw customer churn dataset, partition into train and test splits, encapsulate preprocessing and a default classifier inside a single pipeline, fit on the training split, and record baseline test metrics.

**Acceptance Scenarios**:

1. **Given** raw customer records with non-numeric and blank values, **When** data is ingested and partitioned into train/test sets, **Then** all preprocessing steps reside inside a pipeline and are fitted exclusively on the training set without holdout data leakage.
2. **Given** the baseline pipeline fitted with default hyperparameters, **When** evaluated on the holdout test set, **Then** baseline performance metrics are recorded and the holdout split is sealed from any tuning process.

---

### User Story 2 - Two-Stage Hyperparameter Exploration & Stability Selection (Priority: P2)

As a tech lead and data scientist, I need a two-phase hyperparameter optimization process (broad randomized search followed by focused grid refinement) evaluated with cross-validation on the training set, so that the parameter space is systematically explored and candidates are evaluated for fold stability rather than relying solely on mean scores.

**Why this priority**: Explores high-dimensional parameter spaces efficiently while avoiding overfitting and ensuring that chosen model configurations generalize reliably across data slices.

**Independent Test**: Run a broad randomized cross-validation search over the classifier parameter space on training data, narrow down the search space around the most promising region with a focused grid search, extract cross-validation fold variance from search results, and justify final model selection based on the mean-variance trade-off.

**Acceptance Scenarios**:

1. **Given** the training split and a wide hyperparameter space, **When** a randomized cross-validation search executes using an optimization metric aligned with business priorities, **Then** top-performing parameter candidate regions are identified without accessing test data.
2. **Given** high-potential candidate regions from the randomized search, **When** a focused grid search executes cross-validation around those boundaries, **Then** detailed fold-by-fold score distributions are recorded.
3. **Given** cross-validation results from the candidate configurations, **When** selecting the winning estimator, **Then** both mean performance and fold-to-fold variance are inspected and the 1-standard-error (1-SE) rule is applied to select the candidate with lowest fold variance within 1 SE of the best mean.

---

### User Story 3 - Final Holdout Validation & Spanish Tuning Report (Priority: P3)

As a business stakeholder and technical manager, I need the final tuned model evaluated once on the holdout test set and accompanied by a detailed markdown report written in Spanish (`tuning_report.md`), so that StreamLoop can verify generalization gains, understand the business metric choice, and review production readiness.

**Why this priority**: Delivers the final verification on unseen data and provides the governance documentation required for stakeholder sign-off in the requested language.

**Independent Test**: Evaluate the best-refitted pipeline on the untouched test set using the exact same metrics as the baseline, verify that test data was accessed exactly twice across the entire process, and confirm `tuning_report.md` contains all required analytical sections in Spanish.

**Acceptance Scenarios**:

1. **Given** the final tuned estimator refitted on the full training split, **When** evaluated on the untouched holdout test set, **Then** test metrics are computed using the identical evaluation criteria applied to the baseline.
2. **Given** baseline and tuned evaluation results, **When** generating `tuning_report.md`, **Then** the report is written in Spanish and details baseline vs. tuned metrics, final chosen hyperparameters, business metric rationale, and the stability trade-off analysis.

---

### Edge Cases

- **Blank or Whitespace Data**: Attributes such as `TotalCharges` contain empty spaces for new customers with zero tenure; system must cleanly transform these without row-loss or manual leakage.
- **Network or Offline Ingestion Failure**: If the remote IBM Telco dataset URL is unreachable, system must automatically fall back to the local cached dataset at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- **Extreme Fold Variance**: If top-scoring configurations show high standard deviation across cross-validation folds, selection criteria must apply the 1-SE rule to choose the lowest-variance configuration within 1 SE of the top mean score rather than defaulting to an erratic numerical peak.
- **Class Imbalance Sensitivity**: Churn events represent a minority of subscribers (~26.5%); default accuracy masks poor recall of churners, addressed via algorithmic weighting (`class_weight='balanced'` or `'balanced_subsample'`) in the classifier paired with the $F_2$ optimization metric.
- **Hyperparameter Search Boundary Traps**: If optimal parameters lie at the edge of the initial randomized space, grid refinement must expand or center appropriately around the boundary.
- **Accidental Test Set Leakage**: Test split must never be passed to `fit`, `fit_transform`, or search cross-validation routines; it is strictly accessed once for baseline recording and once for final evaluation.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST ingest customer churn data by querying the specified IBM Telco Customer Churn URL first, falling back to a local copy at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` upon network failure, and handling non-numeric and blank values prior to pipeline ingestion.
- **FR-002**: System MUST split the dataset into stratified train and test partitions prior to any pipeline fitting or hyperparameter exploration.
- **FR-003**: System MUST encapsulate all feature preprocessing (encoding, scaling, missing value handling) and classifier estimation within a unified pipeline object, addressing class imbalance algorithmically via `class_weight` parameters (`'balanced'`, `'balanced_subsample'`) in `RandomForestClassifier` rather than synthetic data resampling.
- **FR-004**: System MUST fit an initial baseline pipeline with default hyperparameters on the training split and record its performance on the holdout test set before initiating tuning.
- **FR-005**: System MUST isolate the holdout test set completely during all hyperparameter search operations, ensuring cross-validation operates exclusively on the training split.
- **FR-006**: System MUST perform a broad initial search over the classifier's supported hyperparameter space using Stratified 5-Fold cross-validation with 50 randomized iterations (`n_iter=50`) executed across parallel workers (`n_jobs=-1`).
- **FR-007**: System MUST utilize the findings from the randomized search to narrow the parameter search space and perform a refined grid search over that focused sub-region.
- **FR-008**: System MUST configure the $F_2$ score (`fbeta` with $\beta=2$ on the churn positive class) as the deliberate cross-validation scoring metric for hyperparameter searches, weighting recall twice as heavily as precision to reflect the business cost asymmetry of lost subscribers without causing excessive false alarms.
- **FR-009**: System MUST enable automatic estimator retraining on the full training set upon search completion (`refit=True`) without manual re-fitting steps.
- **FR-010**: System MUST select the winning hyperparameter configuration using the 1-Standard-Error (1-SE) rule: among all candidate configurations whose mean cross-validation score falls within one standard error ($\text{SE} = \frac{\sigma}{\sqrt{k}}$) of the top-performing configuration, the system MUST select the candidate with the lowest fold standard deviation.
- **FR-011**: System MUST evaluate the final tuned pipeline on the holdout test set exactly once, using the identical performance metrics measured for the baseline.
- **FR-012**: System MUST produce a comprehensive markdown report (`tuning_report.md`) written in Spanish comparing baseline and tuned performance, listing final hyperparameters, explaining scoring metric justification, and articulating the stability trade-off decision.

### Key Entities

- **Customer Record**: The demographic, account, and contractual features representing a subscriber profile (e.g., tenure, contract type, payment method, monthly charges, total charges).
- **Churn Label**: Binary ground truth target indicating subscriber churn (`Yes`) versus retention (`No`).
- **Churn Pipeline**: Unified encapsulation of data preprocessing transformers and downstream classification estimator.
- **Search Result Matrix**: Detailed cross-validation metrics across all evaluated parameter combinations, including mean test scores, standard deviations, and individual fold scores.
- **Tuning Report**: Formal documentation in Spanish (`tuning_report.md`) presenting comparative model performance, parameter specifications, metric alignment, and stability analysis.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Zero data leakage throughout lifecycle: holdout test split is evaluated exactly twice (once for baseline, once for final model) and is never referenced in cross-validation searches.
- **SC-002**: Measurable improvement in business-aligned performance metric ($F_2$ score on the churn class) of the tuned model over the default baseline on the holdout test set.
- **SC-003**: 100% of candidate configurations evaluated during grid search have recorded fold standard deviation metrics for verifiable stability inspection.
- **SC-004**: Two distinct, sequentially dependent search stages (broad randomized search followed by narrowed grid search) executed and documented.
- **SC-005**: Delivery of a complete, Spanish-language tuning report (`tuning_report.md`) satisfying all four required analytical sections (baseline vs. tuned comparison, final hyperparameters, business metric rationale, and stability trade-off analysis).

## Assumptions

- The target dataset is accessible via the IBM Telco Customer Churn raw GitHub repository URL.
- Blank string entries in `TotalCharges` correspond to new customers with zero tenure and will be imputed or converted to numeric zero.
- The chosen classifier is a tree-based ensemble classifier (such as `RandomForestClassifier`) aligned with StreamLoop project constitutional principles.
- The business priority where "losing a customer unnoticed costs far more than sending an unneeded retention offer" is formalized by optimizing the $F_2$ score ($\beta=2$), which places double weighting on churn recall relative to precision.
- Train/test split uses a standard 80/20 ratio with stratification on the target variable to preserve class distribution.
- Preprocessing and classifier components operate within Python >= 3.11 using scikit-learn standard pipeline components managed via `uv`.
