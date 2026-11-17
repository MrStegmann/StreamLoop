---
description: "Task list for systematic churn model hyperparameter tuning implementation"
---

# Tasks: Systematic Churn Model Hyperparameter Tuning

**Input**: Design documents from `/specs/001-tune-churn-model/`  
**Prerequisites**: [plan.md](file:///j:/GitHub/StreamLoop/specs/001-tune-churn-model/plan.md), [spec.md](file:///j:/GitHub/StreamLoop/specs/001-tune-churn-model/spec.md), [research.md](file:///j:/GitHub/StreamLoop/specs/001-tune-churn-model/research.md), [data-model.md](file:///j:/GitHub/StreamLoop/specs/001-tune-churn-model/data-model.md), [contracts/](file:///j:/GitHub/StreamLoop/specs/001-tune-churn-model/contracts/), [constitution.md](file:///j:/GitHub/StreamLoop/.specify/memory/constitution.md)

**Tests**: Test tasks are included per project constitution quality gates and contract verification requirements.

**Organization**: Tasks are grouped by user story to enable independent implementation, testing, and delivery of each story.

## Format: `- [ ] [TaskID] [P?] [Story?] Description with file path`

- **[P]**: Can run in parallel (different files, no dependencies on incomplete tasks)
- **[Story]**: Which user story this task belongs to (`[US1]`, `[US2]`, `[US3]`)
- Every task includes an exact target file path

## Path Conventions

- **Modular library**: `src/streamloop/`
- **Data storage**: `data/raw/`
- **Model artifacts**: `models/`
- **Reports**: `reports/`
- **Test suite**: `tests/` (`contract/`, `integration/`, `unit/`)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization, dependency management with `uv`, and storage structure.

- [X] T001 Configure project dependencies, runtime (Python >= 3.11), and tool settings in pyproject.toml
- [X] T002 Create directory structure for src/streamloop, data/raw, models, reports, and tests per implementation plan
- [X] T003 [P] Place local fallback dataset at data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure and utilities that MUST be complete before ANY user story can be implemented.

**⚠️ CRITICAL**: No user story work can begin until this phase is complete.

- [X] T004 Create package initialization modules in src/streamloop/__init__.py and subpackage __init__.py files
- [X] T005 [P] Implement evaluation metrics and custom F2 scorer (beta=2, pos_label=1) in src/streamloop/evaluation/metrics.py
- [X] T006 [P] Implement leakage boundary monitor to track and enforce holdout test touch count in src/streamloop/evaluation/boundary.py
- [X] T007 [P] Create shared test fixtures and synthetic churn dataset generators in tests/conftest.py

**Checkpoint**: Foundation ready — User Story 1 implementation can now begin.

---

## Phase 3: User Story 1 - Establishing a Leak-Free Baseline Benchmark (Priority: P1) 🎯 MVP

**Goal**: Ingest customer data (remote URL with local fallback), handle whitespace values in `TotalCharges`, partition into stratified 80/20 train/test splits, encapsulate preprocessing and default `RandomForestClassifier` with balanced class weights inside a scikit-learn `Pipeline`, fit on training data, evaluate on holdout test set (Touch 1), and record baseline benchmark metrics.

**Independent Test**: Run data ingestion and splitting with seed 42, fit baseline pipeline strictly on `X_train`, evaluate on `X_test`, and confirm baseline metrics ($F_2$, $F_1$, Precision, Recall, PR-AUC, ROC-AUC, Confusion Matrix) are recorded with test touch count = 1.

### Tests for User Story 1

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T008 [P] [US1] Create schema contract test for Telco dataset columns and types in tests/contract/test_schema_contract.py
- [X] T009 [P] [US1] Create ingestion and fallback unit test simulating offline conditions in tests/unit/test_ingestion_fallback.py
- [X] T010 [P] [US1] Create pipeline builder unit test verifying ColumnTransformer and RandomForest in tests/unit/test_pipeline_builder.py
- [X] T011 [P] [US1] Create leakage boundary unit test verifying single test access during baseline in tests/unit/test_leakage_boundary.py
- [X] T012 [P] [US1] Create ingestion integration test validating data loading and schema in tests/integration/test_ingestion_integration.py

### Implementation for User Story 1

- [X] T013 [US1] Implement resilient data ingestion with remote URL fetch and local CSV fallback in src/streamloop/data/ingestion.py
- [X] T014 [US1] Implement stratified train/test dataset partitioner with customerID removal in src/streamloop/data/split.py
- [X] T015 [US1] Implement ColumnTransformer and baseline RandomForest pipeline builder in src/streamloop/pipeline/builder.py
- [X] T016 [US1] Integrate baseline fitting and Touch 1 holdout evaluation with boundary monitoring in src/streamloop/pipeline/builder.py

**Checkpoint**: At this point, User Story 1 is fully functional and testable independently as the project MVP.

---

## Phase 4: User Story 2 - Two-Stage Hyperparameter Exploration & Stability Selection (Priority: P2)

**Goal**: Systematically explore the classifier hyperparameter space on the training split using a two-stage search (broad `RandomizedSearchCV` with 50 iterations and 5-fold Stratified CV, followed by a focused `GridSearchCV` around promising parameter regions) scored via the $F_2$ metric, and select the winning configuration using the 1-Standard-Error (1-SE) stability rule to balance mean score against fold variance.

**Independent Test**: Execute Stage 1 randomized search and Stage 2 grid search on `X_train`, record fold-to-fold score distributions, apply the 1-SE rule ($\text{SE} = \frac{\sigma}{\sqrt{k}}$) to select the lowest fold variance candidate within 1 SE of top mean score, and refit on the full training split.

### Tests for User Story 2

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T017 [P] [US2] Create unit test for 1-SE stability selection rule and standard error calculation in tests/unit/test_selection_1se.py
- [X] T018 [P] [US2] Create unit test for two-stage hyperparameter search routines in tests/unit/test_search_stages.py

### Implementation for User Story 2

- [X] T019 [US2] Implement 1-Standard-Error (1-SE) stability selection algorithm and refit helper in src/streamloop/tuning/selection.py
- [X] T020 [US2] Implement Stage 1 RandomizedSearchCV and Stage 2 focused GridSearchCV tuning routines in src/streamloop/tuning/search.py

**Checkpoint**: User Stories 1 AND 2 both work independently; tuning produces a verified 1-SE winning model without touching holdout data.

---

## Phase 5: User Story 3 - Final Holdout Validation & Spanish Tuning Report (Priority: P3)

**Goal**: Evaluate the refitted winning pipeline on the holdout test set exactly once (Touch 2), verify test isolation (touch count = 2), serialize the winning model to `models/churn_model_tuned.joblib`, generate the executive Spanish markdown report `reports/tuning_report.md` fulfilling all 4 contract sections, and orchestrate the entire flow via the command-line interface in `src/streamloop/cli.py`.

**Independent Test**: Execute the full CLI pipeline, verify holdout test data is accessed strictly twice, verify `models/churn_model_tuned.joblib` exists, and validate that `reports/tuning_report.md` contains all 4 mandatory Spanish sections with consistent metrics.

### Tests for User Story 3

> **NOTE: Write these tests FIRST, ensure they FAIL before implementation**

- [X] T021 [P] [US3] Create contract test for Spanish tuning report structure and metrics in tests/contract/test_tuning_report_contract.py
- [X] T022 [P] [US3] Create contract test for CLI options and standard exit codes in tests/contract/test_cli_contract.py
- [X] T023 [P] [US3] Create unit test for Spanish markdown report generator in tests/unit/test_spanish_report.py
- [X] T024 [P] [US3] Create end-to-end integration test running full pipeline workflow in tests/integration/test_end_to_end_pipeline.py

### Implementation for User Story 3

- [X] T025 [US3] Implement Spanish markdown tuning report generator with 4 mandatory sections in src/streamloop/reporting/spanish_report.py
- [X] T026 [US3] Implement main CLI entrypoint orchestrating pipeline execution, logging, and exit codes in src/streamloop/cli.py
- [X] T027 [US3] Integrate model serialization to models/churn_model_tuned.joblib and report export in src/streamloop/cli.py

**Checkpoint**: All three user stories are complete, independently verifiable, and fully integrated.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Code quality enforcement, full verification suite execution, quickstart scenario validation, and documentation.

- [X] T028 [P] Run code formatting and linting checks using ruff in pyproject.toml
- [X] T029 [P] Run entire test suite with coverage validation via uv run pytest
- [X] T030 Execute end-to-end CLI validation scenario per quickstart.md
- [X] T031 [P] Update project documentation and usage guide in README.md

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies — can start immediately.
- **Foundational (Phase 2)**: Depends on Setup (Phase 1) completion — BLOCKS all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational (Phase 2) completion. Establishes the baseline pipeline and MVP.
- **User Story 2 (Phase 4)**: Depends on User Story 1 (Phase 3) completion for pipeline builder and data split components.
- **User Story 3 (Phase 5)**: Depends on User Story 2 (Phase 4) completion for winning model selection and tuning metrics.
- **Polish (Phase 6)**: Depends on all user stories being complete.

### User Story Dependencies

```
[Phase 1: Setup] ──> [Phase 2: Foundational]
                             │
                             ▼
                 [Phase 3: US1 - Baseline (MVP)]
                             │
                             ▼
                 [Phase 4: US2 - 2-Stage Tuning & 1-SE]
                             │
                             ▼
                 [Phase 5: US3 - Final Eval, Report & CLI]
                             │
                             ▼
                 [Phase 6: Polish & Verification]
```

### Within Each User Story

1. Write tests FIRST and verify failure before implementation.
2. Implement data structures and business logic before integrations.
3. Verify all tests pass for that story before advancing to the next priority.
4. Verify holdout test set access boundaries.

### Parallel Opportunities

- **Phase 1**: `T003` can run in parallel with `T001` and `T002`.
- **Phase 2**: `T005`, `T006`, and `T007` can all be implemented in parallel once `T004` completes.
- **Phase 3 (US1)**: All tests (`T008`, `T009`, `T010`, `T011`, `T012`) can be written in parallel.
- **Phase 4 (US2)**: Tests `T017` and `T018` can be written in parallel. Selection logic `T019` can be implemented in parallel with search routines `T020`.
- **Phase 5 (US3)**: Tests `T021`, `T022`, `T023`, and `T024` can be written in parallel. Spanish report generator `T025` can be implemented in parallel with CLI options setup in `T026`.
- **Phase 6**: `T028`, `T029`, and `T031` can run in parallel.

---

## Parallel Example: User Story 1

```bash
# Launch all test writing for User Story 1 in parallel:
Task T008: "Create schema contract test for Telco dataset columns and types in tests/contract/test_schema_contract.py"
Task T009: "Create ingestion and fallback unit test simulating offline conditions in tests/unit/test_ingestion_fallback.py"
Task T010: "Create pipeline builder unit test verifying ColumnTransformer and RandomForest in tests/unit/test_pipeline_builder.py"
Task T011: "Create leakage boundary unit test verifying single test access during baseline in tests/unit/test_leakage_boundary.py"
Task T012: "Create ingestion integration test validating data loading and schema in tests/integration/test_ingestion_integration.py"
```

## Parallel Example: User Story 2

```bash
# Launch tests for User Story 2 in parallel:
Task T017: "Create unit test for 1-SE stability selection rule and standard error calculation in tests/unit/test_selection_1se.py"
Task T018: "Create unit test for two-stage hyperparameter search routines in tests/unit/test_search_stages.py"
```

## Parallel Example: User Story 3

```bash
# Launch test and report implementation for User Story 3 in parallel:
Task T021: "Create contract test for Spanish tuning report structure and metrics in tests/contract/test_tuning_report_contract.py"
Task T022: "Create contract test for CLI options and standard exit codes in tests/contract/test_cli_contract.py"
Task T023: "Create unit test for Spanish markdown report generator in tests/unit/test_spanish_report.py"
Task T025: "Implement Spanish markdown tuning report generator with 4 mandatory sections in src/streamloop/reporting/spanish_report.py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup (`T001`–`T003`).
2. Complete Phase 2: Foundational (`T004`–`T007`).
3. Complete Phase 3: User Story 1 (`T008`–`T016`).
4. **STOP and VALIDATE**: Verify baseline pipeline runs leak-free, computes baseline $F_2$, and passes all US1 tests.
5. Demonstrate working baseline reference benchmark.

### Incremental Delivery

1. **Increment 1 (MVP)**: Deliver baseline pipeline and leak-free evaluation benchmark (US1).
2. **Increment 2**: Add Stage 1 randomized search, Stage 2 grid search, and 1-SE stability selection rule (US2).
3. **Increment 3**: Add final holdout evaluation, Spanish tuning report generation, and CLI interface (US3).
4. **Increment 4**: Code formatting, complete test verification, and documentation polish (Phase 6).

---

## Notes

- All tasks follow strict format: `- [ ] [TaskID] [P?] [Story?] Description with file path`.
- Class imbalance is handled algorithmically via `class_weight='balanced'` in `RandomForestClassifier` per constitution and spec.
- Optimization metric is locked to $F_2$ Score ($\beta=2$, `pos_label=1`).
- Holdout test set is quarantined and accessed strictly twice (Touch 1 for baseline, Touch 2 for winning tuned model).
- Tuning report is rendered strictly in Spanish (`reports/tuning_report.md`).
