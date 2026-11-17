# Quickstart Validation Guide: Systematic Churn Model Hyperparameter Tuning

**Branch**: `001-tune-churn-model`  
**Feature**: Systematic Churn Model Hyperparameter Tuning  
**Date**: 2026-11-17  

This guide provides step-by-step instructions to set up the environment, execute the end-to-end churn model tuning pipeline, run unit and integration tests, and validate generated artifacts against the specification contracts.

---

## 1. Prerequisites

- Python 3.11+
- `uv` package and environment manager (`uv --version`)
- Network access for remote dataset download (or presence of local fallback file at `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`)

---

## 2. Environment Setup

Initialize the virtual environment and install declared dependencies using `uv`:

```bash
# Sync environment dependencies and development tools (pytest, ruff) from pyproject.toml
uv sync --extra dev

# Verify environment python version
uv run python --version
```

---

## 3. End-to-End Pipeline Execution

Execute the full hyperparameter optimization and evaluation pipeline via the CLI:

```bash
uv run python -m streamloop.cli \
  --data-url "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" \
  --local-fallback "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv" \
  --report-output "reports/tuning_report.md" \
  --model-output "models/churn_model_tuned.joblib" \
  --seed 42 \
  --stage1-iter 50 \
  --cv-splits 5 \
  --n-jobs -1
```

### Expected Output Highlights:
- Logs indicating successful remote data ingestion (or fallback if offline).
- Stratified 80/20 train/test split.
- Baseline pipeline fit and **Touch 1** holdout test evaluation.
- Stage 1: 50 randomized iterations evaluated with 5-fold cross-validation ($N=250$ fits).
- Stage 2: Focused grid search centered on top candidates ($N \approx 180$ fits).
- 1-SE rule computation selecting the lowest fold variance model within 1 SE of the best mean $F_2$ score.
- Model refit on full training split.
- Final holdout test evaluation (**Touch 2**).
- Zero data leakage verification (test touch counter strictly equals 2).
- Successful serialization of `models/churn_model_tuned.joblib` and creation of `reports/tuning_report.md`.

---

## 4. Validation Scenarios

### Scenario A: Zero Data Leakage & Test Isolation
**Goal**: Verify that holdout test data is accessed strictly twice and never seen during cross-validation or preprocessing fits.
**Command**:
```bash
uv run pytest tests/unit/test_leakage_boundary.py -v
```
**Expected Outcome**: Tests pass, verifying `X_test` was never passed to any intermediate transformer `fit` or search cross-validation loop.

---

### Scenario B: 1-SE Stability Selection Logic
**Goal**: Prove that given candidates with high mean and high variance vs. slightly lower mean and low variance, the 1-SE rule correctly selects the stable candidate.
**Command**:
```bash
uv run pytest tests/unit/test_selection_1se.py -v
```
**Expected Outcome**: Tests pass, verifying correct standard error formula ($\text{SE} = \frac{\sigma}{\sqrt{k}}$), filtering of candidates within 1 SE, and tie-breaking by minimum fold standard deviation.

---

### Scenario C: Offline Ingestion Resiliency
**Goal**: Verify the ingestion fallback mechanism works when network is disconnected or invalid URL provided.
**Command**:
```bash
uv run pytest tests/unit/test_ingestion_fallback.py -v
```
**Expected Outcome**: Ingestion attempts invalid URL, catches network exception, logs warning, and successfully loads the local CSV file.

---

### Scenario D: Spanish Tuning Report Quality Inspection
**Goal**: Validate that `reports/tuning_report.md` fulfills all contract requirements defined in [tuning-report-contract.md](file:///j:/GitHub/StreamLoop/specs/001-tune-churn-model/contracts/tuning-report-contract.md).
**Command**:
```bash
uv run pytest tests/contract/test_tuning_report_contract.py -v
```
**Expected Outcome**: Tests pass, verifying the file exists, contains the 4 required sections in Spanish, includes comparative metric tables, and has no unfilled placeholders.

---

## 5. Full Test Suite Execution

Run the complete test suite with coverage report:

```bash
uv run pytest --maxfail=1 --disable-warnings -v
```
