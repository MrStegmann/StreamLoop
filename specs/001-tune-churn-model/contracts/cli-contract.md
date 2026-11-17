# CLI Interface Contract: StreamLoop Churn Model Tuning

**Branch**: `001-tune-churn-model`  
**Interface**: Command-Line Interface (CLI)  
**Target Module**: `streamloop.cli` / `streamloop-tune`  
**Execution Command**: `uv run python -m streamloop.cli [OPTIONS]`

---

## Command Overview

The CLI coordinates the end-to-end churn model hyperparameter tuning workflow: data ingestion (remote with local fallback), stratified splitting, baseline pipeline fitting and evaluation, two-stage hyperparameter search (RandomizedSearchCV + GridSearchCV) with Stratified 5-Fold cross-validation, 1-SE stability model selection, final single-pass holdout evaluation, and generation of the Spanish tuning report (`tuning_report.md`).

---

## Command Options & Arguments

| Option | Flag | Type | Default | Description |
|---|---|---|---|---|
| `--data-url` | `-u` | string | `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv` | Canonical remote URL for the IBM Telco Customer Churn dataset. |
| `--local-fallback` | `-f` | Path | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Local fallback path when remote download fails or is unreachable. |
| `--report-output` | `-r` | Path | `reports/tuning_report.md` | Path where the final Spanish tuning report markdown file will be saved. |
| `--model-output` | `-m` | Path | `models/churn_model_tuned.joblib` | Path where the final refitted winning pipeline estimator is serialized. |
| `--seed` | `-s` | int | `42` | Global random seed for train/test split, cross-validation, and estimator reproducibility. |
| `--test-size` | `-t` | float | `0.2` | Fraction of dataset reserved for the holdout test set ($0.1 \le t \le 0.4$). |
| `--cv-splits` | `-k` | int | `5` | Number of stratified cross-validation folds. |
| `--stage1-iter` | `-n` | int | `50` | Number of randomized parameter candidates sampled in Stage 1 search. |
| `--n-jobs` | `-j` | int | `-1` | Parallel worker count for cross-validation search routines (-1 uses all available CPU cores). |
| `--verbose` | `-v` | bool flag | `False` | Verbose logging output to stdout. |

---

## Standard Exit Codes

| Exit Code | Reason | Description |
|---|---|---|
| `0` | `SUCCESS` | Pipeline executed successfully, models evaluated, and report generated. |
| `1` | `DATA_INGESTION_ERROR` | Remote URL failed and local fallback file is missing or unreadable. |
| `2` | `DATA_SCHEMA_ERROR` | Ingested dataset violates mandatory schema requirements or missing column headers. |
| `3` | `LEAKAGE_INTEGRITY_ERROR` | Test set touch counter violated (accessed more than twice). |
| `4` | `TUNING_EXECUTION_ERROR` | Exception raised during cross-validation search or 1-SE selection. |
| `5` | `REPORT_GENERATION_ERROR` | Failure generating or writing `tuning_report.md`. |

---

## Standard Output & Logging Format

All logs MUST be deterministic and structured. Standard execution produces output in the following format:

```text
[INFO] StreamLoop Churn Model Tuning Pipeline initialized (seed=42)
[INFO] Ingesting dataset from remote URL: https://raw.githubusercontent.com/...
[INFO] Ingestion successful. Total rows: 7043, columns: 21
[INFO] Partitioning dataset: Train=5634 (26.54% churn), Test=1409 (26.54% churn)
[INFO] Fitting Baseline Pipeline with default hyperparameters...
[INFO] Baseline Holdout Test Evaluation [TOUCH 1]: F2=0.5124, F1=0.5512, Precision=0.6234, Recall=0.4947, PR-AUC=0.5892, ROC-AUC=0.8234
[INFO] Initiating Stage 1: RandomizedSearchCV (50 iterations x 5 folds = 250 fits, metric=F2, n_jobs=-1)...
[INFO] Stage 1 complete. Best mean F2: 0.6921 (params: ...)
[INFO] Initiating Stage 2: Focused GridSearchCV (36 candidates x 5 folds = 180 fits, metric=F2)...
[INFO] Stage 2 complete. Top mean F2: 0.7045 (std=0.0210, SE=0.0094)
[INFO] Applying 1-SE Rule: Selection threshold F2 >= 0.6951
[INFO] Winning configuration selected: Candidate #14 (mean F2=0.6982, std=0.0084, lowest fold variance within 1-SE)
[INFO] Refitting winning pipeline on full training set (5634 samples)...
[INFO] Final Holdout Test Evaluation [TOUCH 2]: F2=0.7102, F1=0.6430, Precision=0.5612, Recall=0.9679, PR-AUC=0.6710, ROC-AUC=0.8654
[INFO] Test set touch count verified: 2 (Baseline=1, Final=1, Leakage=0)
[INFO] Saving tuned model to: models/churn_model_tuned.joblib
[INFO] Generating Spanish tuning report: reports/tuning_report.md
[INFO] Pipeline execution complete. Exit code: 0
```
