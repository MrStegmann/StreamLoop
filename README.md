# StreamLoop: Systematic Churn Model Hyperparameter Tuning

Production-grade, leak-free machine learning pipeline designed to systematically optimize, evaluate, and serialize customer churn prediction models.

---

## Overview

StreamLoop optimizes a `RandomForestClassifier` pipeline on the IBM Telco Customer Churn dataset using:
- **Pipeline Encapsulation**: End-to-end preprocessing (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`) and estimation strictly encapsulated in a scikit-learn `Pipeline`.
- **Zero Data Leakage**: Stratified 80/20 train/test split where the holdout test set is accessed strictly twice (Touch 1 for baseline benchmark, Touch 2 for final tuned validation) and never during cross-validation or preprocessing fits.
- **Metric Optimization**: Primary optimization driven by the **$F_2$ Score** ($\beta=2$, `pos_label=1`), penalizing false negatives (unidentified churners) twice as heavily as false positives.
- **Two-Stage Tuning**: Global exploration via `RandomizedSearchCV` (50 iterations x 5 folds = 250 fits) followed by focused local refinement via `GridSearchCV` (~24-72 fits).
- **1-Standard-Error (1-SE) Stability Rule**: Final selection prioritizes the lowest fold-to-fold variance configuration within 1 SE of the best mean cross-validation score ($\text{SE} = \sigma / \sqrt{k}$), guaranteeing robustness and avoiding overfitting.
- **Executive Reporting**: Automated generation of `reports/tuning_report.md` in Spanish detailing comparative performance, business metrics, and stability rationale.

---

## Prerequisites

- **Python**: `>=3.11`
- **uv**: Package and environment manager (`uv --version`)

---

## Installation & Setup

Sync the virtual environment and install dependencies:

```bash
uv sync --extra dev
```

---

## CLI Usage

Execute the end-to-end tuning pipeline:

```bash
uv run python -m streamloop.cli [OPTIONS]
```

### CLI Options

| Flag | Option | Default | Description |
|---|---|---|---|
| `-u` | `--data-url` | IBM Telco GitHub Raw URL | Remote URL to fetch dataset. |
| `-f` | `--local-fallback` | `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Local fallback path when offline. |
| `-r` | `--report-output` | `reports/tuning_report.md` | Path to generated Spanish markdown report. |
| `-m` | `--model-output` | `models/churn_model_tuned.joblib` | Path to serialized tuned pipeline artifact. |
| `-s` | `--seed` | `42` | Global random seed for reproducible splits and estimators. |
| `-t` | `--test-size` | `0.2` | Fraction reserved for holdout test split. |
| `-k` | `--cv-splits` | `5` | Number of stratified cross-validation folds. |
| `-n` | `--stage1-iter` | `50` | Number of randomized parameter candidates sampled in Stage 1. |
| `-j` | `--n-jobs` | `-1` | Parallel CPU workers for CV search (-1 uses all cores). |
| `-v` | `--verbose` | `False` | Verbose debug logging to stdout. |

---

## Testing & Quality Assurance

Run code linting and formatting checks:

```bash
uv run ruff check .
uv run ruff format --check .
```

Run test suite with code coverage:

```bash
uv run pytest --cov=streamloop tests/
```

---

## Jupyter Notebook (EDA & Tuning Walkthrough)

In accordance with **Constitution Principle V (Dual-Track EDA to Modular Production Pipelines)**, an end-to-end interactive diagnostic notebook is provided:

```bash
uv run jupyter notebook notebooks/churn_model_tuning.ipynb
```

---

## Output Artifacts

- **Jupyter Notebook**: `notebooks/churn_model_tuning.ipynb` — Executed top-to-bottom analysis with EDA, two-stage search plots, 1-SE stability graphs, and comparative evaluation.
- **Visual Artifacts**: `reports/figures/` — Exported high-resolution diagnostic charts (churn distributions, contract rates, 1-SE trade-offs, confusion matrices).
- **Model Artifact**: `models/churn_model_tuned.joblib` — Serialized winning scikit-learn pipeline ready for production scoring.
- **Analytical Report**: `reports/tuning_report.md` — Formal executive report in Spanish containing executive summaries, confusion matrices, business justifications, and 1-SE stability tables.
