# Bug Assessment: TotalCharges Pre-Split Imputation & Baseline Class Weight Default Discrepancy

- **Slug**: churn-imputation-baseline
- **Created**: 2026-11-18
- **Source**: pasted text
- **Verdict**: valid
- **Severity**: high

## Report (verbatim or summarized)

The user reported two interrelated pipeline and baseline modeling issues:
1. `clean_churn_dataset` fills `TotalCharges` with `0.0` before the data split, which means this imputation happens outside the pipeline, leaving the `SimpleImputer` unable to detect the missing values. Missing entries should remain `NaN` and the pipeline should handle the imputation.
2. The baseline model does not use default hyperparameters: `build_churn_pipeline` sets `class_weight='balanced'`, yet the report labels it as 'Default'. The baseline should use `RandomForestClassifier` with its actual default settings (`class_weight=None`), and the hyperparameter tuning process should explore `class_weight` (e.g., `None`, `'balanced'`, `'balanced_subsample'`).

## Symptom

1. **Premature Imputation / Bypassed Pipeline Imputer**: In `src/streamloop/data/ingestion.py`, whitespace strings in `TotalCharges` are replaced with `"0.0"` and coerced via `pd.to_numeric(...).fillna(0.0)`. As a result, the `SimpleImputer(strategy="median")` inside the preprocessor pipeline never encounters missing values (`NaN`) and thus cannot execute its intended median imputation fitted strictly on training data.
2. **Mislabeled Baseline Hyperparameters**: In `src/streamloop/pipeline/builder.py`, `build_churn_pipeline()` hardcodes `"class_weight": "balanced"`. When `train_and_evaluate_baseline()` executes, it logs and documents that it fits a baseline with "default hyperparameters", but `RandomForestClassifier`'s actual default in scikit-learn is `class_weight=None`. Furthermore, the tuning space in `src/streamloop/tuning/search.py` restricts `class_weight` to `["balanced", "balanced_subsample"]`, preventing exploration of the actual default setting (`None`).

## Reproduction

1. Load raw Telco data or run `clean_churn_dataset()` on a DataFrame with whitespace values in `TotalCharges`:
   ```python
   import pandas as pd
   from streamloop.data.ingestion import clean_churn_dataset
   df = pd.DataFrame({"TotalCharges": [" ", " 100.5 ", ""]})
   cleaned = clean_churn_dataset(df)
   assert cleaned["TotalCharges"].isna().sum() == 0  # Missing values were already imputed with 0.0!
   ```
2. Inspect pipeline preprocessor imputer behavior:
   ```python
   from streamloop.pipeline.builder import build_preprocessor
   prep = build_preprocessor()
   # prep.named_transformers_['num'].named_steps['imputer'] will see 0.0 instead of NaN,
   # distorting the median calculation and rendering the imputer ineffective.
   ```
3. Inspect `build_churn_pipeline()` defaults:
   ```python
   from streamloop.pipeline.builder import build_churn_pipeline
   pipe = build_churn_pipeline()
   rf = pipe.named_steps["classifier"]
   assert rf.class_weight == "balanced"  # Not the scikit-learn default (None)
   ```
4. Check tuning search space and report:
   ```python
   from streamloop.tuning.search import STAGE1_PARAM_DISTRIBUTIONS
   print(STAGE1_PARAM_DISTRIBUTIONS["classifier__class_weight"])
   # Outputs ['balanced', 'balanced_subsample'], excluding None
   ```

## Suspected Code Paths

- `src/streamloop/data/ingestion.py:78-85` — `clean_churn_dataset()` prematurely coerces and fills missing/whitespace `TotalCharges` with `0.0`.
- `src/streamloop/pipeline/builder.py:86-90` — `build_churn_pipeline()` sets default `"class_weight": "balanced"` instead of allowing scikit-learn default `None`.
- `src/streamloop/pipeline/builder.py:120-121` — `train_and_evaluate_baseline()` creates a baseline claiming "default hyperparameters" while using `class_weight='balanced'`.
- `src/streamloop/tuning/search.py:25` — `STAGE1_PARAM_DISTRIBUTIONS` lacks `None` in `classifier__class_weight` search space.
- `src/streamloop/reporting/spanish_report.py:158` — Spanish report hardcodes baseline `class_weight` as `'balanced'` instead of `None`.
- `reports/tuning_report.md:69` — Existing report table lists baseline `class_weight` as `'balanced'`.
- `tests/unit/test_pipeline_builder.py:20` — Unit test explicitly asserts `rf.class_weight == "balanced"`.
- `tests/integration/test_ingestion_integration.py:18` — Integration test asserts `df["TotalCharges"].isna().sum() == 0` right after ingestion.

## Root Cause Hypothesis

During the initial implementation, data ingestion was designed to handle whitespace in `TotalCharges` by replacing it with `0.0` to avoid type coercion issues. However, an imputation step in ingestion occurs prior to `train_test_split`, causing premature imputation and blinding the `SimpleImputer(strategy="median")` in the pipeline preprocessor. Concurrently, to immediately address class imbalance, `class_weight="balanced"` was hardcoded as a default parameter in `build_churn_pipeline()`, conflating baseline model definition with hyperparameter tuning choices. Confidence: High.

## Proposed Remediation

**Preferred**:
1. **In `src/streamloop/data/ingestion.py`**:
   - Update `clean_churn_dataset()` so `TotalCharges` whitespace is coerced to `NaN` without `.fillna(0.0)`.
   - Specifically: `cleaned_df["TotalCharges"] = pd.to_numeric(cleaned_df["TotalCharges"], errors="coerce")`.
   - Update `clean_churn_dataset` docstrings accordingly.
2. **In `src/streamloop/pipeline/builder.py`**:
   - Change `build_churn_pipeline()` so that `classifier_params` defaults to `{}` and `RandomForestClassifier` retains its native default `class_weight=None` (or `params` only sets `random_state=random_state` unless `classifier_params` explicitly provides `class_weight`).
   - The baseline evaluated in `train_and_evaluate_baseline()` will thus be a genuine default `RandomForestClassifier(random_state=random_state, class_weight=None)`.
3. **In `src/streamloop/tuning/search.py`**:
   - Update `STAGE1_PARAM_DISTRIBUTIONS["classifier__class_weight"]` to include `[None, "balanced", "balanced_subsample"]` so hyperparameter search explores both unweighted default and weighted configurations.
   - Ensure `build_stage2_param_grid()` preserves the selected `class_weight` (including when it is `None`).
4. **In `src/streamloop/reporting/spanish_report.py`**:
   - Update the table and descriptions to show `class_weight` baseline as `None` (or `'None'`) and the explored space as `[None, 'balanced', 'balanced_subsample']`.
5. **In tests**:
   - Update `tests/unit/test_pipeline_builder.py:20` to verify `rf.class_weight is None`.
   - Update `tests/integration/test_ingestion_integration.py:18` to verify that raw ingestion preserves `NaN` for missing `TotalCharges` (11 rows in the standard Telco dataset) and that the pipeline successfully fits and imputes them during pipeline execution.

**Alternatives**:
- *Add a separate explicit baseline builder function*: e.g., `build_default_baseline_pipeline()` vs `build_churn_pipeline()`. However, keeping `build_churn_pipeline()` adhering to standard scikit-learn defaults (with overrides passed via `classifier_params`) is cleaner, idiomatic, and avoids duplicate pipeline construction logic.

**Files likely to change**:
- `src/streamloop/data/ingestion.py`
- `src/streamloop/pipeline/builder.py`
- `src/streamloop/tuning/search.py`
- `src/streamloop/reporting/spanish_report.py`
- `tests/unit/test_pipeline_builder.py`
- `tests/integration/test_ingestion_integration.py`
- `tests/unit/test_ingestion_fallback.py`
- `reports/tuning_report.md` (regenerated or updated)

**Tests to add or update**:
- `test_clean_churn_dataset_preserves_nan`: Verify that whitespace / empty strings in `TotalCharges` become `NaN` in cleaned data and are not filled with 0.0.
- `test_build_churn_pipeline_default_hyperparameters`: Verify default `RandomForestClassifier` has `class_weight is None`.
- `test_pipeline_imputer_handles_nan_totalcharges`: Verify that `build_churn_pipeline().fit(X_train, y_train)` successfully imputes `NaN` values via `SimpleImputer`.
- Update `test_build_churn_pipeline_structure` in `tests/unit/test_pipeline_builder.py`.
- Update `test_real_dataset_ingestion_and_split` in `tests/integration/test_ingestion_integration.py`.

## Risks & Considerations

- **Baseline Metrics Change**: Evaluating the baseline model with `class_weight=None` instead of `class_weight='balanced'` will result in a lower initial recall / $F_2$ score on the baseline, as unweighted Random Forest typically optimizes accuracy at the expense of minority class recall. This will actually highlight and quantify the substantial impact of tuning and class weighting.
- **Contract & Report Assertions**: Any hardcoded metric checks in tests or contracts for the baseline should be verified to ensure they use valid dynamic metrics or thresholds.
- **Stage 2 Grid Search**: Ensure `stage1_best_params.get("classifier__class_weight")` properly handles `None` without falling back erroneously to `'balanced'`.

## Open Questions

- None. Both bug behaviors and required remediations are clearly defined.
