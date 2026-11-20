# Bug Fix: TotalCharges Pre-Split Imputation & Baseline Class Weight Default Discrepancy

- **Slug**: churn-imputation-baseline
- **Fixed**: 2026-11-18
- **Assessment**: ./assessment.md
- **Status**: applied

## Summary

Removed premature imputation of `TotalCharges` in `clean_churn_dataset` so missing values remain `NaN` for pipeline `SimpleImputer` handling, and eliminated hardcoded `class_weight="balanced"` from `build_churn_pipeline` so the baseline genuinely evaluates scikit-learn's default hyperparameters (`class_weight=None`) while extending tuning exploration to include unweighted and weighted variants.

## Changes

| File | Change | Notes |
|------|--------|-------|
| `src/streamloop/data/ingestion.py` | modified | Coerced `TotalCharges` via `pd.to_numeric(..., errors="coerce")` without `.fillna(0.0)`, leaving missing entries as `NaN`. Updated docstring. |
| `src/streamloop/pipeline/builder.py` | modified | Removed default `"class_weight": "balanced"` in `build_churn_pipeline()`, restoring scikit-learn's native `class_weight=None` for the baseline. |
| `src/streamloop/tuning/search.py` | modified | Added `None` to `STAGE1_PARAM_DISTRIBUTIONS["classifier__class_weight"]` and preserved `class_weight=None` in `build_stage2_param_grid()`. |
| `src/streamloop/reporting/spanish_report.py` | modified | Updated report generation table to show baseline `class_weight` as `None` and explored space as `[None, 'balanced', 'balanced_subsample']`. |
| `reports/tuning_report.md` | modified | Aligned baseline table with `class_weight=None` and updated parameter options. |
| `tests/unit/test_pipeline_builder.py` | modified | Updated baseline assertions to `rf.class_weight is None`; added unit tests for custom classifier params, NaN preservation in ingestion, and imputer execution in pipeline. |
| `tests/integration/test_ingestion_integration.py` | modified | Updated assertion to expect 11 `NaN` rows in raw Telco `TotalCharges` rather than premature 0.0 imputation. |
| `tests/unit/test_search_stages.py` | modified | Added verification that `None` is in `STAGE1_PARAM_DISTRIBUTIONS` and `build_stage2_param_grid` correctly preserves `class_weight=None`. |

## Diff Highlights

```python
# src/streamloop/data/ingestion.py
-        if cleaned_df["TotalCharges"].dtype == object:
-            cleaned_df["TotalCharges"] = cleaned_df["TotalCharges"].replace(
-                r"^\s*$", "0.0", regex=True
-            )
-        cleaned_df["TotalCharges"] = pd.to_numeric(
-            cleaned_df["TotalCharges"], errors="coerce"
-        ).fillna(0.0)
+        cleaned_df["TotalCharges"] = pd.to_numeric(
+            cleaned_df["TotalCharges"], errors="coerce"
+        )
```

```python
# src/streamloop/pipeline/builder.py
     params: dict[str, Any] = {
         "random_state": random_state,
-        "class_weight": "balanced",
     }
```

```python
# src/streamloop/tuning/search.py
-    "classifier__class_weight": ["balanced", "balanced_subsample"],
+    "classifier__class_weight": [None, "balanced", "balanced_subsample"],
```

## Tests Added or Updated

- `tests/unit/test_pipeline_builder.py::test_build_churn_pipeline_structure` — Assert default classifier `class_weight is None`.
- `tests/unit/test_pipeline_builder.py::test_build_churn_pipeline_custom_classifier_params` — Assert `classifier_params` overrides default settings.
- `tests/unit/test_pipeline_builder.py::test_clean_churn_dataset_preserves_nan` — Assert whitespace/empty strings in `TotalCharges` become `NaN` and are not imputed with 0.0.
- `tests/unit/test_pipeline_builder.py::test_pipeline_imputer_handles_nan_totalcharges` — Assert `SimpleImputer` in pipeline successfully fits and imputes missing `TotalCharges`.
- `tests/integration/test_ingestion_integration.py::test_real_dataset_ingestion_and_split` — Assert 11 missing `TotalCharges` rows in Telco dataset are preserved as `NaN`.
- `tests/unit/test_search_stages.py::test_stage1_param_distributions_includes_none_class_weight` — Assert `None` is present in Stage 1 `class_weight` search space.
- `tests/unit/test_search_stages.py::test_build_stage2_param_grid` — Assert Stage 2 grid retains `class_weight=None`.

## Local Verification

- Commands run: `.venv\Scripts\pytest` → 27 passed in 49.17s (all contract, unit, and end-to-end integration tests passing).
- Manual checks: Verified `imputer.statistics_` properly computes median on `TotalCharges` without pre-split contamination.

## Deviations from Assessment

None. All changes strictly implemented the preferred remediation plan from the assessment.

## Follow-ups

- Run `/speckit-bug-test slug=churn-imputation-baseline` to generate verification report and validate fix persistence.
