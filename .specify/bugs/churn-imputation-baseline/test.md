# Bug Verification: TotalCharges Pre-Split Imputation & Baseline Class Weight Default Discrepancy

- **Slug**: churn-imputation-baseline
- **Tested**: 2026-11-18
- **Assessment**: ./assessment.md
- **Fix**: ./fix.md
- **Result**: verified

## Summary

The fix successfully holds: missing and whitespace entries in `TotalCharges` are preserved as `NaN` during data ingestion so that the pipeline's `SimpleImputer` executes median imputation cleanly without data leakage, and `build_churn_pipeline()` uses scikit-learn's authentic default `class_weight=None` while Stage 1 hyperparameter search explores `[None, 'balanced', 'balanced_subsample']`. All 27 tests in the regression suite passed without failure and the linter reported zero issues.

## Checks Performed

| Check | Command / Action | Result | Notes |
|-------|------------------|--------|-------|
| Reproduction (post-fix) | Python script testing `clean_churn_dataset` NaN preservation, `build_churn_pipeline()` default classifier weights, and Stage 1 parameter space | pass | Verified `TotalCharges` retains `NaN`, pipeline `SimpleImputer` computes median across clean training data, and `class_weight is None`. |
| New / updated tests | `.venv\Scripts\pytest tests/unit/test_pipeline_builder.py tests/integration/test_ingestion_integration.py tests/unit/test_search_stages.py -v` | pass | 10 passed in 3.57s covering pipeline structure, parameter overrides, imputer behavior, real Telco dataset 11-row NaN preservation, and Stage 1/2 search spaces. |
| Regression suite | `.venv\Scripts\pytest` | pass | 27 passed in 46.36s across contract, unit, and end-to-end integration test suites. |
| Lint / type-check | `.venv\Scripts\ruff check src tests` | pass | All checks passed; no lint or formatting violations. |

## Output Excerpts

### Post-fix Reproduction Check
```
Reproduction checks passed: missing values preserved as NaN, class_weight is None, tuning space includes None.
```

### New / Updated Test Execution
```
tests/unit/test_pipeline_builder.py::test_build_churn_pipeline_structure PASSED [ 10%]
tests/unit/test_pipeline_builder.py::test_build_churn_pipeline_custom_classifier_params PASSED [ 20%]
tests/unit/test_pipeline_builder.py::test_clean_churn_dataset_preserves_nan PASSED [ 30%]
tests/unit/test_pipeline_builder.py::test_pipeline_imputer_handles_nan_totalcharges PASSED [ 40%]
tests/unit/test_pipeline_builder.py::test_pipeline_fit_and_predict PASSED [ 50%]
tests/unit/test_pipeline_builder.py::test_train_and_evaluate_baseline PASSED [ 60%]
tests/integration/test_ingestion_integration.py::test_real_dataset_ingestion_and_split PASSED [ 70%]
tests/unit/test_search_stages.py::test_stage1_param_distributions_includes_none_class_weight PASSED [ 80%]
tests/unit/test_search_stages.py::test_build_stage2_param_grid PASSED    [ 90%]
tests/unit/test_search_stages.py::test_stage1_search_execution PASSED    [100%]
============================= 10 passed in 3.57s ==============================
```

### Full Regression Suite
```
collected 27 items
============================= 27 passed in 46.36s =============================
```

### Lint Check
```
$ .venv\Scripts\ruff check src tests
All checks passed!
```

## Residual Risks

- Downstream consumers expecting `TotalCharges` to contain no missing values immediately following `clean_churn_dataset()` must ensure the DataFrame is processed through the scikit-learn pipeline or an explicit imputer before invoking models or metrics that cannot handle `np.nan`. This is standard scikit-learn pipeline practice.
- The baseline model's recall and $F_2$ score without class weighting are lower than previously reported with `'balanced'`, which accurately represents unweighted Random Forest baseline performance and better highlights the value added by hyperparameter tuning.

## Recommendation

Close the bug — verified end-to-end. Both premature pre-split imputation and baseline hyperparameter discrepancy are completely resolved, validated with automated reproductions, targeted unit/integration tests, full test suite execution, and code style validation.
