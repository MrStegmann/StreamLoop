"""End-to-end integration test executing the complete tuning pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline

from streamloop.cli import run_pipeline


def test_end_to_end_pipeline_execution(tmp_path: Path, local_dataset_path: Path):
    """Verify full pipeline runs from data loading to report and model export."""
    report_file = tmp_path / "reports" / "tuning_report.md"
    model_file = tmp_path / "models" / "churn_model_tuned.joblib"

    exit_code = run_pipeline(
        data_url="https://invalid.url.to.force.fallback/churn.csv",
        local_fallback=local_dataset_path,
        report_output=report_file,
        model_output=model_file,
        seed=42,
        test_size=0.2,
        cv_splits=2,
        stage1_iter=2,
        n_jobs=1,
        verbose=True,
    )

    assert exit_code == 0
    assert model_file.exists()
    assert report_file.exists()

    # Load and verify serialized model artifact
    loaded_model = joblib.load(model_file)
    assert isinstance(loaded_model, Pipeline)
    assert "preprocessor" in loaded_model.named_steps
    assert "classifier" in loaded_model.named_steps
