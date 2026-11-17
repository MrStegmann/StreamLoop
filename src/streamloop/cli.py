"""Command-Line Interface (CLI) for StreamLoop Churn Model Tuning Pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import joblib

from streamloop.data.ingestion import (
    DEFAULT_LOCAL_FALLBACK,
    DEFAULT_REMOTE_URL,
    IngestionError,
    load_churn_data,
)
from streamloop.data.split import split_churn_dataset
from streamloop.evaluation.boundary import LeakageBoundaryMonitor, LeakageIntegrityError
from streamloop.evaluation.metrics import compute_evaluation_metrics
from streamloop.pipeline.builder import train_and_evaluate_baseline
from streamloop.reporting.spanish_report import generate_spanish_tuning_report
from streamloop.tuning.search import (
    build_stage2_param_grid,
    extract_candidate_evaluations,
    run_stage1_randomized_search,
    run_stage2_grid_search,
)
from streamloop.tuning.selection import refit_winning_pipeline, select_winning_candidate_1se

# Standard Exit Codes
EXIT_SUCCESS: Final[int] = 0
EXIT_DATA_INGESTION_ERROR: Final[int] = 1
EXIT_DATA_SCHEMA_ERROR: Final[int] = 2
EXIT_LEAKAGE_INTEGRITY_ERROR: Final[int] = 3
EXIT_TUNING_EXECUTION_ERROR: Final[int] = 4
EXIT_REPORT_GENERATION_ERROR: Final[int] = 5

logger = logging.getLogger("streamloop")


def configure_logging(verbose: bool = False) -> None:
    """Configure deterministic structured logging format."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(handler)


def build_parser() -> argparse.ArgumentParser:
    """Build command-line argument parser."""
    parser = argparse.ArgumentParser(
        prog="streamloop-tune",
        description="StreamLoop Churn Model Hyperparameter Tuning Pipeline",
    )
    parser.add_argument(
        "-u",
        "--data-url",
        type=str,
        default=DEFAULT_REMOTE_URL,
        help="Canonical remote URL for IBM Telco Customer Churn dataset.",
    )
    parser.add_argument(
        "-f",
        "--local-fallback",
        type=Path,
        default=DEFAULT_LOCAL_FALLBACK,
        help="Local fallback path when remote download fails.",
    )
    parser.add_argument(
        "-r",
        "--report-output",
        type=Path,
        default=Path("reports/tuning_report.md"),
        help="Path where the final Spanish tuning report markdown file will be saved.",
    )
    parser.add_argument(
        "-m",
        "--model-output",
        type=Path,
        default=Path("models/churn_model_tuned.joblib"),
        help="Path where the final refitted winning pipeline estimator is serialized.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )
    parser.add_argument(
        "-t",
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of dataset reserved for the holdout test set.",
    )
    parser.add_argument(
        "-k",
        "--cv-splits",
        type=int,
        default=5,
        help="Number of stratified cross-validation folds.",
    )
    parser.add_argument(
        "-n",
        "--stage1-iter",
        type=int,
        default=50,
        help="Number of randomized parameter candidates sampled in Stage 1 search.",
    )
    parser.add_argument(
        "-j",
        "--n-jobs",
        type=int,
        default=-1,
        help="Parallel worker count for cross-validation search routines (-1 uses all cores).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=False,
        help="Verbose logging output to stdout.",
    )
    return parser


def run_pipeline(
    data_url: str = DEFAULT_REMOTE_URL,
    local_fallback: Path | str = DEFAULT_LOCAL_FALLBACK,
    report_output: Path | str = Path("reports/tuning_report.md"),
    model_output: Path | str = Path("models/churn_model_tuned.joblib"),
    seed: int = 42,
    test_size: float = 0.2,
    cv_splits: int = 5,
    stage1_iter: int = 50,
    n_jobs: int = -1,
    verbose: bool = False,
) -> int:
    """Orchestrate end-to-end churn model tuning pipeline and report generation."""
    configure_logging(verbose)
    logger.info("StreamLoop Churn Model Tuning Pipeline initialized (seed=%d)", seed)

    # 1. Ingest Dataset
    try:
        raw_df = load_churn_data(
            remote_url=data_url,
            local_fallback_path=local_fallback,
        )
        logger.info(
            "Ingestion successful. Total rows: %d, columns: %d",
            len(raw_df),
            len(raw_df.columns),
        )
    except IngestionError as exc:
        logger.error("Data ingestion failed: %s", exc)
        return EXIT_DATA_INGESTION_ERROR
    except ValueError as exc:
        logger.error("Data schema validation failed: %s", exc)
        return EXIT_DATA_SCHEMA_ERROR
    except Exception as exc:
        logger.error("Unexpected ingestion error: %s", exc)
        return EXIT_DATA_INGESTION_ERROR

    # 2. Partition Dataset
    try:
        split = split_churn_dataset(
            raw_df,
            test_size=test_size,
            random_state=seed,
        )
        train_churn_pct = (split.y_train.mean()) * 100
        test_churn_pct = (split.y_test.mean()) * 100
        logger.info(
            "Partitioning dataset: Train=%d (%.2f%% churn), Test=%d (%.2f%% churn)",
            len(split.X_train),
            train_churn_pct,
            len(split.X_test),
            test_churn_pct,
        )
    except Exception as exc:
        logger.error("Dataset partitioning error: %s", exc)
        return EXIT_DATA_SCHEMA_ERROR

    # 3. Baseline Pipeline Fitting & Touch 1 Holdout Evaluation
    monitor = LeakageBoundaryMonitor(max_allowed_touches=2)
    try:
        _, baseline_metrics = train_and_evaluate_baseline(
            split=split,
            monitor=monitor,
            random_state=seed,
        )
        logger.info(
            "Baseline Holdout Test Evaluation [TOUCH 1]: F2=%.4f, F1=%.4f, Precision=%.4f, Recall=%.4f, PR-AUC=%.4f, ROC-AUC=%.4f",
            baseline_metrics.f2_score,
            baseline_metrics.f1_score,
            baseline_metrics.precision,
            baseline_metrics.recall,
            baseline_metrics.pr_auc,
            baseline_metrics.roc_auc,
        )
    except Exception as exc:
        logger.error("Baseline fitting/evaluation failed: %s", exc)
        return EXIT_TUNING_EXECUTION_ERROR

    # 4. Hyperparameter Search (Stage 1 & Stage 2)
    try:
        # Stage 1: Randomized Search
        stage1_search = run_stage1_randomized_search(
            X_train=split.X_train,
            y_train=split.y_train,
            n_iter=stage1_iter,
            cv_splits=cv_splits,
            random_state=seed,
            n_jobs=n_jobs,
        )
        logger.info("Stage 1 complete. Best mean F2: %.4f", stage1_search.best_score_)

        # Stage 2: Focused Grid Search
        param_grid = build_stage2_param_grid(stage1_search.best_params_)
        stage2_search = run_stage2_grid_search(
            X_train=split.X_train,
            y_train=split.y_train,
            param_grid=param_grid,
            cv_splits=cv_splits,
            random_state=seed,
            n_jobs=n_jobs,
        )
        stage2_candidates = extract_candidate_evaluations(
            stage2_search.cv_results_,
            cv_splits=cv_splits,
        )

        # 5. Model Selection using 1-SE Stability Rule
        best_mean_cand, winning_cand, threshold = select_winning_candidate_1se(stage2_candidates)
        logger.info(
            "Stage 2 complete. Top mean F2: %.4f (std=%.4f, SE=%.4f)",
            best_mean_cand.mean_score,
            best_mean_cand.std_score,
            best_mean_cand.standard_error,
        )
        logger.info("Applying 1-SE Rule: Selection threshold F2 >= %.4f", threshold)
        logger.info(
            "Winning configuration selected: Candidate #%d (mean F2=%.4f, std=%.4f, lowest fold variance within 1-SE)",
            winning_cand.candidate_id,
            winning_cand.mean_score,
            winning_cand.std_score,
        )

        # 6. Refit Winning Pipeline on Full Training Split
        logger.info(
            "Refitting winning pipeline on full training set (%d samples)...", len(split.X_train)
        )
        tuned_pipeline = refit_winning_pipeline(
            winning_candidate=winning_cand,
            X_train=split.X_train,
            y_train=split.y_train,
            random_state=seed,
        )
    except Exception as exc:
        logger.error("Tuning or selection failed: %s", exc)
        return EXIT_TUNING_EXECUTION_ERROR

    # 7. Final Holdout Evaluation [Touch 2]
    try:
        monitor.record_touch("Final Tuned Model Holdout Test Evaluation")
        y_pred = tuned_pipeline.predict(split.X_test)
        y_proba = tuned_pipeline.predict_proba(split.X_test)[:, 1]
        tuned_metrics = compute_evaluation_metrics(
            y_true=split.y_test.to_numpy(),
            y_pred=y_pred,
            y_proba=y_proba,
        )
        logger.info(
            "Final Holdout Test Evaluation [TOUCH 2]: F2=%.4f, F1=%.4f, Precision=%.4f, Recall=%.4f, PR-AUC=%.4f, ROC-AUC=%.4f",
            tuned_metrics.f2_score,
            tuned_metrics.f1_score,
            tuned_metrics.precision,
            tuned_metrics.recall,
            tuned_metrics.pr_auc,
            tuned_metrics.roc_auc,
        )
        # Verify isolation
        monitor.verify_final_count(2)
        logger.info("Test set touch count verified: 2 (Baseline=1, Final=1, Leakage=0)")
    except LeakageIntegrityError as exc:
        logger.error("Leakage integrity violation: %s", exc)
        return EXIT_LEAKAGE_INTEGRITY_ERROR
    except Exception as exc:
        logger.error("Final test evaluation failed: %s", exc)
        return EXIT_TUNING_EXECUTION_ERROR

    # 8. Model Serialization
    try:
        model_path = Path(model_output)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(tuned_pipeline, model_path)
        logger.info("Saving tuned model to: %s", model_path)
    except Exception as exc:
        logger.error("Failed to serialize model: %s", exc)
        return EXIT_TUNING_EXECUTION_ERROR

    # 9. Generate Spanish Report
    try:
        report_path = Path(report_output)
        logger.info("Generating Spanish tuning report: %s", report_path)
        generate_spanish_tuning_report(
            output_path=report_path,
            baseline_metrics=baseline_metrics,
            tuned_metrics=tuned_metrics,
            best_mean_candidate=best_mean_cand,
            winning_candidate=winning_cand,
            candidates=stage2_candidates,
            test_touch_count=monitor.touch_count,
            stage1_iter=stage1_iter,
            cv_splits=cv_splits,
        )
    except Exception as exc:
        logger.error("Failed to generate Spanish report: %s", exc)
        return EXIT_REPORT_GENERATION_ERROR

    logger.info("Pipeline execution complete. Exit code: 0")
    return EXIT_SUCCESS


def main() -> None:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()
    code = run_pipeline(
        data_url=args.data_url,
        local_fallback=args.local_fallback,
        report_output=args.report_output,
        model_output=args.model_output,
        seed=args.seed,
        test_size=args.test_size,
        cv_splits=args.cv_splits,
        stage1_iter=args.stage1_iter,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )
    sys.exit(code)


if __name__ == "__main__":
    main()
