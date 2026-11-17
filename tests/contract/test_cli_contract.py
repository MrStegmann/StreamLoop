"""Contract tests for CLI argument parsing and exit codes."""

from __future__ import annotations

from streamloop.cli import (
    EXIT_DATA_INGESTION_ERROR,
    EXIT_DATA_SCHEMA_ERROR,
    EXIT_LEAKAGE_INTEGRITY_ERROR,
    EXIT_REPORT_GENERATION_ERROR,
    EXIT_SUCCESS,
    EXIT_TUNING_EXECUTION_ERROR,
    build_parser,
)


def test_cli_exit_codes_contract():
    """Verify exit code values comply strictly with CLI contract."""
    assert EXIT_SUCCESS == 0
    assert EXIT_DATA_INGESTION_ERROR == 1
    assert EXIT_DATA_SCHEMA_ERROR == 2
    assert EXIT_LEAKAGE_INTEGRITY_ERROR == 3
    assert EXIT_TUNING_EXECUTION_ERROR == 4
    assert EXIT_REPORT_GENERATION_ERROR == 5


def test_cli_parser_defaults():
    """Verify default arguments configured in CLI parser."""
    parser = build_parser()
    args = parser.parse_args([])

    assert args.seed == 42
    assert args.test_size == 0.2
    assert args.cv_splits == 5
    assert args.stage1_iter == 50
    assert args.n_jobs == -1
    assert args.verbose is False
    assert "telco-customer-churn" in args.data_url.lower()


def test_cli_parser_custom_flags():
    """Verify short flags override default arguments."""
    parser = build_parser()
    args = parser.parse_args(
        [
            "-s",
            "99",
            "-t",
            "0.25",
            "-k",
            "3",
            "-n",
            "10",
            "-j",
            "2",
            "-v",
        ]
    )
    assert args.seed == 99
    assert args.test_size == 0.25
    assert args.cv_splits == 3
    assert args.stage1_iter == 10
    assert args.n_jobs == 2
    assert args.verbose is True
