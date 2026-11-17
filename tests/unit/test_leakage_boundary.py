"""Unit tests for leakage boundary monitor and test set touch constraints."""

from __future__ import annotations

import pytest

from streamloop.evaluation.boundary import LeakageBoundaryMonitor, LeakageIntegrityError


def test_boundary_monitor_tracking():
    """Verify monitor tracks touches and allows up to max_allowed_touches."""
    monitor = LeakageBoundaryMonitor(max_allowed_touches=2)
    assert monitor.touch_count == 0

    c1 = monitor.record_touch("Baseline Evaluation")
    assert c1 == 1
    assert monitor.touch_count == 1

    c2 = monitor.record_touch("Final Model Evaluation")
    assert c2 == 2
    assert monitor.touch_count == 2
    assert monitor.verify_final_count(2) is True


def test_boundary_monitor_exceeded():
    """Verify monitor raises LeakageIntegrityError on third touch."""
    monitor = LeakageBoundaryMonitor(max_allowed_touches=2)
    monitor.record_touch("Baseline")
    monitor.record_touch("Final")

    with pytest.raises(LeakageIntegrityError, match="Test set leakage detected"):
        monitor.record_touch("Illegal extra test access")


def test_boundary_monitor_verify_mismatch():
    """Verify exception if verify_final_count does not match actual count."""
    monitor = LeakageBoundaryMonitor(max_allowed_touches=2)
    monitor.record_touch("Baseline")
    with pytest.raises(LeakageIntegrityError, match="Leakage boundary verification failed"):
        monitor.verify_final_count(2)
