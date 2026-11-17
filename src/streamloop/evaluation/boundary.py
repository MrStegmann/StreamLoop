"""Leakage boundary monitor to enforce test set quarantine and isolation."""

from __future__ import annotations


class LeakageIntegrityError(Exception):
    """Exception raised when the holdout test set access constraints are violated."""


class LeakageBoundaryMonitor:
    """Tracks and enforces holdout test set access boundaries to prevent data leakage."""

    def __init__(self, max_allowed_touches: int = 2) -> None:
        self.max_allowed_touches = max_allowed_touches
        self.touch_count = 0
        self.history: list[str] = []

    def record_touch(self, context: str) -> int:
        """Record an access to the holdout test set.

        Args:
            context: Description of why the test set was accessed (e.g., 'Baseline Evaluation', 'Final Model Evaluation').

        Returns:
            The updated touch count.

        Raises:
            LeakageIntegrityError: If touch count exceeds max_allowed_touches.
        """
        self.touch_count += 1
        self.history.append(f"Touch {self.touch_count}: {context}")

        if self.touch_count > self.max_allowed_touches:
            raise LeakageIntegrityError(
                f"Test set leakage detected! Accessed {self.touch_count} times "
                f"(maximum allowed: {self.max_allowed_touches}). History: {self.history}"
            )
        return self.touch_count

    def verify_final_count(self, expected_touches: int = 2) -> bool:
        """Verify that the test set was accessed exactly the expected number of times.

        Args:
            expected_touches: Expected number of accesses (default: 2).

        Returns:
            True if exact match.

        Raises:
            LeakageIntegrityError: If touch count does not match expected_touches.
        """
        if self.touch_count != expected_touches:
            raise LeakageIntegrityError(
                f"Leakage boundary verification failed: Test set accessed {self.touch_count} times, "
                f"expected exactly {expected_touches}."
            )
        return True
