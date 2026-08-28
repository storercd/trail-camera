"""Tests for process-level logging configuration."""

import logging

from process_videos import configure_logging, estimate_remaining_time


def test_configure_logging_should_include_timestamp_in_log_format() -> None:
    """Configure emitted log records with a timestamp."""
    configure_logging("INFO")

    handler = logging.getLogger().handlers[0]

    assert handler.formatter is not None
    assert "%(asctime)s" in handler.formatter._style._fmt


def test_estimate_remaining_time_should_skip_first_file() -> None:
    """Avoid estimating time before a file has completed."""
    assert estimate_remaining_time(elapsed_seconds=0.0, completed_files=0, remaining_files=74) is None


def test_estimate_remaining_time_should_use_average_completed_file_duration() -> None:
    """Estimate remaining duration from the average completed-file duration."""
    estimate = estimate_remaining_time(elapsed_seconds=150.0, completed_files=2, remaining_files=72)

    assert estimate == "1h 30m remaining"
