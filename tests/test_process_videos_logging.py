"""Tests for process-level logging configuration."""

import logging

from process_videos import configure_logging


def test_configure_logging_should_include_timestamp_in_log_format() -> None:
    """Configure emitted log records with a timestamp."""
    configure_logging("INFO")

    handler = logging.getLogger().handlers[0]

    assert handler.formatter is not None
    assert "%(asctime)s" in handler.formatter._style._fmt
