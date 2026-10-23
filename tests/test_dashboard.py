"""Tests for dashboard progress parsing."""

import pytest

from dashboard import parse_progress


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("2026 INFO [3/10] Processing a.mp4 (abc...)", (3, 10)),
        ("2026 INFO copied 25/60 files", (25, 60)),
        ("[12/10] over", (10, 10)),
        ("Found 5 media file(s)", None),
        ("[0/0] nothing", None),
    ],
)
def test_parse_progress(line, expected):
    """Parse progress counts from log lines."""
    assert parse_progress(line) == expected
