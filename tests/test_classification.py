"""Unit tests for classification helpers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classification import build_clipped_log_line


class BuildClippedLogLineTests(unittest.TestCase):
    """Tests for clipped video status-line formatting."""

    def test_should_include_kept_frames_and_percent(self) -> None:
        """Include frame range and kept percentage when source frame count is known."""
        line = build_clipped_log_line(
            index=1,
            total_entries=665,
            output_relative_path="20260203-PICT0005.AVI",
            start_frame=0,
            end_frame=240,
            frames_written=241,
            total_source_frames=1200,
            bucket="interesting",
        )

        self.assertEqual(
            line,
            "[1/665] Clipped 20260203-PICT0005.AVI "
            "frames 0-240 (241/1200 frames kept (20.1%)) -> interesting",
        )

    def test_should_fallback_when_source_frame_count_missing(self) -> None:
        """Omit percentage when source frame count is unavailable."""
        line = build_clipped_log_line(
            index=2,
            total_entries=10,
            output_relative_path="20260101-PICT0001.AVI",
            start_frame=10,
            end_frame=40,
            frames_written=31,
            total_source_frames=0,
            bucket="interesting",
        )

        self.assertEqual(
            line,
            "[2/10] Clipped 20260101-PICT0001.AVI frames 10-40 (31 frames kept) -> interesting",
        )


if __name__ == "__main__":
    unittest.main()
