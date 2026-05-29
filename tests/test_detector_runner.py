"""Tests for detector runner console-noise suppression."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import detector_runner


def test_run_detector_should_suppress_third_party_console_noise(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """Quiet detector mode should hide stdout, stderr, and warnings from the model runner."""
    results_file = tmp_path / "results.json"

    def fake_process_videos(_options: object) -> None:
        print("noisy stdout")
        print("noisy stderr", file=sys.stderr)
        warnings.warn("noisy warning", UserWarning, stacklevel=1)

    monkeypatch.setattr(detector_runner, "process_videos", fake_process_videos)

    detector_runner.run_detector(
        input_dir=tmp_path,
        results_file=results_file,
        model="MDV5A",
        frame_sample=1,
        recursive=False,
        verbose=False,
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""
