"""Tests for detector runner console-noise suppression."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

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

    class DummyProcessVideoOptions:
        pass

    monkeypatch.setattr(detector_runner, "ProcessVideoOptions", DummyProcessVideoOptions)
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


def test_run_detector_should_use_image_batch_for_staged_images(monkeypatch, tmp_path: Path) -> None:
    """Route still images through the detector batch image entrypoint."""
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"image")
    results_file = tmp_path / "results.json"
    calls: dict[str, object] = {}

    monkeypatch.setattr(
        detector_runner,
        "path_utils",
        SimpleNamespace(find_images=lambda _root, recursive=True: [str(image)]),
    )

    def fake_load_and_run_detector_batch(model_name: str, image_file_names: list[str]) -> dict[str, object]:
        calls["model"] = model_name
        calls["image_file_names"] = image_file_names
        return {"images": [{"file": "photo.jpg", "detections": []}]}

    def fake_write_results_to_file(results: object, output_file: str, **kwargs: object) -> None:
        calls["results"] = results
        calls["output_file"] = output_file
        calls["kwargs"] = kwargs

    monkeypatch.setattr(detector_runner, "load_and_run_detector_batch", fake_load_and_run_detector_batch)
    monkeypatch.setattr(detector_runner, "write_results_to_file", fake_write_results_to_file)

    detector_runner.run_detector(
        input_dir=tmp_path,
        results_file=results_file,
        model="MDV5A",
        frame_sample=1,
        recursive=False,
        verbose=False,
    )

    assert calls["model"] == "MDV5A"
    assert calls["image_file_names"] == [str(image)]
    assert calls["output_file"] == str(results_file)
    assert calls["kwargs"] == {"relative_path_base": str(tmp_path), "detector_file": "MDV5A"}


def test_run_detector_should_suppress_image_batch_console_noise(
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    """Quiet detector mode should hide stdout, stderr, and warnings from image batch runs."""
    image = tmp_path / "photo.jpg"
    image.write_bytes(b"image")
    results_file = tmp_path / "results.json"

    monkeypatch.setattr(
        detector_runner,
        "path_utils",
        SimpleNamespace(find_images=lambda _root, recursive=True: [str(image)]),
    )

    def fake_load_and_run_detector_batch(_model_name: str, _image_file_names: list[str]) -> dict[str, object]:
        print("noisy stdout")
        print("noisy stderr", file=sys.stderr)
        warnings.warn("noisy warning", UserWarning, stacklevel=1)
        return {"images": [{"file": "photo.jpg", "detections": []}]}

    def fake_write_results_to_file(_results: object, _output_file: str, **_kwargs: object) -> None:
        return None

    monkeypatch.setattr(detector_runner, "load_and_run_detector_batch", fake_load_and_run_detector_batch)
    monkeypatch.setattr(detector_runner, "write_results_to_file", fake_write_results_to_file)

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
