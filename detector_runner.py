"""MegaDetector execution and result-loading helpers."""

from __future__ import annotations

import json
import logging
import warnings
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Any

try:
    from megadetector.detection.process_video import ProcessVideoOptions as _ProcessVideoOptions
    from megadetector.detection.process_video import process_videos as _process_videos
except ImportError:  # pragma: no cover - depends on runtime environment
    _ProcessVideoOptions = None
    _process_videos = None

try:
    from megadetector.detection.run_detector_batch import (
        load_and_run_detector_batch as _load_and_run_detector_batch,
    )
    from megadetector.detection.run_detector_batch import (
        write_results_to_file as _write_results_to_file,
    )
    from megadetector.utils import path_utils as _path_utils
except ImportError:  # pragma: no cover - depends on runtime environment
    _load_and_run_detector_batch = None
    _write_results_to_file = None
    _path_utils = None

ProcessVideoOptions = _ProcessVideoOptions
process_videos = _process_videos
load_and_run_detector_batch = _load_and_run_detector_batch
write_results_to_file = _write_results_to_file
path_utils = _path_utils

logger = logging.getLogger(__name__)


def _run_quietly(callback: Any, *args: Any, **kwargs: Any) -> Any:
    """Execute a callback while swallowing third-party console noise."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            return callback(*args, **kwargs)


def _run_process_videos_quietly(options: ProcessVideoOptions) -> None:
    """Run MegaDetector while suppressing third-party console noise.

    Raises:
        SystemExit: If MegaDetector video-processing dependencies are unavailable.
    """
    if process_videos is None:
        raise SystemExit("Missing MegaDetector video-processing dependency")
    _run_quietly(process_videos, options)


def _has_image_inputs(input_dir: Path) -> bool:
    """Return whether the staged input directory contains any image files."""
    if path_utils is not None:
        return bool(path_utils.find_images(str(input_dir), recursive=True))

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
    return any(path.is_file() and path.suffix.lower() in image_suffixes for path in input_dir.rglob("*"))


def _run_detector_on_images(input_dir: Path, results_file: Path, model: str, verbose: bool) -> None:
    """Run MegaDetector on a staged image directory.

    Raises:
        SystemExit: If MegaDetector image-processing dependencies are unavailable
            or no images are discovered in the staged directory.
    """
    if load_and_run_detector_batch is None or write_results_to_file is None or path_utils is None:
        raise SystemExit("Missing MegaDetector image-processing dependency")

    image_file_names = path_utils.find_images(str(input_dir), recursive=True)
    if not image_file_names:
        raise SystemExit(f"No images found in {input_dir}")

    logger.debug("Starting MegaDetector image run: input=%s model=%s", input_dir, model)
    if verbose:
        results = load_and_run_detector_batch(model, image_file_names)
        write_results_to_file(
            results,
            str(results_file),
            relative_path_base=str(input_dir),
            detector_file=model,
        )
    else:
        results = _run_quietly(load_and_run_detector_batch, model, image_file_names)
        _run_quietly(
            write_results_to_file,
            results,
            str(results_file),
            relative_path_base=str(input_dir),
            detector_file=model,
        )
    if verbose:
        logger.debug("MegaDetector image run complete. Results written to %s", results_file)


def load_results(results_path: Path) -> dict[str, Any]:
    """Load MegaDetector JSON results from disk.

    Args:
        results_path: Path to the MegaDetector JSON output file.

    Returns:
        dict[str, Any]: Parsed JSON payload.
    """
    with results_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def run_detector(
    input_dir: Path,
    results_file: Path,
    model: str,
    frame_sample: int,
    recursive: bool,
    verbose: bool,
) -> None:
    """Run MegaDetector on the input directory and write JSON results.

    Args:
        input_dir: Directory containing media to process.
        results_file: Destination for MegaDetector JSON output.
        model: MegaDetector model name or file path.
        frame_sample: Process every Nth frame.
        recursive: Whether to scan input recursively.
        verbose: Whether to enable MegaDetector verbose output.

    Raises:
        SystemExit: If required MegaDetector runtime dependencies are unavailable.
    """
    effective_recursive = True
    if _has_image_inputs(input_dir):
        _run_detector_on_images(input_dir, results_file, model, verbose)
        return

    if process_videos is None or ProcessVideoOptions is None:
        raise SystemExit("Missing MegaDetector video-processing dependency")

    logger.debug(
        "Starting MegaDetector run: input=%s model=%s frame_sample=%s recursive=%s",
        input_dir,
        model,
        frame_sample,
        effective_recursive,
    )
    options = ProcessVideoOptions()
    options.input_video_file = str(input_dir)
    options.output_json_file = str(results_file)
    options.model_file = model
    options.frame_sample = frame_sample
    # Always recurse through nested folders for directory inputs.
    options.recursive = effective_recursive
    options.verbose = verbose
    if verbose:
        process_videos(options)
    else:
        _run_process_videos_quietly(options)
    logger.debug("MegaDetector run complete. Results written to %s", results_file)
