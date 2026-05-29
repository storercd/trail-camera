"""MegaDetector execution and result-loading helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from megadetector.detection.process_video import ProcessVideoOptions, process_videos

logger = logging.getLogger(__name__)


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
        input_dir: Directory containing videos to process.
        results_file: Destination for MegaDetector JSON output.
        model: MegaDetector model name or file path.
        frame_sample: Process every Nth frame.
        recursive: Whether to scan input recursively.
        verbose: Whether to enable MegaDetector verbose output.
    """
    effective_recursive = True
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
    process_videos(options)
    logger.debug("MegaDetector run complete. Results written to %s", results_file)
