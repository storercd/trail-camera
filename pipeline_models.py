"""Shared dataclasses for the trail camera processing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class VideoDecision:
    """Stores sorting and scoring information for one processed video."""

    relative_path: str
    output_relative_path: str
    bucket: str
    top_confidence: float | None
    top_category: str | None
    top_frame: int | None
    top_bbox: list[float] | None
    first_interesting_frame: int | None
    last_interesting_frame: int | None
    num_detections: int
    failure: str | None


@dataclass
class AppConfig:
    """Runtime settings loaded from the YAML config file."""

    input_dir: str
    output_dir: str
    model: str
    frame_sample: int
    interesting_threshold: float
    interesting_categories: list[str]
    move_files: bool
    save_uninteresting_files: bool
    clip_interesting_videos: bool
    run_folder_mode: str
    recursive: bool
    detector_verbose: bool
    generate_html_report: bool
    auto_open_html_report: bool
    generate_top_frame_previews: bool
    preview_output_dir: str
    preview_include_uninteresting: bool
    classify_previews_with_speciesnet: bool
    speciesnet_model: str
    speciesnet_geofence: bool
    speciesnet_label_in_filename: bool
    speciesnet_use_crops: bool
    species_crop_output_dir: str
    species_crop_padding: float


@dataclass
class RunPaths:
    """Resolved paths used during a processing run."""

    config_path: Path
    input_dir: Path
    output_root_dir: Path
    output_dir: Path
    run_id: str | None
    metadata_dir: Path
    md_results_path: Path
    summary_path: Path
    html_summary_path: Path
