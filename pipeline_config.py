"""Configuration loading and path resolution for trail camera processing."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yaml

from pipeline_models import AppConfig, RunPaths

DEFAULT_INTERESTING_CATEGORIES = {"1", "2", "3"}
DEFAULT_CONFIG_PATH = "process_videos.config.yaml"


def load_config(config_path: Path) -> AppConfig:
    """Load and validate application settings from a YAML file.

    Args:
        config_path: Path to YAML configuration.

    Returns:
        AppConfig: Parsed and validated configuration values.

    Raises:
        SystemExit: If the config file is missing or has invalid values.
    """
    if not config_path.exists():
        raise SystemExit(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    raw_config = loaded if isinstance(loaded, dict) else {}

    try:
        input_dir = str(raw_config["input_dir"])
        output_dir = str(raw_config["output_dir"])
        model = str(raw_config.get("model", "MDV5A"))
        frame_sample = int(raw_config.get("frame_sample", 5))
        interesting_threshold = float(raw_config.get("interesting_threshold", 0.7))
        categories_raw = raw_config.get("interesting_categories", ["1", "2", "3"])
        if not isinstance(categories_raw, list):
            raise SystemExit("interesting_categories must be a YAML list")
        categories = [str(c).strip() for c in categories_raw if str(c).strip()]
        move_files = bool(raw_config.get("move_files", False))
        save_uninteresting_files = bool(raw_config.get("save_uninteresting_files", True))
        clip_interesting_videos = bool(raw_config.get("clip_interesting_videos", True))
        run_folder_mode = str(raw_config.get("run_folder_mode", "none")).strip().lower()
        recursive = bool(raw_config.get("recursive", False))
        detector_verbose = bool(raw_config.get("detector_verbose", False))
        generate_html_report = bool(raw_config.get("generate_html_report", True))
        auto_open_html_report = bool(raw_config.get("auto_open_html_report", False))
        generate_top_frame_previews = bool(raw_config.get("generate_top_frame_previews", True))
        preview_output_dir = str(raw_config.get("preview_output_dir", "preview_frames"))
        preview_include_uninteresting = bool(raw_config.get("preview_include_uninteresting", False))
        classify_previews_with_speciesnet = bool(raw_config.get("classify_previews_with_speciesnet", True))
        speciesnet_model = str(raw_config.get("speciesnet_model", ""))
        speciesnet_geofence = bool(raw_config.get("speciesnet_geofence", False))
        speciesnet_label_in_filename = bool(raw_config.get("speciesnet_label_in_filename", True))
        speciesnet_use_crops = bool(raw_config.get("speciesnet_use_crops", True))
        species_crop_output_dir = str(raw_config.get("species_crop_output_dir", "preview_species_crops"))
        species_crop_padding = float(raw_config.get("species_crop_padding", 0.15))
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid config file {config_path}: {exc}") from exc

    if frame_sample <= 0:
        raise SystemExit("frame_sample must be greater than 0")
    if not 0.0 <= interesting_threshold <= 1.0:
        raise SystemExit("interesting_threshold must be between 0.0 and 1.0")
    if run_folder_mode not in {"none", "timestamped"}:
        raise SystemExit("run_folder_mode must be either 'none' or 'timestamped'")
    if not 0.0 <= species_crop_padding <= 1.0:
        raise SystemExit("species_crop_padding must be between 0.0 and 1.0")

    return AppConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        frame_sample=frame_sample,
        interesting_threshold=interesting_threshold,
        interesting_categories=categories,
        move_files=move_files,
        save_uninteresting_files=save_uninteresting_files,
        clip_interesting_videos=clip_interesting_videos,
        run_folder_mode=run_folder_mode,
        recursive=recursive,
        detector_verbose=detector_verbose,
        generate_html_report=generate_html_report,
        auto_open_html_report=auto_open_html_report,
        generate_top_frame_previews=generate_top_frame_previews,
        preview_output_dir=preview_output_dir,
        preview_include_uninteresting=preview_include_uninteresting,
        classify_previews_with_speciesnet=classify_previews_with_speciesnet,
        speciesnet_model=speciesnet_model,
        speciesnet_geofence=speciesnet_geofence,
        speciesnet_label_in_filename=speciesnet_label_in_filename,
        speciesnet_use_crops=speciesnet_use_crops,
        species_crop_output_dir=species_crop_output_dir,
        species_crop_padding=species_crop_padding,
    )


def build_run_paths(config_path: Path, config: AppConfig) -> RunPaths:
    """Build resolved filesystem paths used throughout one run.

    Args:
        config_path: Path to the configuration file.
        config: Loaded runtime configuration values.

    Returns:
        RunPaths: All resolved paths needed for processing.
    """
    output_root_dir = Path(config.output_dir).resolve()
    run_id: str | None = None
    if config.run_folder_mode == "timestamped":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root_dir / "runs" / run_id
    else:
        output_dir = output_root_dir

    metadata_dir = output_dir / "metadata"
    return RunPaths(
        config_path=config_path.resolve(),
        input_dir=Path(config.input_dir).resolve(),
        output_root_dir=output_root_dir,
        output_dir=output_dir,
        run_id=run_id,
        metadata_dir=metadata_dir,
        md_results_path=metadata_dir / "megadetector_results.json",
        summary_path=metadata_dir / "summary.json",
        html_summary_path=metadata_dir / "summary.html",
    )


def resolve_preview_output_dir(preview_output_dir: str, run_output_dir: Path) -> Path:
    """Resolve configured preview output location for the current run.

    Args:
        preview_output_dir: Configured preview output directory.
        run_output_dir: Root output directory for the current run.

    Returns:
        Path: Resolved preview output path.
    """
    configured_path = Path(preview_output_dir)
    if configured_path.is_absolute():
        return configured_path.resolve()
    return (run_output_dir / configured_path).resolve()


def resolve_interesting_categories(config: AppConfig) -> set[str]:
    """Normalize interesting category IDs from config.

    Args:
        config: Loaded runtime configuration.

    Returns:
        set[str]: Category IDs used to classify videos as interesting.
    """
    categories = {
        category.strip() for category in config.interesting_categories if category.strip()
    }
    return categories or DEFAULT_INTERESTING_CATEGORIES
