"""Configuration loading and path resolution for trail camera processing."""

from __future__ import annotations

from pathlib import Path

import yaml

from pipeline_models import DEFAULT_PIPELINE_VERSION, AppConfig, RunPaths

DEFAULT_INTERESTING_CATEGORIES = {"1", "2", "3"}
DEFAULT_CONFIG_PATH = "process_videos.config.yaml"


def _parse_string_list(raw_config: dict[str, object], key: str, default: list[str], error_message: str) -> list[str]:
    """Parse a YAML list field and normalize to non-empty strings.

    Returns:
        list[str]: Normalized list entries.

    Raises:
        SystemExit: If the configured value is not a list.
    """
    values_raw = raw_config.get(key, default)
    if not isinstance(values_raw, list):
        raise SystemExit(error_message)
    return [str(value).strip() for value in values_raw if str(value).strip()]


def _parse_camera_date_profile(raw_config: dict[str, object]) -> dict[str, object] | None:
    """Parse optional camera date profile mapping.

    Returns:
        dict[str, object] | None: Camera date profile mapping when configured.

    Raises:
        SystemExit: If a non-mapping value is provided.
    """
    raw_camera_date_profile = raw_config.get("camera_date_profile")
    if raw_camera_date_profile is not None and not isinstance(raw_camera_date_profile, dict):
        raise SystemExit("camera_date_profile must be a YAML mapping when provided")
    return raw_camera_date_profile


def _build_app_config(raw_config: dict[str, object], config_path: Path) -> AppConfig:
    """Build AppConfig from raw YAML values.

    Returns:
        AppConfig: Parsed configuration values.

    Raises:
        SystemExit: If required values are missing or invalid.
    """
    try:
        return AppConfig(
            input_dir=str(raw_config["input_dir"]),
            output_dir=str(raw_config["output_dir"]),
            metadata_db_path=str(raw_config.get("metadata_db_path", "metadata/catalog.sqlite3")),
            pipeline_version=str(raw_config.get("pipeline_version", DEFAULT_PIPELINE_VERSION)).strip(),
            model=str(raw_config.get("model", "MDV5A")),
            frame_sample=int(raw_config.get("frame_sample", 5)),
            interesting_threshold=float(raw_config.get("interesting_threshold", 0.7)),
            interesting_categories=_parse_string_list(
                raw_config,
                key="interesting_categories",
                default=["1", "2", "3"],
                error_message="interesting_categories must be a YAML list",
            ),
            move_files=bool(raw_config.get("move_files", False)),
            save_uninteresting_files=bool(raw_config.get("save_uninteresting_files", True)),
            clip_interesting_videos=bool(raw_config.get("clip_interesting_videos", True)),
            recursive=bool(raw_config.get("recursive", False)),
            detector_verbose=bool(raw_config.get("detector_verbose", False)),
            generate_html_report=bool(raw_config.get("generate_html_report", True)),
            auto_open_html_report=bool(raw_config.get("auto_open_html_report", False)),
            write_json_exports=bool(raw_config.get("write_json_exports", True)),
            preview_output_dir=str(raw_config.get("preview_output_dir", "preview_frames")),
            preview_include_uninteresting=bool(raw_config.get("preview_include_uninteresting", False)),
            speciesnet_model=str(raw_config.get("speciesnet_model", "")),
            speciesnet_geofence=bool(raw_config.get("speciesnet_geofence", False)),
            speciesnet_label_in_filename=bool(raw_config.get("speciesnet_label_in_filename", True)),
            species_crop_output_dir=str(raw_config.get("species_crop_output_dir", "preview_species_crops")),
            species_crop_padding=float(raw_config.get("species_crop_padding", 0.15)),
            capture_date_source=str(raw_config.get("capture_date_source", "filesystem")).strip().lower(),
            camera_date_profile=_parse_camera_date_profile(raw_config),
            excluded_megadetector_categories=_parse_string_list(
                raw_config,
                key="excluded_megadetector_categories",
                default=["3"],
                error_message="excluded_megadetector_categories must be a YAML list",
            ),
            uninteresting_species_labels=_parse_string_list(
                raw_config,
                key="uninteresting_species_labels",
                default=["domestic dog"],
                error_message="uninteresting_species_labels must be a YAML list",
            ),
            generic_species_labels_to_skip=_parse_string_list(
                raw_config,
                key="generic_species_labels_to_skip",
                default=[],
                error_message="generic_species_labels_to_skip must be a YAML list",
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid config file {config_path}: {exc}") from exc


def _validate_config(config: AppConfig) -> None:
    """Validate logical constraints for loaded configuration.

    Raises:
        SystemExit: If any config field has an invalid value.
    """
    if config.frame_sample <= 0:
        raise SystemExit("frame_sample must be greater than 0")
    if not 0.0 <= config.interesting_threshold <= 1.0:
        raise SystemExit("interesting_threshold must be between 0.0 and 1.0")
    if not 0.0 <= config.species_crop_padding <= 1.0:
        raise SystemExit("species_crop_padding must be between 0.0 and 1.0")
    if not config.pipeline_version:
        raise SystemExit("pipeline_version must be a non-empty string")
    if config.capture_date_source not in {"filesystem", "camera_overlay"}:
        raise SystemExit("capture_date_source must be either 'filesystem' or 'camera_overlay'")


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
    config = _build_app_config(raw_config, config_path)
    _validate_config(config)
    return config


def build_run_paths(config_path: Path, config: AppConfig) -> RunPaths:
    """Build resolved filesystem paths used throughout one run.

    Args:
        config_path: Path to the configuration file.
        config: Loaded runtime configuration values.

    Returns:
        RunPaths: All resolved paths needed for processing.
    """
    output_root_dir = Path(config.output_dir).resolve()
    configured_db_path = Path(config.metadata_db_path)
    metadata_db_path = (
        configured_db_path.resolve()
        if configured_db_path.is_absolute()
        else (output_root_dir / configured_db_path).resolve()
    )
    output_dir = output_root_dir

    canonical_videos_dir = output_root_dir / "videos"
    metadata_dir = output_dir / "metadata"
    return RunPaths(
        config_path=config_path.resolve(),
        input_dir=Path(config.input_dir).resolve(),
        output_root_dir=output_root_dir,
        output_dir=output_dir,
        canonical_videos_dir=canonical_videos_dir,
        metadata_dir=metadata_dir,
        metadata_db_path=metadata_db_path,
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
