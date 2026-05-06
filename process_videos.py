"""Main entrypoint for trail-camera video processing and report generation."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from classification import classify_and_sort_videos, compute_bucket_counts
from detector_runner import load_results, run_detector
from file_ops import validate_and_find_videos
from metadata_store import (
    SpeciesClassificationRecord,
    delete_species_classification_record,
    initialize_metadata_store,
    upsert_species_classification_record,
)
from pipeline_config import (
    DEFAULT_CONFIG_PATH,
    build_run_paths,
    load_config,
    resolve_interesting_categories,
    resolve_preview_output_dir,
)
from pipeline_models import VideoDecision
from preview_frames import PreviewExtractionStats, TopFrameRecord, extract_top_frames
from processing_modes import (
    ProcessingSource,
    build_bucket_output_paths,
    build_preview_output_paths,
    build_report_output_paths,
    build_species_crop_output_dirs,
    build_video_storage_dir,
    collect_species_artifact_maps,
    ingest_videos_into_catalog,
    load_reprocess_sources,
    record_processing_results,
    stage_single_processing_input,
)
from reporting import (
    generate_report_videos,
    open_file_in_default_app,
    write_html_summary_from_catalog,
    write_sqlite_snapshot_export,
)


@dataclass
class RunAccumulator:
    """Mutable containers for one end-to-end processing run."""

    all_decisions: list[VideoDecision]
    all_staged_video_ids: dict[str, str]
    all_report_video_paths: dict[str, Path]
    all_preview_paths: dict[str, Path]
    all_crop_paths: dict[str, Path]
    temp_species_report_paths: list[Path]
    processing_sources_by_decision: dict[int, ProcessingSource]
    aggregated_preview_stats: PreviewExtractionStats


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the video processing workflow.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run MegaDetector on videos and sort originals using values "
            "loaded from a YAML config file."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to YAML configuration file "
            f"(default: {DEFAULT_CONFIG_PATH})"
        ),
    )
    parser.add_argument(
        "--mode",
        default="new-only",
        choices=("new-only", "reprocess-existing", "force-reprocess", "report-only"),
        help=(
            "Processing mode: new-only (new videos), reprocess-existing "
            "(outdated or failed), force-reprocess (all), or report-only "
            "(generate reports only)"
            "(default: new-only)"
        ),
    )
    return parser.parse_args()


def _load_species_entries_by_video_id(
    species_classification_report_path: Path,
    video_ids_by_output_path: dict[str, str],
) -> dict[str, dict[str, Any]]:
    """Load species report entries keyed by canonical video ID.

    Returns:
        dict[str, dict[str, Any]]: Species entries keyed by canonical video ID.
    """
    try:
        with species_classification_report_path.open("r", encoding="utf-8") as handle:
            report_payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}

    entries_by_video_id: dict[str, dict[str, Any]] = {}
    for entry in report_payload.get("entries", []):
        if not isinstance(entry, dict):
            continue
        output_relative_path = entry.get("source_relative_path")
        if not isinstance(output_relative_path, str) or not output_relative_path:
            continue
        video_id = video_ids_by_output_path.get(output_relative_path)
        if video_id is not None:
            entries_by_video_id[video_id] = entry
    return entries_by_video_id


def _extract_species_top_fields(entry: dict[str, Any]) -> tuple[str | None, float | None, str | None]:
    """Extract normalized top species fields from one species entry.

    Returns:
        tuple[str | None, float | None, str | None]: (top_label, top_score, top_raw_class).
    """
    top = entry.get("top_classification") or {}
    top_label = top.get("label") if isinstance(top, dict) else None
    top_raw_class = top.get("raw_class") if isinstance(top, dict) else None
    try:
        top_score = (
            float(top.get("score"))
            if isinstance(top, dict) and top.get("score") is not None
            else None
        )
    except (TypeError, ValueError):
        top_score = None
    return (
        str(top_label) if top_label is not None else None,
        top_score,
        str(top_raw_class) if top_raw_class is not None else None,
    )


def sync_species_classifications_to_catalog(
    species_classification_report_path: Path,
    metadata_db_path: Path,
    video_ids_by_output_path: dict[str, str],
) -> None:
    """Persist species classifications from merged JSON report into SQLite catalog."""
    if not species_classification_report_path.exists():
        for video_id in video_ids_by_output_path.values():
            delete_species_classification_record(metadata_db_path, video_id)
        return

    entries_by_video_id = _load_species_entries_by_video_id(
        species_classification_report_path,
        video_ids_by_output_path,
    )

    for video_id in video_ids_by_output_path.values():
        entry = entries_by_video_id.get(video_id)
        if entry is None:
            delete_species_classification_record(metadata_db_path, video_id)
            continue

        top_label, top_score, top_raw_class = _extract_species_top_fields(entry)
        raw_candidates = entry.get("candidates")
        candidates = raw_candidates if isinstance(raw_candidates, list) else []
        upsert_species_classification_record(
            db_path=metadata_db_path,
            record=SpeciesClassificationRecord(
                video_id=video_id,
                top_label=top_label,
                top_score=top_score,
                top_raw_class=top_raw_class,
                candidates_json=json.dumps(candidates),
            ),
        )


def extract_preview_frames_for_decisions(
    decisions: list[VideoDecision],
    input_dir: Path,
    preview_output_dir: Path,
    include_uninteresting: bool,
    classify_with_speciesnet: bool,
    speciesnet_model: str,
    speciesnet_geofence: bool,
    speciesnet_label_in_filename: bool,
    speciesnet_use_crops: bool,
    species_crop_output_dir: Path,
    species_crop_padding: float,
    species_classification_report_path: Path,
    preview_output_paths: dict[str, Path] | None = None,
    species_crop_output_dirs: dict[str, Path] | None = None,
) -> PreviewExtractionStats:
    """Extract preview images for selected decisions.

    Args:
        decisions: Per-video decisions from classification.
        input_dir: Root folder containing source videos.
        preview_output_dir: Destination folder for preview images.
        include_uninteresting: Include uninteresting videos when true.
        classify_with_speciesnet: Run species classification on preview images.
        speciesnet_model: SpeciesNet model identifier, empty for default.
        speciesnet_geofence: Use geofence filtering in SpeciesNet.
        speciesnet_label_in_filename: Append species label and score to filenames.
        speciesnet_use_crops: Use MegaDetector bboxes to classify cropped images.
        species_crop_output_dir: Destination folder for saved classification crops.
        species_crop_padding: Extra context around bbox as normalized padding.
        species_classification_report_path: Output JSON path for full species candidates and scores.
        preview_output_paths: Optional per-video preview output paths keyed by staged relative path.
        species_crop_output_dirs: Optional per-video crop output dirs keyed by staged relative path.

    Returns:
        PreviewExtractionStats: Frame extraction summary counters.
    """
    records = [
        TopFrameRecord(
            relative_path=decision.relative_path,
            output_relative_path=decision.output_relative_path,
            top_frame=decision.top_frame,
            top_confidence=decision.top_confidence,
            bucket=decision.bucket,
            top_bbox=decision.top_bbox,
        )
        for decision in decisions
    ]
    return extract_top_frames(
        records=records,
        input_dir=input_dir,
        output_dir=preview_output_dir,
        include_uninteresting=include_uninteresting,
        classify_with_speciesnet=classify_with_speciesnet,
        speciesnet_model=speciesnet_model or None,
        speciesnet_geofence=speciesnet_geofence,
        include_label_in_filename=speciesnet_label_in_filename,
        speciesnet_use_crops=speciesnet_use_crops,
        species_crop_output_dir=species_crop_output_dir,
        species_crop_padding=species_crop_padding,
        species_classification_report_path=species_classification_report_path,
        preview_output_paths=preview_output_paths,
        species_crop_output_dirs=species_crop_output_dirs,
    )


def process_single_video(
    source: ProcessingSource,
    staging_dir: Path,
    md_results_path: Path,
    config: Any,
    paths: Any,
    categories: list[str],
    effective_move_files: bool,
    preview_output_dir: Path,
    species_crop_output_dir: Path,
    species_classification_report_path: Path,
    video_index: int,
) -> tuple[
    list[VideoDecision],
    dict[str, str],
    dict[str, Path],
    PreviewExtractionStats,
    dict[str, Path],
    dict[str, Path],
    Path,
]:
    """Process a single video through all stages.

    Args:
        source: ProcessingSource with source path and video_id.
        staging_dir: Temporary directory for staging.
        md_results_path: Path to MegaDetector results JSON.
        config: Pipeline configuration.
        paths: Run paths object.
        categories: Interesting categories.
        effective_move_files: Whether to move files.
        preview_output_dir: Preview output directory.
        species_crop_output_dir: Species crop output directory.
        species_classification_report_path: Final species classification report path.
        video_index: Index of this video (for unique temp file naming).

    Returns:
        tuple: (
            decisions,
            staged_video_ids,
            report_video_paths,
            preview_stats,
            preview_paths,
            crop_paths,
            temp_species_report,
        ).
    """
    # Stage this single video
    processing_input_dir, staged_video_ids = stage_single_processing_input(
        source=source,
        staging_dir=staging_dir,
    )

    # Use a temporary species classification report for this video to avoid overwrites
    temp_species_report = staging_dir.parent / f"species_classifications_{video_index}.json"

    # Run MegaDetector on this single video
    run_detector(
        input_dir=processing_input_dir,
        results_file=md_results_path,
        model=config.model,
        frame_sample=config.frame_sample,
        recursive=config.recursive,
        verbose=config.detector_verbose,
    )

    # Load this video's results
    results = load_results(md_results_path)
    image_entries: list[dict[str, Any]] = results.get("images", [])

    # Classify this video
    decisions = classify_and_sort_videos(
        image_entries=image_entries,
        input_dir=processing_input_dir,
        output_dir=paths.output_dir,
        interesting_categories=categories,
        threshold=config.interesting_threshold,
        move_files=effective_move_files,
        save_uninteresting_files=config.save_uninteresting_files,
        clip_interesting_videos=config.clip_interesting_videos,
        clip_buffer_frames=config.frame_sample,
        bucket_output_paths=build_bucket_output_paths(
            staged_video_ids=staged_video_ids,
            canonical_videos_dir=paths.canonical_videos_dir,
        ),
    )

    # Generate report videos for this video
    report_video_paths = generate_report_videos(
        decisions=decisions,
        run_output_dir=paths.output_dir,
        source_video_paths=build_bucket_output_paths(
            staged_video_ids=staged_video_ids,
            canonical_videos_dir=paths.canonical_videos_dir,
        ),
        output_video_paths=build_report_output_paths(
            decisions=decisions,
            staged_video_ids=staged_video_ids,
            canonical_videos_dir=paths.canonical_videos_dir,
        ),
    )

    # Extract preview frames using temporary species classification file
    preview_stats = extract_preview_frames_for_decisions(
        decisions=decisions,
        input_dir=processing_input_dir,
        preview_output_dir=preview_output_dir,
        include_uninteresting=config.preview_include_uninteresting,
        classify_with_speciesnet=True,
        speciesnet_model=config.speciesnet_model,
        speciesnet_geofence=config.speciesnet_geofence,
        speciesnet_label_in_filename=config.speciesnet_label_in_filename,
        speciesnet_use_crops=True,
        species_crop_output_dir=species_crop_output_dir,
        species_crop_padding=config.species_crop_padding,
        species_classification_report_path=temp_species_report,
        preview_output_paths=build_preview_output_paths(
            decisions=decisions,
            staged_video_ids=staged_video_ids,
            canonical_videos_dir=paths.canonical_videos_dir,
        ),
        species_crop_output_dirs=build_species_crop_output_dirs(
            decisions=decisions,
            staged_video_ids=staged_video_ids,
            canonical_videos_dir=paths.canonical_videos_dir,
        ),
    )

    # Collect species artifacts for this video
    preview_paths, crop_paths = collect_species_artifact_maps(
        temp_species_report,
    )

    # Clear staging directory for next video
    for item in processing_input_dir.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_symlink():
            item.unlink()

    return (
        decisions,
        staged_video_ids,
        report_video_paths,
        preview_stats,
        preview_paths,
        crop_paths,
        temp_species_report,
    )


def merge_species_classification_reports(
    temp_report_paths: list[Path],
    final_report_path: Path,
) -> None:
    """Merge temporary species classification reports into a single final report.

    Args:
        temp_report_paths: List of temporary report file paths.
        final_report_path: Destination for merged report.
    """
    all_entries: list[dict[str, Any]] = []
    for temp_path in temp_report_paths:
        if not temp_path.exists():
            continue
        try:
            with temp_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            all_entries.extend(data.get("entries", []))
        except (json.JSONDecodeError, OSError):
            pass

    final_report_path.parent.mkdir(parents=True, exist_ok=True)
    with final_report_path.open("w", encoding="utf-8") as handle:
        json.dump({"entries": all_entries}, handle, indent=2)


def _print_runtime_header(paths: Any, config: Any, mode: str) -> None:
    """Print run configuration for traceability."""
    print(f"Using config file: {paths.config_path}")
    print(f"Input directory: {paths.input_dir}")
    print(f"Output root directory: {paths.output_root_dir}")
    print(f"Run output directory: {paths.output_dir}")
    print(f"Pipeline version: {config.pipeline_version}")
    print(f"Metadata database: {paths.metadata_db_path}")
    print(f"Processing mode: {mode}")
    if mode != "report-only":
        print("Processing model: per-video streaming")


def _write_optional_reports(paths: Any, config: Any) -> None:
    """Write optional summary exports and HTML reports based on config."""
    if config.write_json_exports:
        write_sqlite_snapshot_export(
            summary_path=paths.summary_path,
            config=config,
            config_path=paths.config_path,
            run_output_dir=paths.output_dir,
            metadata_db_path=paths.metadata_db_path,
        )
        print(f"Wrote summary metadata to {paths.summary_path}")
    else:
        print("JSON exports disabled by config")

    if not config.generate_html_report:
        print("HTML summary disabled by config")
        return

    write_html_summary_from_catalog(
        html_summary_path=paths.html_summary_path,
        metadata_db_path=paths.metadata_db_path,
    )
    print(f"Wrote HTML summary to {paths.html_summary_path}")
    if config.auto_open_html_report:
        if open_file_in_default_app(paths.html_summary_path):
            print(f"Opened HTML summary in default app: {paths.html_summary_path}")
        else:
            print(f"Failed to open HTML summary automatically: {paths.html_summary_path}")


def _load_processing_sources_for_mode(mode: str, paths: Any, config: Any) -> list[ProcessingSource]:
    """Resolve source list based on processing mode.

    Returns:
        list[ProcessingSource]: Sources selected for this processing mode.
    """
    if mode == "new-only":
        discovered_videos = validate_and_find_videos(paths.input_dir, config.recursive)
        ingested_count, newly_persisted_count, newly_discovered_sources = ingest_videos_into_catalog(
            videos=discovered_videos,
            canonical_videos_dir=paths.canonical_videos_dir,
            metadata_db_path=paths.metadata_db_path,
            camera_date_profile=(
                config.camera_date_profile
                if config.capture_date_source == "camera_overlay"
                else None
            ),
        )
        print(
            "Catalog ingestion complete. "
            f"videos={ingested_count}, "
            f"new_canonical_originals={newly_persisted_count}, "
            f"new_videos={len(newly_discovered_sources)}"
        )
        return newly_discovered_sources
    if mode == "reprocess-existing":
        sources = load_reprocess_sources(
            metadata_db_path=paths.metadata_db_path,
            current_pipeline_version=config.pipeline_version,
            filter_mode="existing",
        )
        print(
            f"Loaded {len(sources)} outdated or failed videos for reprocessing "
            f"(current pipeline version: {config.pipeline_version})"
        )
        return sources
    if mode == "force-reprocess":
        sources = load_reprocess_sources(
            metadata_db_path=paths.metadata_db_path,
            filter_mode="force",
        )
        print(f"Loaded {len(sources)} canonical originals for force reprocessing")
        return sources
    return []


def _resolve_effective_move_files(mode: str, move_files: bool) -> bool:
    """Disable move_files for reprocessing modes.

    Returns:
        bool: Effective move_files setting for the current mode.
    """
    if mode in ("reprocess-existing", "force-reprocess") and move_files:
        print(f"Disabling move_files for {mode} mode")
        return False
    return move_files


def _new_run_accumulator() -> RunAccumulator:
    """Build a fresh mutable result accumulator.

    Returns:
        RunAccumulator: Initialized accumulator for one run.
    """
    return RunAccumulator(
        all_decisions=[],
        all_staged_video_ids={},
        all_report_video_paths={},
        all_preview_paths={},
        all_crop_paths={},
        temp_species_report_paths=[],
        processing_sources_by_decision={},
        aggregated_preview_stats=PreviewExtractionStats(
            total_candidates=0,
            extracted=0,
            skipped=0,
            failed=0,
            classified=0,
            classification_failed=0,
        ),
    )


def _apply_collision_remaps(
    all_decisions: list[VideoDecision],
    decisions: list[VideoDecision],
    staged_video_ids: dict[str, str],
) -> dict[str, str]:
    """Append deterministic suffixes when output-relative path collisions are detected.

    Returns:
        dict[str, str]: Mapping from original output path to collision-safe output path.
    """
    seen_output_paths = {d.output_relative_path for d in all_decisions}
    collision_remaps: dict[str, str] = {}
    for decision in decisions:
        old_path = decision.output_relative_path
        if old_path not in seen_output_paths:
            continue
        video_id = staged_video_ids.get(decision.relative_path, "unknown")
        new_path = f"{old_path}__{video_id[:8]}"
        decision.output_relative_path = new_path
        collision_remaps[old_path] = new_path
    return collision_remaps


def _rewrite_species_source_paths(temp_species_report: Path, collision_remaps: dict[str, str]) -> None:
    """Rewrite source_relative_path values in per-video species reports after collisions."""
    if not collision_remaps or not temp_species_report.exists():
        return
    try:
        with temp_species_report.open("r", encoding="utf-8") as handle:
            species_data = json.load(handle)
        for entry in species_data.get("entries", []):
            old_path = entry.get("source_relative_path")
            if old_path in collision_remaps:
                entry["source_relative_path"] = collision_remaps[old_path]
        with temp_species_report.open("w", encoding="utf-8") as handle:
            json.dump(species_data, handle, indent=2)
    except (json.JSONDecodeError, OSError):
        return


def _merge_path_map_with_remaps(
    destination: dict[str, Path],
    source_map: dict[str, Path],
    collision_remaps: dict[str, str],
) -> None:
    """Merge path map while remapping colliding keys."""
    for old_key, path_value in source_map.items():
        new_key = collision_remaps.get(old_key, old_key)
        destination[new_key] = path_value


def _accumulate_preview_stats(target: PreviewExtractionStats, source: PreviewExtractionStats) -> None:
    """Accumulate preview extraction counters."""
    target.total_candidates += source.total_candidates
    target.extracted += source.extracted
    target.skipped += source.skipped
    target.failed += source.failed
    target.classified += source.classified
    target.classification_failed += source.classification_failed


def _print_video_result(source: ProcessingSource, decisions: list[VideoDecision]) -> None:
    """Print one-line completion status for a processed source."""
    if not decisions:
        print(f"  [✓] {source.source.name} ({source.video_id[:8]}...) → no animals detected")
        return
    decision = decisions[0]
    confidence = decision.top_confidence
    confidence_suffix = f" (confidence: {confidence:.3f})" if confidence is not None else ""
    print(
        f"  [✓] {source.source.name} ({source.video_id[:8]}...) → "
        f"{decision.bucket}{confidence_suffix}"
    )


def _process_sources_sequential(
    processing_sources: list[ProcessingSource],
    paths: Any,
    config: Any,
    categories: list[str],
    effective_move_files: bool,
    preview_output_dir: Path,
    species_crop_output_dir: Path,
    species_classification_report_path: Path,
) -> RunAccumulator:
    """Process all selected sources and return accumulated artifacts and decisions.

    Returns:
        RunAccumulator: Collected decisions, artifacts, source mapping, and stats.
    """
    accumulator = _new_run_accumulator()
    with tempfile.TemporaryDirectory(prefix="processing_inputs_") as temp_dir:
        staging_dir = Path(temp_dir)
        md_results_path = paths.md_results_path
        for index, source in enumerate(processing_sources, 1):
            print(
                f"[{index}/{len(processing_sources)}] Processing {source.source.name} "
                f"({source.video_id[:8]}...)"
            )
            try:
                (
                    decisions,
                    staged_video_ids,
                    report_video_paths,
                    preview_stats,
                    preview_paths,
                    crop_paths,
                    temp_species_report,
                ) = process_single_video(
                    source=source,
                    staging_dir=staging_dir,
                    md_results_path=md_results_path,
                    config=config,
                    paths=paths,
                    categories=categories,
                    effective_move_files=effective_move_files,
                    preview_output_dir=preview_output_dir,
                    species_crop_output_dir=species_crop_output_dir,
                    species_classification_report_path=species_classification_report_path,
                    video_index=index,
                )
            except Exception as exc:
                print(f"  [✗] Error processing {source.source.name}: {exc}")
                raise

            collision_remaps = _apply_collision_remaps(
                accumulator.all_decisions,
                decisions,
                staged_video_ids,
            )
            decision_start_idx = len(accumulator.all_decisions)
            accumulator.all_decisions.extend(decisions)
            for decision_offset, _decision in enumerate(decisions):
                accumulator.processing_sources_by_decision[decision_start_idx + decision_offset] = source
            accumulator.all_staged_video_ids.update(staged_video_ids)
            _merge_path_map_with_remaps(
                accumulator.all_report_video_paths,
                report_video_paths,
                collision_remaps,
            )
            _merge_path_map_with_remaps(
                accumulator.all_preview_paths,
                preview_paths,
                collision_remaps,
            )
            _merge_path_map_with_remaps(
                accumulator.all_crop_paths,
                crop_paths,
                collision_remaps,
            )
            _rewrite_species_source_paths(temp_species_report, collision_remaps)
            accumulator.temp_species_report_paths.append(temp_species_report)
            _accumulate_preview_stats(accumulator.aggregated_preview_stats, preview_stats)
            _print_video_result(source, decisions)
    return accumulator


def _build_video_id_maps(
    accumulator: RunAccumulator,
    canonical_videos_dir: Path,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Map output-relative paths to canonical artifacts and video IDs.

    Returns:
        tuple[dict[str, Path], dict[str, str]]: Bucketed artifact path map and video_id map.
    """
    bucketed_video_paths_final: dict[str, Path] = {}
    video_ids_by_output_path: dict[str, str] = {}
    for decision_idx, decision in enumerate(accumulator.all_decisions):
        source = accumulator.processing_sources_by_decision.get(decision_idx)
        if source is None:
            continue
        video_id = source.video_id
        suffix = Path(decision.relative_path).suffix.lower() or ".bin"
        canonical_dir = build_video_storage_dir(canonical_videos_dir, video_id)
        bucketed_video_paths_final[decision.output_relative_path] = canonical_dir / f"interesting{suffix}"
        video_ids_by_output_path[decision.output_relative_path] = video_id
    return bucketed_video_paths_final, video_ids_by_output_path


def main() -> int:
    """Run the end-to-end video processing workflow with per-video streaming.

    Returns:
        int: Process exit code.
    """
    args = parse_args()
    config = load_config(Path(args.config))
    paths = build_run_paths(Path(args.config), config)
    _print_runtime_header(paths, config, args.mode)

    paths.canonical_videos_dir.mkdir(parents=True, exist_ok=True)
    initialize_metadata_store(paths.metadata_db_path)

    if args.mode == "report-only":
        _write_optional_reports(paths, config)
        print("Report generation complete from sqlite catalog and artifact files.")
        return 0

    processing_sources = _load_processing_sources_for_mode(args.mode, paths, config)
    if not processing_sources:
        print(f"No videos selected for processing in mode '{args.mode}'. Exiting.")
        return 0

    categories = resolve_interesting_categories(config)
    effective_move_files = _resolve_effective_move_files(args.mode, config.move_files)
    preview_output_dir = resolve_preview_output_dir(config.preview_output_dir, paths.output_dir)
    species_crop_output_dir = resolve_preview_output_dir(config.species_crop_output_dir, paths.output_dir)
    species_classification_report_path = paths.metadata_dir / "species_classifications.json"

    accumulator = _process_sources_sequential(
        processing_sources=processing_sources,
        paths=paths,
        config=config,
        categories=categories,
        effective_move_files=effective_move_files,
        preview_output_dir=preview_output_dir,
        species_crop_output_dir=species_crop_output_dir,
        species_classification_report_path=species_classification_report_path,
    )

    merge_species_classification_reports(
        temp_report_paths=accumulator.temp_species_report_paths,
        final_report_path=species_classification_report_path,
    )
    bucketed_video_paths_final, video_ids_by_output_path = _build_video_id_maps(
        accumulator,
        paths.canonical_videos_dir,
    )

    record_processing_results(
        decisions=accumulator.all_decisions,
        staged_video_ids=accumulator.all_staged_video_ids,
        metadata_db_path=paths.metadata_db_path,
        pipeline_version=config.pipeline_version,
        mode=args.mode,
        save_uninteresting_files=config.save_uninteresting_files,
        bucketed_video_paths=bucketed_video_paths_final,
        report_video_paths=accumulator.all_report_video_paths,
        preview_image_paths=accumulator.all_preview_paths,
        species_crop_paths=accumulator.all_crop_paths,
        video_ids_by_output_path=video_ids_by_output_path,
    )
    sync_species_classifications_to_catalog(
        species_classification_report_path=species_classification_report_path,
        metadata_db_path=paths.metadata_db_path,
        video_ids_by_output_path=video_ids_by_output_path,
    )

    print(
        "Preview extraction complete. "
        f"candidates={accumulator.aggregated_preview_stats.total_candidates}, "
        f"extracted={accumulator.aggregated_preview_stats.extracted}, "
        f"skipped={accumulator.aggregated_preview_stats.skipped}, "
        f"failed={accumulator.aggregated_preview_stats.failed}, "
        f"classified={accumulator.aggregated_preview_stats.classified}, "
        f"classification_failed={accumulator.aggregated_preview_stats.classification_failed}"
    )
    _write_optional_reports(paths, config)

    counts = compute_bucket_counts(accumulator.all_decisions)
    print(
        "Finished processing all videos. "
        f"interesting={counts['interesting']}, "
        f"uninteresting={counts['uninteresting']}, "
        f"failed={counts['failed']}"
    )
    print(f"Raw MegaDetector output: {paths.md_results_path}")
    if config.write_json_exports:
        print(f"Summary report: {paths.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
