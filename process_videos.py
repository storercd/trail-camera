from __future__ import annotations

import argparse
import json
import tempfile
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
        choices=("new-only", "reprocess-existing", "report-only"),
        help=(
            "Processing mode: new-only, reprocess-existing, or report-only "
            "(default: new-only)"
        ),
    )
    return parser.parse_args()


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

    try:
        with species_classification_report_path.open("r", encoding="utf-8") as handle:
            report_payload = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return

    entries_by_video_id: dict[str, dict[str, Any]] = {}
    for entry in report_payload.get("entries", []):
        if not isinstance(entry, dict):
            continue
        output_relative_path = entry.get("source_relative_path")
        if not isinstance(output_relative_path, str) or not output_relative_path:
            continue
        video_id = video_ids_by_output_path.get(output_relative_path)
        if video_id is None:
            continue
        entries_by_video_id[video_id] = entry

    for video_id in video_ids_by_output_path.values():
        entry = entries_by_video_id.get(video_id)
        if entry is None:
            delete_species_classification_record(metadata_db_path, video_id)
            continue

        top = entry.get("top_classification") or {}
        top_label = top.get("label") if isinstance(top, dict) else None
        top_raw_class = top.get("raw_class") if isinstance(top, dict) else None
        try:
            top_score = float(top.get("score")) if isinstance(top, dict) and top.get("score") is not None else None
        except (TypeError, ValueError):
            top_score = None

        raw_candidates = entry.get("candidates")
        candidates = raw_candidates if isinstance(raw_candidates, list) else []
        upsert_species_classification_record(
            db_path=metadata_db_path,
            record=SpeciesClassificationRecord(
                video_id=video_id,
                top_label=str(top_label) if top_label is not None else None,
                top_score=top_score,
                top_raw_class=str(top_raw_class) if top_raw_class is not None else None,
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
        tuple: (decisions, staged_video_ids, report_video_paths, preview_stats, preview_paths, crop_paths, temp_species_report).
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

    return decisions, staged_video_ids, report_video_paths, preview_stats, preview_paths, crop_paths, temp_species_report


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


def main() -> int:
    """Run the end-to-end video processing workflow with per-video streaming.

    Returns:
        int: Process exit code.
    """
    args = parse_args()
    config = load_config(Path(args.config))
    paths = build_run_paths(Path(args.config), config)

    print(f"Using config file: {paths.config_path}")
    print(f"Input directory: {paths.input_dir}")
    print(f"Output root directory: {paths.output_root_dir}")
    print(f"Run output directory: {paths.output_dir}")
    print(f"Pipeline version: {config.pipeline_version}")
    print(f"Metadata database: {paths.metadata_db_path}")
    print(f"Processing mode: {args.mode}")
    if args.mode != "report-only":
        print("Processing model: per-video streaming")

    paths.canonical_videos_dir.mkdir(parents=True, exist_ok=True)
    initialize_metadata_store(paths.metadata_db_path)

    if args.mode == "report-only":
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

        if config.generate_html_report:
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
        else:
            print("HTML summary disabled by config")

        print("Report generation complete from sqlite catalog and artifact files.")
        return 0

    processing_sources: list[ProcessingSource]
    if args.mode == "new-only":
        discovered_videos = validate_and_find_videos(paths.input_dir, config.recursive)
        ingested_count, newly_persisted_count, newly_discovered_sources = ingest_videos_into_catalog(
            videos=discovered_videos,
            canonical_videos_dir=paths.canonical_videos_dir,
            metadata_db_path=paths.metadata_db_path,
        )
        print(
            "Catalog ingestion complete. "
            f"videos={ingested_count}, "
            f"new_canonical_originals={newly_persisted_count}, "
            f"new_videos={len(newly_discovered_sources)}"
        )
        processing_sources = newly_discovered_sources
    else:
        processing_sources = load_reprocess_sources(paths.metadata_db_path)
        print(f"Loaded {len(processing_sources)} canonical originals for reprocessing")

    if not processing_sources:
        print(f"No videos selected for processing in mode '{args.mode}'. Exiting.")
        return 0

    categories = resolve_interesting_categories(config)

    effective_move_files = config.move_files
    if args.mode == "reprocess-existing" and config.move_files:
        print("Disabling move_files for reprocess-existing mode")
        effective_move_files = False

    # Setup output directories
    preview_output_dir = resolve_preview_output_dir(config.preview_output_dir, paths.output_dir)
    species_crop_output_dir = resolve_preview_output_dir(
        config.species_crop_output_dir,
        paths.output_dir,
    )
    species_classification_report_path = paths.metadata_dir / "species_classifications.json"

    # Accumulate results across all videos
    all_decisions: list[VideoDecision] = []
    all_staged_video_ids: dict[str, str] = {}
    all_report_video_paths: dict[str, Path] = {}
    all_preview_paths: dict[str, Path] = {}
    all_crop_paths: dict[str, Path] = {}
    temp_species_report_paths: list[Path] = []
    processing_sources_by_decision: dict[int, ProcessingSource] = {}  # decision_index -> source
    aggregated_preview_stats = PreviewExtractionStats(
        total_candidates=0,
        extracted=0,
        skipped=0,
        failed=0,
        classified=0,
        classification_failed=0,
    )

    # Process each video through all stages sequentially
    with tempfile.TemporaryDirectory(prefix="processing_inputs_") as temp_dir:
        staging_dir = Path(temp_dir)
        md_results_path = paths.md_results_path

        for index, source in enumerate(processing_sources, 1):
            print(
                f"[{index}/{len(processing_sources)}] Processing {source.source.name} "
                f"({source.video_id[:8]}...)"
            )

            try:
                decisions, staged_video_ids, report_video_paths, preview_stats, preview_paths, crop_paths, temp_species_report = (
                    process_single_video(
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
                )

                # Handle output_relative_path collisions by appending video_id
                # when the same path is already in use
                seen_output_paths = {d.output_relative_path for d in all_decisions}
                collision_remaps: dict[str, str] = {}  # old_path -> new_path
                for decision in decisions:
                    old_path = decision.output_relative_path
                    if old_path in seen_output_paths:
                        # Collision detected, append shortened video_id
                        video_id = staged_video_ids.get(decision.relative_path, "unknown")
                        new_path = f"{old_path}__{video_id[:8]}"
                        decision.output_relative_path = new_path
                        collision_remaps[old_path] = new_path

                # Accumulate results
                decision_start_idx = len(all_decisions)
                all_decisions.extend(decisions)
                # Track which source each decision came from
                for i, decision in enumerate(decisions):
                    processing_sources_by_decision[decision_start_idx + i] = source
                all_staged_video_ids.update(staged_video_ids)

                # Remap report paths, applying collision suffixes to keys
                for old_key, video_value in report_video_paths.items():
                    new_key = collision_remaps.get(old_key, old_key)
                    all_report_video_paths[new_key] = video_value
                for old_key, video_value in preview_paths.items():
                    new_key = collision_remaps.get(old_key, old_key)
                    all_preview_paths[new_key] = video_value
                for old_key, video_value in crop_paths.items():
                    new_key = collision_remaps.get(old_key, old_key)
                    all_crop_paths[new_key] = video_value

                # Update species classification entries with new output_relative_path if there were collisions
                if collision_remaps and temp_species_report.exists():
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
                        pass

                temp_species_report_paths.append(temp_species_report)

                # Accumulate stats
                aggregated_preview_stats.total_candidates += preview_stats.total_candidates
                aggregated_preview_stats.extracted += preview_stats.extracted
                aggregated_preview_stats.skipped += preview_stats.skipped
                aggregated_preview_stats.failed += preview_stats.failed
                aggregated_preview_stats.classified += preview_stats.classified
                aggregated_preview_stats.classification_failed += (
                    preview_stats.classification_failed
                )

                print(
                    f"  [✓] {decisions[0].relative_path} → {decisions[0].bucket} "
                    f"(confidence: {decisions[0].top_confidence:.3f})"
                )
            except Exception as e:
                print(f"  [✗] Error processing {source.source.name}: {e}")
                raise

    # Merge all temporary species classification reports into final report
    merge_species_classification_reports(
        temp_report_paths=temp_species_report_paths,
        final_report_path=species_classification_report_path,
    )

    # Build bucketed_video_paths and video_ids_by_output_path keyed by output_relative_path
    # (after collision handling), mapping each decision to its correct video_id and canonical path
    bucketed_video_paths_final: dict[str, Path] = {}
    video_ids_by_output_path: dict[str, str] = {}
    for decision_idx, decision in enumerate(all_decisions):
        source = processing_sources_by_decision.get(decision_idx)
        if source:
            video_id = source.video_id
            suffix = Path(decision.relative_path).suffix.lower() or ".bin"
            canonical_dir = build_video_storage_dir(paths.canonical_videos_dir, video_id)
            bucketed_video_paths_final[decision.output_relative_path] = canonical_dir / f"interesting{suffix}"
            video_ids_by_output_path[decision.output_relative_path] = video_id

    # Record all accumulated results to database
    record_processing_results(
        decisions=all_decisions,
        staged_video_ids=all_staged_video_ids,
        metadata_db_path=paths.metadata_db_path,
        pipeline_version=config.pipeline_version,
        mode=args.mode,
        save_uninteresting_files=config.save_uninteresting_files,
        bucketed_video_paths=bucketed_video_paths_final,
        report_video_paths=all_report_video_paths,
        preview_image_paths=all_preview_paths,
        species_crop_paths=all_crop_paths,
        video_ids_by_output_path=video_ids_by_output_path,
    )

    # Persist species classifications in SQLite so report generation is decoupled from processing.
    sync_species_classifications_to_catalog(
        species_classification_report_path=species_classification_report_path,
        metadata_db_path=paths.metadata_db_path,
        video_ids_by_output_path=video_ids_by_output_path,
    )

    print(
        "Preview extraction complete. "
        f"candidates={aggregated_preview_stats.total_candidates}, "
        f"extracted={aggregated_preview_stats.extracted}, "
        f"skipped={aggregated_preview_stats.skipped}, "
        f"failed={aggregated_preview_stats.failed}, "
        f"classified={aggregated_preview_stats.classified}, "
        f"classification_failed={aggregated_preview_stats.classification_failed}"
    )

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

    if config.generate_html_report:
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
    else:
        print("HTML summary disabled by config")

    counts = compute_bucket_counts(all_decisions)
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
