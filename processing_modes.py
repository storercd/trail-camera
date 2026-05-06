"""Processing mode helpers for catalog ingestion and result persistence."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from file_ops import (
    build_video_storage_dir,
    compute_sha256,
    find_existing_canonical_original_path,
    get_capture_date,
    persist_canonical_original,
)
from metadata_store import (
    ProcessingStateRecord,
    VideoCatalogRecord,
    get_processing_bucket,
    get_stored_original_records,
    sync_artifact_path,
    upsert_processing_state_record,
    upsert_video_record,
    video_exists,
)
from pipeline_models import VideoDecision


@dataclass
class ProcessingSource:
    """Video selected for processing with resolved canonical identity."""

    source: Path
    video_id: str


def ingest_videos_into_catalog(
    videos: list[Path],
    canonical_videos_dir: Path,
    metadata_db_path: Path,
    camera_date_profile: dict[str, object] | None = None,
) -> tuple[int, int, list[ProcessingSource]]:
    """Compute IDs, persist canonical originals, and upsert catalog rows.

    Args:
        videos: Source videos discovered for this run.
        canonical_videos_dir: Root directory for canonical per-video storage.
        metadata_db_path: SQLite metadata catalog path.
        camera_date_profile: Optional camera overlay OCR profile.

    Returns:
        tuple[int, int, list[ProcessingSource]]: Ingested count, newly persisted originals,
            and newly discovered sources with IDs.
    """
    newly_persisted = 0
    newly_discovered_sources: list[ProcessingSource] = []
    total_videos = len(videos)
    progress_every = 10
    for index, source in enumerate(videos, start=1):
        if index == 1 or index % progress_every == 0 or index == total_videos:
            print(
                f"[ingest {index}/{total_videos}] hashing and cataloging {source.name}",
                flush=True,
            )

        video_id = compute_sha256(source)
        was_known_video_id = video_exists(metadata_db_path, video_id)
        stored_original_path: Path | None = None
        if was_known_video_id:
            # Keep uninteresting videos out of canonical storage to avoid re-growing disk usage.
            current_bucket = get_processing_bucket(metadata_db_path, video_id)
            if current_bucket == "uninteresting":
                continue

            existing_original = find_existing_canonical_original_path(canonical_videos_dir, video_id)
            if existing_original is not None:
                stored_original_path = existing_original

        if stored_original_path is None:
            had_existing_original = (
                find_existing_canonical_original_path(canonical_videos_dir, video_id) is not None
            )
            stored_original_path = persist_canonical_original(
                videos_root_dir=canonical_videos_dir,
                video_id=video_id,
                source=source,
            )
            if not had_existing_original:
                newly_persisted += 1

        if not was_known_video_id:
            newly_discovered_sources.append(ProcessingSource(source=source, video_id=video_id))

        stat = source.stat()
        upsert_video_record(
            db_path=metadata_db_path,
            record=VideoCatalogRecord(
                video_id=video_id,
                original_filename=source.name,
                capture_date=get_capture_date(source, camera_date_profile=camera_date_profile),
                filesize_bytes=int(stat.st_size),
                source_ext=source.suffix.lower(),
                stored_original_path=str(stored_original_path),
            ),
        )

    return len(videos), newly_persisted, newly_discovered_sources


def load_reprocess_sources(
    metadata_db_path: Path,
    current_pipeline_version: str | None = None,
    filter_mode: str = "force",
) -> list[ProcessingSource]:
    """Load canonical stored originals for reprocessing.

    Args:
        metadata_db_path: SQLite metadata catalog path.
        current_pipeline_version: Current pipeline version for filtering.
            Required when filter_mode is "existing".
        filter_mode: Filtering strategy:
            - "force": Load all videos regardless of status/version.
            - "existing": Load only videos with outdated pipeline_version or status="failed".

    Returns:
        list[ProcessingSource]: Canonical source files selected for reprocessing.

    Raises:
        ValueError: When filter_mode is "existing" and current_pipeline_version is missing.
    """
    import sqlite3

    all_records = dict(get_stored_original_records(metadata_db_path))

    if filter_mode == "force":
        return [
            ProcessingSource(source=path, video_id=video_id)
            for video_id, path in all_records.items()
            if path.exists()
        ]

    # filter_mode == "existing": only outdated or failed videos
    if not current_pipeline_version:
        raise ValueError("current_pipeline_version required for filter_mode='existing'")

    filtered_records: dict[str, Path] = {}
    with sqlite3.connect(metadata_db_path) as connection:
        rows = connection.execute(
            """
            SELECT video_id FROM processing_state
            WHERE pipeline_version != ? OR status = 'failed'
            """,
            (current_pipeline_version,),
        ).fetchall()

    outdated_or_failed_ids = {row[0] for row in rows}

    for video_id, path in all_records.items():
        if video_id in outdated_or_failed_ids and path.exists():
            filtered_records[video_id] = path

    return [
        ProcessingSource(source=path, video_id=video_id)
        for video_id, path in filtered_records.items()
    ]


def stage_single_processing_input(
    source: ProcessingSource,
    staging_dir: Path,
) -> tuple[Path, dict[str, str]]:
    """Stage a single processing source into a temporary directory.

    Creates symlinks when possible and falls back to file copies.

    Args:
        source: Source file and canonical ID to stage.
        staging_dir: Destination staging directory to use.

    Returns:
        tuple[Path, dict[str, str]]: Staging dir and staged filename -> video_id map.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)

    candidate = staging_dir / source.source.name
    if candidate.exists():
        suffix_index = 1
        while True:
            candidate = staging_dir / f"{source.source.stem}_{suffix_index}{source.source.suffix}"
            if not candidate.exists():
                break
            suffix_index += 1

    try:
        candidate.symlink_to(source.source.resolve())
    except OSError:
        shutil.copy2(source.source, candidate)

    return staging_dir, {candidate.name: source.video_id}


def stage_processing_inputs(
    sources: list[ProcessingSource],
    staging_dir: Path,
) -> tuple[Path, dict[str, str]]:
    """Stage selected processing sources into one temporary directory.

    Creates symlinks when possible and falls back to file copies.

    Args:
        sources: Source files and canonical IDs to stage.
        staging_dir: Destination staging directory.

    Returns:
        tuple[Path, dict[str, str]]: Staging dir and staged filename -> video_id map.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    staged_video_ids: dict[str, str] = {}

    for source_item in sources:
        source = source_item.source
        candidate = staging_dir / source.name
        if candidate.exists():
            suffix_index = 1
            while True:
                candidate = staging_dir / f"{source.stem}_{suffix_index}{source.suffix}"
                if not candidate.exists():
                    break
                suffix_index += 1

        try:
            candidate.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, candidate)
        staged_video_ids[candidate.name] = source_item.video_id

    return staging_dir, staged_video_ids


def build_bucket_output_paths(
    staged_video_ids: dict[str, str],
    canonical_videos_dir: Path,
) -> dict[str, Path]:
    """Build canonical bucketed video output paths by staged relative path.

    Returns:
        dict[str, Path]: Canonical bucketed output paths keyed by staged relative path.
    """
    output_paths: dict[str, Path] = {}
    for relative_path, video_id in staged_video_ids.items():
        suffix = Path(relative_path).suffix.lower() or ".bin"
        canonical_dir = build_video_storage_dir(canonical_videos_dir, video_id)
        output_paths[relative_path] = canonical_dir / f"interesting{suffix}"
    return output_paths


def build_report_output_paths(
    decisions: list[VideoDecision],
    staged_video_ids: dict[str, str],
    canonical_videos_dir: Path,
) -> dict[str, Path]:
    """Build canonical report video output paths keyed by output-relative path.

    Returns:
        dict[str, Path]: Canonical report paths keyed by output-relative path.
    """
    output_paths: dict[str, Path] = {}
    for decision in decisions:
        if decision.bucket != "interesting":
            continue
        video_id = staged_video_ids.get(decision.relative_path)
        if video_id is None:
            continue
        canonical_dir = build_video_storage_dir(canonical_videos_dir, video_id)
        output_paths[decision.output_relative_path] = canonical_dir / "report.mp4"
    return output_paths


def build_preview_output_paths(
    decisions: list[VideoDecision],
    staged_video_ids: dict[str, str],
    canonical_videos_dir: Path,
) -> dict[str, Path]:
    """Build canonical preview image output paths keyed by staged relative path.

    Returns:
        dict[str, Path]: Canonical preview image paths keyed by staged relative path.
    """
    output_paths: dict[str, Path] = {}
    for decision in decisions:
        video_id = staged_video_ids.get(decision.relative_path)
        if video_id is None:
            continue
        canonical_dir = build_video_storage_dir(canonical_videos_dir, video_id)
        output_paths[decision.relative_path] = canonical_dir / "preview.jpg"
    return output_paths


def build_species_crop_output_dirs(
    decisions: list[VideoDecision],
    staged_video_ids: dict[str, str],
    canonical_videos_dir: Path,
) -> dict[str, Path]:
    """Build canonical species-crop output dirs keyed by staged relative path.

    Returns:
        dict[str, Path]: Canonical crop output dirs keyed by staged relative path.
    """
    output_dirs: dict[str, Path] = {}
    for decision in decisions:
        video_id = staged_video_ids.get(decision.relative_path)
        if video_id is None:
            continue
        output_dirs[decision.relative_path] = build_video_storage_dir(canonical_videos_dir, video_id)
    return output_dirs


def resolve_canonical_artifacts_for_decision(
    decision: VideoDecision,
    save_uninteresting_files: bool,
    bucketed_video_paths: dict[str, Path],
    report_video_paths: dict[str, Path],
    preview_image_paths: dict[str, Path],
    species_crop_paths: dict[str, Path],
) -> tuple[Path | None, Path | None, Path | None, Path | None, bool]:
    """Resolve canonical artifact paths for one decision.

    Returns:
        tuple[Path | None, Path | None, Path | None, Path | None, bool]:
            Bucketed video, report video, preview image, species crop, and
            whether canonical source should be pruned for an uninteresting video.
    """
    should_have_output = decision.bucket != "uninteresting" or save_uninteresting_files
    bucketed_output_path = None
    if should_have_output:
        bucketed_candidate = bucketed_video_paths.get(decision.output_relative_path)
        if bucketed_candidate is not None and bucketed_candidate.exists():
            bucketed_output_path = bucketed_candidate

    report_video_path = report_video_paths.get(decision.output_relative_path)
    if report_video_path is not None and not report_video_path.exists():
        report_video_path = None

    preview_image_path = preview_image_paths.get(decision.output_relative_path)
    if preview_image_path is not None and not preview_image_path.exists():
        preview_image_path = None

    species_crop_path = species_crop_paths.get(decision.output_relative_path)
    if species_crop_path is not None and not species_crop_path.exists():
        species_crop_path = None

    should_prune_source = decision.bucket == "uninteresting" and not save_uninteresting_files
    return (
        bucketed_output_path,
        report_video_path,
        preview_image_path,
        species_crop_path,
        should_prune_source,
    )


def sync_decision_artifacts(
    metadata_db_path: Path,
    video_id: str,
    bucketed_output_path: Path | None,
    report_video_path: Path | None,
    preview_image_path: Path | None,
    species_crop_path: Path | None,
) -> None:
    """Sync all artifact path records for a processed decision."""
    sync_artifact_path(
        db_path=metadata_db_path,
        video_id=video_id,
        artifact_type="bucketed_video",
        artifact_path=bucketed_output_path,
    )
    sync_artifact_path(
        db_path=metadata_db_path,
        video_id=video_id,
        artifact_type="report_video",
        artifact_path=report_video_path,
    )
    sync_artifact_path(
        db_path=metadata_db_path,
        video_id=video_id,
        artifact_type="preview_image",
        artifact_path=preview_image_path,
    )
    sync_artifact_path(
        db_path=metadata_db_path,
        video_id=video_id,
        artifact_type="species_crop",
        artifact_path=species_crop_path,
    )


def set_relocated_path(
    destination_map: dict[str, Path],
    key: str,
    artifact_path: Path | None,
) -> None:
    """Set relocated artifact map entry when path exists."""
    if artifact_path is not None:
        destination_map[key] = artifact_path


def prune_uninteresting_canonical_sources(
    metadata_db_path: Path,
    video_ids: set[str],
) -> None:
    """Delete canonical source files for uninteresting videos when requested."""
    if not video_ids:
        return

    stored_records = dict(get_stored_original_records(metadata_db_path))
    for video_id in sorted(video_ids):
        stored_original = stored_records.get(video_id)
        if stored_original is None:
            continue
        if stored_original.exists() and stored_original.is_file():
            stored_original.unlink()


def record_processing_results(
    decisions: list[VideoDecision],
    staged_video_ids: dict[str, str],
    metadata_db_path: Path,
    pipeline_version: str,
    mode: str,
    save_uninteresting_files: bool,
    bucketed_video_paths: dict[str, Path] | None = None,
    report_video_paths: dict[str, Path] | None = None,
    preview_image_paths: dict[str, Path] | None = None,
    species_crop_paths: dict[str, Path] | None = None,
    video_ids_by_output_path: dict[str, str] | None = None,
) -> tuple[dict[str, Path], dict[str, Path], dict[str, Path], dict[str, Path]]:
    """Persist processing_state rows and sync primary output artifact paths.

    Args:
        decisions: Classification decisions for this processing pass.
        staged_video_ids: Mapping from staged relative path to canonical video ID.
        metadata_db_path: SQLite metadata catalog path.
        pipeline_version: Current logical pipeline version.
        mode: Processing mode (`new-only` or `reprocess-existing`).
        save_uninteresting_files: Whether uninteresting videos are persisted.
        bucketed_video_paths: Bucketed video paths by staged relative path.
        report_video_paths: Generated report-video paths by output-relative path.
        preview_image_paths: Generated preview-image paths by output-relative path.
        species_crop_paths: Generated species-crop paths by output-relative path.
        video_ids_by_output_path: Optional override mapping output-relative path to video ID.
            When provided, takes precedence over staged_video_ids for per-decision lookups.

    Returns:
        tuple[dict[str, Path], dict[str, Path], dict[str, Path], dict[str, Path]]:
            Resolved report-video, bucketed-video, preview-image, and species-crop
            paths keyed by output-relative path.
    """
    resolved_bucketed_videos = bucketed_video_paths or {}
    resolved_report_videos = report_video_paths or {}
    resolved_preview_images = preview_image_paths or {}
    resolved_species_crops = species_crop_paths or {}
    uninteresting_video_ids_to_prune: set[str] = set()
    relocated_report_videos: dict[str, Path] = {}
    relocated_bucketed_videos: dict[str, Path] = {}
    relocated_preview_images: dict[str, Path] = {}
    relocated_species_crops: dict[str, Path] = {}

    for decision in decisions:
        if video_ids_by_output_path is not None:
            video_id = video_ids_by_output_path.get(decision.output_relative_path)
        else:
            video_id = staged_video_ids.get(decision.relative_path)
        if video_id is None:
            continue

        status = "failed" if decision.bucket == "failed" else "processed"
        upsert_processing_state_record(
            metadata_db_path,
            ProcessingStateRecord(
                video_id=video_id,
                pipeline_version=pipeline_version,
                mode=mode,
                bucket=decision.bucket,
                top_confidence=decision.top_confidence,
                top_category=decision.top_category,
                top_frame=decision.top_frame,
                status=status,
            ),
        )

        (
            bucketed_output_path,
            report_video_path,
            preview_image_path,
            species_crop_path,
            should_prune_source,
        ) = resolve_canonical_artifacts_for_decision(
            decision=decision,
            save_uninteresting_files=save_uninteresting_files,
            bucketed_video_paths=resolved_bucketed_videos,
            report_video_paths=resolved_report_videos,
            preview_image_paths=resolved_preview_images,
            species_crop_paths=resolved_species_crops,
        )

        sync_decision_artifacts(
            metadata_db_path=metadata_db_path,
            video_id=video_id,
            bucketed_output_path=bucketed_output_path,
            report_video_path=report_video_path,
            preview_image_path=preview_image_path,
            species_crop_path=species_crop_path,
        )

        set_relocated_path(
            destination_map=relocated_bucketed_videos,
            key=decision.output_relative_path,
            artifact_path=bucketed_output_path,
        )
        set_relocated_path(
            destination_map=relocated_report_videos,
            key=decision.output_relative_path,
            artifact_path=report_video_path,
        )
        set_relocated_path(
            destination_map=relocated_preview_images,
            key=decision.output_relative_path,
            artifact_path=preview_image_path,
        )
        set_relocated_path(
            destination_map=relocated_species_crops,
            key=decision.output_relative_path,
            artifact_path=species_crop_path,
        )

        if should_prune_source:
            uninteresting_video_ids_to_prune.add(video_id)

    prune_uninteresting_canonical_sources(
        metadata_db_path=metadata_db_path,
        video_ids=uninteresting_video_ids_to_prune,
    )

    return (
        relocated_report_videos,
        relocated_bucketed_videos,
        relocated_preview_images,
        relocated_species_crops,
    )


def collect_species_artifact_maps(
    species_classification_report_path: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    """Collect preview and species-crop paths keyed by output-relative video path.

    Args:
        species_classification_report_path: SpeciesNet JSON report path.

    Returns:
        tuple[dict[str, Path], dict[str, Path]]: Preview and species-crop maps.
    """
    preview_paths: dict[str, Path] = {}
    crop_paths: dict[str, Path] = {}
    if not species_classification_report_path.exists():
        return preview_paths, crop_paths

    with species_classification_report_path.open("r", encoding="utf-8") as handle:
        report_payload = json.load(handle)

    for entry in report_payload.get("entries", []):
        if not isinstance(entry, dict):
            continue
        source_relative_path = entry.get("source_relative_path")
        if not isinstance(source_relative_path, str) or not source_relative_path:
            continue

        preview_path = entry.get("preview_image_final")
        if isinstance(preview_path, str) and preview_path:
            preview_paths[source_relative_path] = Path(preview_path)

        used_species_crop = bool(entry.get("used_species_crop", False))
        crop_path = entry.get("classification_input_image")
        if used_species_crop and isinstance(crop_path, str) and crop_path:
            crop_paths[source_relative_path] = Path(crop_path)

    return preview_paths, crop_paths
