"""Tests for processing mode helpers and in-place replacement behavior."""

import sqlite3
from pathlib import Path

from file_ops import build_video_storage_dir
from metadata_store import (
    ProcessingStateRecord,
    VideoCatalogRecord,
    get_stored_original_records,
    initialize_metadata_store,
    sync_artifact_path,
    upsert_processing_state_record,
    upsert_video_record,
)
from pipeline_models import VideoDecision
from processing_modes import (
    ingest_videos_into_catalog,
    load_reprocess_sources,
    record_processing_results,
)


def test_ingest_videos_into_catalog_should_return_only_new_sources(tmp_path: Path) -> None:
    """Return newly discovered sources only after initial catalog ingestion."""
    source = tmp_path / "PICT0001.AVI"
    source.write_bytes(b"video-content")
    canonical_root = tmp_path / "videos"
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    first_ingested, first_new_persisted, first_new_sources = ingest_videos_into_catalog(
        videos=[source],
        canonical_videos_dir=canonical_root,
        metadata_db_path=db_path,
    )
    second_ingested, second_new_persisted, second_new_sources = ingest_videos_into_catalog(
        videos=[source],
        canonical_videos_dir=canonical_root,
        metadata_db_path=db_path,
    )

    assert first_ingested == 1
    assert first_new_persisted == 1
    assert len(first_new_sources) == 1
    assert second_ingested == 1
    assert second_new_persisted == 0
    assert second_new_sources == []


def test_ingest_videos_into_catalog_should_not_repersist_known_uninteresting(
    tmp_path: Path,
) -> None:
    """Skip canonical re-persist when a known video was already marked uninteresting."""
    source = tmp_path / "PICT0001.AVI"
    source.write_bytes(b"video-content")
    canonical_root = tmp_path / "videos"
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    ingest_videos_into_catalog(
        videos=[source],
        canonical_videos_dir=canonical_root,
        metadata_db_path=db_path,
    )
    video_id, stored_original = get_stored_original_records(db_path)[0]

    upsert_processing_state_record(
        db_path,
        ProcessingStateRecord(
            video_id=video_id,
            pipeline_version="0.1.0",
            mode="new-only",
            bucket="uninteresting",
            top_confidence=None,
            top_category=None,
            top_frame=None,
            status="processed",
        ),
    )

    stored_original.unlink()
    _, second_new_persisted, second_new_sources = ingest_videos_into_catalog(
        videos=[source],
        canonical_videos_dir=canonical_root,
        metadata_db_path=db_path,
    )

    assert second_new_persisted == 0
    assert second_new_sources == []
    assert not stored_original.exists()


def test_load_reprocess_sources_should_skip_missing_files(tmp_path: Path) -> None:
    """Load only existing canonical originals in reprocess-existing mode."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    existing = tmp_path / "existing.AVI"
    existing.write_bytes(b"ok")

    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="vid-existing",
            original_filename="existing.AVI",
            capture_date="2026-05-05",
            filesize_bytes=2,
            source_ext=".avi",
            stored_original_path=str(existing),
        ),
    )
    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="vid-missing",
            original_filename="missing.AVI",
            capture_date="2026-05-05",
            filesize_bytes=2,
            source_ext=".avi",
            stored_original_path=str(tmp_path / "missing.AVI"),
        ),
    )

    sources = load_reprocess_sources(db_path)

    assert len(sources) == 1
    assert sources[0].video_id == "vid-existing"
    assert sources[0].source == existing


def test_record_processing_results_should_clear_unsaved_uninteresting_artifact(
    tmp_path: Path,
) -> None:
    """Remove stale bucketed artifact when uninteresting outputs are disabled."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="vid-001",
            original_filename="PICT0001.AVI",
            capture_date="2026-05-05",
            filesize_bytes=10,
            source_ext=".avi",
            stored_original_path=str(tmp_path / "source.avi"),
        ),
    )

    old_artifact = tmp_path / "old_bucketed.AVI"
    old_artifact.write_bytes(b"old")
    sync_artifact_path(db_path, "vid-001", "bucketed_video", old_artifact)

    decision = VideoDecision(
        relative_path="stage.AVI",
        output_relative_path="uninteresting_output.AVI",
        bucket="uninteresting",
        top_confidence=None,
        top_category=None,
        top_frame=None,
        top_bbox=None,
        first_interesting_frame=None,
        last_interesting_frame=None,
        num_detections=0,
        failure=None,
    )

    record_processing_results(
        decisions=[decision],
        staged_video_ids={"stage.AVI": "vid-001"},
        metadata_db_path=db_path,
        pipeline_version="0.1.1",
        mode="reprocess-existing",
        save_uninteresting_files=False,
    )

    assert not old_artifact.exists()

    with sqlite3.connect(db_path) as connection:
        artifact = connection.execute(
            "SELECT path FROM artifacts WHERE video_id=? AND artifact_type=?",
            ("vid-001", "bucketed_video"),
        ).fetchone()
        state = connection.execute(
            "SELECT pipeline_version, mode, bucket FROM processing_state WHERE video_id=?",
            ("vid-001",),
        ).fetchone()

    assert artifact is None
    assert state == ("0.1.1", "reprocess-existing", "uninteresting")


def test_record_processing_results_should_prune_canonical_for_unsaved_uninteresting(
    tmp_path: Path,
) -> None:
    """Delete canonical original for uninteresting videos when unsaved outputs are requested."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    canonical = tmp_path / "videos" / "aa" / "bb" / "vid-001" / "source.avi"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"video")

    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="vid-001",
            original_filename="PICT0001.AVI",
            capture_date="2026-05-05",
            filesize_bytes=5,
            source_ext=".avi",
            stored_original_path=str(canonical),
        ),
    )

    decision = VideoDecision(
        relative_path="stage.AVI",
        output_relative_path="uninteresting_output.AVI",
        bucket="uninteresting",
        top_confidence=None,
        top_category=None,
        top_frame=None,
        top_bbox=None,
        first_interesting_frame=None,
        last_interesting_frame=None,
        num_detections=0,
        failure=None,
    )

    record_processing_results(
        decisions=[decision],
        staged_video_ids={"stage.AVI": "vid-001"},
        metadata_db_path=db_path,
        pipeline_version="0.1.1",
        mode="new-only",
        save_uninteresting_files=False,
    )

    assert not canonical.exists()


def test_record_processing_results_should_record_direct_canonical_artifacts(
    tmp_path: Path,
) -> None:
    """Record already-direct canonical artifact paths in metadata."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    canonical_source = tmp_path / "videos" / "aa" / "bb" / "vid-001" / "source.avi"
    canonical_source.parent.mkdir(parents=True, exist_ok=True)
    canonical_source.write_bytes(b"source")

    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="vid-001",
            original_filename="PICT0001.AVI",
            capture_date="2026-05-05",
            filesize_bytes=6,
            source_ext=".avi",
            stored_original_path=str(canonical_source),
        ),
    )

    canonical_dir = build_video_storage_dir(tmp_path / "videos", "vid-001")
    canonical_dir.mkdir(parents=True, exist_ok=True)
    bucketed = canonical_dir / "interesting.avi"
    report_video = canonical_dir / "report.mp4"
    preview_image = canonical_dir / "preview.jpg"
    species_crop = canonical_dir / "species_crop.jpg"
    bucketed.write_bytes(b"bucketed")
    report_video.write_bytes(b"report")
    preview_image.write_bytes(b"preview")
    species_crop.write_bytes(b"crop")

    decision = VideoDecision(
        relative_path="stage.AVI",
        output_relative_path="20260505-PICT0001.AVI",
        bucket="interesting",
        top_confidence=0.99,
        top_category="1",
        top_frame=42,
        top_bbox=[0.1, 0.1, 0.5, 0.5],
        first_interesting_frame=10,
        last_interesting_frame=80,
        num_detections=1,
        failure=None,
    )

    record_processing_results(
        decisions=[decision],
        staged_video_ids={"stage.AVI": "vid-001"},
        metadata_db_path=db_path,
        pipeline_version="0.1.1",
        mode="new-only",
        save_uninteresting_files=False,
        bucketed_video_paths={decision.relative_path: bucketed},
        report_video_paths={decision.output_relative_path: report_video},
        preview_image_paths={decision.output_relative_path: preview_image},
        species_crop_paths={decision.output_relative_path: species_crop},
    )

    assert (canonical_dir / "interesting.avi").exists()
    assert (canonical_dir / "report.mp4").exists()
    assert (canonical_dir / "preview.jpg").exists()
    assert (canonical_dir / "species_crop.jpg").exists()
