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


def test_record_processing_results_should_map_video_ids_by_output_path_when_relative_paths_collide(
    tmp_path: Path,
) -> None:
    """Persist artifact records to the correct video when staged relative paths collide."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    canonical_root = tmp_path / "videos"
    for video_id in ("vid-001", "vid-002"):
        canonical_source = build_video_storage_dir(canonical_root, video_id) / "source.avi"
        canonical_source.parent.mkdir(parents=True, exist_ok=True)
        canonical_source.write_bytes(b"source")
        upsert_video_record(
            db_path,
            VideoCatalogRecord(
                video_id=video_id,
                original_filename="source.avi",
                capture_date="2026-05-05",
                filesize_bytes=6,
                source_ext=".avi",
                stored_original_path=str(canonical_source),
            ),
        )

    decision1 = VideoDecision(
        relative_path="source.avi",
        output_relative_path="20260505-source.avi",
        bucket="interesting",
        top_confidence=0.9,
        top_category="1",
        top_frame=10,
        top_bbox=[0.1, 0.1, 0.5, 0.5],
        first_interesting_frame=5,
        last_interesting_frame=20,
        num_detections=1,
        failure=None,
    )
    decision2 = VideoDecision(
        relative_path="source.avi",
        output_relative_path="20260505-source.avi__vid002",
        bucket="interesting",
        top_confidence=0.8,
        top_category="1",
        top_frame=11,
        top_bbox=[0.2, 0.2, 0.4, 0.4],
        first_interesting_frame=6,
        last_interesting_frame=21,
        num_detections=1,
        failure=None,
    )

    vid1_dir = build_video_storage_dir(canonical_root, "vid-001")
    vid2_dir = build_video_storage_dir(canonical_root, "vid-002")
    vid1_bucketed = vid1_dir / "interesting.avi"
    vid1_report = vid1_dir / "report.mp4"
    vid1_preview = vid1_dir / "preview.jpg"
    vid1_crop = vid1_dir / "preview_crop.jpg"
    vid2_bucketed = vid2_dir / "interesting.avi"
    vid2_report = vid2_dir / "report.mp4"
    vid2_preview = vid2_dir / "preview.jpg"
    vid2_crop = vid2_dir / "preview_crop.jpg"

    for artifact in (
        vid1_bucketed,
        vid1_report,
        vid1_preview,
        vid1_crop,
        vid2_bucketed,
        vid2_report,
        vid2_preview,
        vid2_crop,
    ):
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_bytes(b"artifact")

    record_processing_results(
        decisions=[decision1, decision2],
        staged_video_ids={"source.avi": "vid-002"},
        metadata_db_path=db_path,
        pipeline_version="0.1.1",
        mode="reprocess-existing",
        save_uninteresting_files=False,
        bucketed_video_paths={
            decision1.output_relative_path: vid1_bucketed,
            decision2.output_relative_path: vid2_bucketed,
        },
        report_video_paths={
            decision1.output_relative_path: vid1_report,
            decision2.output_relative_path: vid2_report,
        },
        preview_image_paths={
            decision1.output_relative_path: vid1_preview,
            decision2.output_relative_path: vid2_preview,
        },
        species_crop_paths={
            decision1.output_relative_path: vid1_crop,
            decision2.output_relative_path: vid2_crop,
        },
        video_ids_by_output_path={
            decision1.output_relative_path: "vid-001",
            decision2.output_relative_path: "vid-002",
        },
    )

    with sqlite3.connect(db_path) as connection:
        vid1_rows = connection.execute(
            "SELECT artifact_type, path FROM artifacts WHERE video_id=? ORDER BY artifact_type",
            ("vid-001",),
        ).fetchall()
        vid2_rows = connection.execute(
            "SELECT artifact_type, path FROM artifacts WHERE video_id=? ORDER BY artifact_type",
            ("vid-002",),
        ).fetchall()

    assert dict(vid1_rows) == {
        "bucketed_video": str(vid1_bucketed),
        "preview_image": str(vid1_preview),
        "report_video": str(vid1_report),
        "species_crop": str(vid1_crop),
    }
    assert dict(vid2_rows) == {
        "bucketed_video": str(vid2_bucketed),
        "preview_image": str(vid2_preview),
        "report_video": str(vid2_report),
        "species_crop": str(vid2_crop),
    }
