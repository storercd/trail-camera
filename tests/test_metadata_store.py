"""Tests for SQLite metadata store initialization."""

import sqlite3
from pathlib import Path

import pytest

from metadata_store import (
    ProcessingStateRecord,
    SpeciesClassificationRecord,
    VideoCatalogRecord,
    delete_species_classification_record,
    fetch_catalog_snapshot,
    get_stored_original_paths,
    get_stored_original_records,
    initialize_metadata_store,
    sync_artifact_path,
    upsert_processing_state_record,
    upsert_species_classification_record,
    upsert_video_record,
    video_exists,
)


def _table_exists(db_path: Path, table_name: str) -> bool:
    with sqlite3.connect(db_path) as connection:
        result = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        ).fetchone()
    return result is not None


def test_initialize_metadata_store_should_create_expected_tables(tmp_path: Path) -> None:
    """Create the sqlite catalog and required core tables."""
    db_path = tmp_path / "catalog.sqlite3"

    initialize_metadata_store(db_path)

    assert db_path.exists()
    assert _table_exists(db_path, "videos")
    assert _table_exists(db_path, "processing_state")
    assert _table_exists(db_path, "artifacts")
    assert _table_exists(db_path, "species_classifications")


def test_initialize_metadata_store_should_fail_when_migration_missing(tmp_path: Path) -> None:
    """Raise SystemExit when the migration bootstrap file cannot be found."""
    db_path = tmp_path / "catalog.sqlite3"
    missing_migrations_dir = tmp_path / "missing_migrations"

    with pytest.raises(SystemExit):
        initialize_metadata_store(db_path, migrations_dir=missing_migrations_dir)


def test_upsert_video_record_should_insert_and_update(tmp_path: Path) -> None:
    """Insert and then update a canonical video catalog row by video_id."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="abc123",
            original_filename="PICT0001.AVI",
            capture_date="2026-05-01",
            filesize_bytes=123,
            source_ext=".avi",
            stored_original_path="/tmp/videos/ab/c1/abc123/source.avi",
        ),
    )
    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="abc123",
            original_filename="PICT0001-copy.AVI",
            capture_date="2026-05-01",
            filesize_bytes=123,
            source_ext=".avi",
            stored_original_path="/tmp/videos/ab/c1/abc123/source.avi",
        ),
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT original_filename, filesize_bytes, source_ext FROM videos WHERE video_id=?",
            ("abc123",),
        ).fetchone()

    assert row == ("PICT0001-copy.AVI", 123, ".avi")


def test_video_exists_and_stored_paths_should_reflect_catalog(tmp_path: Path) -> None:
    """Return identity existence and canonical original paths from videos table."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="xyz789",
            original_filename="PICT0100.AVI",
            capture_date="2026-05-03",
            filesize_bytes=456,
            source_ext=".avi",
            stored_original_path="/tmp/videos/xy/z7/xyz789/source.avi",
        ),
    )

    assert video_exists(db_path, "xyz789") is True
    assert video_exists(db_path, "missing") is False
    assert get_stored_original_paths(db_path) == [Path("/tmp/videos/xy/z7/xyz789/source.avi")]
    assert get_stored_original_records(db_path) == [
        ("xyz789", Path("/tmp/videos/xy/z7/xyz789/source.avi")),
    ]


def test_upsert_processing_state_record_should_write_latest_values(tmp_path: Path) -> None:
    """Persist processing-state details keyed by canonical video ID."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="state001",
            original_filename="PICT0200.AVI",
            capture_date="2026-05-05",
            filesize_bytes=1,
            source_ext=".avi",
            stored_original_path="/tmp/videos/st/at/state001/source.avi",
        ),
    )

    upsert_processing_state_record(
        db_path,
        ProcessingStateRecord(
            video_id="state001",
            pipeline_version="0.1.0",
            mode="new-only",
            bucket="interesting",
            top_confidence=0.9,
            top_category="1",
            top_frame=12,
            status="processed",
        ),
    )

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT pipeline_version, mode, bucket, top_frame, status FROM processing_state WHERE video_id=?",
            ("state001",),
        ).fetchone()

    assert row == ("0.1.0", "new-only", "interesting", 12, "processed")


def test_sync_artifact_path_should_remove_stale_file_when_replaced(tmp_path: Path) -> None:
    """Delete previous artifact file when syncing a replacement path."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="art001",
            original_filename="PICT0300.AVI",
            capture_date="2026-05-05",
            filesize_bytes=1,
            source_ext=".avi",
            stored_original_path="/tmp/videos/ar/t0/art001/source.avi",
        ),
    )

    old_artifact = tmp_path / "old.mp4"
    new_artifact = tmp_path / "new.mp4"
    old_artifact.write_bytes(b"old")
    new_artifact.write_bytes(b"new")

    sync_artifact_path(db_path, "art001", "bucketed_video", old_artifact)
    sync_artifact_path(db_path, "art001", "bucketed_video", new_artifact)

    assert not old_artifact.exists()
    assert new_artifact.exists()

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT path FROM artifacts WHERE video_id=? AND artifact_type=?",
            ("art001", "bucketed_video"),
        ).fetchone()

    assert row == (str(new_artifact),)


def test_fetch_catalog_snapshot_should_return_table_payloads(tmp_path: Path) -> None:
    """Return dictionary snapshots for videos, processing_state, and artifacts."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="snap001",
            original_filename="PICT0400.AVI",
            capture_date="2026-05-05",
            filesize_bytes=100,
            source_ext=".avi",
            stored_original_path="/tmp/videos/sn/ap/snap001/source.avi",
        ),
    )
    upsert_processing_state_record(
        db_path,
        ProcessingStateRecord(
            video_id="snap001",
            pipeline_version="0.2.0",
            mode="new-only",
            bucket="interesting",
            top_confidence=0.8,
            top_category="1",
            top_frame=10,
            status="processed",
        ),
    )
    artifact = tmp_path / "artifact.mp4"
    artifact.write_bytes(b"x")
    sync_artifact_path(db_path, "snap001", "report_video", artifact)

    snapshot = fetch_catalog_snapshot(db_path)

    assert len(snapshot["videos"]) == 1
    assert len(snapshot["processing_state"]) == 1
    assert len(snapshot["artifacts"]) == 1
    assert len(snapshot["species_classifications"]) == 0
    assert snapshot["videos"][0]["video_id"] == "snap001"


def test_species_classification_record_should_upsert_and_delete(tmp_path: Path) -> None:
    """Persist and remove species classification records keyed by video ID."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="species001",
            original_filename="PICT0600.AVI",
            capture_date="2026-05-05",
            filesize_bytes=100,
            source_ext=".avi",
            stored_original_path="/tmp/videos/sp/ec/species001/source.avi",
        ),
    )

    upsert_species_classification_record(
        db_path=db_path,
        record=SpeciesClassificationRecord(
            video_id="species001",
            top_label="domestic dog",
            top_score=0.648,
            top_raw_class="animal;canid;domestic dog",
            candidates_json='[{"label":"domestic dog","score":0.648}]',
        ),
    )

    snapshot = fetch_catalog_snapshot(db_path)
    assert len(snapshot["species_classifications"]) == 1
    assert snapshot["species_classifications"][0]["video_id"] == "species001"

    delete_species_classification_record(db_path, "species001")

    snapshot_after_delete = fetch_catalog_snapshot(db_path)
    assert len(snapshot_after_delete["species_classifications"]) == 0
