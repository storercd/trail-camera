"""Tests for SQLite metadata store initialization."""

import sqlite3
from pathlib import Path

import pytest

from metadata_store import (
    ProcessingStateRecord,
    SpeciesClassificationRecord,
    VideoCatalogRecord,
    delete_species_classification_record,
    demote_videos_to_uninteresting,
    fetch_catalog_snapshot,
    get_stored_original_paths,
    get_stored_original_records,
    initialize_metadata_store,
    purge_species_classification_labels,
    set_video_favorite,
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
    assert _table_exists(db_path, "favorites")


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


def test_set_video_favorite_should_toggle_and_validate_video_id(tmp_path: Path) -> None:
    """Persist and remove favorite records for known videos only."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="fav001",
            original_filename="PICT0700.AVI",
            capture_date="2026-05-05",
            filesize_bytes=100,
            source_ext=".avi",
            stored_original_path="/tmp/videos/fa/v0/fav001/source.avi",
        ),
    )

    assert set_video_favorite(db_path, "missing", True) is False
    assert set_video_favorite(db_path, "fav001", True) is True

    snapshot = fetch_catalog_snapshot(db_path)
    assert len(snapshot["favorites"]) == 1
    assert snapshot["favorites"][0]["video_id"] == "fav001"

    assert set_video_favorite(db_path, "fav001", False) is True
    snapshot_after_clear = fetch_catalog_snapshot(db_path)
    assert snapshot_after_clear["favorites"] == []


def test_purge_species_classification_labels_should_preserve_favorites(tmp_path: Path) -> None:
    """Delete matching species rows except for favorited videos."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    for video_id in ("fav001", "drop001"):
        upsert_video_record(
            db_path,
            VideoCatalogRecord(
                video_id=video_id,
                original_filename=f"{video_id}.avi",
                capture_date="2026-05-05",
                filesize_bytes=100,
                source_ext=".avi",
                stored_original_path=f"/tmp/videos/{video_id}/source.avi",
            ),
        )
        upsert_species_classification_record(
            db_path=db_path,
            record=SpeciesClassificationRecord(
                video_id=video_id,
                top_label="blank",
                top_score=0.95,
                top_raw_class="animal;blank",
                candidates_json='[{"label":"blank","score":0.95}]',
            ),
        )

    assert set_video_favorite(db_path, "fav001", True) is True

    deleted_rows = purge_species_classification_labels(db_path, ["blank"], preserve_favorites=True)

    assert deleted_rows == 1
    snapshot = fetch_catalog_snapshot(db_path)
    remaining_video_ids = {row["video_id"] for row in snapshot["species_classifications"]}
    assert remaining_video_ids == {"fav001"}


def test_demote_videos_to_uninteresting_should_clear_report_artifacts(tmp_path: Path) -> None:
    """Demote videos and remove report-visible artifacts from the catalog."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    for video_id in ("drop001", "keep001"):
        upsert_video_record(
            db_path,
            VideoCatalogRecord(
                video_id=video_id,
                original_filename=f"{video_id}.avi",
                capture_date="2026-05-05",
                filesize_bytes=100,
                source_ext=".avi",
                stored_original_path=f"/tmp/videos/{video_id}/source.avi",
            ),
        )

        upsert_processing_state_record(
            db_path,
            ProcessingStateRecord(
                video_id=video_id,
                pipeline_version="0.1.0",
                mode="new-only",
                bucket="interesting",
                top_confidence=0.9,
                top_category="1",
                top_frame=12,
                status="processed",
            ),
        )

    for artifact_type in ("bucketed_video", "report_video", "preview_image", "species_crop"):
        artifact_path = tmp_path / f"drop001-{artifact_type}.bin"
        artifact_path.write_bytes(b"x")
        sync_artifact_path(db_path, "drop001", artifact_type, artifact_path)

    updated_rows, deleted_artifacts = demote_videos_to_uninteresting(db_path, ["drop001"])

    assert updated_rows == 1
    assert deleted_artifacts == 4
    snapshot = fetch_catalog_snapshot(db_path)
    processing_by_video = {row["video_id"]: row for row in snapshot["processing_state"]}

    assert processing_by_video["drop001"]["bucket"] == "uninteresting"
    assert processing_by_video["keep001"]["bucket"] == "interesting"
    assert snapshot["artifacts"] == []
    assert list(tmp_path.glob("drop001-*.bin")) == []
