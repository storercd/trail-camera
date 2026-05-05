"""SQLite metadata catalog bootstrap and migration helpers."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class VideoCatalogRecord:
    """Metadata row for canonical video identity storage."""

    video_id: str
    original_filename: str
    capture_date: str | None
    filesize_bytes: int
    source_ext: str
    stored_original_path: str


@dataclass
class ProcessingStateRecord:
    """Current processing-state row for one canonical video."""

    video_id: str
    pipeline_version: str
    mode: str
    bucket: str | None
    top_confidence: float | None
    top_category: str | None
    top_frame: int | None
    status: str


def initialize_metadata_store(db_path: Path, migrations_dir: Path | None = None) -> None:
    """Create metadata database and apply required bootstrap migrations.

    Args:
        db_path: Target sqlite database file path.
        migrations_dir: Directory containing SQL migration scripts.

    Raises:
        SystemExit: If the required migration script is missing.
    """
    migration_root = migrations_dir or Path(__file__).resolve().parent / "migrations"
    initial_schema_path = migration_root / "001_initial.sql"
    if not initial_schema_path.exists():
        raise SystemExit(f"Missing migration file: {initial_schema_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        with initial_schema_path.open("r", encoding="utf-8") as handle:
            schema_sql = handle.read()
        connection.executescript(schema_sql)
        connection.commit()


def upsert_video_record(db_path: Path, record: VideoCatalogRecord) -> None:
    """Insert or update canonical metadata for a discovered source video.

    Args:
        db_path: Metadata sqlite database path.
        record: Canonical video metadata to store.
    """
    now_utc = datetime.now(UTC).isoformat()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO videos (
                video_id,
                original_filename,
                capture_date,
                filesize_bytes,
                source_ext,
                stored_original_path,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                original_filename=excluded.original_filename,
                capture_date=excluded.capture_date,
                filesize_bytes=excluded.filesize_bytes,
                source_ext=excluded.source_ext,
                stored_original_path=excluded.stored_original_path
            """,
            (
                record.video_id,
                record.original_filename,
                record.capture_date,
                record.filesize_bytes,
                record.source_ext,
                record.stored_original_path,
                now_utc,
            ),
        )
        connection.commit()


def video_exists(db_path: Path, video_id: str) -> bool:
    """Return whether a video ID already exists in the catalog.

    Args:
        db_path: Metadata sqlite database path.
        video_id: Canonical video identifier.

    Returns:
        bool: True when the catalog already contains the given ID.
    """
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM videos WHERE video_id=? LIMIT 1",
            (video_id,),
        ).fetchone()
    return row is not None


def get_processing_bucket(db_path: Path, video_id: str) -> str | None:
    """Return the current processing bucket for a video ID.

    Args:
        db_path: Metadata sqlite database path.
        video_id: Canonical video identifier.

    Returns:
        str | None: Stored bucket value, or None when no state row exists.
    """
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            "SELECT bucket FROM processing_state WHERE video_id=? LIMIT 1",
            (video_id,),
        ).fetchone()
    if row is None:
        return None
    value = row[0]
    return str(value) if value is not None else None


def get_stored_original_paths(db_path: Path) -> list[Path]:
    """Fetch stored canonical original video paths from the catalog.

    Args:
        db_path: Metadata sqlite database path.

    Returns:
        list[Path]: Stored canonical original file paths.
    """
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT stored_original_path FROM videos ORDER BY created_at ASC",
        ).fetchall()
    return [Path(str(row[0])) for row in rows if row and row[0]]


def get_stored_original_records(db_path: Path) -> list[tuple[str, Path]]:
    """Fetch `(video_id, stored_original_path)` rows from catalog.

    Args:
        db_path: Metadata sqlite database path.

    Returns:
        list[tuple[str, Path]]: Video IDs with canonical original paths.
    """
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT video_id, stored_original_path FROM videos ORDER BY created_at ASC",
        ).fetchall()
    return [(str(row[0]), Path(str(row[1]))) for row in rows if row and row[0] and row[1]]


def upsert_processing_state_record(db_path: Path, record: ProcessingStateRecord) -> None:
    """Insert or update one processing-state row.

    Args:
        db_path: Metadata sqlite database path.
        record: Processing-state values for one video.
    """
    processed_at = datetime.now(UTC).isoformat()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO processing_state (
                video_id,
                pipeline_version,
                processed_at,
                mode,
                bucket,
                top_confidence,
                top_category,
                top_frame,
                status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                pipeline_version=excluded.pipeline_version,
                processed_at=excluded.processed_at,
                mode=excluded.mode,
                bucket=excluded.bucket,
                top_confidence=excluded.top_confidence,
                top_category=excluded.top_category,
                top_frame=excluded.top_frame,
                status=excluded.status
            """,
            (
                record.video_id,
                record.pipeline_version,
                processed_at,
                record.mode,
                record.bucket,
                record.top_confidence,
                record.top_category,
                record.top_frame,
                record.status,
            ),
        )
        connection.commit()


def sync_artifact_path(
    db_path: Path,
    video_id: str,
    artifact_type: str,
    artifact_path: Path | None,
) -> None:
    """Upsert artifact path and remove stale file when replacing/deleting.

    Args:
        db_path: Metadata sqlite database path.
        video_id: Canonical video identifier.
        artifact_type: Logical artifact key (for example `bucketed_video`).
        artifact_path: New artifact path; set to None to clear existing artifact.
    """
    now_utc = datetime.now(UTC).isoformat()
    with sqlite3.connect(db_path) as connection:
        existing = connection.execute(
            "SELECT path FROM artifacts WHERE video_id=? AND artifact_type=?",
            (video_id, artifact_type),
        ).fetchone()
        existing_path = Path(str(existing[0])) if existing and existing[0] else None

        if artifact_path is None:
            if existing_path is not None and existing_path.exists() and existing_path.is_file():
                existing_path.unlink()
            connection.execute(
                "DELETE FROM artifacts WHERE video_id=? AND artifact_type=?",
                (video_id, artifact_type),
            )
            connection.commit()
            return

        if (
            existing_path is not None
            and existing_path != artifact_path
            and existing_path.exists()
            and existing_path.is_file()
        ):
            existing_path.unlink()

        connection.execute(
            """
            INSERT INTO artifacts (video_id, artifact_type, path, updated_at, exists_flag)
            VALUES (?, ?, ?, ?, 1)
            ON CONFLICT(video_id, artifact_type) DO UPDATE SET
                path=excluded.path,
                updated_at=excluded.updated_at,
                exists_flag=excluded.exists_flag
            """,
            (video_id, artifact_type, str(artifact_path), now_utc),
        )
        connection.commit()


def fetch_catalog_snapshot(db_path: Path) -> dict[str, list[dict[str, Any]]]:
    """Fetch full catalog tables as JSON-serializable dictionaries.

    Args:
        db_path: Metadata sqlite database path.

    Returns:
        dict[str, list[dict[str, Any]]]: Snapshot payload for videos,
            processing_state, and artifacts tables.
    """
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        videos = [dict(row) for row in connection.execute("SELECT * FROM videos ORDER BY created_at ASC")]
        processing_state = [
            dict(row)
            for row in connection.execute("SELECT * FROM processing_state ORDER BY processed_at ASC")
        ]
        artifacts = [dict(row) for row in connection.execute("SELECT * FROM artifacts ORDER BY updated_at ASC")]

    return {
        "videos": videos,
        "processing_state": processing_state,
        "artifacts": artifacts,
    }
