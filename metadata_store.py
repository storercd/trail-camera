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


@dataclass
class SpeciesClassificationRecord:
    """Stored SpeciesNet classification payload for one canonical video."""

    video_id: str
    top_label: str | None
    top_score: float | None
    top_raw_class: str | None
    candidates_json: str


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
        # Forward-compatible migration for repositories with existing catalogs.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS species_classifications (
                video_id TEXT PRIMARY KEY,
                top_label TEXT,
                top_score REAL,
                top_raw_class TEXT,
                candidates_json TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS favorites (
                video_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
            """
        )
        connection.commit()


def ensure_favorites_table(db_path: Path) -> None:
    """Ensure the favorites table exists for legacy catalogs.

    This keeps report queries compatible with catalogs created before
    the favorites feature was introduced.
    """
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS favorites (
                video_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                FOREIGN KEY (video_id) REFERENCES videos(video_id) ON DELETE CASCADE
            )
            """
        )
        connection.commit()


def set_video_favorite(db_path: Path, video_id: str, is_favorite: bool) -> bool:
    """Set or clear favorite state for a known video.

    Returns:
        bool: True when the video exists and was updated; False otherwise.
    """
    now_utc = datetime.now(UTC).isoformat()
    ensure_favorites_table(db_path)
    with sqlite3.connect(db_path) as connection:
        exists = connection.execute(
            "SELECT 1 FROM videos WHERE video_id=? LIMIT 1",
            (video_id,),
        ).fetchone()
        if exists is None:
            return False
        if is_favorite:
            connection.execute(
                """
                INSERT INTO favorites (video_id, created_at)
                VALUES (?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    created_at=excluded.created_at
                """,
                (video_id, now_utc),
            )
        else:
            connection.execute(
                "DELETE FROM favorites WHERE video_id=?",
                (video_id,),
            )
        connection.commit()
    return True


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


def delete_video_records(db_path: Path, video_ids: set[str]) -> int:
    """Delete catalog rows for the given video IDs.

    Args:
        db_path: Metadata sqlite database path.
        video_ids: Canonical video identifiers to delete.

    Returns:
        int: Number of video rows deleted.
    """
    if not video_ids:
        return 0

    deleted_count = 0
    with sqlite3.connect(db_path) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        for video_id in sorted(video_ids):
            deleted_count += connection.execute(
                "DELETE FROM videos WHERE video_id=?",
                (video_id,),
            ).rowcount
        connection.commit()
    return deleted_count


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


def upsert_species_classification_record(
    db_path: Path,
    record: SpeciesClassificationRecord,
) -> None:
    """Insert or update SpeciesNet classification metadata for one video."""
    now_utc = datetime.now(UTC).isoformat()
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            INSERT INTO species_classifications (
                video_id,
                top_label,
                top_score,
                top_raw_class,
                candidates_json,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(video_id) DO UPDATE SET
                top_label=excluded.top_label,
                top_score=excluded.top_score,
                top_raw_class=excluded.top_raw_class,
                candidates_json=excluded.candidates_json,
                updated_at=excluded.updated_at
            """,
            (
                record.video_id,
                record.top_label,
                record.top_score,
                record.top_raw_class,
                record.candidates_json,
                now_utc,
            ),
        )
        connection.commit()


def delete_species_classification_record(db_path: Path, video_id: str) -> None:
    """Delete SpeciesNet classification metadata for one video."""
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "DELETE FROM species_classifications WHERE video_id=?",
            (video_id,),
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
        species_classifications = [
            dict(row)
            for row in connection.execute(
                "SELECT * FROM species_classifications ORDER BY updated_at ASC"
            )
        ]
        favorites = [
            dict(row)
            for row in connection.execute("SELECT * FROM favorites ORDER BY created_at ASC")
        ]

    return {
        "videos": videos,
        "processing_state": processing_state,
        "artifacts": artifacts,
        "species_classifications": species_classifications,
        "favorites": favorites,
    }


def list_catalog_videos(
    db_path: Path,
    current_pipeline_version: str,
    date_from: str | None = None,
    date_to: str | None = None,
    bucket: str | None = None,
    species: str | None = None,
    min_confidence: float | None = None,
    max_confidence: float | None = None,
    needs_reprocess: bool | None = None,
    is_favorite: bool | None = None,
    sort_by: str = "capture_date",
    sort_dir: str = "desc",
    page: int = 1,
    page_size: int = 25,
) -> tuple[list[dict[str, Any]], int]:
    """Return paginated catalog rows for the reporting app list view."""
    ensure_favorites_table(db_path)
    sort_columns = {
        "capture_date": "COALESCE(v.capture_date, '')",
        "processed_at": "COALESCE(ps.processed_at, '')",
        "species": "LOWER(COALESCE(sc.top_label, ''))",
        "confidence": "COALESCE(sc.top_score, ps.top_confidence, -1)",
        "bucket": "LOWER(COALESCE(ps.bucket, ''))",
    }
    order_column = sort_columns.get(sort_by, sort_columns["capture_date"])
    order_direction = "ASC" if str(sort_dir).lower() == "asc" else "DESC"

    filters: list[str] = []
    params: list[Any] = []
    if date_from:
        filters.append("v.capture_date >= ?")
        params.append(date_from)
    if date_to:
        filters.append("v.capture_date <= ?")
        params.append(date_to)
    if bucket:
        filters.append("ps.bucket = ?")
        params.append(bucket)
    if species:
        filters.append("LOWER(COALESCE(sc.top_label, '')) LIKE ?")
        params.append(f"%{species.lower()}%")
    if min_confidence is not None:
        filters.append("COALESCE(sc.top_score, ps.top_confidence) >= ?")
        params.append(min_confidence)
    if max_confidence is not None:
        filters.append("COALESCE(sc.top_score, ps.top_confidence) <= ?")
        params.append(max_confidence)
    if needs_reprocess is True:
        filters.append("(ps.pipeline_version IS NULL OR ps.pipeline_version != ?)")
        params.append(current_pipeline_version)
    elif needs_reprocess is False:
        filters.append("ps.pipeline_version = ?")
        params.append(current_pipeline_version)
    if is_favorite is True:
        filters.append("f.video_id IS NOT NULL")
    elif is_favorite is False:
        filters.append("f.video_id IS NULL")

    where_clause = f"WHERE {' AND '.join(filters)}" if filters else ""
    safe_page = max(1, int(page))
    safe_page_size = max(1, min(int(page_size), 100))
    offset = (safe_page - 1) * safe_page_size

    count_sql = f"""
        SELECT COUNT(DISTINCT v.video_id)
        FROM videos v
        LEFT JOIN processing_state ps ON ps.video_id = v.video_id
        LEFT JOIN species_classifications sc ON sc.video_id = v.video_id
        LEFT JOIN favorites f ON f.video_id = v.video_id
        {where_clause}
    """

    rows_sql = f"""
        SELECT
            v.video_id,
            v.original_filename,
            v.capture_date,
            v.filesize_bytes,
            v.stored_original_path,
            ps.pipeline_version,
            ps.processed_at,
            ps.mode,
            ps.bucket,
            ps.top_confidence,
            ps.top_category,
            ps.top_frame,
            ps.status,
            sc.top_label,
            sc.top_score,
            sc.top_raw_class,
            sc.candidates_json,
            CASE WHEN f.video_id IS NULL THEN 0 ELSE 1 END AS is_favorite,
            MAX(CASE WHEN a.artifact_type = 'bucketed_video' THEN a.path END) AS bucketed_video_path,
            MAX(CASE WHEN a.artifact_type = 'report_video' THEN a.path END) AS report_video_path,
            MAX(CASE WHEN a.artifact_type = 'preview_image' THEN a.path END) AS preview_image_path,
            MAX(CASE WHEN a.artifact_type = 'species_crop' THEN a.path END) AS species_crop_path
        FROM videos v
        LEFT JOIN processing_state ps ON ps.video_id = v.video_id
        LEFT JOIN species_classifications sc ON sc.video_id = v.video_id
        LEFT JOIN favorites f ON f.video_id = v.video_id
        LEFT JOIN artifacts a ON a.video_id = v.video_id
        {where_clause}
        GROUP BY
            v.video_id,
            v.original_filename,
            v.capture_date,
            v.filesize_bytes,
            v.stored_original_path,
            ps.pipeline_version,
            ps.processed_at,
            ps.mode,
            ps.bucket,
            ps.top_confidence,
            ps.top_category,
            ps.top_frame,
            ps.status,
            sc.top_label,
            sc.top_score,
            sc.top_raw_class,
            sc.candidates_json,
            f.video_id
        ORDER BY {order_column} {order_direction}, v.video_id ASC
        LIMIT ? OFFSET ?
    """

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        total_count = int(connection.execute(count_sql, params).fetchone()[0])
        rows = [
            dict(row)
            for row in connection.execute(rows_sql, [*params, safe_page_size, offset]).fetchall()
        ]

    return rows, total_count


def get_catalog_video_detail(
    db_path: Path,
    video_id: str,
    current_pipeline_version: str,
) -> dict[str, Any] | None:
    """Return one video detail row for the reporting app."""
    ensure_favorites_table(db_path)
    # Use a direct detail query so list ordering/filtering logic stays separate.
    detail_sql = """
        SELECT
            v.video_id,
            v.original_filename,
            v.capture_date,
            v.filesize_bytes,
            v.stored_original_path,
            ps.pipeline_version,
            ps.processed_at,
            ps.mode,
            ps.bucket,
            ps.top_confidence,
            ps.top_category,
            ps.top_frame,
            ps.status,
            sc.top_label,
            sc.top_score,
            sc.top_raw_class,
            sc.candidates_json,
            CASE WHEN f.video_id IS NULL THEN 0 ELSE 1 END AS is_favorite,
            MAX(CASE WHEN a.artifact_type = 'bucketed_video' THEN a.path END) AS bucketed_video_path,
            MAX(CASE WHEN a.artifact_type = 'report_video' THEN a.path END) AS report_video_path,
            MAX(CASE WHEN a.artifact_type = 'preview_image' THEN a.path END) AS preview_image_path,
            MAX(CASE WHEN a.artifact_type = 'species_crop' THEN a.path END) AS species_crop_path
        FROM videos v
        LEFT JOIN processing_state ps ON ps.video_id = v.video_id
        LEFT JOIN species_classifications sc ON sc.video_id = v.video_id
        LEFT JOIN favorites f ON f.video_id = v.video_id
        LEFT JOIN artifacts a ON a.video_id = v.video_id
        WHERE v.video_id = ?
        GROUP BY
            v.video_id,
            v.original_filename,
            v.capture_date,
            v.filesize_bytes,
            v.stored_original_path,
            ps.pipeline_version,
            ps.processed_at,
            ps.mode,
            ps.bucket,
            ps.top_confidence,
            ps.top_category,
            ps.top_frame,
            ps.status,
            sc.top_label,
            sc.top_score,
            sc.top_raw_class,
                sc.candidates_json,
                f.video_id
    """

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(detail_sql, (video_id,)).fetchone()
    if row is None:
        return None

    payload = dict(row)
    payload["needs_reprocess"] = payload.get("pipeline_version") != current_pipeline_version
    payload["is_favorite"] = bool(payload.get("is_favorite"))
    return payload
