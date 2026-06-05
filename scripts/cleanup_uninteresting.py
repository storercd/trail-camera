"""Cleanup utility to demote configured uninteresting detections and delete artifacts."""

from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from metadata_store import purge_species_classification_labels
from pipeline_config import DEFAULT_CONFIG_PATH, build_run_paths, load_config


@dataclass
class CleanupTarget:
    """Matched video row selected for cleanup."""

    video_id: str
    stored_original_path: Path
    top_category: str | None
    top_label: str | None
    bucket: str | None
    derivative_artifact_count: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed cleanup CLI arguments.
    """
    parser = argparse.ArgumentParser(
        description="Mark configured uninteresting videos and remove generated output artifacts.",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML configuration file.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List cleanup matches without modifying files or database rows.",
    )
    parser.add_argument(
        "--include-already-cleaned",
        action="store_true",
        help=(
            "Include matches that are already uninteresting and have no remaining derivative artifacts."
        ),
    )
    return parser.parse_args()


def _normalize_categories(values: list[str]) -> set[str]:
    return {value.strip() for value in values if value.strip()}


def _normalize_labels(values: list[str]) -> set[str]:
    return {value.strip().lower() for value in values if value.strip()}


def _unlink_if_exists(path: Path) -> bool:
    if path.exists() and path.is_file():
        path.unlink()
        return True
    return False


def _build_where_clause(categories: set[str], labels: set[str]) -> tuple[str, list[str]]:
    clauses: list[str] = []
    params: list[str] = []
    if categories:
        placeholders = ",".join("?" for _ in categories)
        clauses.append(f"ps.top_category IN ({placeholders})")
        params.extend(sorted(categories))
    if labels:
        placeholders = ",".join("?" for _ in labels)
        clauses.append(f"LOWER(COALESCE(sc.top_label, '')) IN ({placeholders})")
        params.extend(sorted(labels))
    if not clauses:
        return "", []
    return "WHERE (" + " OR ".join(clauses) + ") AND f.video_id IS NULL", params


def load_cleanup_targets(
    db_path: Path,
    excluded_categories: set[str],
    uninteresting_labels: set[str],
) -> list[CleanupTarget]:
    """Fetch videos matching configured uninteresting filters.

    Returns:
        list[CleanupTarget]: Matched catalog rows and derivative artifact counts.
    """
    where_clause, params = _build_where_clause(excluded_categories, uninteresting_labels)
    if not where_clause:
        return []

    query = f"""
        SELECT
            v.video_id,
            v.stored_original_path,
            ps.top_category,
            sc.top_label,
            ps.bucket,
            SUM(
                CASE WHEN a.artifact_type IN ('bucketed_video', 'report_video', 'preview_image', 'species_crop')
                THEN 1 ELSE 0 END
            ) AS derivative_artifact_count
        FROM videos v
        LEFT JOIN processing_state ps ON ps.video_id = v.video_id
        LEFT JOIN species_classifications sc ON sc.video_id = v.video_id
        LEFT JOIN artifacts a ON a.video_id = v.video_id
        LEFT JOIN favorites f ON f.video_id = v.video_id
        {where_clause}
        GROUP BY v.video_id, v.stored_original_path, ps.top_category, sc.top_label, ps.bucket
        ORDER BY v.video_id ASC
    """
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(query, params).fetchall()

    return [
        CleanupTarget(
            video_id=str(row[0]),
            stored_original_path=Path(str(row[1])),
            top_category=str(row[2]) if row[2] is not None else None,
            top_label=str(row[3]) if row[3] is not None else None,
            bucket=str(row[4]) if row[4] is not None else None,
            derivative_artifact_count=int(row[5] or 0),
        )
        for row in rows
    ]


def _is_target_pending(target: CleanupTarget, save_uninteresting_files: bool) -> bool:
    """Return whether a matched target still has cleanup work pending."""
    if target.bucket != "uninteresting":
        return True
    if target.derivative_artifact_count > 0:
        return True
    if not save_uninteresting_files and target.stored_original_path.exists():
        return True
    return False


def split_targets_by_pending_state(
    targets: list[CleanupTarget],
    save_uninteresting_files: bool,
) -> tuple[list[CleanupTarget], list[CleanupTarget]]:
    """Split targets into pending-work and already-cleaned buckets.

    Returns:
        tuple[list[CleanupTarget], list[CleanupTarget]]: Pending targets and already-cleaned targets.
    """
    pending: list[CleanupTarget] = []
    already_cleaned: list[CleanupTarget] = []
    for target in targets:
        if _is_target_pending(target, save_uninteresting_files):
            pending.append(target)
        else:
            already_cleaned.append(target)
    return pending, already_cleaned


def cleanup_targets(db_path: Path, targets: list[CleanupTarget], save_uninteresting_files: bool) -> tuple[int, int]:
    """Apply DB updates and file cleanup for matched videos.

    Returns:
        tuple[int, int]: (deleted_artifact_files, deleted_source_files).
    """
    if not targets:
        return 0, 0

    deleted_artifact_files = 0
    deleted_source_files = 0
    video_ids = [target.video_id for target in targets]
    updated_at = datetime.now(UTC).isoformat()

    with sqlite3.connect(db_path) as connection:
        for video_id in video_ids:
            artifact_rows = connection.execute(
                """
                SELECT path FROM artifacts
                WHERE video_id = ?
                  AND artifact_type IN (
                      'bucketed_video',
                      'report_video',
                      'preview_image',
                      'species_crop'
                  )
                """,
                (video_id,),
            ).fetchall()
            for artifact_row in artifact_rows:
                artifact_path = Path(str(artifact_row[0]))
                if _unlink_if_exists(artifact_path):
                    deleted_artifact_files += 1

            connection.execute(
                """
                DELETE FROM artifacts
                WHERE video_id = ?
                  AND artifact_type IN (
                      'bucketed_video',
                      'report_video',
                      'preview_image',
                      'species_crop'
                  )
                """,
                (video_id,),
            )
            connection.execute(
                """
                UPDATE processing_state
                SET bucket = 'uninteresting', processed_at = ?
                WHERE video_id = ?
                """,
                (updated_at, video_id),
            )

        connection.commit()

    if not save_uninteresting_files:
        for target in targets:
            if _unlink_if_exists(target.stored_original_path):
                deleted_source_files += 1

    return deleted_artifact_files, deleted_source_files


def main() -> int:
    """Run configured uninteresting cleanup against the existing catalog.

    Returns:
        int: Process exit code.
    """
    args = parse_args()
    config_path = Path(args.config)
    config = load_config(config_path)
    paths = build_run_paths(config_path, config)

    excluded_categories = _normalize_categories(config.excluded_megadetector_categories)
    uninteresting_labels = _normalize_labels(config.uninteresting_species_labels)
    all_matches = load_cleanup_targets(
        db_path=paths.metadata_db_path,
        excluded_categories=excluded_categories,
        uninteresting_labels=uninteresting_labels,
    )
    pending_targets, already_cleaned_targets = split_targets_by_pending_state(
        targets=all_matches,
        save_uninteresting_files=config.save_uninteresting_files,
    )
    targets = all_matches if args.include_already_cleaned else pending_targets

    print(
        "Configured cleanup filters: "
        f"excluded_categories={sorted(excluded_categories)}, "
        f"uninteresting_species_labels={sorted(uninteresting_labels)}"
    )
    print(
        "Matches summary: "
        f"all={len(all_matches)}, pending={len(pending_targets)}, already_cleaned={len(already_cleaned_targets)}"
    )
    if not args.include_already_cleaned:
        print("Showing pending matches only. Use --include-already-cleaned to view all matches.")
    print(f"Matched videos: {len(targets)}")
    if not targets:
        return 0

    for target in targets[:20]:
        print(
            f"- {target.video_id} "
            "("
            f"top_category={target.top_category or 'n/a'}, "
            f"top_label={target.top_label or 'n/a'}, "
            f"bucket={target.bucket or 'n/a'}, "
            f"derivative_artifacts={target.derivative_artifact_count}"
            ")"
        )
    if len(targets) > 20:
        print(f"... {len(targets) - 20} additional matches")

    if args.dry_run:
        print("Dry run complete. No changes were made.")
        return 0

    deleted_artifact_files, deleted_source_files = cleanup_targets(
        db_path=paths.metadata_db_path,
        targets=targets,
        save_uninteresting_files=config.save_uninteresting_files,
    )
    deleted_species_rows = purge_species_classification_labels(
        db_path=paths.metadata_db_path,
        labels=config.uninteresting_species_labels,
        preserve_favorites=True,
    )
    print(
        "Cleanup complete. "
        f"deleted_artifact_files={deleted_artifact_files}, deleted_source_files={deleted_source_files}, "
        f"deleted_species_rows={deleted_species_rows}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
