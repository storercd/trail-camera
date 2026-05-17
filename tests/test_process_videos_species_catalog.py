"""Tests for syncing species classification JSON into SQLite catalog."""

import json
from pathlib import Path

from metadata_store import VideoCatalogRecord, fetch_catalog_snapshot, initialize_metadata_store, upsert_video_record
from pipeline_models import VideoDecision
from process_videos import apply_uninteresting_species_filters, sync_species_classifications_to_catalog


def test_sync_species_classifications_to_catalog_should_upsert_rows(tmp_path: Path) -> None:
    """Upsert species classifications for output-relative paths mapped to video IDs."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    for video_id in ("vid-001", "vid-002"):
        upsert_video_record(
            db_path,
            VideoCatalogRecord(
                video_id=video_id,
                original_filename="source.avi",
                capture_date="2026-05-05",
                filesize_bytes=1,
                source_ext=".avi",
                stored_original_path=str(tmp_path / video_id / "source.avi"),
            ),
        )

    report_path = tmp_path / "species_classifications.json"
    report_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "source_relative_path": "20260505-source.avi",
                        "top_classification": {
                            "label": "domestic dog",
                            "score": 0.648,
                            "raw_class": "animal;canid;domestic dog",
                        },
                        "candidates": [
                            {"label": "domestic dog", "score": 0.648},
                            {"label": "canid", "score": 0.325},
                        ],
                    },
                    {
                        "source_relative_path": "20260505-source.avi__vid002",
                        "top_classification": {
                            "label": "bird",
                            "score": 0.818,
                            "raw_class": "animal;bird",
                        },
                        "candidates": [
                            {"label": "bird", "score": 0.818},
                        ],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    sync_species_classifications_to_catalog(
        species_classification_report_path=report_path,
        metadata_db_path=db_path,
        video_ids_by_output_path={
            "20260505-source.avi": "vid-001",
            "20260505-source.avi__vid002": "vid-002",
        },
    )

    snapshot = fetch_catalog_snapshot(db_path)
    rows = snapshot["species_classifications"]
    assert len(rows) == 2
    by_video = {row["video_id"]: row for row in rows}
    assert by_video["vid-001"]["top_label"] == "domestic dog"
    assert by_video["vid-002"]["top_label"] == "bird"


def test_sync_species_classifications_to_catalog_should_delete_when_report_missing(
    tmp_path: Path,
) -> None:
    """Clear species rows for processed videos when the merged report is unavailable."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    for video_id in ("vid-001", "vid-002"):
        upsert_video_record(
            db_path,
            VideoCatalogRecord(
                video_id=video_id,
                original_filename="source.avi",
                capture_date="2026-05-05",
                filesize_bytes=1,
                source_ext=".avi",
                stored_original_path=str(tmp_path / video_id / "source.avi"),
            ),
        )

    # Seed one record first.
    report_path = tmp_path / "species_classifications.json"
    report_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "source_relative_path": "20260505-source.avi",
                        "top_classification": {"label": "dog", "score": 0.5, "raw_class": "animal;dog"},
                        "candidates": [{"label": "dog", "score": 0.5}],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    sync_species_classifications_to_catalog(
        species_classification_report_path=report_path,
        metadata_db_path=db_path,
        video_ids_by_output_path={"20260505-source.avi": "vid-001"},
    )

    report_path.unlink()
    sync_species_classifications_to_catalog(
        species_classification_report_path=report_path,
        metadata_db_path=db_path,
        video_ids_by_output_path={
            "20260505-source.avi": "vid-001",
            "20260505-source.avi__vid002": "vid-002",
        },
    )

    snapshot = fetch_catalog_snapshot(db_path)
    assert snapshot["species_classifications"] == []


def test_apply_uninteresting_species_filters_should_demote_matching_labels(tmp_path: Path) -> None:
    """Demote interesting decisions when top species label is configured as uninteresting."""
    report_path = tmp_path / "species_classifications.json"
    report_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "source_relative_path": "20260505-camera.avi",
                        "top_classification": {
                            "label": "domestic dog",
                            "score": 0.92,
                            "raw_class": "animal;canid;domestic dog",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    decisions = [
        VideoDecision(
            relative_path="camera.avi",
            output_relative_path="20260505-camera.avi",
            bucket="interesting",
            top_confidence=0.88,
            top_category="1",
            top_frame=12,
            top_bbox=[0.1, 0.1, 0.3, 0.3],
            first_interesting_frame=10,
            last_interesting_frame=15,
            num_detections=1,
            failure=None,
        )
    ]

    demoted = apply_uninteresting_species_filters(
        decisions=decisions,
        species_classification_report_path=report_path,
        uninteresting_species_labels=["domestic dog"],
    )

    assert demoted == {"20260505-camera.avi"}
    assert decisions[0].bucket == "uninteresting"
