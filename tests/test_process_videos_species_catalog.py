"""Tests for syncing species classification JSON into SQLite catalog."""

import json
from argparse import Namespace
from pathlib import Path

import process_videos
from metadata_store import (
    ProcessingStateRecord,
    VideoCatalogRecord,
    fetch_catalog_snapshot,
    initialize_metadata_store,
    set_video_favorite,
    sync_artifact_path,
    upsert_processing_state_record,
    upsert_video_record,
)
from pipeline_models import VideoDecision
from process_videos import (
    apply_uninteresting_species_filters,
    purge_uninteresting_species_records,
    sync_species_classifications_to_catalog,
)


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


def test_apply_uninteresting_species_filters_should_skip_protected_paths(tmp_path: Path) -> None:
    """Keep protected outputs interesting even when labels match demotion rules."""
    report_path = tmp_path / "species_classifications.json"
    report_path.write_text(
        json.dumps(
            {
                "entries": [
                    {
                        "source_relative_path": "20260505-camera.avi",
                        "top_classification": {
                            "label": "blank",
                            "score": 0.99,
                            "raw_class": "animal;blank",
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
        uninteresting_species_labels=["blank"],
        protected_output_paths={"20260505-camera.avi"},
    )

    assert demoted == set()
    assert decisions[0].bucket == "interesting"


def test_purge_uninteresting_species_records_should_preserve_favorites(tmp_path: Path) -> None:
    """Remove matching species rows and hide those videos while keeping favorites intact."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    for video_id in ("fav001", "drop001"):
        upsert_video_record(
            db_path,
            VideoCatalogRecord(
                video_id=video_id,
                original_filename=f"{video_id}.avi",
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
                        "source_relative_path": "fav.avi",
                        "top_classification": {
                            "label": "blank",
                            "score": 0.9,
                            "raw_class": "animal;blank",
                        },
                        "candidates": [{"label": "blank", "score": 0.9}],
                    },
                    {
                        "source_relative_path": "drop.avi",
                        "top_classification": {
                            "label": "blank",
                            "score": 0.8,
                            "raw_class": "animal;blank",
                        },
                        "candidates": [{"label": "blank", "score": 0.8}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    sync_species_classifications_to_catalog(
        species_classification_report_path=report_path,
        metadata_db_path=db_path,
        video_ids_by_output_path={"fav.avi": "fav001", "drop.avi": "drop001"},
    )
    for video_id in ("fav001", "drop001"):
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
    artifact_path = tmp_path / "drop-report.mp4"
    artifact_path.write_bytes(b"x")
    sync_artifact_path(db_path, "drop001", "report_video", artifact_path)
    assert set_video_favorite(db_path, "fav001", True) is True

    deleted_rows = purge_uninteresting_species_records(db_path, ["blank"])

    assert deleted_rows == 1
    snapshot = fetch_catalog_snapshot(db_path)
    remaining_video_ids = {row["video_id"] for row in snapshot["species_classifications"]}
    assert remaining_video_ids == {"fav001"}
    processing_by_video = {row["video_id"]: row for row in snapshot["processing_state"]}
    assert processing_by_video["drop001"]["bucket"] == "uninteresting"
    assert processing_by_video["fav001"]["bucket"] == "interesting"
    assert artifact_path.exists() is False


def test_main_should_purge_species_rows_when_no_videos_selected(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Run catalog maintenance on the zero-video branch before exiting."""
    config_path = tmp_path / "config.yaml"
    output_dir = tmp_path / "output"
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    config_path.write_text(
        "\n".join(
            [
                f"input_dir: {input_dir}",
                f"output_dir: {output_dir}",
                "metadata_db_path: metadata/catalog.sqlite3",
                "pipeline_version: test-v1",
                "model: MDV5A",
                "frame_sample: 5",
                "interesting_threshold: 0.7",
                'interesting_categories: ["1"]',
                'excluded_megadetector_categories: []',
                'uninteresting_species_labels: ["blank"]',
                "move_files: false",
                "save_uninteresting_files: false",
                "clip_interesting_videos: true",
                "recursive: false",
                "detector_verbose: false",
                "generate_html_report: false",
                "auto_open_html_report: false",
                "write_json_exports: false",
                "preview_output_dir: preview_frames",
                "preview_include_uninteresting: false",
                'speciesnet_model: ""',
                "speciesnet_geofence: false",
                "speciesnet_label_in_filename: true",
                "species_crop_output_dir: preview_species_crops",
                "species_crop_padding: 0.15",
                "capture_date_source: filesystem",
            ]
        ),
        encoding="utf-8",
    )

    db_path = output_dir / "metadata" / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    for video_id in ("fav001", "drop001"):
        upsert_video_record(
            db_path,
            VideoCatalogRecord(
                video_id=video_id,
                original_filename=f"{video_id}.avi",
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
                        "source_relative_path": "fav.avi",
                        "top_classification": {
                            "label": "blank",
                            "score": 0.9,
                            "raw_class": "animal;blank",
                        },
                        "candidates": [{"label": "blank", "score": 0.9}],
                    },
                    {
                        "source_relative_path": "drop.avi",
                        "top_classification": {
                            "label": "blank",
                            "score": 0.8,
                            "raw_class": "animal;blank",
                        },
                        "candidates": [{"label": "blank", "score": 0.8}],
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    sync_species_classifications_to_catalog(
        species_classification_report_path=report_path,
        metadata_db_path=db_path,
        video_ids_by_output_path={"fav.avi": "fav001", "drop.avi": "drop001"},
    )
    assert set_video_favorite(db_path, "fav001", True) is True

    monkeypatch.setattr(
        process_videos,
        "parse_args",
        lambda: Namespace(config=str(config_path), log_level="INFO", mode="new-only"),
    )

    exit_code = process_videos.main()

    assert exit_code == 0
    snapshot = fetch_catalog_snapshot(db_path)
    remaining_video_ids = {row["video_id"] for row in snapshot["species_classifications"]}
    assert remaining_video_ids == {"fav001"}
