"""Tests for SQLite-backed JSON export writing."""

import json
from pathlib import Path

from metadata_store import VideoCatalogRecord, initialize_metadata_store, upsert_video_record
from pipeline_models import AppConfig
from reporting import write_sqlite_snapshot_export


def test_write_sqlite_snapshot_export_should_write_summary_payload(tmp_path: Path) -> None:
    """Write summary JSON with SQLite snapshot and table-count metadata."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id="export001",
            original_filename="PICT0500.AVI",
            capture_date="2026-05-05",
            filesize_bytes=200,
            source_ext=".avi",
            stored_original_path="/tmp/videos/ex/po/export001/source.avi",
        ),
    )

    summary_path = tmp_path / "summary.json"
    config = AppConfig(
        input_dir=str(tmp_path / "input"),
        output_dir=str(tmp_path / "output"),
        metadata_db_path=str(db_path),
        pipeline_version="0.3.0",
        model="MDV5A",
        frame_sample=5,
        interesting_threshold=0.7,
        interesting_categories=["1", "2", "3"],
        move_files=False,
        save_uninteresting_files=False,
        clip_interesting_videos=True,
        recursive=True,
        detector_verbose=False,
        generate_html_report=True,
        auto_open_html_report=False,
        write_json_exports=True,
        preview_output_dir="preview_frames",
        preview_include_uninteresting=False,
        speciesnet_model="",
        speciesnet_geofence=False,
        speciesnet_label_in_filename=True,
        species_crop_output_dir="preview_species_crops",
        species_crop_padding=0.15,
        capture_date_source="filesystem",
        camera_date_profile=None,
        excluded_megadetector_categories=["3"],
        uninteresting_species_labels=["domestic dog"],
    )
    config_path = tmp_path / "config.yaml"
    config_path.write_text("input_dir: input\noutput_dir: output\n", encoding="utf-8")

    write_sqlite_snapshot_export(
        summary_path=summary_path,
        config=config,
        config_path=config_path,
        run_output_dir=tmp_path / "output",
        metadata_db_path=db_path,
    )

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["pipeline_version"] == "0.3.0"
    assert payload["counts"]["videos"] == 1
    assert payload["snapshot"]["videos"][0]["video_id"] == "export001"
