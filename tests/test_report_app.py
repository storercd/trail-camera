"""Tests for the local Flask reporting app."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from metadata_store import (
    ProcessingStateRecord,
    SpeciesClassificationRecord,
    VideoCatalogRecord,
    initialize_metadata_store,
    sync_artifact_path,
    upsert_processing_state_record,
    upsert_species_classification_record,
    upsert_video_record,
)
from report_app import create_app


def _seed_video(
    db_path: Path,
    root: Path,
    video_id: str,
    capture_date: str,
    bucket: str,
    pipeline_version: str,
    label: str,
    score: float,
) -> None:
    canonical_dir = root / "videos" / video_id[:2] / video_id[2:4] / video_id
    canonical_dir.mkdir(parents=True, exist_ok=True)
    source_path = canonical_dir / "source.avi"
    bucketed = canonical_dir / "interesting.avi"
    report_video = canonical_dir / "report.mp4"
    preview = canonical_dir / f"preview_species-{label.replace(' ', '_')}.jpg"
    crop = canonical_dir / "preview_crop.jpg"
    for path in (source_path, bucketed, report_video, preview, crop):
        path.write_bytes(b"artifact")

    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id=video_id,
            original_filename="source.avi",
            capture_date=capture_date,
            filesize_bytes=1,
            source_ext=".avi",
            stored_original_path=str(source_path),
        ),
    )
    upsert_processing_state_record(
        db_path,
        ProcessingStateRecord(
            video_id=video_id,
            pipeline_version=pipeline_version,
            mode="reprocess-existing",
            bucket=bucket,
            top_confidence=score,
            top_category="1",
            top_frame=10,
            status="processed",
        ),
    )
    upsert_species_classification_record(
        db_path=db_path,
        record=SpeciesClassificationRecord(
            video_id=video_id,
            top_label=label,
            top_score=score,
            top_raw_class=f"animal;{label}",
            candidates_json=json.dumps([
                {"label": label, "score": score},
                {"label": "backup", "score": 0.1},
            ]),
        ),
    )
    sync_artifact_path(db_path, video_id, "bucketed_video", bucketed)
    sync_artifact_path(db_path, video_id, "report_video", report_video)
    sync_artifact_path(db_path, video_id, "preview_image", preview)
    sync_artifact_path(db_path, video_id, "species_crop", crop)


def _write_config(tmp_path: Path, db_path: Path) -> Path:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            [
                f"input_dir: {tmp_path / 'input'}",
                f"output_dir: {tmp_path}",
                f"metadata_db_path: {db_path}",
                "pipeline_version: 0.2.0",
                "model: MDV5A",
                "frame_sample: 5",
                "interesting_threshold: 0.7",
                "interesting_categories: ['1', '2', '3']",
                "move_files: false",
                "save_uninteresting_files: false",
                "clip_interesting_videos: true",
                "recursive: true",
                "detector_verbose: false",
                "generate_html_report: true",
                "auto_open_html_report: false",
                "write_json_exports: true",
                "preview_output_dir: preview_frames",
                "preview_include_uninteresting: false",
                "speciesnet_model: ''",
                "speciesnet_geofence: false",
                "speciesnet_label_in_filename: true",
                "species_crop_output_dir: preview_species_crops",
                "species_crop_padding: 0.15",
                   "generic_species_labels_to_skip: ['bird']",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def test_report_app_should_render_filtered_list_and_detail(tmp_path: Path) -> None:
    """Render list/detail views with filters, needs-reprocess state, and artifact serving."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    _seed_video(db_path, tmp_path, "aaa11111", "2026-05-01", "interesting", "0.1.0", "dog", 0.91)
    _seed_video(db_path, tmp_path, "bbb22222", "2026-05-02", "interesting", "0.2.0", "bird", 0.82)

    app = create_app(_write_config(tmp_path, db_path))
    client = app.test_client()

    list_response = client.get("/?species=dog&needs_reprocess=yes&sort_by=confidence&sort_dir=desc")
    assert list_response.status_code == 200
    list_body = list_response.get_data(as_text=True)
    assert "dog" in list_body
    assert "Needs Reprocess" in list_body
    assert "bbb22222" not in list_body
    assert "/artifact/aaa11111/species_crop" in list_body

    detail_response = client.get("/video/aaa11111")
    assert detail_response.status_code == 200
    detail_body = detail_response.get_data(as_text=True)
    assert "Open Original Camera Video" in detail_body
    assert "Open Clipped Original-Format Video" in detail_body
    assert "Open Browser-Compatible MP4" in detail_body
    assert "Open Top Detection Frame Image" in detail_body
    assert "Open Species Crop Image" in detail_body
    assert "backup" in detail_body

    artifact_response = client.get("/artifact/aaa11111/preview_image")
    assert artifact_response.status_code == 200
    assert artifact_response.data == b"artifact"

    source_response = client.get("/artifact/aaa11111/source_video")
    assert source_response.status_code == 200
    assert source_response.data == b"artifact"


def test_report_app_should_default_to_interesting_bucket(tmp_path: Path) -> None:
    """Default list view should show interesting videos unless bucket filter is overridden."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    _seed_video(db_path, tmp_path, "ccc33333", "2026-05-03", "interesting", "0.2.0", "fox", 0.88)
    _seed_video(db_path, tmp_path, "ddd44444", "2026-05-04", "uninteresting", "0.2.0", "empty", 0.01)

    app = create_app(_write_config(tmp_path, db_path))
    client = app.test_client()

    default_response = client.get("/")
    assert default_response.status_code == 200
    default_body = default_response.get_data(as_text=True)
    assert "ccc33333" in default_body
    assert "ddd44444" not in default_body

    all_response = client.get("/?bucket=")
    assert all_response.status_code == 200
    all_body = all_response.get_data(as_text=True)
    assert "ccc33333" in all_body
    assert "ddd44444" in all_body


def test_report_app_should_toggle_favorite_and_filter_results(tmp_path: Path) -> None:
    """Toggle favorites from routes and filter list view to favorite videos only."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    _seed_video(db_path, tmp_path, "eee55555", "2026-05-05", "interesting", "0.2.0", "deer", 0.84)
    _seed_video(db_path, tmp_path, "fff66666", "2026-05-06", "interesting", "0.2.0", "raccoon", 0.78)

    app = create_app(_write_config(tmp_path, db_path))
    client = app.test_client()

    favorite_response = client.post(
        "/video/eee55555/favorite",
        data={"action": "set", "favorite": "1", "next": "/?is_favorite=yes"},
    )
    assert favorite_response.status_code == 302

    favorites_only_response = client.get("/?is_favorite=yes")
    assert favorites_only_response.status_code == 200
    favorites_only_body = favorites_only_response.get_data(as_text=True)
    assert "eee55555" in favorites_only_body
    assert "fff66666" not in favorites_only_body
    assert "★ Favorite" in favorites_only_body

    detail_response = client.get("/video/eee55555")
    assert detail_response.status_code == 200
    detail_body = detail_response.get_data(as_text=True)
    assert "★ Favorited" in detail_body

    unfavorite_response = client.post(
        "/video/eee55555/favorite",
        data={"action": "clear", "next": "/?is_favorite=yes"},
    )
    assert unfavorite_response.status_code == 302

    favorites_after_clear_response = client.get("/?is_favorite=yes")
    assert favorites_after_clear_response.status_code == 200
    assert "eee55555" not in favorites_after_clear_response.get_data(as_text=True)


def test_report_app_should_support_legacy_catalog_without_favorites_table(tmp_path: Path) -> None:
    """List view should auto-create favorites table for pre-feature catalogs."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    _seed_video(db_path, tmp_path, "ggg77777", "2026-05-07", "interesting", "0.2.0", "bobcat", 0.93)

    with sqlite3.connect(db_path) as connection:
        connection.execute("DROP TABLE favorites")
        connection.commit()

    app = create_app(_write_config(tmp_path, db_path))
    client = app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert "ggg77777" in response.get_data(as_text=True)


def test_report_app_should_render_top_candidate_counts_for_filtered_set(tmp_path: Path) -> None:
    """Statistics should count top labels across the full filtered set, not only the current page."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)
    _seed_video(db_path, tmp_path, "hhh88888", "2026-05-08", "interesting", "0.2.0", "coyote", 0.96)
    _seed_video(db_path, tmp_path, "iii99999", "2026-05-08", "interesting", "0.2.0", "squirrel", 0.91)
    _seed_video(db_path, tmp_path, "jjj00000", "2026-05-08", "interesting", "0.2.0", "squirrel", 0.83)

    app = create_app(_write_config(tmp_path, db_path))
    client = app.test_client()

    response = client.get("/?date_from=2026-05-08&date_to=2026-05-08&page_size=1&sort_by=confidence&sort_dir=desc")
    assert response.status_code == 200

    body = response.get_data(as_text=True)
    assert "Top Candidate Counts" in body
    assert "coyote <strong>1</strong>" in body
    assert "squirrel <strong>2</strong>" in body
    assert "Counts use the most likely candidate for every video matching the current filters" in body


def test_report_app_should_render_all_candidate_counts_without_top_limit(tmp_path: Path) -> None:
    """Statistics should include labels beyond the former top-results cap."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    labels = [
        "antelope",
        "badger",
        "bobcat",
        "chipmunk",
        "coyote",
        "crow",
        "deer",
        "dog",
        "fox",
        "hawk",
        "mouse",
        "rabbit",
        "raccoon",
    ]
    for index, label in enumerate(labels, start=1):
        _seed_video(
            db_path,
            tmp_path,
            f"vid{index:05d}",
            "2026-05-09",
            "interesting",
            "0.2.0",
            label,
            0.8,
        )

    app = create_app(_write_config(tmp_path, db_path))
    client = app.test_client()

    response = client.get("/?date_from=2026-05-09&date_to=2026-05-09")
    assert response.status_code == 200

    body = response.get_data(as_text=True)
    assert "antelope <strong>1</strong>" in body
    assert "raccoon <strong>1</strong>" in body

    def test_report_app_should_display_primary_species_candidate_before_generic_labels(tmp_path: Path) -> None:
        """Detail candidates should show the stored primary species before generic labels."""
        db_path = tmp_path / "catalog.sqlite3"
        initialize_metadata_store(db_path)
        _seed_video(
            db_path,
            tmp_path,
            "kkk11111",
            "2026-06-02",
            "interesting",
            "0.2.0",
            "steller's jay",
            0.326,
        )

        with sqlite3.connect(db_path) as connection:
            connection.execute(
                """
                UPDATE species_classifications
                SET candidates_json=?
                WHERE video_id=?
                """,
                (
                    json.dumps(
                        [
                            {"label": "bird", "score": 0.38},
                            {"label": "steller's jay", "score": 0.326},
                            {"label": "green jay", "score": 0.013},
                        ]
                    ),
                    "kkk11111",
                ),
            )
            connection.commit()

        app = create_app(_write_config(tmp_path, db_path))
        client = app.test_client()

        response = client.get("/video/kkk11111")

        assert response.status_code == 200
        body = response.get_data(as_text=True)
        assert body.index("steller&#39;s jay (0.326)") < body.index("bird (0.380)")
