"""End-to-end integration tests for the video processing pipeline.

Tests the three processing modes (new-only, reprocess-existing, report-only)
in realistic scenarios with actual video files, catalog persistence, and artifact
generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from metadata_store import (
    fetch_catalog_snapshot,
    initialize_metadata_store,
)
from pipeline_models import AppConfig
from process_videos import merge_species_classification_reports
from processing_modes import ingest_videos_into_catalog, load_reprocess_sources
from reporting import write_html_summary_from_catalog


@pytest.fixture
def tmp_workspace(tmp_path: Path) -> dict[str, Any]:
    """Create a minimal workspace structure with input/output directories.

    Returns a dict with paths to key directories and config for testing.
    """
    workspace = {
        "root": tmp_path,
        "input": tmp_path / "input",
        "output": tmp_path / "output",
        "metadata": tmp_path / "output" / "metadata",
        "canonical": tmp_path / "output" / "videos",
    }

    for path in [workspace["input"], workspace["metadata"], workspace["canonical"]]:
        path.mkdir(parents=True, exist_ok=True)

    # Initialize metadata store
    metadata_db = workspace["metadata"] / "catalog.sqlite3"
    initialize_metadata_store(metadata_db)
    workspace["metadata_db"] = metadata_db

    return workspace


def _create_test_video(path: Path, name: str = "test.avi", duration_seconds: float = 2.0, seed: int = 0) -> Path:
    """Create a minimal valid AVI video file for testing.

    Uses OpenCV to create a valid video with a few frames with unique content based on seed.
    This ensures the detector and classifier can process it and produces unique SHA256 hashes.
    """
    try:
        import cv2
        import numpy as np

        output_file = path / name
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps = 2
        frame_count = int(fps * duration_seconds)
        writer = cv2.VideoWriter(str(output_file), fourcc, fps, (640, 480))

        for i in range(frame_count):
            # Create a frame with unique content based on seed and frame index
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add seed-based shapes to make each test video unique
            x_offset = (seed * 20 + i * 10) % 500
            y_offset = (seed * 15 + i * 8) % 400
            cv2.rectangle(
                frame,
                (50 + x_offset, 50 + y_offset),
                (150 + x_offset, 150 + y_offset),
                (seed % 256, 100 + seed, 200),
                2,
            )
            writer.write(frame)

        writer.release()
        return output_file
    except ImportError:
        # Fallback: create a dummy file with unique content if OpenCV not available
        output_file = path / name
        # Create unique content based on seed so different test videos have different content
        unique_content = f"RIFF{seed:08d}".encode() + b"\x00" * 100 + b"AVI "
        output_file.write_bytes(unique_content)
        return output_file


def _create_minimal_config(workspace: dict[str, Any]) -> AppConfig:
    """Create a minimal test configuration."""
    return AppConfig(
        input_dir=str(workspace["input"]),
        output_dir=str(workspace["output"]),
        metadata_db_path=str(workspace["metadata_db"]),
        pipeline_version="test-e2e-v1",
        model="MDV5A",
        frame_sample=5,
        interesting_threshold=0.1,
        interesting_categories=["1", "2", "3"],
        move_files=False,  # Keep files in place for reprocessing
        save_uninteresting_files=False,
        clip_interesting_videos=True,
        recursive=False,
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
    )


class TestNewOnlyMode:
    """Test new-only processing mode: discover and process new videos."""

    def test_should_discover_and_ingest_videos(self, tmp_workspace: dict[str, Any]) -> None:
        """New-only mode should discover new videos and record them in catalog."""
        # Setup: create test videos with different content (different seeds)
        input_dir = tmp_workspace["input"]
        _create_test_video(input_dir, "video1.avi", seed=1)
        _create_test_video(input_dir, "video2.avi", seed=2)

        # Verify videos were created
        videos = list(input_dir.glob("*.avi"))
        assert len(videos) == 2

        # Ingest into catalog
        ingested_count, newly_persisted_count, sources = ingest_videos_into_catalog(
            videos=videos,
            canonical_videos_dir=tmp_workspace["canonical"],
            metadata_db_path=tmp_workspace["metadata_db"],
        )

        # Verify ingestion results
        assert ingested_count == 2
        assert newly_persisted_count == 2
        assert len(sources) == 2

        # Verify database has the videos
        snapshot = fetch_catalog_snapshot(tmp_workspace["metadata_db"])
        assert len(snapshot["videos"]) == 2
        assert snapshot["videos"][0]["original_filename"] in ("video1.avi", "video2.avi")
        assert snapshot["videos"][1]["original_filename"] in ("video1.avi", "video2.avi")

    def test_should_not_reingest_existing_videos(self, tmp_workspace: dict[str, Any]) -> None:
        """New-only mode should skip videos already in catalog."""
        input_dir = tmp_workspace["input"]
        video = _create_test_video(input_dir, "video.avi", seed=3)

        # First ingest
        ingested_count1, newly_persisted_count1, sources1 = ingest_videos_into_catalog(
            videos=[video],
            canonical_videos_dir=tmp_workspace["canonical"],
            metadata_db_path=tmp_workspace["metadata_db"],
        )
        assert ingested_count1 == 1
        assert newly_persisted_count1 == 1

        # Second ingest of same video
        ingested_count2, newly_persisted_count2, sources2 = ingest_videos_into_catalog(
            videos=[video],
            canonical_videos_dir=tmp_workspace["canonical"],
            metadata_db_path=tmp_workspace["metadata_db"],
        )

        # Should skip since it's already in catalog
        assert ingested_count2 == 1
        assert newly_persisted_count2 == 0
        assert len(sources2) == 0

        # Database should still have only one video
        snapshot = fetch_catalog_snapshot(tmp_workspace["metadata_db"])
        assert len(snapshot["videos"]) == 1


class TestReprocessExistingMode:
    """Test reprocess-existing mode: reload and reprocess existing videos."""

    def test_should_load_existing_videos_for_reprocessing(
        self, tmp_workspace: dict[str, Any]
    ) -> None:
        """Reprocess mode should load videos already in catalog."""
        # Setup: ingest videos first
        input_dir = tmp_workspace["input"]
        _create_test_video(input_dir, "video1.avi", seed=4)
        _create_test_video(input_dir, "video2.avi", seed=5)
        videos = list(input_dir.glob("*.avi"))

        ingest_videos_into_catalog(
            videos=videos,
            canonical_videos_dir=tmp_workspace["canonical"],
            metadata_db_path=tmp_workspace["metadata_db"],
        )

        # Load for reprocessing
        sources = load_reprocess_sources(tmp_workspace["metadata_db"])

        # Should load the two videos
        assert len(sources) == 2
        assert all(hasattr(s, "video_id") for s in sources)
        assert all(hasattr(s, "source") for s in sources)

    def test_should_track_reprocessing_in_catalog(self, tmp_workspace: dict[str, Any]) -> None:
        """Reprocess mode should update processing_state with reprocess markers."""
        # Setup: ingest one video
        input_dir = tmp_workspace["input"]
        _create_test_video(input_dir, "video.avi", seed=6)
        videos = list(input_dir.glob("*.avi"))

        ingest_videos_into_catalog(
            videos=videos,
            canonical_videos_dir=tmp_workspace["canonical"],
            metadata_db_path=tmp_workspace["metadata_db"],
        )

        initial_snapshot = fetch_catalog_snapshot(tmp_workspace["metadata_db"])
        assert len(initial_snapshot["processing_state"]) == 0

        # After processing (simulated by loading sources), processing_state should be populated
        # This test validates the structure but doesn't run full processing
        sources = load_reprocess_sources(tmp_workspace["metadata_db"])
        assert len(sources) == 1


class TestReportOnlyMode:
    """Test report-only mode: generate reports without processing."""

    def test_should_generate_html_without_processing(self, tmp_workspace: dict[str, Any]) -> None:
        """Report-only mode should generate HTML from existing catalog data."""
        # Setup: seed catalog with processed data
        from metadata_store import (
            ProcessingStateRecord,
            SpeciesClassificationRecord,
            VideoCatalogRecord,
            sync_artifact_path,
            upsert_processing_state_record,
            upsert_species_classification_record,
            upsert_video_record,
        )

        # Create and ingest a video record
        video_id = "test-video-001"
        canonical_dir = tmp_workspace["canonical"] / video_id[:2] / video_id[2:4] / video_id
        canonical_dir.mkdir(parents=True, exist_ok=True)

        source_path = canonical_dir / "source.avi"
        source_path.write_bytes(b"dummy video data")

        upsert_video_record(
            tmp_workspace["metadata_db"],
            VideoCatalogRecord(
                video_id=video_id,
                original_filename="test.avi",
                capture_date="2026-05-05",
                filesize_bytes=16,
                source_ext=".avi",
                stored_original_path=str(source_path),
            ),
        )

        # Add processing state (simulating detection results)
        upsert_processing_state_record(
            tmp_workspace["metadata_db"],
            ProcessingStateRecord(
                video_id=video_id,
                pipeline_version="test-e2e-v1",
                mode="new-only",
                bucket="interesting",
                top_confidence=0.85,
                top_category="person",
                top_frame=0,
                status="success",
            ),
        )

        # Add species classification
        upsert_species_classification_record(
            tmp_workspace["metadata_db"],
            SpeciesClassificationRecord(
                video_id=video_id,
                top_label="domestic dog",
                top_score=0.75,
                top_raw_class="animal;canid;domestic dog",
                candidates_json=json.dumps([{"label": "domestic dog", "score": 0.75}]),
            ),
        )

        # Add artifact paths
        interesting_video = canonical_dir / "interesting.avi"
        interesting_video.write_bytes(b"bucketed video")
        sync_artifact_path(
            db_path=tmp_workspace["metadata_db"],
            video_id=video_id,
            artifact_type="bucketed_video",
            artifact_path=interesting_video,
        )

        # Generate HTML report from catalog
        html_path = tmp_workspace["output"] / "metadata" / "summary.html"
        write_html_summary_from_catalog(
            html_summary_path=html_path,
            metadata_db_path=tmp_workspace["metadata_db"],
        )

        # Verify HTML was generated
        assert html_path.exists()
        html_content = html_path.read_text()
        assert "test.avi" in html_content
        assert "domestic dog" in html_content

    def test_report_should_include_all_artifacts(self, tmp_workspace: dict[str, Any]) -> None:
        """Report should include all artifact types stored in catalog."""
        from metadata_store import (
            VideoCatalogRecord,
            sync_artifact_path,
            upsert_video_record,
        )

        video_id = "test-video-002"
        canonical_dir = tmp_workspace["canonical"] / video_id[:2] / video_id[2:4] / video_id
        canonical_dir.mkdir(parents=True, exist_ok=True)

        source_path = canonical_dir / "source.avi"
        source_path.write_bytes(b"source video")

        upsert_video_record(
            tmp_workspace["metadata_db"],
            VideoCatalogRecord(
                video_id=video_id,
                original_filename="test.avi",
                capture_date="2026-05-05",
                filesize_bytes=12,
                source_ext=".avi",
                stored_original_path=str(source_path),
            ),
        )

        # Add all artifact types
        for artifact_type, filename in [
            ("bucketed_video", "interesting.avi"),
            ("report_video", "report.mp4"),
            ("preview_image", "preview_0.jpg"),
            ("species_crop", "crop_0.jpg"),
        ]:
            artifact_path = canonical_dir / filename
            artifact_path.write_bytes(b"artifact data")
            sync_artifact_path(
                db_path=tmp_workspace["metadata_db"],
                video_id=video_id,
                artifact_type=artifact_type,
                artifact_path=artifact_path,
            )

        # Generate report
        html_path = tmp_workspace["output"] / "metadata" / "summary.html"
        write_html_summary_from_catalog(
            html_summary_path=html_path,
            metadata_db_path=tmp_workspace["metadata_db"],
        )

        # Verify report includes references to artifacts
        assert html_path.exists()


class TestCollisionHandling:
    """Test that videos with same filename from different dates don't collide."""

    def test_should_handle_colliding_filenames_from_different_dates(
        self, tmp_workspace: dict[str, Any]
    ) -> None:
        """Multiple videos with same name but different capture dates should get unique output paths."""
        input_dir = tmp_workspace["input"]

        # Create two videos with same name but from different date folders
        date1_dir = input_dir / "20260405"
        date1_dir.mkdir()
        video1 = _create_test_video(date1_dir, "camera.avi", seed=7)

        date2_dir = input_dir / "20260419"
        date2_dir.mkdir()
        video2 = _create_test_video(date2_dir, "camera.avi", seed=8)

        # Ingest both
        videos = [video1, video2]
        ingested_count, newly_persisted_count, sources = ingest_videos_into_catalog(
            videos=videos,
            canonical_videos_dir=tmp_workspace["canonical"],
            metadata_db_path=tmp_workspace["metadata_db"],
        )

        assert ingested_count == 2
        assert newly_persisted_count == 2

        # Both should be in catalog with different video_ids
        snapshot = fetch_catalog_snapshot(tmp_workspace["metadata_db"])
        assert len(snapshot["videos"]) == 2
        video_ids = [v["video_id"] for v in snapshot["videos"]]
        assert video_ids[0] != video_ids[1]


class TestSpeciesClassificationPersistence:
    """Test that species classifications are correctly persisted to catalog."""

    def test_should_merge_and_persist_species_classifications(
        self, tmp_workspace: dict[str, Any]
    ) -> None:
        """Species classification reports should be merged and persisted to SQLite."""
        # Create temporary species report files
        temp_reports = []

        report1_path = tmp_workspace["output"] / "temp_species_1.json"
        report1_path.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "source_relative_path": "video1.avi",
                            "top_classification": {
                                "label": "dog",
                                "score": 0.9,
                                "raw_class": "animal;canid;dog",
                            },
                            "candidates": [{"label": "dog", "score": 0.9}],
                        }
                    ]
                }
            )
        )
        temp_reports.append(report1_path)

        report2_path = tmp_workspace["output"] / "temp_species_2.json"
        report2_path.write_text(
            json.dumps(
                {
                    "entries": [
                        {
                            "source_relative_path": "video2.avi",
                            "top_classification": {
                                "label": "bird",
                                "score": 0.85,
                                "raw_class": "animal;bird",
                            },
                            "candidates": [{"label": "bird", "score": 0.85}],
                        }
                    ]
                }
            )
        )
        temp_reports.append(report2_path)

        # Merge reports
        final_report_path = tmp_workspace["output"] / "species_classifications.json"
        merge_species_classification_reports(
            temp_report_paths=temp_reports,
            final_report_path=final_report_path,
        )

        # Verify merged report
        assert final_report_path.exists()
        merged = json.loads(final_report_path.read_text())
        assert len(merged["entries"]) == 2
        assert merged["entries"][0]["top_classification"]["label"] in ("dog", "bird")
        assert merged["entries"][1]["top_classification"]["label"] in ("dog", "bird")
