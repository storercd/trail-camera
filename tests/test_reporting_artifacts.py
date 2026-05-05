"""Tests for report video generation and HTML rendering integration."""

import json
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
from pipeline_models import VideoDecision
from reporting import generate_report_videos, write_html_summary, write_html_summary_from_catalog


def _decision(path: str, bucket: str = "interesting") -> VideoDecision:
    return VideoDecision(
        relative_path=path,
        output_relative_path=path,
        bucket=bucket,
        top_confidence=0.9,
        top_category="1",
        top_frame=10,
        top_bbox=None,
        first_interesting_frame=5,
        last_interesting_frame=20,
        num_detections=2,
        failure=None,
    )


def test_generate_report_videos_should_return_generated_paths(
    tmp_path: Path,
) -> None:
    """Generate report video paths for interesting decisions with existing sources."""
    run_output_dir = tmp_path / "output"
    source_video = run_output_dir / "interesting" / "clip1.AVI"
    source_video.parent.mkdir(parents=True, exist_ok=True)
    source_video.write_bytes(b"video")

    def fake_transcode(source: Path, output: Path) -> Path:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b"mp4")
        return output

    result = generate_report_videos(
        [_decision("clip1.AVI")],
        run_output_dir,
        transcode_fn=fake_transcode,
    )

    assert "clip1.AVI" in result
    assert result["clip1.AVI"].exists()


def test_write_html_summary_should_use_pre_generated_report_videos(
    tmp_path: Path,
) -> None:
    """Render HTML using provided report video paths without transcoding."""
    html_path = tmp_path / "summary.html"
    run_output_dir = tmp_path / "output"
    source_video = run_output_dir / "interesting" / "clip2.AVI"
    source_video.parent.mkdir(parents=True, exist_ok=True)
    source_video.write_bytes(b"video")

    report_video = run_output_dir / "report_videos" / "clip2.mp4"
    report_video.parent.mkdir(parents=True, exist_ok=True)
    report_video.write_bytes(b"mp4")

    write_html_summary(
        html_summary_path=html_path,
        decisions=[_decision("clip2.AVI")],
        run_output_dir=run_output_dir,
        species_classification_report_path=tmp_path / "species.json",
        report_video_paths={"clip2.AVI": report_video},
    )

    assert html_path.exists()
    contents = html_path.read_text(encoding="utf-8")
    assert "clip2.mp4" in contents


def test_write_html_summary_from_catalog_should_render_species_and_artifacts(
    tmp_path: Path,
) -> None:
    """Render HTML from SQLite snapshot and on-disk artifacts without in-memory decisions."""
    db_path = tmp_path / "catalog.sqlite3"
    initialize_metadata_store(db_path)

    video_id = "vid-catalog-001"
    canonical_dir = tmp_path / "videos" / "aa" / "bb" / video_id
    canonical_dir.mkdir(parents=True, exist_ok=True)
    source_path = canonical_dir / "source.avi"
    bucketed = canonical_dir / "interesting.avi"
    report_video = canonical_dir / "report.mp4"
    preview = canonical_dir / "preview_species-dog_sp0.500.jpg"
    crop = canonical_dir / "preview_crop.jpg"
    for path in (source_path, bucketed, report_video, preview, crop):
        path.write_bytes(b"x")

    upsert_video_record(
        db_path,
        VideoCatalogRecord(
            video_id=video_id,
            original_filename="source.avi",
            capture_date="2026-05-05",
            filesize_bytes=1,
            source_ext=".avi",
            stored_original_path=str(source_path),
        ),
    )
    upsert_processing_state_record(
        db_path,
        ProcessingStateRecord(
            video_id=video_id,
            pipeline_version="0.1.0",
            mode="reprocess-existing",
            bucket="interesting",
            top_confidence=0.9,
            top_category="1",
            top_frame=10,
            status="processed",
        ),
    )
    sync_artifact_path(db_path, video_id, "bucketed_video", bucketed)
    sync_artifact_path(db_path, video_id, "report_video", report_video)
    sync_artifact_path(db_path, video_id, "preview_image", preview)
    sync_artifact_path(db_path, video_id, "species_crop", crop)
    upsert_species_classification_record(
        db_path=db_path,
        record=SpeciesClassificationRecord(
            video_id=video_id,
            top_label="dog",
            top_score=0.5,
            top_raw_class="animal;dog",
            candidates_json=json.dumps([
                {"label": "dog", "score": 0.5},
                {"label": "canid", "score": 0.3},
            ]),
        ),
    )

    html_path = tmp_path / "summary.html"
    write_html_summary_from_catalog(
        html_summary_path=html_path,
        metadata_db_path=db_path,
    )

    contents = html_path.read_text(encoding="utf-8")
    assert "Species source: sqlite catalog" in contents
    assert "Open native clip" in contents
    assert "dog" in contents
    assert "canid" in contents
