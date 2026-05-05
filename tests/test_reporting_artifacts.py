"""Tests for report video generation and HTML rendering integration."""

from pathlib import Path

from pipeline_models import VideoDecision
from reporting import generate_report_videos, write_html_summary


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
