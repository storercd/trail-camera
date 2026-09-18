"""Tests for report video generation."""

from pathlib import Path

from pipeline_models import VideoDecision
from reporting import generate_report_videos


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
