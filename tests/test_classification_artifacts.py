"""Regression tests for classification artifact write behavior."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from classification import classify_and_sort_videos
from video_clipping import ClipResult


def test_classify_and_sort_videos_should_write_clipped_video_to_exact_destination(
    tmp_path: Path,
) -> None:
    """Write clip output to mapped canonical destination without unique suffixing."""
    input_dir = tmp_path / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    source = input_dir / "source.avi"
    source.write_bytes(b"video")

    output_dir = tmp_path / "output"
    expected_destination = tmp_path / "canonical" / "vid-001" / "interesting.avi"

    image_entries = [
        {
            "file": "source.avi",
            "detections": [
                {
                    "category": "1",
                    "conf": 0.95,
                    "frame_number": 12,
                    "bbox": [0.1, 0.1, 0.5, 0.5],
                }
            ],
        }
    ]

    captured_output_path: list[Path] = []

    def fake_clip_video_by_frame_window(
        source_video: Path,
        output_video: Path,
        first_frame: int,
        last_frame: int,
        buffer_frames: int,
    ) -> ClipResult:
        captured_output_path.append(output_video)
        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_video.write_bytes(b"clip")
        return ClipResult(
            output_path=output_video,
            start_frame=0,
            end_frame=20,
            frames_written=21,
            total_source_frames=100,
        )

    with patch("video_clipping.clip_video_by_frame_window", new=fake_clip_video_by_frame_window):
        decisions = classify_and_sort_videos(
            image_entries=image_entries,
            input_dir=input_dir,
            output_dir=output_dir,
            interesting_categories={"1"},
            threshold=0.1,
            move_files=False,
            save_uninteresting_files=False,
            clip_interesting_videos=True,
            clip_buffer_frames=5,
            bucket_output_paths={"source.avi": expected_destination},
        )

    assert len(decisions) == 1
    assert decisions[0].bucket == "interesting"
    assert captured_output_path == [expected_destination]
    assert expected_destination.exists()
