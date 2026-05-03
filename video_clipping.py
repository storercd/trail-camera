"""Utilities for clipping videos by frame range."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class ClipResult:
    """Result metadata for a clipped video output."""

    output_path: Path
    start_frame: int
    end_frame: int
    frames_written: int


def _codec_for_suffix(suffix: str) -> str:
    """Choose a broadly compatible codec based on file extension.

    Returns:
        str: FourCC codec identifier.
    """
    suffix = suffix.lower()
    if suffix == ".avi":
        return "MJPG"
    if suffix in {".mp4", ".mov", ".m4v"}:
        return "mp4v"
    return "XVID"


def clip_video_by_frame_window(
    source_video: Path,
    output_video: Path,
    first_frame: int,
    last_frame: int,
    buffer_frames: int,
) -> ClipResult | None:
    """Write a clipped video window around detections.

    Args:
        source_video: Source video path.
        output_video: Destination video path.
        first_frame: First interesting frame index.
        last_frame: Last interesting frame index.
        buffer_frames: Number of frames to include before and after the window.

    Returns:
        ClipResult | None: Clip metadata on success, otherwise None.
    """
    capture = cv2.VideoCapture(str(source_video))
    if not capture.isOpened():
        return None

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    if total_frames <= 0 or width <= 0 or height <= 0:
        capture.release()
        return None

    if fps <= 0:
        fps = 30.0

    clip_start = max(0, int(first_frame) - int(buffer_frames))
    clip_end = min(total_frames - 1, int(last_frame) + int(buffer_frames))
    if clip_end < clip_start:
        capture.release()
        return None

    output_video.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*_codec_for_suffix(output_video.suffix))
    writer = cv2.VideoWriter(str(output_video), fourcc, fps, (width, height))
    if not writer.isOpened():
        capture.release()
        return None

    capture.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
    frames_written = 0

    for _ in range(clip_start, clip_end + 1):
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        frames_written += 1

    writer.release()
    capture.release()

    if frames_written <= 0:
        try:
            output_video.unlink(missing_ok=True)
        except OSError:
            pass
        return None

    return ClipResult(
        output_path=output_video,
        start_frame=clip_start,
        end_frame=clip_start + frames_written - 1,
        frames_written=frames_written,
    )
