"""Utilities for extracting preview images from selected video frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass
class TopFrameRecord:
    """Per-video frame selection metadata for preview extraction."""

    relative_path: str
    top_frame: int | None
    top_confidence: float | None
    bucket: str


@dataclass
class PreviewExtractionStats:
    """Outcome counters for one preview extraction run."""

    total_candidates: int
    extracted: int
    skipped: int
    failed: int


def build_output_path(output_dir: Path, record: TopFrameRecord) -> Path:
    """Build destination image path for a selected frame.

    Args:
        output_dir: Root preview output directory.
        record: Frame metadata for one video.

    Returns:
        Path: Destination path for preview image.
    """
    relative_video = Path(record.relative_path)
    parent = output_dir / relative_video.parent
    frame_label = record.top_frame if record.top_frame is not None else -1
    confidence_label = (
        f"{record.top_confidence:.3f}"
        if isinstance(record.top_confidence, (float, int))
        else "na"
    )
    filename = f"{relative_video.stem}_frame{frame_label}_conf{confidence_label}.jpg"
    return parent / filename


def extract_frame(video_path: Path, frame_number: int, output_image: Path) -> bool:
    """Extract and write one frame from a video.

    Args:
        video_path: Source video path.
        frame_number: Zero-based frame index.
        output_image: Destination image path.

    Returns:
        bool: True when frame extraction and write succeed.
    """
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False

    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_number))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return False

    output_image.parent.mkdir(parents=True, exist_ok=True)
    return bool(cv2.imwrite(str(output_image), frame))


def extract_top_frames(
    records: list[TopFrameRecord],
    input_dir: Path,
    output_dir: Path,
    include_uninteresting: bool,
) -> PreviewExtractionStats:
    """Extract top-frame preview images for selected records.

    Args:
        records: Candidate records to process.
        input_dir: Root directory where source videos exist.
        output_dir: Root directory for generated preview images.
        include_uninteresting: Include uninteresting bucket records when true.

    Returns:
        PreviewExtractionStats: Aggregated extraction counters.
    """
    extracted = 0
    skipped = 0
    failed = 0

    filtered_records = [
        record
        for record in records
        if include_uninteresting or record.bucket == "interesting"
    ]

    total = len(filtered_records)
    for index, record in enumerate(filtered_records, start=1):
        if not record.relative_path or record.top_frame is None or int(record.top_frame) < 0:
            skipped += 1
            print(f"[{index}/{total}] Skipped {record.relative_path}: no valid top_frame")
            continue

        source_video = input_dir / record.relative_path
        if not source_video.exists():
            failed += 1
            print(f"[{index}/{total}] Failed {record.relative_path}: video not found")
            continue

        output_image = build_output_path(output_dir, record)
        if not extract_frame(source_video, int(record.top_frame), output_image):
            failed += 1
            print(f"[{index}/{total}] Failed {record.relative_path}: frame extraction error")
            continue

        extracted += 1
        print(f"[{index}/{total}] Wrote {output_image}")

    return PreviewExtractionStats(
        total_candidates=total,
        extracted=extracted,
        skipped=skipped,
        failed=failed,
    )
