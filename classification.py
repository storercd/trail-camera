"""Video classification and output-bucketing helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from file_ops import build_dated_relative_output_path, copy_or_move, make_unique_destination
from pipeline_models import VideoDecision
from video_clipping import clip_video_by_frame_window


def build_clipped_log_line(
    index: int,
    total_entries: int,
    output_relative_path: str,
    start_frame: int,
    end_frame: int,
    frames_written: int,
    total_source_frames: int,
    bucket: str,
) -> str:
    """Build a human-readable clip status line with coverage details.

    Args:
        index: Current 1-based video index.
        total_entries: Total number of entries in this run.
        output_relative_path: Output path under the destination bucket.
        start_frame: First frame index written to output.
        end_frame: Last frame index written to output.
        frames_written: Number of frames written to the clip.
        total_source_frames: Total frame count in the source video.
        bucket: Destination bucket name.

    Returns:
        str: Formatted status line for logging.
    """
    if total_source_frames > 0:
        kept_pct = (frames_written / total_source_frames) * 100.0
        coverage_text = f"{frames_written}/{total_source_frames} frames kept ({kept_pct:.1f}%)"
    else:
        coverage_text = f"{frames_written} frames kept"

    return (
        f"[{index}/{total_entries}] Clipped {output_relative_path} "
        f"frames {start_frame}-{end_frame} ({coverage_text}) -> {bucket}"
    )


def analyze_video_result(
    image_entry: dict[str, Any],
    interesting_categories: set[str],
    threshold: float,
) -> VideoDecision:
    """Classify one MegaDetector image record into an output bucket.

    Args:
        image_entry: One image record from the MegaDetector output.
        interesting_categories: Category IDs treated as interesting.
        threshold: Minimum confidence for interesting detections.

    Returns:
        VideoDecision: Classification and summary fields for the video.
    """
    rel_path = image_entry["file"]
    if "failure" in image_entry:
        return VideoDecision(
            relative_path=rel_path,
            output_relative_path=rel_path,
            bucket="failed",
            top_confidence=None,
            top_category=None,
            top_frame=None,
            top_bbox=None,
            first_interesting_frame=None,
            last_interesting_frame=None,
            num_detections=0,
            failure=str(image_entry["failure"]),
        )

    detections = image_entry.get("detections") or []
    best, interesting_count, first_interesting_frame, last_interesting_frame = (
        collect_interesting_detection_stats(detections, interesting_categories, threshold)
    )

    if best is None:
        return VideoDecision(
            relative_path=rel_path,
            output_relative_path=rel_path,
            bucket="uninteresting",
            top_confidence=None,
            top_category=None,
            top_frame=None,
            top_bbox=None,
            first_interesting_frame=None,
            last_interesting_frame=None,
            num_detections=0,
            failure=None,
        )

    top_bbox = best.get("bbox")
    normalized_bbox = None
    if isinstance(top_bbox, list) and len(top_bbox) == 4:
        try:
            normalized_bbox = [float(value) for value in top_bbox]
        except (TypeError, ValueError):
            normalized_bbox = None

    return VideoDecision(
        relative_path=rel_path,
        output_relative_path=rel_path,
        bucket="interesting",
        top_confidence=float(best.get("conf", 0.0)),
        top_category=str(best.get("category", "")),
        top_frame=int(best.get("frame_number", -1)),
        top_bbox=normalized_bbox,
        first_interesting_frame=first_interesting_frame,
        last_interesting_frame=last_interesting_frame,
        num_detections=interesting_count,
        failure=None,
    )


def collect_interesting_detection_stats(
    detections: list[dict[str, Any]],
    interesting_categories: set[str],
    threshold: float,
) -> tuple[dict[str, Any] | None, int, int | None, int | None]:
    """Collect best detection and frame bounds for interesting detections.

    Returns:
        tuple[dict[str, Any] | None, int, int | None, int | None]:
            Best detection, count, first interesting frame, and last interesting frame.
    """
    best: dict[str, Any] | None = None
    interesting_count = 0
    first_interesting_frame: int | None = None
    last_interesting_frame: int | None = None

    for det in detections:
        confidence = float(det.get("conf", 0.0))
        category = str(det.get("category", ""))
        if category not in interesting_categories or confidence < threshold:
            continue

        interesting_count += 1
        frame_number = int(det.get("frame_number", -1))
        if frame_number >= 0:
            if first_interesting_frame is None or frame_number < first_interesting_frame:
                first_interesting_frame = frame_number
            if last_interesting_frame is None or frame_number > last_interesting_frame:
                last_interesting_frame = frame_number

        if best is None or confidence > float(best.get("conf", 0.0)):
            best = det

    return best, interesting_count, first_interesting_frame, last_interesting_frame


def classify_and_sort_videos(
    image_entries: list[dict[str, Any]],
    input_dir: Path,
    output_dir: Path,
    interesting_categories: set[str],
    threshold: float,
    move_files: bool,
    save_uninteresting_files: bool,
    clip_interesting_videos: bool,
    clip_buffer_frames: int,
) -> list[VideoDecision]:
    """Classify video results and copy/move original files into output buckets.

    Args:
        image_entries: MegaDetector image records loaded from results JSON.
        input_dir: Root input directory for source videos.
        output_dir: Root output directory for bucketed files.
        interesting_categories: Category IDs considered interesting.
        threshold: Minimum confidence for interesting detections.
        move_files: If True, move files instead of copying.
        save_uninteresting_files: If False, skip writing uninteresting videos.
        clip_interesting_videos: If True, clip interesting videos to detection window.
        clip_buffer_frames: Buffer to add before first and after last interesting frame.

    Returns:
        list[VideoDecision]: Per-video decisions used for reporting.
    """
    decisions: list[VideoDecision] = []
    total_entries = len(image_entries)

    for index, image_entry in enumerate(image_entries, start=1):
        decision = analyze_video_result(image_entry, interesting_categories, threshold)
        source = input_dir / decision.relative_path
        if source.exists():
            decision.output_relative_path = build_dated_relative_output_path(
                source=source,
                relative_path=decision.relative_path,
            )
        should_write_file = decision.bucket != "uninteresting" or save_uninteresting_files

        if not should_write_file:
            print(
                f"[{index}/{total_entries}] Skipped {decision.relative_path}: "
                "uninteresting output disabled"
            )
            decisions.append(decision)
            continue

        if source.exists():
            destination = output_dir / decision.bucket / decision.output_relative_path
            bucket_root = output_dir / decision.bucket
            if (
                clip_interesting_videos
                and decision.bucket == "interesting"
                and decision.first_interesting_frame is not None
                and decision.last_interesting_frame is not None
            ):
                destination.parent.mkdir(parents=True, exist_ok=True)
                clip_destination = make_unique_destination(destination)
                clip_result = clip_video_by_frame_window(
                    source_video=source,
                    output_video=clip_destination,
                    first_frame=decision.first_interesting_frame,
                    last_frame=decision.last_interesting_frame,
                    buffer_frames=clip_buffer_frames,
                )
                if clip_result is not None:
                    decision.output_relative_path = str(clip_destination.relative_to(bucket_root))
                    if move_files:
                        source.unlink(missing_ok=True)
                    print(
                        build_clipped_log_line(
                            index=index,
                            total_entries=total_entries,
                            output_relative_path=decision.output_relative_path,
                            start_frame=clip_result.start_frame,
                            end_frame=clip_result.end_frame,
                            frames_written=clip_result.frames_written,
                            total_source_frames=clip_result.total_source_frames,
                            bucket=decision.bucket,
                        )
                    )
                else:
                    written_path = copy_or_move(source, destination, move=move_files)
                    decision.output_relative_path = str(written_path.relative_to(bucket_root))
                    action = "Moved" if move_files else "Copied"
                    print(
                        f"[{index}/{total_entries}] {action} {decision.output_relative_path} -> {decision.bucket} "
                        "(clip failed, saved full video)"
                    )
            else:
                written_path = copy_or_move(source, destination, move=move_files)
                decision.output_relative_path = str(written_path.relative_to(bucket_root))
                action = "Moved" if move_files else "Copied"
                print(f"[{index}/{total_entries}] {action} {decision.output_relative_path} -> {decision.bucket}")
        else:
            print(f"[{index}/{total_entries}] Source missing for {decision.relative_path}; skipping copy/move")
        decisions.append(decision)

    return decisions


def compute_bucket_counts(decisions: list[VideoDecision]) -> dict[str, int]:
    """Count decisions by output bucket.

    Args:
        decisions: Per-video decisions from one run.

    Returns:
        dict[str, int]: Counts for interesting, uninteresting, and failed buckets.
    """
    return {
        "interesting": sum(1 for decision in decisions if decision.bucket == "interesting"),
        "uninteresting": sum(1 for decision in decisions if decision.bucket == "uninteresting"),
        "failed": sum(1 for decision in decisions if decision.bucket == "failed"),
    }
