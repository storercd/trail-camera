#!/usr/bin/env python3
"""Run MegaDetector on videos and sort them by whether they contain interesting content."""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from megadetector.detection.process_video import ProcessVideoOptions, process_videos

VIDEO_EXTENSIONS = {
    ".avi",
    ".mp4",
    ".mov",
    ".mkv",
    ".mpeg",
    ".mpg",
    ".wmv",
    ".m4v",
}

DEFAULT_INTERESTING_CATEGORIES = {"1", "2", "3"}


@dataclass
class VideoDecision:
    """Stores sorting and scoring information for one processed video."""

    relative_path: str
    bucket: str
    top_confidence: float | None
    top_category: str | None
    top_frame: int | None
    num_detections: int
    failure: str | None


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the video processing workflow.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run MegaDetector on videos in an input folder and sort originals into "
            "output/interesting, output/uninteresting, and output/failed."
        )
    )
    parser.add_argument("--input-dir", default="input", help="Folder containing videos")
    parser.add_argument("--output-dir", default="output", help="Folder to receive sorted videos")
    parser.add_argument(
        "--model",
        default="MDV5A",
        help="MegaDetector model identifier or .pt path (default: MDV5A)",
    )
    parser.add_argument(
        "--frame-sample",
        type=int,
        default=5,
        help="Process every Nth frame (default: 5)",
    )
    parser.add_argument(
        "--interesting-threshold",
        type=float,
        default=0.7,
        help="Detection confidence threshold for classifying a video as interesting (default: 0.7)",
    )
    parser.add_argument(
        "--interesting-categories",
        default="1,2,3",
        help=(
            "Comma-separated category IDs considered interesting. "
            "MD default labels are 1=animal, 2=person, 3=vehicle."
        ),
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying them into output buckets",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan input directory for videos",
    )
    return parser.parse_args()


def find_videos(input_dir: Path, recursive: bool) -> list[Path]:
    """Find video files in the input directory.

    Args:
        input_dir: Directory to scan.
        recursive: Whether to search subdirectories recursively.

    Returns:
        list[Path]: Sorted list of matching video files.
    """
    iterator = input_dir.rglob("*") if recursive else input_dir.glob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


def load_results(results_path: Path) -> dict[str, Any]:
    """Load MegaDetector JSON results from disk.

    Args:
        results_path: Path to the MegaDetector JSON output file.

    Returns:
        dict[str, Any]: Parsed JSON payload.
    """
    with results_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_unique_destination(dest: Path) -> Path:
    """Generate a unique destination path when a file already exists.

    Args:
        dest: Desired destination path.

    Returns:
        Path: A path that does not currently exist.
    """
    if not dest.exists():
        return dest

    stem = dest.stem
    suffix = dest.suffix
    index = 1
    while True:
        candidate = dest.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def copy_or_move(src: Path, dst: Path, move: bool) -> None:
    """Copy or move a file to its destination, avoiding collisions.

    Args:
        src: Source file path.
        dst: Target file path.
        move: If True, move the file; otherwise copy it.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst = make_unique_destination(dst)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)


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
            bucket="failed",
            top_confidence=None,
            top_category=None,
            top_frame=None,
            num_detections=0,
            failure=str(image_entry["failure"]),
        )

    detections = image_entry.get("detections") or []
    best: dict[str, Any] | None = None
    interesting_count = 0

    for det in detections:
        confidence = float(det.get("conf", 0.0))
        category = str(det.get("category", ""))
        if category in interesting_categories and confidence >= threshold:
            interesting_count += 1
            if best is None or confidence > float(best.get("conf", 0.0)):
                best = det

    if best is None:
        return VideoDecision(
            relative_path=rel_path,
            bucket="uninteresting",
            top_confidence=None,
            top_category=None,
            top_frame=None,
            num_detections=0,
            failure=None,
        )

    return VideoDecision(
        relative_path=rel_path,
        bucket="interesting",
        top_confidence=float(best.get("conf", 0.0)),
        top_category=str(best.get("category", "")),
        top_frame=int(best.get("frame_number", -1)),
        num_detections=interesting_count,
        failure=None,
    )


def run_detector(
    input_dir: Path,
    results_file: Path,
    model: str,
    frame_sample: int,
    recursive: bool,
) -> None:
    """Run MegaDetector on the input directory and write JSON results.

    Args:
        input_dir: Directory containing videos to process.
        results_file: Destination for MegaDetector JSON output.
        model: MegaDetector model name or file path.
        frame_sample: Process every Nth frame.
        recursive: Whether to scan input recursively.
    """
    print(
        "Starting MegaDetector run: "
        f"input={input_dir}, model={model}, frame_sample={frame_sample}, recursive={recursive}"
    )
    options = ProcessVideoOptions()
    options.input_video_file = str(input_dir)
    options.output_json_file = str(results_file)
    options.model_file = model
    options.frame_sample = frame_sample
    options.recursive = recursive
    options.verbose = False
    process_videos(options)
    print(f"MegaDetector run complete. Results written to {results_file}")


def write_summary(summary_path: Path, decisions: list[VideoDecision], args: argparse.Namespace) -> None:
    """Write a JSON summary file for the run.

    Args:
        summary_path: Output path for the summary JSON.
        decisions: Per-video decisions generated from detector output.
        args: Parsed command-line arguments used for the run.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "input_dir": str(Path(args.input_dir).resolve()),
        "output_dir": str(Path(args.output_dir).resolve()),
        "model": args.model,
        "frame_sample": args.frame_sample,
        "interesting_threshold": args.interesting_threshold,
        "interesting_categories": sorted(
            c.strip() for c in args.interesting_categories.split(",") if c.strip()
        ),
        "videos": [decision.__dict__ for decision in decisions],
        "counts": {
            "interesting": sum(1 for d in decisions if d.bucket == "interesting"),
            "uninteresting": sum(1 for d in decisions if d.bucket == "uninteresting"),
            "failed": sum(1 for d in decisions if d.bucket == "failed"),
            "total": len(decisions),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    """Run the end-to-end video processing workflow.

    Returns:
        int: Process exit code.

    Raises:
        SystemExit: If the input directory is invalid or no videos are found.
    """
    args = parse_args()

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    videos = find_videos(input_dir, args.recursive)
    print(f"Found {len(videos)} video(s) to process")
    if not videos:
        raise SystemExit(f"No videos found in {input_dir}")

    categories = {
        c.strip() for c in args.interesting_categories.split(",") if c.strip()
    } or DEFAULT_INTERESTING_CATEGORIES

    metadata_dir = output_dir / "metadata"
    md_results_path = metadata_dir / "megadetector_results.json"
    summary_path = metadata_dir / "summary.json"

    run_detector(
        input_dir=input_dir,
        results_file=md_results_path,
        model=args.model,
        frame_sample=args.frame_sample,
        recursive=args.recursive,
    )

    print("Loading MegaDetector results for classification")
    results = load_results(md_results_path)
    image_entries: list[dict[str, Any]] = results.get("images", [])
    print(f"Loaded {len(image_entries)} video result record(s)")

    decisions: list[VideoDecision] = []
    total_entries = len(image_entries)
    for index, image_entry in enumerate(image_entries, start=1):
        decision = analyze_video_result(
            image_entry=image_entry,
            interesting_categories=categories,
            threshold=args.interesting_threshold,
        )
        source = input_dir / decision.relative_path
        if source.exists():
            destination = output_dir / decision.bucket / decision.relative_path
            copy_or_move(source, destination, move=args.move)
            action = "Moved" if args.move else "Copied"
            print(
                f"[{index}/{total_entries}] {action} {decision.relative_path} -> {decision.bucket}"
            )
        else:
            print(
                f"[{index}/{total_entries}] Source missing for {decision.relative_path}; skipping copy/move"
            )
        decisions.append(decision)

    write_summary(summary_path=summary_path, decisions=decisions, args=args)
    print(f"Wrote summary metadata to {summary_path}")

    counts = {
        "interesting": sum(1 for d in decisions if d.bucket == "interesting"),
        "uninteresting": sum(1 for d in decisions if d.bucket == "uninteresting"),
        "failed": sum(1 for d in decisions if d.bucket == "failed"),
    }
    print(
        "Finished sorting videos. "
        f"interesting={counts['interesting']}, "
        f"uninteresting={counts['uninteresting']}, "
        f"failed={counts['failed']}"
    )
    print(f"Raw MegaDetector output: {md_results_path}")
    print(f"Summary report: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
