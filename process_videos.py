#!/usr/bin/env python3
"""Run MegaDetector on videos and sort them by whether they contain interesting content."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from megadetector.detection.process_video import ProcessVideoOptions, process_videos

from preview_frames import PreviewExtractionStats, TopFrameRecord, extract_top_frames
from video_clipping import clip_video_by_frame_window, transcode_video_for_web

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
DEFAULT_CONFIG_PATH = "process_videos.config.yaml"


@dataclass
class VideoDecision:
    """Stores sorting and scoring information for one processed video."""

    relative_path: str
    bucket: str
    top_confidence: float | None
    top_category: str | None
    top_frame: int | None
    top_bbox: list[float] | None
    first_interesting_frame: int | None
    last_interesting_frame: int | None
    num_detections: int
    failure: str | None


@dataclass
class AppConfig:
    """Runtime settings loaded from the YAML config file."""

    input_dir: str
    output_dir: str
    model: str
    frame_sample: int
    interesting_threshold: float
    interesting_categories: list[str]
    move_files: bool
    save_uninteresting_files: bool
    clip_interesting_videos: bool
    run_folder_mode: str
    recursive: bool
    detector_verbose: bool
    generate_html_report: bool
    generate_top_frame_previews: bool
    preview_output_dir: str
    preview_include_uninteresting: bool
    classify_previews_with_speciesnet: bool
    speciesnet_model: str
    speciesnet_geofence: bool
    speciesnet_label_in_filename: bool
    speciesnet_use_crops: bool
    species_crop_output_dir: str
    species_crop_padding: float


@dataclass
class RunPaths:
    """Resolved paths used during a processing run."""

    config_path: Path
    input_dir: Path
    output_root_dir: Path
    output_dir: Path
    run_id: str | None
    metadata_dir: Path
    md_results_path: Path
    summary_path: Path
    html_summary_path: Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the video processing workflow.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run MegaDetector on videos and sort originals using values "
            "loaded from a YAML config file."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=(
            "Path to YAML configuration file "
            f"(default: {DEFAULT_CONFIG_PATH})"
        ),
    )
    return parser.parse_args()


def load_config(config_path: Path) -> AppConfig:
    """Load and validate application settings from a YAML file.

    Args:
        config_path: Path to YAML configuration.

    Returns:
        AppConfig: Parsed and validated configuration values.

    Raises:
        SystemExit: If the config file is missing or has invalid values.
    """
    if not config_path.exists():
        raise SystemExit(f"Config file does not exist: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    raw_config = loaded if isinstance(loaded, dict) else {}

    try:
        input_dir = str(raw_config["input_dir"])
        output_dir = str(raw_config["output_dir"])
        model = str(raw_config.get("model", "MDV5A"))
        frame_sample = int(raw_config.get("frame_sample", 5))
        interesting_threshold = float(raw_config.get("interesting_threshold", 0.7))
        categories_raw = raw_config.get("interesting_categories", ["1", "2", "3"])
        if not isinstance(categories_raw, list):
            raise SystemExit("interesting_categories must be a YAML list")
        categories = [str(c).strip() for c in categories_raw if str(c).strip()]
        move_files = bool(raw_config.get("move_files", False))
        save_uninteresting_files = bool(raw_config.get("save_uninteresting_files", True))
        clip_interesting_videos = bool(raw_config.get("clip_interesting_videos", True))
        run_folder_mode = str(raw_config.get("run_folder_mode", "none")).strip().lower()
        recursive = bool(raw_config.get("recursive", False))
        detector_verbose = bool(raw_config.get("detector_verbose", False))
        generate_html_report = bool(raw_config.get("generate_html_report", True))
        generate_top_frame_previews = bool(raw_config.get("generate_top_frame_previews", True))
        preview_output_dir = str(raw_config.get("preview_output_dir", "preview_frames"))
        preview_include_uninteresting = bool(raw_config.get("preview_include_uninteresting", False))
        classify_previews_with_speciesnet = bool(raw_config.get("classify_previews_with_speciesnet", True))
        speciesnet_model = str(raw_config.get("speciesnet_model", ""))
        speciesnet_geofence = bool(raw_config.get("speciesnet_geofence", False))
        speciesnet_label_in_filename = bool(raw_config.get("speciesnet_label_in_filename", True))
        speciesnet_use_crops = bool(raw_config.get("speciesnet_use_crops", True))
        species_crop_output_dir = str(raw_config.get("species_crop_output_dir", "preview_species_crops"))
        species_crop_padding = float(raw_config.get("species_crop_padding", 0.15))
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"Invalid config file {config_path}: {exc}") from exc

    if frame_sample <= 0:
        raise SystemExit("frame_sample must be greater than 0")
    if not 0.0 <= interesting_threshold <= 1.0:
        raise SystemExit("interesting_threshold must be between 0.0 and 1.0")
    if run_folder_mode not in {"none", "timestamped"}:
        raise SystemExit("run_folder_mode must be either 'none' or 'timestamped'")
    if not 0.0 <= species_crop_padding <= 1.0:
        raise SystemExit("species_crop_padding must be between 0.0 and 1.0")

    return AppConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        model=model,
        frame_sample=frame_sample,
        interesting_threshold=interesting_threshold,
        interesting_categories=categories,
        move_files=move_files,
        save_uninteresting_files=save_uninteresting_files,
        clip_interesting_videos=clip_interesting_videos,
        run_folder_mode=run_folder_mode,
        recursive=recursive,
        detector_verbose=detector_verbose,
        generate_html_report=generate_html_report,
        generate_top_frame_previews=generate_top_frame_previews,
        preview_output_dir=preview_output_dir,
        preview_include_uninteresting=preview_include_uninteresting,
        classify_previews_with_speciesnet=classify_previews_with_speciesnet,
        speciesnet_model=speciesnet_model,
        speciesnet_geofence=speciesnet_geofence,
        speciesnet_label_in_filename=speciesnet_label_in_filename,
        speciesnet_use_crops=speciesnet_use_crops,
        species_crop_output_dir=species_crop_output_dir,
        species_crop_padding=species_crop_padding,
    )


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


def run_detector(
    input_dir: Path,
    results_file: Path,
    model: str,
    frame_sample: int,
    recursive: bool,
    verbose: bool,
) -> None:
    """Run MegaDetector on the input directory and write JSON results.

    Args:
        input_dir: Directory containing videos to process.
        results_file: Destination for MegaDetector JSON output.
        model: MegaDetector model name or file path.
        frame_sample: Process every Nth frame.
        recursive: Whether to scan input recursively.
        verbose: Whether to enable MegaDetector verbose output.
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
    options.verbose = verbose
    process_videos(options)
    print(f"MegaDetector run complete. Results written to {results_file}")


def write_summary(
    summary_path: Path,
    decisions: list[VideoDecision],
    config: AppConfig,
    config_path: Path,
    preview_stats: PreviewExtractionStats | None,
    preview_output_dir: Path,
    run_output_dir: Path,
    run_id: str | None,
    species_classification_report_path: Path,
) -> None:
    """Write a JSON summary file for the run.

    Args:
        summary_path: Output path for the summary JSON.
        decisions: Per-video decisions generated from detector output.
        config: Runtime configuration used for this run.
        config_path: Path to the config file used for this run.
        preview_stats: Optional statistics from preview frame extraction.
        preview_output_dir: Resolved output folder for preview frames.
        run_output_dir: Resolved root output folder for this run.
        run_id: Timestamp-based run ID when run_folder_mode is timestamped.
        species_classification_report_path: Path to detailed species classification report JSON.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config_file": str(config_path.resolve()),
        "input_dir": str(Path(config.input_dir).resolve()),
        "output_root_dir": str(Path(config.output_dir).resolve()),
        "output_dir": str(run_output_dir),
        "run_folder_mode": config.run_folder_mode,
        "run_id": run_id,
        "model": config.model,
        "frame_sample": config.frame_sample,
        "interesting_threshold": config.interesting_threshold,
        "interesting_categories": sorted(config.interesting_categories),
        "move_files": config.move_files,
        "save_uninteresting_files": config.save_uninteresting_files,
        "clip_interesting_videos": config.clip_interesting_videos,
        "recursive": config.recursive,
        "detector_verbose": config.detector_verbose,
        "generate_html_report": config.generate_html_report,
        "generate_top_frame_previews": config.generate_top_frame_previews,
        "preview_output_dir": str(preview_output_dir),
        "preview_include_uninteresting": config.preview_include_uninteresting,
        "classify_previews_with_speciesnet": config.classify_previews_with_speciesnet,
        "speciesnet_model": config.speciesnet_model,
        "speciesnet_geofence": config.speciesnet_geofence,
        "speciesnet_label_in_filename": config.speciesnet_label_in_filename,
        "speciesnet_use_crops": config.speciesnet_use_crops,
        "species_crop_output_dir": str(resolve_preview_output_dir(config.species_crop_output_dir, run_output_dir)),
        "species_crop_padding": config.species_crop_padding,
        "species_classification_report": str(species_classification_report_path),
        "videos": [decision.__dict__ for decision in decisions],
        "counts": {
            "interesting": sum(1 for d in decisions if d.bucket == "interesting"),
            "uninteresting": sum(1 for d in decisions if d.bucket == "uninteresting"),
            "failed": sum(1 for d in decisions if d.bucket == "failed"),
            "total": len(decisions),
        },
    }
    if preview_stats is not None:
        payload["preview_frames"] = {
            "total_candidates": preview_stats.total_candidates,
            "extracted": preview_stats.extracted,
            "skipped": preview_stats.skipped,
            "failed": preview_stats.failed,
            "classified": preview_stats.classified,
            "classification_failed": preview_stats.classification_failed,
        }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def relpath_from(base_dir: Path, target: Path) -> str:
    """Compute a display-friendly relative path when possible.

    Args:
        base_dir: Base directory for relative paths.
        target: Target path to relativize.

    Returns:
        str: Relative path when possible, otherwise absolute path.
    """
    try:
        return os.path.relpath(target.resolve(), start=base_dir.resolve())
    except OSError:
        return str(target.resolve())


def render_candidate_list(candidates: list[dict[str, Any]], limit: int = 5) -> str:
    """Render top candidate species labels as HTML list items.

    Args:
        candidates: Candidate rows from SpeciesNet output.
        limit: Maximum number of candidates to render.

    Returns:
        str: HTML fragment for candidate list items.
    """
    items: list[str] = []
    for candidate in candidates[:limit]:
        label = str(candidate.get("label", "unknown"))
        try:
            score = float(candidate.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        items.append(f"<li>{html_escape(label)} <span>{score:.3f}</span></li>")
    if not items:
        items.append("<li>No candidates</li>")
    return "".join(items)


def html_escape(value: Any) -> str:
    """Escape plain text for safe insertion into HTML.

    Args:
        value: Value to escape.

    Returns:
        str: Escaped text.
    """
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_html_summary(
    html_summary_path: Path,
    decisions: list[VideoDecision],
    run_output_dir: Path,
    species_classification_report_path: Path,
) -> None:
    """Write an HTML summary report for interesting detections.

    Args:
        html_summary_path: Destination HTML file path.
        decisions: Per-video decisions generated from detector output.
        run_output_dir: Root output directory for the current run.
        species_classification_report_path: JSON report created by SpeciesNet postprocessing.
    """
    html_summary_path.parent.mkdir(parents=True, exist_ok=True)

    species_entries: dict[str, dict[str, Any]] = {}
    if species_classification_report_path.exists():
        with species_classification_report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        for entry in report.get("entries", []):
            relative_path = entry.get("source_relative_path")
            if isinstance(relative_path, str) and relative_path:
                species_entries[relative_path] = entry

    interesting_decisions = [d for d in decisions if d.bucket == "interesting"]
    web_video_dir = run_output_dir / "report_videos"
    row_fragments: list[str] = []
    for decision in interesting_decisions:
        species_entry = species_entries.get(decision.relative_path, {})
        top_classification = species_entry.get("top_classification") or {}
        top_label = html_escape(top_classification.get("label", "unknown"))
        escaped_relative_path = html_escape(decision.relative_path)
        try:
            top_score = float(top_classification.get("score", 0.0))
        except (TypeError, ValueError):
            top_score = 0.0

        video_path = run_output_dir / decision.bucket / decision.relative_path
        web_video_path = web_video_dir / f"{Path(decision.relative_path).stem}.mp4"
        web_video_href = ""
        if video_path.exists():
            transcoded_path = transcode_video_for_web(video_path, web_video_path)
            if transcoded_path is not None:
                web_video_href = html_escape(
                    relpath_from(html_summary_path.parent, transcoded_path)
                )
        preview_path = species_entry.get("preview_image_final")
        crop_path = species_entry.get("classification_input_image")
        preview_href = (
            html_escape(relpath_from(html_summary_path.parent, Path(str(preview_path))))
            if preview_path
            else ""
        )
        crop_href = (
            html_escape(relpath_from(html_summary_path.parent, Path(str(crop_path))))
            if crop_path
            else ""
        )
        video_href = html_escape(relpath_from(html_summary_path.parent, video_path))

        candidates_html = render_candidate_list(species_entry.get("candidates", []))
        row_fragments.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{html_escape(decision.relative_path)}</td>",
                    f"<td>{top_label}<br><small>{top_score:.3f}</small></td>",
                    "<td>",
                    (
                        "".join(
                            [
                                "<div class=\"media-stack\">",
                                f"<video controls preload=\"metadata\" src=\"{web_video_href}\"></video>",
                                "<div class=\"link-row\">",
                                f"<a href=\"{web_video_href}\" target=\"_blank\">Open browser video</a>",
                                f"<a href=\"{video_href}\" target=\"_blank\">Open native clip</a>",
                                "</div>",
                                "</div>",
                            ]
                        )
                        if web_video_href
                        else f"<a href=\"{video_href}\" target=\"_blank\">Open native clip</a>"
                    ),
                    "</td>",
                    (
                        "".join(
                            [
                                "<td><div class=\"media-stack\">",
                                (
                                    f"<a href=\"{crop_href}\" target=\"_blank\">"
                                    f"<img class=\"thumb\" src=\"{crop_href}\" "
                                    f"alt=\"Species crop for {escaped_relative_path}\"></a>"
                                ),
                                f"<a href=\"{crop_href}\" target=\"_blank\">species crop</a>",
                                "</div></td>",
                            ]
                        )
                        if crop_href
                        else "<td>n/a</td>"
                    ),
                    (
                        "".join(
                            [
                                "<td><div class=\"media-stack\">",
                                (
                                    f"<a href=\"{preview_href}\" target=\"_blank\">"
                                    f"<img class=\"thumb\" src=\"{preview_href}\" "
                                    f"alt=\"Preview image for {escaped_relative_path}\"></a>"
                                ),
                                f"<a href=\"{preview_href}\" target=\"_blank\">preview image</a>",
                                "</div></td>",
                            ]
                        )
                        if preview_href
                        else "<td>n/a</td>"
                    ),
                    f"<td><ul>{candidates_html}</ul></td>",
                    "</tr>",
                ]
            )
        )

    page = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
    <meta charset=\"utf-8\">
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
    <title>Trail Camera Summary</title>
    <style>
        :root {{
            color-scheme: light;
            --bg: #f4efe6;
            --panel: #fffaf2;
            --ink: #1f1c17;
            --muted: #6f665b;
            --line: #d8cdbd;
            --accent: #2e6f40;
            --accent-soft: #e3f0e6;
        }}
        body {{
            margin: 0;
            font-family: Georgia, \"Iowan Old Style\", serif;
            background: radial-gradient(circle at top, #fff9ef 0%, var(--bg) 58%);
            color: var(--ink);
        }}
        main {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 32px 20px 48px;
        }}
        h1 {{
            margin: 0 0 8px;
            font-size: 2.4rem;
        }}
        p {{
            color: var(--muted);
            margin: 0 0 24px;
        }}
        .panel {{
            background: color-mix(in srgb, var(--panel) 94%, white);
            border: 1px solid var(--line);
            border-radius: 18px;
            overflow: hidden;
            box-shadow: 0 18px 44px rgba(44, 36, 20, 0.08);
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 14px 16px;
            vertical-align: top;
            border-bottom: 1px solid var(--line);
            text-align: left;
        }}
        th {{
            background: #efe5d5;
            font-size: 0.92rem;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }}
        tr:nth-child(even) td {{
            background: rgba(255, 255, 255, 0.45);
        }}
        a {{
            color: var(--accent);
            text-decoration: none;
            font-weight: 600;
        }}
        a:hover {{
            text-decoration: underline;
        }}
        video, .thumb {{
            width: 100%;
            max-width: 240px;
            border-radius: 12px;
            border: 1px solid var(--line);
            background: #000;
            display: block;
        }}
        ul {{
            margin: 0;
            padding-left: 18px;
        }}
        li span {{
            color: var(--muted);
        }}
        .media-stack {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}
        .link-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }}
        .meta {{
            display: inline-block;
            margin: 0 10px 10px 0;
            padding: 8px 12px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.95rem;
        }}
        @media (max-width: 900px) {{
            table, thead, tbody, th, td, tr {{
                display: block;
            }}
            thead {{
                display: none;
            }}
            tr {{
                border-bottom: 1px solid var(--line);
            }}
            td {{
                border-bottom: none;
                padding-top: 8px;
                padding-bottom: 8px;
            }}
        }}
    </style>
</head>
<body>
    <main>
        <h1>Trail Camera Summary</h1>
        <p>Interesting detections with clipped videos, best-frame previews, crop images, and SpeciesNet candidates.</p>
        <div>
            <span class=\"meta\">Interesting videos: {len(interesting_decisions)}</span>
            <span class=\"meta\">Species report: {html_escape(species_classification_report_path.name)}</span>
        </div>
        <section class=\"panel\">
            <table>
                <thead>
                    <tr>
                        <th>Video</th>
                        <th>Top ID</th>
                        <th>Clipped Video</th>
                        <th>Crop</th>
                        <th>Preview</th>
                        <th>Candidates</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(row_fragments) or '<tr><td colspan="6">No interesting videos found.</td></tr>'}
                </tbody>
            </table>
        </section>
    </main>
</body>
</html>
"""

    with html_summary_path.open("w", encoding="utf-8") as handle:
        handle.write(page)


def build_run_paths(config_path: Path, config: AppConfig) -> RunPaths:
    """Build resolved filesystem paths used throughout one run.

    Args:
        config_path: Path to the configuration file.
        config: Loaded runtime configuration values.

    Returns:
        RunPaths: All resolved paths needed for processing.
    """
    output_root_dir = Path(config.output_dir).resolve()
    run_id: str | None = None
    if config.run_folder_mode == "timestamped":
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_root_dir / "runs" / run_id
    else:
        output_dir = output_root_dir

    metadata_dir = output_dir / "metadata"
    return RunPaths(
        config_path=config_path.resolve(),
        input_dir=Path(config.input_dir).resolve(),
        output_root_dir=output_root_dir,
        output_dir=output_dir,
        run_id=run_id,
        metadata_dir=metadata_dir,
        md_results_path=metadata_dir / "megadetector_results.json",
        summary_path=metadata_dir / "summary.json",
        html_summary_path=metadata_dir / "summary.html",
    )


def resolve_preview_output_dir(preview_output_dir: str, run_output_dir: Path) -> Path:
    """Resolve configured preview output location for the current run.

    Args:
        preview_output_dir: Configured preview output directory.
        run_output_dir: Root output directory for the current run.

    Returns:
        Path: Resolved preview output path.
    """
    configured_path = Path(preview_output_dir)
    if configured_path.is_absolute():
        return configured_path.resolve()
    return (run_output_dir / configured_path).resolve()


def validate_and_find_videos(input_dir: Path, recursive: bool) -> list[Path]:
    """Validate input directory and return matching video files.

    Args:
        input_dir: Directory expected to contain videos.
        recursive: Whether to search subdirectories.

    Returns:
        list[Path]: Video files discovered for processing.

    Raises:
        SystemExit: If input directory is missing/invalid or no videos are found.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    videos = find_videos(input_dir, recursive)
    print(f"Found {len(videos)} video(s) to process")
    if not videos:
        raise SystemExit(f"No videos found in {input_dir}")
    return videos


def resolve_interesting_categories(config: AppConfig) -> set[str]:
    """Normalize interesting category IDs from config.

    Args:
        config: Loaded runtime configuration.

    Returns:
        set[str]: Category IDs used to classify videos as interesting.
    """
    categories = {
        category.strip() for category in config.interesting_categories if category.strip()
    }
    return categories or DEFAULT_INTERESTING_CATEGORIES


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
        should_write_file = decision.bucket != "uninteresting" or save_uninteresting_files

        if not should_write_file:
            print(
                f"[{index}/{total_entries}] Skipped {decision.relative_path}: "
                "uninteresting output disabled"
            )
            decisions.append(decision)
            continue

        if source.exists():
            destination = output_dir / decision.bucket / decision.relative_path
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
                    if move_files:
                        source.unlink(missing_ok=True)
                    print(
                        f"[{index}/{total_entries}] Clipped {decision.relative_path} "
                        f"frames {clip_result.start_frame}-{clip_result.end_frame} -> {decision.bucket}"
                    )
                else:
                    copy_or_move(source, destination, move=move_files)
                    action = "Moved" if move_files else "Copied"
                    print(
                        f"[{index}/{total_entries}] {action} {decision.relative_path} -> {decision.bucket} "
                        "(clip failed, saved full video)"
                    )
            else:
                copy_or_move(source, destination, move=move_files)
                action = "Moved" if move_files else "Copied"
                print(f"[{index}/{total_entries}] {action} {decision.relative_path} -> {decision.bucket}")
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


def extract_preview_frames_for_decisions(
    decisions: list[VideoDecision],
    input_dir: Path,
    preview_output_dir: Path,
    include_uninteresting: bool,
    classify_with_speciesnet: bool,
    speciesnet_model: str,
    speciesnet_geofence: bool,
    speciesnet_label_in_filename: bool,
    speciesnet_use_crops: bool,
    species_crop_output_dir: Path,
    species_crop_padding: float,
    species_classification_report_path: Path,
) -> PreviewExtractionStats:
    """Extract preview images for selected decisions.

    Args:
        decisions: Per-video decisions from classification.
        input_dir: Root folder containing source videos.
        preview_output_dir: Destination folder for preview images.
        include_uninteresting: Include uninteresting videos when true.
        classify_with_speciesnet: Run species classification on preview images.
        speciesnet_model: SpeciesNet model identifier, empty for default.
        speciesnet_geofence: Use geofence filtering in SpeciesNet.
        speciesnet_label_in_filename: Append species label and score to filenames.
        speciesnet_use_crops: Use MegaDetector bboxes to classify cropped images.
        species_crop_output_dir: Destination folder for saved classification crops.
        species_crop_padding: Extra context around bbox as normalized padding.
        species_classification_report_path: Output JSON path for full species candidates and scores.

    Returns:
        PreviewExtractionStats: Frame extraction summary counters.
    """
    records = [
        TopFrameRecord(
            relative_path=decision.relative_path,
            top_frame=decision.top_frame,
            top_confidence=decision.top_confidence,
            bucket=decision.bucket,
            top_bbox=decision.top_bbox,
        )
        for decision in decisions
    ]
    return extract_top_frames(
        records=records,
        input_dir=input_dir,
        output_dir=preview_output_dir,
        include_uninteresting=include_uninteresting,
        classify_with_speciesnet=classify_with_speciesnet,
        speciesnet_model=speciesnet_model or None,
        speciesnet_geofence=speciesnet_geofence,
        include_label_in_filename=speciesnet_label_in_filename,
        speciesnet_use_crops=speciesnet_use_crops,
        species_crop_output_dir=species_crop_output_dir,
        species_crop_padding=species_crop_padding,
        species_classification_report_path=species_classification_report_path,
    )


def main() -> int:
    """Run the end-to-end video processing workflow.

    Returns:
        int: Process exit code.
    """
    args = parse_args()
    config = load_config(Path(args.config))
    paths = build_run_paths(Path(args.config), config)

    print(f"Using config file: {paths.config_path}")
    print(f"Input directory: {paths.input_dir}")
    print(f"Output root directory: {paths.output_root_dir}")
    print(f"Run output directory: {paths.output_dir}")
    if paths.run_id is not None:
        print(f"Run ID: {paths.run_id}")

    validate_and_find_videos(paths.input_dir, config.recursive)
    categories = resolve_interesting_categories(config)

    run_detector(
        input_dir=paths.input_dir,
        results_file=paths.md_results_path,
        model=config.model,
        frame_sample=config.frame_sample,
        recursive=config.recursive,
        verbose=config.detector_verbose,
    )

    print("Loading MegaDetector results for classification")
    results = load_results(paths.md_results_path)
    image_entries: list[dict[str, Any]] = results.get("images", [])
    print(f"Loaded {len(image_entries)} video result record(s)")
    decisions = classify_and_sort_videos(
        image_entries=image_entries,
        input_dir=paths.input_dir,
        output_dir=paths.output_dir,
        interesting_categories=categories,
        threshold=config.interesting_threshold,
        move_files=config.move_files,
        save_uninteresting_files=config.save_uninteresting_files,
        clip_interesting_videos=config.clip_interesting_videos,
        clip_buffer_frames=config.frame_sample,
    )

    preview_stats: PreviewExtractionStats | None = None
    preview_output_dir = resolve_preview_output_dir(config.preview_output_dir, paths.output_dir)
    species_crop_output_dir = resolve_preview_output_dir(config.species_crop_output_dir, paths.output_dir)
    species_classification_report_path = paths.metadata_dir / "species_classifications.json"
    if config.generate_top_frame_previews:
        print(f"Extracting top-frame previews to {preview_output_dir}")
        if config.classify_previews_with_speciesnet and config.speciesnet_use_crops:
            print(f"Species classification crops will be saved to {species_crop_output_dir}")
        preview_stats = extract_preview_frames_for_decisions(
            decisions=decisions,
            input_dir=paths.input_dir,
            preview_output_dir=preview_output_dir,
            include_uninteresting=config.preview_include_uninteresting,
            classify_with_speciesnet=config.classify_previews_with_speciesnet,
            speciesnet_model=config.speciesnet_model,
            speciesnet_geofence=config.speciesnet_geofence,
            speciesnet_label_in_filename=config.speciesnet_label_in_filename,
            speciesnet_use_crops=config.speciesnet_use_crops,
            species_crop_output_dir=species_crop_output_dir,
            species_crop_padding=config.species_crop_padding,
            species_classification_report_path=species_classification_report_path,
        )
        print(
            "Preview extraction complete. "
            f"candidates={preview_stats.total_candidates}, "
            f"extracted={preview_stats.extracted}, "
            f"skipped={preview_stats.skipped}, "
            f"failed={preview_stats.failed}, "
            f"classified={preview_stats.classified}, "
            f"classification_failed={preview_stats.classification_failed}"
        )
    else:
        print("Preview extraction disabled by config")

    write_summary(
        summary_path=paths.summary_path,
        decisions=decisions,
        config=config,
        config_path=paths.config_path,
        preview_stats=preview_stats,
        preview_output_dir=preview_output_dir,
        run_output_dir=paths.output_dir,
        run_id=paths.run_id,
        species_classification_report_path=species_classification_report_path,
    )
    print(f"Wrote summary metadata to {paths.summary_path}")
    if config.generate_html_report:
        write_html_summary(
            html_summary_path=paths.html_summary_path,
            decisions=decisions,
            run_output_dir=paths.output_dir,
            species_classification_report_path=species_classification_report_path,
        )
        print(f"Wrote HTML summary to {paths.html_summary_path}")
    else:
        print("HTML summary disabled by config")

    counts = compute_bucket_counts(decisions)
    print(
        "Finished sorting videos. "
        f"interesting={counts['interesting']}, "
        f"uninteresting={counts['uninteresting']}, "
        f"failed={counts['failed']}"
    )
    print(f"Raw MegaDetector output: {paths.md_results_path}")
    print(f"Summary report: {paths.summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
