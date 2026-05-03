#!/usr/bin/env python3
"""Run MegaDetector on videos and sort them by whether they contain interesting content."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from classification import classify_and_sort_videos, compute_bucket_counts
from detector_runner import load_results, run_detector
from file_ops import validate_and_find_videos
from pipeline_config import (
    DEFAULT_CONFIG_PATH,
    build_run_paths,
    load_config,
    resolve_interesting_categories,
    resolve_preview_output_dir,
)
from pipeline_models import VideoDecision
from preview_frames import PreviewExtractionStats, TopFrameRecord, extract_top_frames
from reporting import open_file_in_default_app, write_html_summary, write_summary


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
        species_crop_output_dir=species_crop_output_dir,
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
        if config.auto_open_html_report:
            if open_file_in_default_app(paths.html_summary_path):
                print(f"Opened HTML summary in default app: {paths.html_summary_path}")
            else:
                print(f"Failed to open HTML summary automatically: {paths.html_summary_path}")
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
