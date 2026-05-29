"""Utilities for extracting preview images from selected video frames."""

from __future__ import annotations

import json
import logging
import re
from contextlib import redirect_stderr
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any

import cv2

logger = logging.getLogger(__name__)

_opencv_noise_suppressed = False


def _suppress_opencv_noise_once() -> None:
    """Best-effort suppression of OpenCV log noise for decoder warnings."""
    global _opencv_noise_suppressed
    if _opencv_noise_suppressed:
        return

    try:
        if hasattr(cv2, "setLogLevel") and hasattr(cv2, "LOG_LEVEL_ERROR"):
            cv2.setLogLevel(cv2.LOG_LEVEL_ERROR)
    except Exception:
        # Keep processing resilient across OpenCV versions/platform builds.
        pass
    _opencv_noise_suppressed = True


@dataclass
class TopFrameRecord:
    """Per-video frame selection metadata for preview extraction."""

    relative_path: str
    output_relative_path: str
    top_frame: int | None
    top_confidence: float | None
    bucket: str
    top_bbox: list[float] | None = None


@dataclass
class PreviewExtractionStats:
    """Outcome counters for one preview extraction run."""

    total_candidates: int
    extracted: int
    skipped: int
    failed: int
    classified: int = 0
    classification_failed: int = 0


@dataclass
class SpeciesClassification:
    """Top SpeciesNet classification result for one preview image."""

    label: str
    score: float
    raw_class: str


def sanitize_label_for_filename(label: str) -> str:
    """Create a filesystem-safe, compact label segment for filenames.

    Returns:
        str: Normalized label suitable for use in filenames.
    """
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    if not cleaned:
        return "unknown"
    return cleaned[:48]


def parse_speciesnet_top_class(classifications: Any) -> SpeciesClassification | None:
    """Extract top class label and score from SpeciesNet classification payload.

    Returns:
        SpeciesClassification | None: Top class details, or None if unavailable.
    """
    if not isinstance(classifications, dict):
        return None

    classes = classifications.get("classes")
    scores = classifications.get("scores")
    if not isinstance(classes, list) or not isinstance(scores, list):
        return None
    if not classes or not scores or len(classes) != len(scores):
        return None

    try:
        top_index = max(range(len(scores)), key=lambda idx: float(scores[idx]))
        raw_class = str(classes[top_index])
        score = float(scores[top_index])
    except (TypeError, ValueError):
        return None

    label = raw_class.split(";")[-1].strip() or "unknown"
    return SpeciesClassification(label=label, score=score, raw_class=raw_class)


def parse_speciesnet_candidates(classifications: Any) -> list[dict[str, Any]]:
    """Convert SpeciesNet classifications payload into candidate rows.

    Returns:
        list[dict[str, Any]]: Candidate list with raw class, display label, and score.
    """
    if not isinstance(classifications, dict):
        return []

    classes = classifications.get("classes")
    scores = classifications.get("scores")
    if not isinstance(classes, list) or not isinstance(scores, list):
        return []

    candidates: list[dict[str, Any]] = []
    for raw_class, score in zip(classes, scores, strict=False):
        try:
            score_value = float(score)
        except (TypeError, ValueError):
            continue
        raw_class_value = str(raw_class)
        candidates.append(
            {
                "raw_class": raw_class_value,
                "label": raw_class_value.split(";")[-1].strip() or "unknown",
                "score": score_value,
            }
        )
    return candidates


def create_species_crop(
    image_path: Path,
    bbox: list[float] | None,
    output_dir: Path,
    padding: float,
) -> Path | None:
    """Create and save a bbox crop for species classification.

    Args:
        image_path: Source preview image path.
        bbox: Normalized bbox values [x, y, width, height].
        output_dir: Destination directory for crop images.
        padding: Extra normalized padding around bbox.

    Returns:
        Path | None: Saved crop path, or None when crop generation fails.
    """
    if bbox is None or len(bbox) != 4:
        return None

    frame = cv2.imread(str(image_path))
    if frame is None:
        return None

    height, width = frame.shape[:2]
    try:
        x, y, w, h = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None

    x1 = max(0.0, x - padding * w)
    y1 = max(0.0, y - padding * h)
    x2 = min(1.0, x + w + padding * w)
    y2 = min(1.0, y + h + padding * h)
    if x2 <= x1 or y2 <= y1:
        return None

    px1 = int(round(x1 * width))
    py1 = int(round(y1 * height))
    px2 = int(round(x2 * width))
    py2 = int(round(y2 * height))
    if px2 <= px1 or py2 <= py1:
        return None

    crop = frame[py1:py2, px1:px2]
    if crop.size == 0:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    crop_path = output_dir / f"{image_path.stem}_crop{image_path.suffix}"
    if not cv2.imwrite(str(crop_path), crop):
        return None

    return crop_path


def classify_preview_images_with_speciesnet(
    image_paths: list[Path],
    model_name: str | None,
    geofence: bool,
) -> tuple[dict[Path, SpeciesClassification], dict[Path, list[dict[str, Any]]], int]:
    """Classify extracted preview images with SpeciesNet.

    Returns:
        tuple[dict[Path, SpeciesClassification], dict[Path, list[dict[str, Any]]], int]:
            Top classifications, full candidate rows by path, and failures.
    """
    if not image_paths:
        return {}, {}, 0

    from speciesnet import DEFAULT_MODEL, SpeciesNet

    resolved_paths = [path.resolve() for path in image_paths]
    model_id = model_name or DEFAULT_MODEL
    classifier = SpeciesNet(model_id, components="classifier", geofence=geofence)
    result = classifier.classify(filepaths=[str(path) for path in resolved_paths], progress_bars=False)

    predictions = result.get("predictions", []) if isinstance(result, dict) else []
    by_path: dict[Path, SpeciesClassification] = {}
    candidates_by_path: dict[Path, list[dict[str, Any]]] = {}
    failures = 0

    for prediction in predictions:
        if not isinstance(prediction, dict):
            failures += 1
            continue

        filepath = prediction.get("filepath")
        if filepath is None:
            failures += 1
            continue

        top_class = parse_speciesnet_top_class(prediction.get("classifications"))
        if top_class is None:
            failures += 1
            continue

        resolved_path = Path(str(filepath)).resolve()
        by_path[resolved_path] = top_class
        candidates_by_path[resolved_path] = parse_speciesnet_candidates(prediction.get("classifications"))

    failures += max(0, len(resolved_paths) - len(by_path))
    return by_path, candidates_by_path, failures


def write_species_classification_report(
    report_path: Path,
    report_data: dict[str, Any],
) -> None:
    """Write detailed SpeciesNet classification report JSON.

    Args:
        report_path: Output JSON file path.
        report_data: Serializable report payload.
    """
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report_data, handle, indent=2)


def build_output_path(output_dir: Path, record: TopFrameRecord) -> Path:
    """Build destination image path for a selected frame.

    Args:
        output_dir: Root preview output directory.
        record: Frame metadata for one video.

    Returns:
        Path: Destination path for preview image.
    """
    relative_video = Path(record.output_relative_path)
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
    _suppress_opencv_noise_once()

    with redirect_stderr(StringIO()):
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


def process_record_for_preview(
    record: TopFrameRecord,
    input_dir: Path,
    output_dir: Path,
    speciesnet_use_crops: bool,
    species_crop_output_dir: Path | None,
    species_crop_padding: float,
    preview_output_paths: dict[str, Path] | None = None,
    species_crop_output_dirs: dict[str, Path] | None = None,
) -> tuple[str, Path | None, Path | None, str]:
    """Extract preview image and optional species crop for one record.

    Returns:
        tuple[str, Path | None, Path | None, str]: status, preview path, classification target, log message.
    """
    if not record.relative_path or record.top_frame is None or int(record.top_frame) < 0:
        return "skipped", None, None, f"Skipped {record.relative_path}: no valid top_frame"

    source_video = input_dir / record.relative_path
    if not source_video.exists():
        return "failed", None, None, f"Failed {record.relative_path}: video not found"

    output_image = build_output_path(output_dir, record)
    if preview_output_paths is not None:
        mapped_output = preview_output_paths.get(record.relative_path)
        if mapped_output is not None:
            output_image = mapped_output
    if not extract_frame(source_video, int(record.top_frame), output_image):
        return "failed", None, None, f"Failed {record.relative_path}: frame extraction error"

    classification_target = output_image.resolve()
    if speciesnet_use_crops and species_crop_output_dir is not None:
        crop_output_dir = species_crop_output_dir
        if species_crop_output_dirs is not None:
            crop_output_dir = species_crop_output_dirs.get(record.relative_path, crop_output_dir)
        crop_path = create_species_crop(
            image_path=output_image,
            bbox=record.top_bbox,
            output_dir=crop_output_dir,
            padding=species_crop_padding,
        )
        if crop_path is not None:
            classification_target = crop_path.resolve()
            message = f"Wrote {output_image} and species crop {crop_path}"
            return "extracted", output_image, classification_target, message

    return "extracted", output_image, classification_target, f"Wrote {output_image}"


def run_speciesnet_postprocessing(
    extracted_paths: list[Path],
    classification_targets: dict[Path, Path],
    source_paths_by_preview: dict[Path, str],
    speciesnet_model: str | None,
    speciesnet_geofence: bool,
    include_label_in_filename: bool,
    speciesnet_use_crops: bool,
    species_crop_padding: float,
    species_classification_report_path: Path | None,
) -> tuple[int, int]:
    """Classify preview images, optionally rename files, and write a detailed report.

    Returns:
        tuple[int, int]: classified count and classification failure count.

    Raises:
        SystemExit: If SpeciesNet is not installed in the active Python environment.
    """
    try:
        classifications, candidates_by_path, classification_failed = classify_preview_images_with_speciesnet(
            image_paths=list(classification_targets.values()),
            model_name=speciesnet_model,
            geofence=speciesnet_geofence,
        )
        classified = len(classifications)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Missing required SpeciesNet dependency. Install runtime packages from requirements.txt"
        ) from exc
    except Exception as exc:
        classifications = {}
        candidates_by_path = {}
        classification_failed = len(extracted_paths)
        logger.warning("SpeciesNet classification failed: %s", exc)
        classified = 0

    renamed_paths: dict[Path, Path] = {
        output_image.resolve(): output_image.resolve() for output_image in extracted_paths
    }

    if include_label_in_filename:
        for output_image in extracted_paths:
            classification_target = classification_targets.get(output_image.resolve(), output_image.resolve())
            classification = classifications.get(classification_target)
            if classification is None:
                continue
            safe_label = sanitize_label_for_filename(classification.label)
            score_label = f"{classification.score:.3f}"
            renamed = output_image.with_name(
                f"{output_image.stem}_species-{safe_label}_sp{score_label}{output_image.suffix}"
            )
            if renamed.exists():
                renamed.unlink()
            output_image.rename(renamed)
            renamed_paths[output_image.resolve()] = renamed.resolve()

    if species_classification_report_path is not None:
        report_entries: list[dict[str, Any]] = []
        for output_image in extracted_paths:
            output_image_resolved = output_image.resolve()
            classification_target = classification_targets.get(output_image_resolved, output_image_resolved)
            top_class = classifications.get(classification_target)
            candidates = candidates_by_path.get(classification_target, [])
            report_entries.append(
                {
                    "source_relative_path": source_paths_by_preview.get(output_image_resolved, ""),
                    "preview_image_original": str(output_image_resolved),
                    "preview_image_final": str(renamed_paths.get(output_image_resolved, output_image_resolved)),
                    "classification_input_image": str(classification_target),
                    "used_species_crop": classification_target != output_image_resolved,
                    "top_classification": None
                    if top_class is None
                    else {
                        "label": top_class.label,
                        "score": top_class.score,
                        "raw_class": top_class.raw_class,
                    },
                    "candidates": candidates,
                }
            )

        report_payload = {
            "speciesnet_model": speciesnet_model,
            "speciesnet_geofence": speciesnet_geofence,
            "speciesnet_use_crops": speciesnet_use_crops,
            "species_crop_padding": species_crop_padding,
            "total_entries": len(report_entries),
            "entries": report_entries,
        }
        write_species_classification_report(species_classification_report_path, report_payload)
        logger.debug("Wrote species classification report: %s", species_classification_report_path)

    return classified, classification_failed


def extract_top_frames(
    records: list[TopFrameRecord],
    input_dir: Path,
    output_dir: Path,
    include_uninteresting: bool,
    classify_with_speciesnet: bool = False,
    speciesnet_model: str | None = None,
    speciesnet_geofence: bool = False,
    include_label_in_filename: bool = True,
    speciesnet_use_crops: bool = True,
    species_crop_output_dir: Path | None = None,
    species_crop_padding: float = 0.15,
    species_classification_report_path: Path | None = None,
    preview_output_paths: dict[str, Path] | None = None,
    species_crop_output_dirs: dict[str, Path] | None = None,
) -> PreviewExtractionStats:
    """Extract top-frame preview images for selected records.

    Args:
        records: Candidate records to process.
        input_dir: Root directory where source videos exist.
        output_dir: Root directory for generated preview images.
        include_uninteresting: Include uninteresting bucket records when true.
        classify_with_speciesnet: Classify extracted images with SpeciesNet.
        speciesnet_model: SpeciesNet model identifier, or None for default.
        speciesnet_geofence: Apply SpeciesNet geofencing if supported.
        include_label_in_filename: Rename previews to include top class and score.
        speciesnet_use_crops: Use bbox crops for species classification when possible.
        species_crop_output_dir: Destination folder for saved crop images.
        species_crop_padding: Extra normalized padding around bbox.
        species_classification_report_path: Optional path for detailed classification JSON.
        preview_output_paths: Optional per-video preview output paths keyed by staged relative path.
        species_crop_output_dirs: Optional per-video crop output dirs keyed by staged relative path.

    Returns:
        PreviewExtractionStats: Aggregated extraction counters.
    """
    extracted = 0
    skipped = 0
    failed = 0
    classified = 0
    classification_failed = 0
    extracted_paths: list[Path] = []
    classification_targets: dict[Path, Path] = {}
    source_paths_by_preview: dict[Path, str] = {}

    filtered_records = [
        record
        for record in records
        if include_uninteresting or record.bucket == "interesting"
    ]

    total = len(filtered_records)
    for _index, record in enumerate(filtered_records, start=1):
        status, output_image, classification_target, message = process_record_for_preview(
            record=record,
            input_dir=input_dir,
            output_dir=output_dir,
            speciesnet_use_crops=speciesnet_use_crops,
            species_crop_output_dir=species_crop_output_dir,
            species_crop_padding=species_crop_padding,
            preview_output_paths=preview_output_paths,
            species_crop_output_dirs=species_crop_output_dirs,
        )

        if status == "skipped":
            skipped += 1
            logger.debug(message)
            continue
        if status == "failed":
            failed += 1
            logger.debug(message)
            continue

        extracted += 1
        assert output_image is not None
        assert classification_target is not None
        extracted_paths.append(output_image)
        classification_targets[output_image.resolve()] = classification_target
        source_paths_by_preview[output_image.resolve()] = record.output_relative_path
        logger.debug(message)

    if classify_with_speciesnet and extracted_paths:
        logger.debug("Running SpeciesNet classification on extracted previews")
        classified, classification_failed = run_speciesnet_postprocessing(
            extracted_paths=extracted_paths,
            classification_targets=classification_targets,
            source_paths_by_preview=source_paths_by_preview,
            speciesnet_model=speciesnet_model,
            speciesnet_geofence=speciesnet_geofence,
            include_label_in_filename=include_label_in_filename,
            speciesnet_use_crops=speciesnet_use_crops,
            species_crop_padding=species_crop_padding,
            species_classification_report_path=species_classification_report_path,
        )

    return PreviewExtractionStats(
        total_candidates=total,
        extracted=extracted,
        skipped=skipped,
        failed=failed,
        classified=classified,
        classification_failed=classification_failed,
    )
