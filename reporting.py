"""Reporting helpers for trail camera processing runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from file_ops import is_image_file
from metadata_store import fetch_catalog_snapshot
from pipeline_models import AppConfig, VideoDecision

if TYPE_CHECKING:
    pass


def write_sqlite_snapshot_export(
    summary_path: Path,
    config: AppConfig,
    config_path: Path,
    run_output_dir: Path,
    metadata_db_path: Path,
) -> None:
    """Write compatibility summary JSON as a snapshot exported from SQLite.

    Args:
        summary_path: Destination JSON path.
        config: Runtime configuration values.
        config_path: Config file path used for the run.
        run_output_dir: Current run output directory.
        metadata_db_path: SQLite metadata catalog path.
    """
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = fetch_catalog_snapshot(metadata_db_path)
    payload = {
        "config_file": str(config_path.resolve()),
        "input_dir": str(Path(config.input_dir).resolve()),
        "output_root_dir": str(Path(config.output_dir).resolve()),
        "output_dir": str(run_output_dir),
        "metadata_db_path": str(metadata_db_path),
        "pipeline_version": config.pipeline_version,
        "snapshot": snapshot,
        "counts": {
            "videos": len(snapshot["videos"]),
            "processing_state": len(snapshot["processing_state"]),
            "artifacts": len(snapshot["artifacts"]),
            "species_classifications": len(snapshot["species_classifications"]),
            "favorites": len(snapshot["favorites"]),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def relpath_from(base_dir: Path, target: Path) -> str:
    """Compute a display-friendly relative path when possible.

    Returns:
        str: Relative path when possible, otherwise absolute path.
    """
    try:
        return os.path.relpath(target.resolve(), start=base_dir.resolve())
    except OSError:
        return str(target.resolve())


def _build_artifacts_by_video(snapshot_artifacts: list[dict[str, Any]]) -> dict[str, dict[str, Path]]:
    """Build artifact map keyed by video_id and artifact_type.

    Returns:
        dict[str, dict[str, Path]]: Artifact paths by video and artifact type.
    """
    artifacts_by_video: dict[str, dict[str, Path]] = {}
    for row in snapshot_artifacts:
        video_id = row.get("video_id")
        artifact_type = row.get("artifact_type")
        artifact_path = row.get("path")
        if not isinstance(video_id, str) or not isinstance(artifact_type, str):
            continue
        if not isinstance(artifact_path, str) or not artifact_path:
            continue
        artifacts_by_video.setdefault(video_id, {})[artifact_type] = Path(artifact_path)
    return artifacts_by_video


def generate_report_videos(
    decisions: list[VideoDecision],
    run_output_dir: Path,
    transcode_fn: Any | None = None,
    source_video_paths: dict[str, Path] | None = None,
    output_video_paths: dict[str, Path] | None = None,
) -> dict[str, Path]:
    """Generate browser-playable report videos for interesting decisions.

    Args:
        decisions: Per-video processing decisions.
        run_output_dir: Root output directory for current run.
        transcode_fn: Optional transcode callable used for testing.
        source_video_paths: Optional source clip paths keyed by output-relative path.
        output_video_paths: Optional output report paths keyed by output-relative path.

    Returns:
        dict[str, Path]: Mapping from decision output-relative path to generated report video path.
    """
    generated_paths: dict[str, Path] = {}
    web_video_dir = run_output_dir / "report_videos"
    if transcode_fn is None:
        from video_clipping import transcode_video_for_web

        resolved_transcode = transcode_video_for_web
    else:
        resolved_transcode = transcode_fn

    for decision in decisions:
        if decision.bucket != "interesting":
            continue

        source_video_path = run_output_dir / decision.bucket / decision.output_relative_path
        if source_video_paths is not None:
            source_video_path = source_video_paths.get(decision.relative_path, source_video_path)
        if not source_video_path.exists():
            continue
        if is_image_file(source_video_path):
            continue

        web_video_path = web_video_dir / f"{Path(decision.output_relative_path).stem}.mp4"
        if output_video_paths is not None:
            web_video_path = output_video_paths.get(decision.output_relative_path, web_video_path)
        transcoded_path = resolved_transcode(source_video_path, web_video_path)
        if transcoded_path is not None:
            generated_paths[decision.output_relative_path] = transcoded_path

    return generated_paths
