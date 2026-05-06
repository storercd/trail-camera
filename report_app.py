"""Local reporting app backed by SQLite metadata and per-video artifacts."""

from __future__ import annotations

import argparse
import json
import mimetypes
from math import ceil
from pathlib import Path
from typing import Any

from flask import Flask, abort, render_template, request, send_file, url_for

from metadata_store import get_catalog_video_detail, list_catalog_videos
from pipeline_config import DEFAULT_CONFIG_PATH, build_run_paths, load_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the local reporting app.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Run the local trail camera reporting app.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="Path to YAML configuration file.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument("--debug", action="store_true", help="Enable Flask debug mode.")
    return parser.parse_args()


def _parse_optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_needs_reprocess(value: str | None) -> bool | None:
    if value == "yes":
        return True
    if value == "no":
        return False
    return None


def _load_candidates(raw_candidates: str | None) -> list[dict[str, Any]]:
    if not raw_candidates:
        return []
    try:
        payload = json.loads(raw_candidates)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, list):
        return []
    return [candidate for candidate in payload if isinstance(candidate, dict)]


def _build_display_names(rows: list[dict[str, Any]]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        capture_date = row.get("capture_date")
        original = row.get("original_filename") or row.get("video_id")
        base_name = f"{capture_date}-{original}" if capture_date else str(original)
        row["base_display_name"] = base_name
        counts[base_name] = counts.get(base_name, 0) + 1

    seen: dict[str, int] = {}
    for row in rows:
        base_name = row["base_display_name"]
        index = seen.get(base_name, 0)
        seen[base_name] = index + 1
        if counts[base_name] > 1 and index > 0:
            row["display_name"] = f"{base_name}__{str(row['video_id'])[:8]}"
        else:
            row["display_name"] = base_name


def create_app(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Flask:
    """Create the local reporting app instance.

    Returns:
        Flask: Configured Flask application.
    """
    resolved_config_path = Path(config_path)
    config = load_config(resolved_config_path)
    paths = build_run_paths(resolved_config_path, config)

    app = Flask(__name__)
    app.config["METADATA_DB_PATH"] = paths.metadata_db_path
    app.config["CURRENT_PIPELINE_VERSION"] = config.pipeline_version

    @app.get("/")
    def list_videos() -> str:
        filters = {
            "date_from": request.args.get("date_from", ""),
            "date_to": request.args.get("date_to", ""),
            "bucket": request.args.get("bucket", ""),
            "species": request.args.get("species", ""),
            "min_confidence": request.args.get("min_confidence", ""),
            "max_confidence": request.args.get("max_confidence", ""),
            "needs_reprocess": request.args.get("needs_reprocess", ""),
            "sort_by": request.args.get("sort_by", "capture_date"),
            "sort_dir": request.args.get("sort_dir", "desc"),
            "page_size": request.args.get("page_size", "24"),
        }
        page = max(1, request.args.get("page", type=int, default=1))
        page_size = max(1, min(request.args.get("page_size", type=int, default=24), 100))

        rows, total_count = list_catalog_videos(
            db_path=app.config["METADATA_DB_PATH"],
            current_pipeline_version=app.config["CURRENT_PIPELINE_VERSION"],
            date_from=filters["date_from"] or None,
            date_to=filters["date_to"] or None,
            bucket=filters["bucket"] or None,
            species=filters["species"] or None,
            min_confidence=_parse_optional_float(filters["min_confidence"]),
            max_confidence=_parse_optional_float(filters["max_confidence"]),
            needs_reprocess=_parse_needs_reprocess(filters["needs_reprocess"]),
            sort_by=filters["sort_by"],
            sort_dir=filters["sort_dir"],
            page=page,
            page_size=page_size,
        )
        for row in rows:
            row["needs_reprocess"] = row.get("pipeline_version") != app.config["CURRENT_PIPELINE_VERSION"]
        _build_display_names(rows)

        total_pages = max(1, ceil(total_count / page_size)) if total_count else 1

        def pagination_url(target_page: int) -> str:
            args = request.args.to_dict(flat=True)
            args["page"] = str(target_page)
            args["page_size"] = str(page_size)
            return url_for("list_videos", **args)

        return render_template(
            "list.html",
            videos=rows,
            filters=filters,
            total_count=total_count,
            page=page,
            total_pages=total_pages,
            current_pipeline_version=app.config["CURRENT_PIPELINE_VERSION"],
            bucket_options=["interesting", "uninteresting", "failed"],
            sort_options=[
                ("capture_date", "Capture Date"),
                ("processed_at", "Processed At"),
                ("species", "Species"),
                ("confidence", "Confidence"),
                ("bucket", "Bucket"),
            ],
            page_sizes=[12, 24, 48, 96],
            pagination_url=pagination_url,
        )

    @app.get("/video/<video_id>")
    def video_detail(video_id: str) -> str:
        video = get_catalog_video_detail(
            db_path=app.config["METADATA_DB_PATH"],
            video_id=video_id,
            current_pipeline_version=app.config["CURRENT_PIPELINE_VERSION"],
        )
        if video is None:
            abort(404)
        _build_display_names([video])
        candidates = _load_candidates(video.get("candidates_json"))
        return render_template("detail.html", video=video, candidates=candidates)

    @app.get("/artifact/<video_id>/<artifact_type>")
    def artifact_file(video_id: str, artifact_type: str):
        video = get_catalog_video_detail(
            db_path=app.config["METADATA_DB_PATH"],
            video_id=video_id,
            current_pipeline_version=app.config["CURRENT_PIPELINE_VERSION"],
        )
        if video is None:
            abort(404)
        path_key = f"{artifact_type}_path"
        raw_path = video.get(path_key)
        if not isinstance(raw_path, str) or not raw_path:
            abort(404)
        artifact_path = Path(raw_path)
        if not artifact_path.exists() or not artifact_path.is_file():
            abort(404)
        mimetype, _ = mimetypes.guess_type(str(artifact_path))
        return send_file(artifact_path, mimetype=mimetype or "application/octet-stream")

    return app


def main() -> int:
    """Run the local Flask reporting app.

    Returns:
        int: Process exit code.
    """
    args = parse_args()
    app = create_app(args.config)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
