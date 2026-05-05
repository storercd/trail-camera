"""Local reporting app backed by SQLite metadata and per-video artifacts."""

from __future__ import annotations

import argparse
import json
import mimetypes
from math import ceil
from pathlib import Path
from typing import Any

from flask import Flask, abort, render_template_string, request, send_file, url_for

from metadata_store import get_catalog_video_detail, list_catalog_videos
from pipeline_config import DEFAULT_CONFIG_PATH, build_run_paths, load_config

LIST_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Trail Camera Catalog</title>
    <style>
        :root {
            color-scheme: light;
            --bg: #f3efe6;
            --panel: #fffaf2;
            --ink: #201b15;
            --muted: #6d655d;
            --line: #d8cdbd;
            --accent: #2f6e49;
            --accent-soft: #e0efdf;
            --warning: #8c5a1f;
            --warning-soft: #f6e7cf;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Georgia, "Iowan Old Style", serif;
            background: linear-gradient(180deg, #fff8ec 0%, var(--bg) 100%);
            color: var(--ink);
        }
        main {
            max-width: 1280px;
            margin: 0 auto;
            padding: 28px 20px 48px;
        }
        h1 {
            margin: 0 0 8px;
            font-size: 2.4rem;
        }
        p.lead {
            margin: 0 0 20px;
            color: var(--muted);
        }
        .meta-row, .filter-grid, .card-grid, .pagination {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        .meta-chip, .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 12px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.95rem;
        }
        .panel {
            margin-top: 18px;
            background: color-mix(in srgb, var(--panel) 95%, white);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: 0 16px 40px rgba(35, 27, 16, 0.08);
            overflow: hidden;
        }
        .filters {
            padding: 18px;
            border-bottom: 1px solid var(--line);
            background: rgba(255,255,255,0.45);
        }
        .filter-grid > label {
            display: flex;
            flex-direction: column;
            gap: 6px;
            min-width: 140px;
            color: var(--muted);
            font-size: 0.9rem;
        }
        input, select {
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px solid var(--line);
            background: white;
            color: var(--ink);
            font: inherit;
        }
        .actions {
            margin-top: 14px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        .button {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            padding: 10px 14px;
            border-radius: 10px;
            border: 1px solid var(--accent);
            background: var(--accent);
            color: white;
            text-decoration: none;
            font-weight: 600;
        }
        .button.secondary {
            background: transparent;
            color: var(--accent);
        }
        .results-header {
            padding: 18px 18px 0;
        }
        .card-grid {
            padding: 18px;
        }
        .video-card {
            flex: 1 1 360px;
            min-width: 320px;
            background: rgba(255,255,255,0.78);
            border: 1px solid var(--line);
            border-radius: 16px;
            overflow: hidden;
        }
        .thumb {
            width: 100%;
            aspect-ratio: 16 / 9;
            object-fit: cover;
            display: block;
            background: #0e0e0e;
        }
        .card-body {
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .title-row {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: start;
        }
        .title-row h2 {
            margin: 0;
            font-size: 1.15rem;
            line-height: 1.25;
        }
        .muted {
            color: var(--muted);
        }
        .warning {
            background: var(--warning-soft);
            color: var(--warning);
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 10px;
            font-size: 0.95rem;
        }
        .stats strong {
            display: block;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            color: var(--muted);
            margin-bottom: 4px;
        }
        .links {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
        }
        .links a, .pagination a {
            color: var(--accent);
            text-decoration: none;
            font-weight: 600;
        }
        .pagination {
            padding: 0 18px 18px;
            align-items: center;
        }
        .pagination .current {
            font-weight: 600;
        }
        @media (max-width: 800px) {
            .video-card { min-width: 100%; }
            .stats { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <main>
        <h1>Trail Camera Catalog</h1>
        <p class="lead">Filter, sort, and review processed videos directly from the SQLite catalog and per-video artifact files.</p>
        <div class="meta-row">
            <span class="meta-chip">Current pipeline version: {{ current_pipeline_version }}</span>
            <span class="meta-chip">Total matches: {{ total_count }}</span>
            <span class="meta-chip">Page {{ page }} of {{ total_pages }}</span>
        </div>

        <section class="panel">
            <form class="filters" method="get">
                <div class="filter-grid">
                    <label>Date From<input type="date" name="date_from" value="{{ filters.date_from }}"></label>
                    <label>Date To<input type="date" name="date_to" value="{{ filters.date_to }}"></label>
                    <label>Bucket
                        <select name="bucket">
                            <option value="">All</option>
                            {% for value in bucket_options %}
                            <option value="{{ value }}" {% if filters.bucket == value %}selected{% endif %}>{{ value }}</option>
                            {% endfor %}
                        </select>
                    </label>
                    <label>Species<input type="text" name="species" value="{{ filters.species }}" placeholder="dog, bird, squirrel"></label>
                    <label>Min Confidence<input type="number" min="0" max="1" step="0.001" name="min_confidence" value="{{ filters.min_confidence }}"></label>
                    <label>Max Confidence<input type="number" min="0" max="1" step="0.001" name="max_confidence" value="{{ filters.max_confidence }}"></label>
                    <label>Needs Reprocess
                        <select name="needs_reprocess">
                            <option value="">All</option>
                            <option value="yes" {% if filters.needs_reprocess == 'yes' %}selected{% endif %}>Yes</option>
                            <option value="no" {% if filters.needs_reprocess == 'no' %}selected{% endif %}>No</option>
                        </select>
                    </label>
                    <label>Sort By
                        <select name="sort_by">
                            {% for key, label in sort_options %}
                            <option value="{{ key }}" {% if filters.sort_by == key %}selected{% endif %}>{{ label }}</option>
                            {% endfor %}
                        </select>
                    </label>
                    <label>Direction
                        <select name="sort_dir">
                            <option value="desc" {% if filters.sort_dir == 'desc' %}selected{% endif %}>Descending</option>
                            <option value="asc" {% if filters.sort_dir == 'asc' %}selected{% endif %}>Ascending</option>
                        </select>
                    </label>
                    <label>Page Size
                        <select name="page_size">
                            {% for size in page_sizes %}
                            <option value="{{ size }}" {% if filters.page_size == size|string %}selected{% endif %}>{{ size }}</option>
                            {% endfor %}
                        </select>
                    </label>
                </div>
                <div class="actions">
                    <button class="button" type="submit">Apply Filters</button>
                    <a class="button secondary" href="{{ url_for('list_videos') }}">Reset</a>
                </div>
            </form>

            <div class="results-header">
                {% if not videos %}
                <p class="muted">No videos match the current filters.</p>
                {% endif %}
            </div>
            <div class="card-grid">
                {% for video in videos %}
                <article class="video-card">
                    {% if video.preview_image_path %}
                    <a href="{{ url_for('video_detail', video_id=video.video_id) }}"><img class="thumb" src="{{ url_for('artifact_file', video_id=video.video_id, artifact_type='preview_image') }}" alt="Preview for {{ video.display_name }}"></a>
                    {% else %}
                    <div class="thumb"></div>
                    {% endif %}
                    <div class="card-body">
                        <div class="title-row">
                            <div>
                                <h2><a href="{{ url_for('video_detail', video_id=video.video_id) }}" style="color: inherit; text-decoration: none;">{{ video.display_name }}</a></h2>
                                <div class="muted">{{ video.video_id[:12] }}{% if video.video_id|length > 12 %}...{% endif %}</div>
                            </div>
                            {% if video.needs_reprocess %}
                            <span class="pill warning">Needs Reprocess</span>
                            {% endif %}
                        </div>
                        <div class="stats">
                            <div><strong>Species</strong>{{ video.top_label or 'unknown' }}</div>
                            <div><strong>Confidence</strong>{{ '%.3f'|format(video.top_score if video.top_score is not none else (video.top_confidence or 0.0)) }}</div>
                            <div><strong>Bucket</strong>{{ video.bucket or 'n/a' }}</div>
                            <div><strong>Capture Date</strong>{{ video.capture_date or 'unknown' }}</div>
                        </div>
                        <div class="links">
                            <a href="{{ url_for('video_detail', video_id=video.video_id) }}">View Details</a>
                            {% if video.bucketed_video_path %}<a href="{{ url_for('artifact_file', video_id=video.video_id, artifact_type='bucketed_video') }}" target="_blank">Open Clip</a>{% endif %}
                            {% if video.report_video_path %}<a href="{{ url_for('artifact_file', video_id=video.video_id, artifact_type='report_video') }}" target="_blank">Open Browser Video</a>{% endif %}
                        </div>
                    </div>
                </article>
                {% endfor %}
            </div>

            {% if total_pages > 1 %}
            <div class="pagination">
                {% if page > 1 %}
                <a href="{{ pagination_url(page - 1) }}">Previous</a>
                {% endif %}
                <span class="current">Page {{ page }} / {{ total_pages }}</span>
                {% if page < total_pages %}
                <a href="{{ pagination_url(page + 1) }}">Next</a>
                {% endif %}
            </div>
            {% endif %}
        </section>
    </main>
</body>
</html>
"""

DETAIL_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{{ video.display_name }} - Trail Camera Catalog</title>
    <style>
        :root {
            color-scheme: light;
            --bg: #f3efe6;
            --panel: #fffaf2;
            --ink: #201b15;
            --muted: #6d655d;
            --line: #d8cdbd;
            --accent: #2f6e49;
            --accent-soft: #e0efdf;
            --warning: #8c5a1f;
            --warning-soft: #f6e7cf;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Georgia, "Iowan Old Style", serif;
            background: linear-gradient(180deg, #fff8ec 0%, var(--bg) 100%);
            color: var(--ink);
        }
        main {
            max-width: 1180px;
            margin: 0 auto;
            padding: 28px 20px 48px;
        }
        .back {
            color: var(--accent);
            text-decoration: none;
            font-weight: 600;
        }
        h1 {
            margin: 14px 0 8px;
            font-size: 2.3rem;
        }
        .meta-row, .media-grid, .artifact-links {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
        }
        .pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 8px 12px;
            border-radius: 999px;
            background: var(--accent-soft);
            color: var(--accent);
            font-size: 0.95rem;
        }
        .warning { background: var(--warning-soft); color: var(--warning); }
        .panel {
            margin-top: 18px;
            background: color-mix(in srgb, var(--panel) 95%, white);
            border: 1px solid var(--line);
            border-radius: 18px;
            box-shadow: 0 16px 40px rgba(35, 27, 16, 0.08);
            overflow: hidden;
            padding: 18px;
        }
        .media-grid > section { flex: 1 1 320px; }
        video, img {
            width: 100%;
            border-radius: 14px;
            border: 1px solid var(--line);
            background: #0e0e0e;
            display: block;
        }
        .kv {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin-top: 16px;
        }
        .kv div strong {
            display: block;
            color: var(--muted);
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 4px;
        }
        .artifact-links a {
            color: var(--accent);
            text-decoration: none;
            font-weight: 600;
        }
        ol, ul { margin: 0; padding-left: 20px; }
        @media (max-width: 800px) { .kv { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <main>
        <a class="back" href="{{ url_for('list_videos') }}">← Back to Catalog</a>
        <h1>{{ video.display_name }}</h1>
        <div class="meta-row">
            <span class="pill">Bucket: {{ video.bucket or 'n/a' }}</span>
            <span class="pill">Species: {{ video.top_label or 'unknown' }}</span>
            <span class="pill">Confidence: {{ '%.3f'|format(video.top_score if video.top_score is not none else (video.top_confidence or 0.0)) }}</span>
            {% if video.needs_reprocess %}<span class="pill warning">Needs Reprocess</span>{% endif %}
        </div>

        <section class="panel">
            <div class="artifact-links">
                {% if video.bucketed_video_path %}<a href="{{ url_for('artifact_file', video_id=video.video_id, artifact_type='bucketed_video') }}" target="_blank">Open Native Clip</a>{% endif %}
                {% if video.report_video_path %}<a href="{{ url_for('artifact_file', video_id=video.video_id, artifact_type='report_video') }}" target="_blank">Open Browser Video</a>{% endif %}
                {% if video.preview_image_path %}<a href="{{ url_for('artifact_file', video_id=video.video_id, artifact_type='preview_image') }}" target="_blank">Open Preview</a>{% endif %}
                {% if video.species_crop_path %}<a href="{{ url_for('artifact_file', video_id=video.video_id, artifact_type='species_crop') }}" target="_blank">Open Species Crop</a>{% endif %}
            </div>
            <div class="kv">
                <div><strong>Video ID</strong>{{ video.video_id }}</div>
                <div><strong>Capture Date</strong>{{ video.capture_date or 'unknown' }}</div>
                <div><strong>Original Filename</strong>{{ video.original_filename }}</div>
                <div><strong>Pipeline Version</strong>{{ video.pipeline_version or 'unprocessed' }}</div>
                <div><strong>Processed At</strong>{{ video.processed_at or 'n/a' }}</div>
                <div><strong>Mode</strong>{{ video.mode or 'n/a' }}</div>
            </div>
        </section>

        <section class="panel media-grid">
            <section>
                <h2>Browser Video</h2>
                {% if video.report_video_path %}
                <video controls preload="metadata" src="{{ url_for('artifact_file', video_id=video.video_id, artifact_type='report_video') }}"></video>
                {% else %}<p>No browser-ready video.</p>{% endif %}
            </section>
            <section>
                <h2>Preview</h2>
                {% if video.preview_image_path %}
                <img src="{{ url_for('artifact_file', video_id=video.video_id, artifact_type='preview_image') }}" alt="Preview image">
                {% else %}<p>No preview image.</p>{% endif %}
            </section>
            <section>
                <h2>Species Crop</h2>
                {% if video.species_crop_path %}
                <img src="{{ url_for('artifact_file', video_id=video.video_id, artifact_type='species_crop') }}" alt="Species crop">
                {% else %}<p>No species crop.</p>{% endif %}
            </section>
        </section>

        <section class="panel">
            <h2>Candidates</h2>
            {% if candidates %}
            <ol>
                {% for candidate in candidates %}
                <li>{{ candidate.label }} ({{ '%.3f'|format(candidate.score) }})</li>
                {% endfor %}
            </ol>
            {% else %}
            <p>No species candidates stored.</p>
            {% endif %}
        </section>
    </main>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the local reporting app."""
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
    """Create the local reporting app instance."""
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

        return render_template_string(
            LIST_TEMPLATE,
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
        return render_template_string(DETAIL_TEMPLATE, video=video, candidates=candidates)

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
    """Run the local Flask reporting app."""
    args = parse_args()
    app = create_app(args.config)
    app.run(host=args.host, port=args.port, debug=args.debug)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
