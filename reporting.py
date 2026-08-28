"""Reporting helpers for trail camera processing runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
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


def render_candidate_list(candidates: list[dict[str, Any]], limit: int = 5) -> str:
    """Render top candidate species labels as HTML list items.

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
    report_video_paths: dict[str, Path] | None = None,
    bucketed_video_paths: dict[str, Path] | None = None,
    preview_image_paths: dict[str, Path] | None = None,
    species_crop_paths: dict[str, Path] | None = None,
) -> None:
    """Write an HTML summary report for interesting detections."""
    html_summary_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_report_videos = report_video_paths or {}
    resolved_bucketed_videos = bucketed_video_paths or {}
    resolved_preview_images = preview_image_paths or {}
    resolved_species_crops = species_crop_paths or {}

    species_entries: dict[str, dict[str, Any]] = {}
    if species_classification_report_path.exists():
        with species_classification_report_path.open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        for entry in report.get("entries", []):
            relative_path = entry.get("source_relative_path")
            if isinstance(relative_path, str) and relative_path:
                species_entries[relative_path] = entry

    interesting_decisions = [d for d in decisions if d.bucket == "interesting"]
    row_fragments: list[str] = []
    for decision in interesting_decisions:
        species_entry = species_entries.get(decision.output_relative_path, {})
        top_classification = species_entry.get("top_classification") or {}
        top_label = html_escape(top_classification.get("label", "unknown"))
        escaped_relative_path = html_escape(decision.output_relative_path)
        try:
            top_score = float(top_classification.get("score", 0.0))
        except (TypeError, ValueError):
            top_score = 0.0

        video_path = resolved_bucketed_videos.get(
            decision.output_relative_path,
            run_output_dir / decision.bucket / decision.output_relative_path,
        )
        web_video_href = ""
        transcoded_path = resolved_report_videos.get(decision.output_relative_path)
        if transcoded_path is not None and transcoded_path.exists():
            web_video_href = html_escape(relpath_from(html_summary_path.parent, transcoded_path))
        preview_path = resolved_preview_images.get(decision.output_relative_path)
        if preview_path is None:
            raw_preview_path = species_entry.get("preview_image_final")
            preview_path = Path(str(raw_preview_path)) if raw_preview_path else None

        crop_path = resolved_species_crops.get(decision.output_relative_path)
        if crop_path is None:
            raw_crop_path = species_entry.get("classification_input_image")
            crop_path = Path(str(raw_crop_path)) if raw_crop_path else None

        preview_href = (
            html_escape(relpath_from(html_summary_path.parent, preview_path))
            if preview_path
            else ""
        )
        crop_href = (
            html_escape(relpath_from(html_summary_path.parent, crop_path))
            if crop_path
            else ""
        )
        video_href = html_escape(relpath_from(html_summary_path.parent, video_path))

        candidates_html = render_candidate_list(species_entry.get("candidates", []))
        row_fragments.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{html_escape(decision.output_relative_path)}</td>",
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


def write_html_summary_from_catalog(
    html_summary_path: Path,
    metadata_db_path: Path,
) -> None:
    """Write an HTML summary report from SQLite catalog + artifact files only."""
    html_summary_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = fetch_catalog_snapshot(metadata_db_path)

    videos_by_id = {
        str(row.get("video_id")): row
        for row in snapshot.get("videos", [])
        if row.get("video_id")
    }
    species_by_id = {
        str(row.get("video_id")): row
        for row in snapshot.get("species_classifications", [])
        if row.get("video_id")
    }

    artifacts_by_video = _build_artifacts_by_video(snapshot.get("artifacts", []))

    processing_rows = [
        row
        for row in snapshot.get("processing_state", [])
        if isinstance(row.get("video_id"), str) and row.get("bucket") == "interesting"
    ]

    base_name_counts: dict[str, int] = {}
    for row in processing_rows:
        video_id = str(row["video_id"])
        video_meta = videos_by_id.get(video_id, {})
        original = str(video_meta.get("original_filename") or video_id)
        capture_date = video_meta.get("capture_date")
        base_name = f"{capture_date}-{original}" if capture_date else original
        base_name_counts[base_name] = base_name_counts.get(base_name, 0) + 1

    row_fragments: list[str] = []
    seen_base_names: dict[str, int] = {}
    for row in processing_rows:
        video_id = str(row["video_id"])
        video_meta = videos_by_id.get(video_id, {})
        original = str(video_meta.get("original_filename") or video_id)
        capture_date = video_meta.get("capture_date")
        base_name = f"{capture_date}-{original}" if capture_date else original
        seen_index = seen_base_names.get(base_name, 0)
        seen_base_names[base_name] = seen_index + 1
        if base_name_counts.get(base_name, 0) > 1 and seen_index > 0:
            display_name = f"{base_name}__{video_id[:8]}"
        else:
            display_name = base_name

        species_row = species_by_id.get(video_id, {})
        top_label = html_escape(species_row.get("top_label") or "unknown")
        try:
            top_score = float(species_row.get("top_score") or 0.0)
        except (TypeError, ValueError):
            top_score = 0.0

        raw_candidates = species_row.get("candidates_json")
        candidates: list[dict[str, Any]] = []
        if isinstance(raw_candidates, str) and raw_candidates:
            try:
                loaded_candidates = json.loads(raw_candidates)
                if isinstance(loaded_candidates, list):
                    candidates = [c for c in loaded_candidates if isinstance(c, dict)]
            except json.JSONDecodeError:
                candidates = []

        artifact_map = artifacts_by_video.get(video_id, {})
        native_video_path = artifact_map.get("bucketed_video")
        web_video_path = artifact_map.get("report_video")
        preview_path = artifact_map.get("preview_image")
        crop_path = artifact_map.get("species_crop")

        native_video_href = (
            html_escape(relpath_from(html_summary_path.parent, native_video_path))
            if native_video_path is not None and native_video_path.exists()
            else ""
        )
        web_video_href = (
            html_escape(relpath_from(html_summary_path.parent, web_video_path))
            if web_video_path is not None and web_video_path.exists()
            else ""
        )
        preview_href = (
            html_escape(relpath_from(html_summary_path.parent, preview_path))
            if preview_path is not None and preview_path.exists()
            else ""
        )
        crop_href = (
            html_escape(relpath_from(html_summary_path.parent, crop_path))
            if crop_path is not None and crop_path.exists()
            else ""
        )

        escaped_display_name = html_escape(display_name)
        candidates_html = render_candidate_list(candidates)
        row_fragments.append(
            "".join(
                [
                    "<tr>",
                    f"<td>{escaped_display_name}</td>",
                    f"<td>{top_label}<br><small>{top_score:.3f}</small></td>",
                    "<td>",
                    (
                        "".join(
                            [
                                '<div class="media-stack">',
                                f'<video controls preload="metadata" src="{web_video_href}"></video>',
                                '<div class="link-row">',
                                f'<a href="{web_video_href}" target="_blank">Open browser video</a>',
                                f'<a href="{native_video_href}" target="_blank">Open native clip</a>',
                                "</div>",
                                "</div>",
                            ]
                        )
                        if web_video_href and native_video_href
                        else (
                            f'<a href="{native_video_href}" target="_blank">Open native clip</a>'
                            if native_video_href
                            else "n/a"
                        )
                    ),
                    "</td>",
                    (
                        "".join(
                            [
                                '<td><div class="media-stack">',
                                (
                                    f'<a href="{crop_href}" target="_blank">'
                                    f'<img class="thumb" src="{crop_href}" '
                                    f'alt="Species crop for {escaped_display_name}"></a>'
                                ),
                                f'<a href="{crop_href}" target="_blank">species crop</a>',
                                "</div></td>",
                            ]
                        )
                        if crop_href
                        else "<td>n/a</td>"
                    ),
                    (
                        "".join(
                            [
                                '<td><div class="media-stack">',
                                (
                                    f'<a href="{preview_href}" target="_blank">'
                                    f'<img class="thumb" src="{preview_href}" '
                                    f'alt="Preview image for {escaped_display_name}"></a>'
                                ),
                                f'<a href="{preview_href}" target="_blank">preview image</a>',
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
            <span class=\"meta\">Interesting videos: {len(processing_rows)}</span>
            <span class=\"meta\">Species source: sqlite catalog</span>
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


def open_file_in_default_app(path: Path) -> bool:
    """Open a file using the platform's default application.

    Returns:
        bool: True if the launch command was started successfully.
    """
    if not path.exists():
        return False

    try:
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
            return True
        if os.name == "nt":
            os.startfile(str(path))
            return True
        if os.name == "posix":
            subprocess.Popen(["xdg-open", str(path)])
            return True
    except (AttributeError, OSError, ValueError):
        return False

    return False
