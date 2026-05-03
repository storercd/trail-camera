"""Reporting helpers for trail camera processing runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from pipeline_models import AppConfig, VideoDecision
from preview_frames import PreviewExtractionStats
from video_clipping import transcode_video_for_web


def write_summary(
    summary_path: Path,
    decisions: list[VideoDecision],
    config: AppConfig,
    config_path: Path,
    preview_stats: PreviewExtractionStats | None,
    preview_output_dir: Path,
    run_output_dir: Path,
    run_id: str | None,
    species_crop_output_dir: Path,
    species_classification_report_path: Path,
) -> None:
    """Write a JSON summary file for the run."""
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
        "auto_open_html_report": config.auto_open_html_report,
        "generate_top_frame_previews": config.generate_top_frame_previews,
        "preview_output_dir": str(preview_output_dir),
        "preview_include_uninteresting": config.preview_include_uninteresting,
        "classify_previews_with_speciesnet": config.classify_previews_with_speciesnet,
        "speciesnet_model": config.speciesnet_model,
        "speciesnet_geofence": config.speciesnet_geofence,
        "speciesnet_label_in_filename": config.speciesnet_label_in_filename,
        "speciesnet_use_crops": config.speciesnet_use_crops,
        "species_crop_output_dir": str(species_crop_output_dir),
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
) -> None:
    """Write an HTML summary report for interesting detections."""
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
