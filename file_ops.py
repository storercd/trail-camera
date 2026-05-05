"""Filesystem helpers for trail camera processing."""

from __future__ import annotations

import hashlib
import shutil
from datetime import datetime
from pathlib import Path

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


def build_video_storage_dir(videos_root_dir: Path, video_id: str) -> Path:
    """Build canonical storage directory for one video ID.

    Uses a two-level shard prefix to keep filesystem fan-out bounded.

    Args:
        videos_root_dir: Root folder containing canonical per-video folders.
        video_id: Stable video identifier (expected sha256 hex string).

    Returns:
        Path: Canonical per-video storage directory.
    """
    normalized_video_id = video_id.strip().lower()
    if len(normalized_video_id) >= 4:
        return videos_root_dir / normalized_video_id[:2] / normalized_video_id[2:4] / normalized_video_id
    return videos_root_dir / normalized_video_id


def build_canonical_original_path(videos_root_dir: Path, video_id: str, source: Path) -> Path:
    """Build canonical destination path for a stored original source video.

    Args:
        videos_root_dir: Root folder containing canonical per-video folders.
        video_id: Stable video identifier.
        source: Original source video path.

    Returns:
        Path: Canonical original video path under the video storage folder.
    """
    canonical_dir = build_video_storage_dir(videos_root_dir, video_id)
    suffix = source.suffix if source.suffix else ".bin"
    return canonical_dir / f"source{suffix.lower()}"


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute sha256 hex digest for a file.

    Args:
        file_path: File to hash.
        chunk_size: Bytes read per iteration.

    Returns:
        str: Lowercase sha256 hex digest.
    """
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def get_capture_date(source: Path) -> str | None:
    """Return source capture date in YYYY-MM-DD format when available.

    Args:
        source: Source video path.

    Returns:
        str | None: Capture date string, or None if stat cannot be read.
    """
    try:
        stat = source.stat()
    except OSError:
        return None

    created_timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
    return datetime.fromtimestamp(created_timestamp).strftime("%Y-%m-%d")


def find_existing_canonical_original_path(videos_root_dir: Path, video_id: str) -> Path | None:
    """Find an already-stored canonical original file for a video ID.

    Args:
        videos_root_dir: Root folder containing canonical per-video folders.
        video_id: Stable video identifier.

    Returns:
        Path | None: Existing source file path if present.
    """
    canonical_dir = build_video_storage_dir(videos_root_dir, video_id)
    if not canonical_dir.exists():
        return None

    for candidate in sorted(canonical_dir.glob("source.*")):
        if candidate.is_file():
            return candidate
    return None


def persist_canonical_original(
    videos_root_dir: Path,
    video_id: str,
    source: Path,
) -> Path:
    """Persist a canonical original source file for the given video ID.

    If the canonical original already exists, this function returns the existing
    path without rewriting it.

    Args:
        videos_root_dir: Root folder containing canonical per-video folders.
        video_id: Stable video identifier.
        source: Source video file to persist.

    Returns:
        Path: Canonical original file path.
    """
    existing = find_existing_canonical_original_path(videos_root_dir, video_id)
    if existing is not None:
        return existing

    destination = build_canonical_original_path(videos_root_dir, video_id, source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def find_videos(input_dir: Path, recursive: bool) -> list[Path]:
    """Find video files in the input directory.

    Args:
        input_dir: Directory to scan.
        recursive: Whether to search subdirectories recursively.

    Returns:
        list[Path]: Sorted list of matching video files.
    """
    # Always recurse through subfolders to support complex input trees.
    iterator = input_dir.rglob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)


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


def copy_or_move(src: Path, dst: Path, move: bool) -> Path:
    """Copy or move a file to its destination, avoiding collisions.

    Args:
        src: Source file path.
        dst: Target file path.
        move: If True, move the file; otherwise copy it.

    Returns:
        Path: The final destination path written after uniqueness handling.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst = make_unique_destination(dst)
    if move:
        shutil.move(str(src), str(dst))
    else:
        shutil.copy2(src, dst)
    return dst


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


def build_dated_relative_output_path(source: Path, relative_path: str) -> str:
    """Prefix a relative file name with the source file's creation date.

    Args:
        source: Source file path used to derive creation timestamp.
        relative_path: Original relative path from the input tree.

    Returns:
        str: Relative path with a YYYYMMDD- prefixed filename.
    """
    relative = Path(relative_path)
    try:
        stat = source.stat()
    except OSError:
        return str(relative)

    created_timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
    date_prefix = datetime.fromtimestamp(created_timestamp).strftime("%Y%m%d")
    if relative.name.startswith(f"{date_prefix}-"):
        return relative.name

    # Flatten output naming; do not preserve source folder structure.
    return f"{date_prefix}-{relative.name}"
