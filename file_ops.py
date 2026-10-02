"""Filesystem helpers for trail camera processing."""

from __future__ import annotations

import hashlib
import logging
import shutil
import struct
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
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

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
    ".webp",
}

MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS


logger = logging.getLogger(__name__)


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


def _get_filesystem_capture_date(source: Path) -> str | None:
    """Return file creation/modify date as YYYY-MM-DD.

    Args:
        source: Source video path.

    Returns:
        str | None: Filesystem-derived date string, or None when stat fails.
    """
    try:
        stat = source.stat()
    except OSError:
        return None

    created_timestamp = getattr(stat, "st_birthtime", stat.st_mtime)
    return datetime.fromtimestamp(created_timestamp).strftime("%Y-%m-%d")


_QUICKTIME_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)


def _iter_mp4_boxes(data: bytes, start: int, end: int) -> Iterator[tuple[str, int, int]]:
    """Yield (box_type, payload_start, payload_end) for top-level boxes in a byte range."""
    pos = start
    while pos + 8 <= end:
        size = struct.unpack(">I", data[pos : pos + 4])[0]
        box_type = data[pos + 4 : pos + 8].decode("latin1")
        header_size = 8
        if size == 1:
            if pos + 16 > end:
                return
            size = struct.unpack(">Q", data[pos + 8 : pos + 16])[0]
            header_size = 16
        elif size == 0:
            size = end - pos
        if size < header_size or pos + size > end:
            return
        yield box_type, pos + header_size, pos + size
        pos += size


def _find_mp4_box(data: bytes, start: int, end: int, box_type: str) -> tuple[int, int] | None:
    """Return (payload_start, payload_end) for the first matching top-level box, else None."""
    for found_type, payload_start, payload_end in _iter_mp4_boxes(data, start, end):
        if found_type == box_type:
            return payload_start, payload_end
    return None


def _get_video_container_capture_date(source: Path) -> str | None:
    """Return capture date from an MP4/MOV container's mvhd creation time.

    QuickTime-family containers (.mp4, .mov, .m4v) store a creation timestamp
    in the moov/mvhd atom as seconds since 1904-01-01 UTC. This is set by the
    camera at record time and survives copies/moves, unlike filesystem
    timestamps or filename-based heuristics.

    Returns:
        str | None: Capture date string from container metadata, else None.
    """
    try:
        with source.open("rb") as handle:
            data = handle.read()
    except OSError:
        return None

    moov = _find_mp4_box(data, 0, len(data), "moov")
    if moov is None:
        return None
    mvhd = _find_mp4_box(data, moov[0], moov[1], "mvhd")
    if mvhd is None:
        return None

    payload_start, payload_end = mvhd
    if payload_end - payload_start < 8:
        return None
    version = data[payload_start]
    try:
        if version == 1:
            if payload_end - payload_start < 12:
                return None
            creation_seconds = struct.unpack(">Q", data[payload_start + 4 : payload_start + 12])[0]
        else:
            creation_seconds = struct.unpack(">I", data[payload_start + 4 : payload_start + 8])[0]
    except struct.error:
        return None

    if creation_seconds <= 0:
        return None
    try:
        capture_dt = _QUICKTIME_EPOCH + timedelta(seconds=creation_seconds)
    except OverflowError:
        return None
    return capture_dt.strftime("%Y-%m-%d")


def get_capture_date(source: Path) -> str | None:
    """Return source capture date in YYYY-MM-DD format when available.

    Args:
        source: Source file path.

    Returns:
        str | None: Capture date from video container metadata when available,
            else filesystem creation/modify date.
    """
    if source.suffix.lower() in VIDEO_EXTENSIONS:
        extracted = _get_video_container_capture_date(source)
        if extracted is not None:
            return extracted
    return _get_filesystem_capture_date(source)


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


def overwrite_canonical_original(
    videos_root_dir: Path,
    video_id: str,
    source: Path,
) -> Path:
    """Force-rewrite canonical original source file for a video ID.

    Args:
        videos_root_dir: Root folder containing canonical per-video folders.
        video_id: Stable video identifier.
        source: Source video file to persist.

    Returns:
        Path: Canonical original file path.
    """
    destination = build_canonical_original_path(videos_root_dir, video_id, source)
    destination.parent.mkdir(parents=True, exist_ok=True)

    for existing in destination.parent.glob("source.*"):
        if existing.is_file() and existing != destination:
            existing.unlink()

    shutil.copy2(source, destination)
    return destination


def is_video_file(path: Path) -> bool:
    """Return whether a path has a known video extension.

    Returns:
        bool: True when the suffix matches a configured video extension.
    """
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: Path) -> bool:
    """Return whether a path has a known image extension.

    Returns:
        bool: True when the suffix matches a configured image extension.
    """
    return path.suffix.lower() in IMAGE_EXTENSIONS


def find_media_files(input_dir: Path, recursive: bool) -> list[Path]:
    """Find supported media files in the input directory.

    Args:
        input_dir: Directory to scan.
        recursive: Whether to search subdirectories recursively.

    Returns:
        list[Path]: Sorted list of matching media files.
    """
    # Always recurse through subfolders to support complex input trees.
    iterator = input_dir.rglob("*")
    return sorted(p for p in iterator if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS)


def find_videos(input_dir: Path, recursive: bool) -> list[Path]:
    """Find video files in the input directory.

    Returns:
        list[Path]: Sorted list of matching video files.
    """
    return [path for path in find_media_files(input_dir, recursive) if is_video_file(path)]


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


def validate_and_find_media_files(input_dir: Path, recursive: bool) -> list[Path]:
    """Validate input directory and return matching media files.

    Args:
        input_dir: Directory expected to contain media files.
        recursive: Whether to search subdirectories.

    Returns:
        list[Path]: Media files discovered for processing.

    Raises:
        SystemExit: If input directory is missing/invalid or no videos are found.
    """
    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    media_files = find_media_files(input_dir, recursive)
    logger.info("Found %s media file(s) to process", len(media_files))
    if not media_files:
        raise SystemExit(f"No media files found in {input_dir}")
    return media_files


def validate_and_find_videos(input_dir: Path, recursive: bool) -> list[Path]:
    """Validate input directory and return matching video files.

    Returns:
        list[Path]: Video files discovered for processing.

    Raises:
        SystemExit: If no videos are found in the input directory.
    """
    videos = find_videos(input_dir, recursive)
    logger.info("Found %s video(s) to process", len(videos))
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
