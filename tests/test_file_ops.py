"""Tests for canonical per-video storage path helpers."""

import hashlib
import struct
from datetime import datetime, timezone
from pathlib import Path

from file_ops import (
    build_canonical_original_path,
    build_video_storage_dir,
    compute_sha256,
    find_media_files,
    find_videos,
    get_capture_date,
    persist_canonical_original,
)

_QUICKTIME_EPOCH = datetime(1904, 1, 1, tzinfo=timezone.utc)


def _build_box(box_type: bytes, payload: bytes) -> bytes:
    """Build one ISO-BMFF/QuickTime box with a 32-bit size header."""
    size = 8 + len(payload)
    return struct.pack(">I", size) + box_type + payload


def _build_mvhd(creation_time: datetime, version: int = 0) -> bytes:
    """Build a minimal mvhd payload with the given creation_time and version."""
    seconds = int((creation_time - _QUICKTIME_EPOCH).total_seconds())
    flags = b"\x00\x00\x00"
    if version == 1:
        return bytes([1]) + flags + struct.pack(">Q", seconds) + b"\x00" * 24
    return bytes([0]) + flags + struct.pack(">I", seconds) + b"\x00" * 20


def _build_mp4(mvhd_payload: bytes) -> bytes:
    """Build a minimal mp4 byte stream containing moov/mvhd."""
    mvhd_box = _build_box(b"mvhd", mvhd_payload)
    moov_box = _build_box(b"moov", mvhd_box)
    return moov_box


def test_build_video_storage_dir_should_shard_sha256_path() -> None:
    """Build a sharded canonical path from a sha256-like video ID."""
    videos_root = Path("/tmp/videos")
    video_id = "abcdef0123456789"

    path = build_video_storage_dir(videos_root, video_id)

    assert path == Path("/tmp/videos/ab/cd/abcdef0123456789")


def test_build_video_storage_dir_should_normalize_case() -> None:
    """Normalize uppercase IDs to lowercase in canonical storage paths."""
    videos_root = Path("/tmp/videos")
    video_id = "ABCD"

    path = build_video_storage_dir(videos_root, video_id)

    assert path == Path("/tmp/videos/ab/cd/abcd")


def test_build_canonical_original_path_should_use_source_extension() -> None:
    """Preserve source extension (lowercased) for canonical original video names."""
    videos_root = Path("/tmp/videos")
    video_id = "abcdef0123456789"
    source = Path("PICT0005.AVI")

    path = build_canonical_original_path(videos_root, video_id, source)

    assert path == Path("/tmp/videos/ab/cd/abcdef0123456789/source.avi")


def test_compute_sha256_should_match_known_digest(tmp_path: Path) -> None:
    """Generate stable sha256 digest for file identity."""
    target = tmp_path / "sample.bin"
    payload = b"trail-camera"
    target.write_bytes(payload)

    digest = compute_sha256(target)
    expected = hashlib.sha256(payload).hexdigest()

    assert digest == expected


def test_persist_canonical_original_should_copy_once(tmp_path: Path) -> None:
    """Avoid rewriting canonical original when it already exists."""
    videos_root = tmp_path / "videos"
    source = tmp_path / "PICT0001.AVI"
    source.write_bytes(b"v1")

    first = persist_canonical_original(videos_root, "a" * 64, source)
    source.write_bytes(b"v2")
    second = persist_canonical_original(videos_root, "a" * 64, source)

    assert first == second
    assert first.read_bytes() == b"v1"


def test_find_media_files_should_include_images_and_videos(tmp_path: Path) -> None:
    """Discover both video and image files in the input tree."""
    video = tmp_path / "clip.AVI"
    image = tmp_path / "photo.JPG"
    other = tmp_path / "notes.txt"
    video.write_bytes(b"video")
    image.write_bytes(b"image")
    other.write_bytes(b"text")

    media_files = find_media_files(tmp_path, recursive=True)

    assert media_files == [video, image]


def test_find_videos_should_exclude_images(tmp_path: Path) -> None:
    """Keep legacy video-only discovery behavior available."""
    video = tmp_path / "clip.AVI"
    image = tmp_path / "photo.JPG"
    video.write_bytes(b"video")
    image.write_bytes(b"image")

    videos = find_videos(tmp_path, recursive=True)

    assert videos == [video]


def test_get_capture_date_should_read_mp4_mvhd_version0(tmp_path: Path) -> None:
    """Extract capture date from a version-0 mvhd creation_time box."""
    video = tmp_path / "clip.mp4"
    creation_time = datetime(2026, 9, 25, 21, 42, 20, tzinfo=timezone.utc)
    video.write_bytes(_build_mp4(_build_mvhd(creation_time, version=0)))

    assert get_capture_date(video) == "2026-09-25"


def test_get_capture_date_should_read_mp4_mvhd_version1(tmp_path: Path) -> None:
    """Extract capture date from a version-1 (64-bit) mvhd creation_time box."""
    video = tmp_path / "clip.mov"
    creation_time = datetime(2026, 9, 25, 21, 42, 20, tzinfo=timezone.utc)
    video.write_bytes(_build_mp4(_build_mvhd(creation_time, version=1)))

    assert get_capture_date(video) == "2026-09-25"


def test_get_capture_date_should_fall_back_to_filesystem_for_non_mp4(tmp_path: Path) -> None:
    """Fall back to filesystem date when the container has no moov/mvhd box."""
    video = tmp_path / "clip.avi"
    video.write_bytes(b"RIFF" + b"\x00" * 32 + b"AVI ")

    capture_date = get_capture_date(video)

    stat = video.stat()
    expected = datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_mtime)).strftime("%Y-%m-%d")
    assert capture_date == expected


def test_get_capture_date_should_fall_back_when_mvhd_missing(tmp_path: Path) -> None:
    """Fall back to filesystem date when moov exists but mvhd does not."""
    video = tmp_path / "clip.mp4"
    empty_moov = _build_box(b"moov", b"")
    video.write_bytes(empty_moov)

    capture_date = get_capture_date(video)

    stat = video.stat()
    expected = datetime.fromtimestamp(getattr(stat, "st_birthtime", stat.st_mtime)).strftime("%Y-%m-%d")
    assert capture_date == expected
