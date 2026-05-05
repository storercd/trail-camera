"""Tests for canonical per-video storage path helpers."""

import hashlib
from pathlib import Path

from file_ops import (
    build_canonical_original_path,
    build_video_storage_dir,
    compute_sha256,
    persist_canonical_original,
)


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
