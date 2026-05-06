"""Filesystem helpers for trail camera processing."""

from __future__ import annotations

import hashlib
import re
import shutil
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

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


def _parse_profile_date_match(match: re.Match[str], date_order: str) -> str | None:
    """Parse regex date groups into ISO date according to configured component order.

    Returns:
        str | None: Parsed ISO date when groups are valid.
    """
    try:
        a, b, c = match.group(1), match.group(2), match.group(3)
        if date_order == "mdy":
            month, day, year = int(a), int(b), int(c)
        elif date_order == "dmy":
            day, month, year = int(a), int(b), int(c)
        elif date_order == "ymd":
            year, month, day = int(a), int(b), int(c)
        else:
            return None
        return datetime(year=year, month=month, day=day).strftime("%Y-%m-%d")
    except (TypeError, ValueError):
        return None


def _parse_profile_roi(profile: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Parse and validate normalized profile ROI.

    Returns:
        tuple[float, float, float, float] | None: (x0, y0, x1, y1) normalized ROI.
    """
    roi = profile.get("roi")
    if not isinstance(roi, list) or len(roi) != 4:
        return None
    try:
        x0, y0, x1, y1 = (float(v) for v in roi)
    except (TypeError, ValueError):
        return None
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        return None
    return x0, y0, x1, y1


def _sample_frame_indices(total_frames: int, sample_frames: int) -> list[int]:
    """Build deterministic sample indices across the clip timeline.

    Returns:
        list[int]: Frame indices to sample.
    """
    if total_frames <= 0:
        return [0]
    sample_count = max(1, min(sample_frames, total_frames))
    step = max(1, total_frames // sample_count)
    return [min(i * step, total_frames - 1) for i in range(sample_count)]


def _extract_date_from_crop(
    gray_image: Any,
    pattern: re.Pattern[str],
    date_order: str,
    ocr_config: str,
    cv2: Any,
    pytesseract: Any,
) -> str | None:
    """Try OCR transforms on one cropped region and return parsed date.

    Returns:
        str | None: Extracted ISO date when OCR + regex succeeds.
    """
    transforms = (
        lambda img: img,
        lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
        lambda img: cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
    )
    for transform in transforms:
        processed = transform(gray_image)
        text = pytesseract.image_to_string(processed, config=ocr_config)
        match = pattern.search(text)
        if match is None:
            continue
        iso_date = _parse_profile_date_match(match, date_order)
        if iso_date is not None:
            return iso_date
    return None


def _extract_profile_capture_date(source: Path, profile: dict[str, Any]) -> str | None:
    """Extract capture date from on-frame overlay using profile-guided OCR.

    Returns:
        str | None: Extracted ISO date when OCR succeeds, else None.
    """
    try:
        import cv2
        import pytesseract
    except ImportError:
        return None

    roi = _parse_profile_roi(profile)
    date_regex = str(profile.get("date_regex", ""))
    date_order = str(profile.get("date_order", "mdy")).lower()
    sample_frames = int(profile.get("sample_frames", 5))
    psm = int(profile.get("ocr_psm", 6))
    if roi is None or not date_regex:
        return None
    x0, y0, x1, y1 = roi

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = _sample_frame_indices(total_frames, sample_frames)

    pattern = re.compile(date_regex)
    candidates: list[str] = []
    cfg = (
        f"--psm {psm} "
        "-c tessedit_char_whitelist=0123456789-/:. "
        "-c preserve_interword_spaces=1"
    )
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue

        h, w = frame.shape[:2]
        x_start, x_end = int(w * x0), int(w * x1)
        y_start, y_end = int(h * y0), int(h * y1)
        crop = frame[y_start:y_end, x_start:x_end]
        if crop.size == 0:
            continue

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        iso_date = _extract_date_from_crop(
            gray_image=gray,
            pattern=pattern,
            date_order=date_order,
            ocr_config=cfg,
            cv2=cv2,
            pytesseract=pytesseract,
        )
        if iso_date is not None:
            candidates.append(iso_date)

    cap.release()
    if not candidates:
        return None

    return Counter(candidates).most_common(1)[0][0]


def get_capture_date(source: Path, camera_date_profile: dict[str, Any] | None = None) -> str | None:
    """Return source capture date in YYYY-MM-DD format when available.

    Args:
        source: Source video path.
        camera_date_profile: Optional camera overlay OCR profile.

    Returns:
        str | None: Capture date string from profile OCR or filesystem fallback.
    """
    if camera_date_profile:
        extracted = _extract_profile_capture_date(source, camera_date_profile)
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
