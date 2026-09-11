"""Memory-card ingest helpers for trail-camera input staging."""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from file_ops import compute_sha256, find_media_files, is_image_file, is_video_file, make_unique_destination
from pipeline_config import DEFAULT_CONFIG_PATH, load_config

DEFAULT_CARD_MOUNT_ROOT = Path("/Volumes")
DEFAULT_CARD_MARKER_DIRNAME = "DCIM"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryCardIngestResult:
    """Summary of one memory-card ingest run."""

    card_root: Path
    copied_files: tuple[Path, ...]
    imported_files: int
    deleted_files: int
    ejected: bool


def find_memory_card_mount(mount_root: Path) -> Path | None:
    """Return the first mounted volume that looks like a memory card."""
    if not mount_root.exists():
        return None

    for volume_path in sorted(path for path in mount_root.iterdir() if path.is_dir()):
        if (volume_path / DEFAULT_CARD_MARKER_DIRNAME).is_dir():
            return volume_path

    return None


def iter_memory_card_media_files(card_root: Path) -> list[Path]:
    """Return supported media files under a mounted card, excluding hidden paths."""
    media_files = find_media_files(card_root, recursive=True)
    return sorted(path for path in media_files if not _is_hidden_card_path(path, card_root))


def _is_hidden_card_path(path: Path, card_root: Path) -> bool:
    """Return whether card-relative path contains hidden path components."""
    relative_parts = path.relative_to(card_root).parts
    return any(part.startswith(".") for part in relative_parts)


def build_copy_plan(source_files: list[Path], destination_dir: Path) -> list[tuple[Path, Path]]:
    """Build source/target pairs while avoiding filename collisions."""
    copy_plan: list[tuple[Path, Path]] = []
    for source_path in source_files:
        destination = make_unique_destination(destination_dir / source_path.name)
        copy_plan.append((source_path, destination))
    return copy_plan


def copy_files(copy_plan: list[tuple[Path, Path]]) -> None:
    """Copy all planned files into the destination directory."""
    total_files = len(copy_plan)
    for index, (source_path, target_path) in enumerate(copy_plan, start=1):
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)
        if index == 1 or index == total_files or index % 25 == 0:
            LOGGER.info("copied %s/%s files", index, total_files)


def verify_copied_files(copy_plan: list[tuple[Path, Path]], *, use_sha256: bool = False) -> None:
    """Verify copied files by size and optional SHA-256 checksum."""
    for source_path, target_path in copy_plan:
        if not target_path.exists():
            raise ValueError(f"Copied file is missing: {target_path}")
        if source_path.stat().st_size != target_path.stat().st_size:
            raise ValueError(f"Copied file size mismatch: {source_path.name}")
        if use_sha256 and compute_sha256(source_path) != compute_sha256(target_path):
            raise ValueError(f"Copied file checksum mismatch: {source_path.name}")


def delete_memory_card_files(source_files: list[Path], *, card_root: Path) -> int:
    """Delete source files and prune empty directories from the card."""
    deleted_files = 0
    for source_path in source_files:
        if source_path.exists():
            source_path.unlink()
            deleted_files += 1

    for directory_path in sorted(
        (path for path in card_root.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        try:
            directory_path.rmdir()
        except OSError:
            continue

    return deleted_files


def eject_memory_card(card_root: Path) -> None:
    """Eject the mounted memory card with diskutil."""
    diskutil_path = shutil.which("diskutil")
    if diskutil_path is None:
        raise FileNotFoundError("Required tool not found on PATH: diskutil")
    subprocess.run(
        [diskutil_path, "eject", str(card_root)],
        capture_output=True,
        check=True,
        text=True,
    )


def run_memory_card_ingest(
    destination_dir: Path,
    *,
    mount_root: Path = DEFAULT_CARD_MOUNT_ROOT,
    delete_source: bool = True,
    use_sha256: bool = False,
    eject_card: bool = True,
) -> MemoryCardIngestResult | None:
    """Copy card media into destination and optionally delete imported source files."""
    card_root = find_memory_card_mount(mount_root)
    if card_root is None:
        LOGGER.info("no memory card detected in %s", mount_root)
        return None

    source_files = iter_memory_card_media_files(card_root)
    if not source_files:
        LOGGER.info("no importable media files found on memory card %s", card_root)
        return MemoryCardIngestResult(
            card_root=card_root,
            copied_files=(),
            imported_files=0,
            deleted_files=0,
            ejected=False,
        )

    destination_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info("detected memory card at %s", card_root)
    LOGGER.info("found %s importable media file(s)", len(source_files))

    copy_plan = build_copy_plan(source_files, destination_dir)
    copy_files(copy_plan)
    verify_copied_files(copy_plan, use_sha256=use_sha256)

    deleted_files = 0
    if delete_source:
        deleted_files = delete_memory_card_files(source_files, card_root=card_root)
        LOGGER.info("deleted %s imported file(s) from memory card", deleted_files)

    ejected = False
    if eject_card:
        eject_memory_card(card_root)
        LOGGER.info("ejected memory card at %s", card_root)
        ejected = True

    copied_targets = tuple(target_path for _, target_path in copy_plan)
    return MemoryCardIngestResult(
        card_root=card_root,
        copied_files=copied_targets,
        imported_files=len(copied_targets),
        deleted_files=deleted_files,
        ejected=ejected,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for memory-card ingest."""
    parser = argparse.ArgumentParser(
        description=(
            "Copy media from a mounted memory card into trail-camera input_dir and "
            "delete source files only after verification."
        )
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to YAML configuration file (default: {DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--mount-root",
        default=str(DEFAULT_CARD_MOUNT_ROOT),
        help=f"Root directory to scan for mounted cards (default: {DEFAULT_CARD_MOUNT_ROOT})",
    )
    parser.add_argument(
        "--destination-dir",
        default="",
        help="Override ingest destination directory (default: config input_dir)",
    )
    parser.add_argument(
        "--no-delete",
        action="store_true",
        help="Keep source files on the memory card after successful copy verification",
    )
    parser.add_argument(
        "--verify-sha256",
        action="store_true",
        help="Run SHA-256 verification in addition to file-size checks",
    )
    parser.add_argument(
        "--no-eject",
        action="store_true",
        help="Skip ejecting the memory card when ingest completes",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        type=str.upper,
        help="Logging verbosity (default: INFO)",
    )
    return parser.parse_args()


def _resolve_destination_dir(config_path: Path, destination_override: str) -> Path:
    """Resolve destination directory from CLI override or pipeline config."""
    if destination_override.strip():
        return Path(destination_override).expanduser().resolve()

    config = load_config(config_path)
    return Path(config.input_dir).expanduser().resolve()


def _log_file_mix(source_files: tuple[Path, ...]) -> None:
    """Log imported file counts grouped by media type."""
    videos = sum(1 for path in source_files if is_video_file(path))
    images = sum(1 for path in source_files if is_image_file(path))
    LOGGER.info("imported %s media file(s): %s video(s), %s image(s)", len(source_files), videos, images)


def main() -> None:
    """Run memory-card ingest into trail-camera input staging."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )

    config_path = Path(args.config).expanduser().resolve()
    destination_dir = _resolve_destination_dir(config_path, args.destination_dir)
    result = run_memory_card_ingest(
        destination_dir=destination_dir,
        mount_root=Path(args.mount_root).expanduser().resolve(),
        delete_source=not args.no_delete,
        use_sha256=args.verify_sha256,
        eject_card=not args.no_eject,
    )

    if result is None:
        return

    _log_file_mix(result.copied_files)
    LOGGER.info("memory-card ingest completed into %s", destination_dir)


if __name__ == "__main__":
    main()
