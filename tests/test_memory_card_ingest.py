"""Tests for memory-card ingest workflow."""

from pathlib import Path

import pytest

import memory_card_ingest


def test_run_memory_card_ingest_should_return_none_without_card(tmp_path: Path) -> None:
    """Return None when no mounted memory card is detected."""
    mount_root = tmp_path / "Volumes"
    mount_root.mkdir(parents=True)

    result = memory_card_ingest.run_memory_card_ingest(
        destination_dir=tmp_path / "input",
        mount_root=mount_root,
    )

    assert result is None


def test_run_memory_card_ingest_should_copy_and_delete_sources(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Copy media into destination and delete source files after verification."""
    mount_root = tmp_path / "Volumes"
    card_root = mount_root / "SDCARD"
    source_dir = card_root / "DCIM" / "100MEDIA"
    source_dir.mkdir(parents=True)
    source_video = source_dir / "clip.AVI"
    source_image = source_dir / "frame.JPG"
    source_video.write_bytes(b"video-bytes")
    source_image.write_bytes(b"image-bytes")
    eject_calls: list[Path] = []

    def fake_eject(card_root: Path) -> None:
        eject_calls.append(card_root)

    monkeypatch.setattr(memory_card_ingest, "eject_memory_card", fake_eject)

    destination_dir = tmp_path / "input"
    result = memory_card_ingest.run_memory_card_ingest(
        destination_dir=destination_dir,
        mount_root=mount_root,
    )

    assert result is not None
    assert result.imported_files == 2
    assert result.deleted_files == 2
    assert sorted(path.name for path in result.copied_files) == ["clip.AVI", "frame.JPG"]
    assert sorted(path.name for path in destination_dir.iterdir()) == ["clip.AVI", "frame.JPG"]
    assert not source_video.exists()
    assert not source_image.exists()
    assert result.ejected is True
    assert eject_calls == [card_root]


def test_run_memory_card_ingest_should_not_delete_on_verification_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Leave source files on the card when verification fails."""
    mount_root = tmp_path / "Volumes"
    card_root = mount_root / "SDCARD"
    source_dir = card_root / "DCIM" / "100MEDIA"
    source_dir.mkdir(parents=True)
    source_video = source_dir / "clip.AVI"
    source_video.write_bytes(b"video-bytes")
    eject_calls: list[Path] = []

    def fail_verification(*_args, **_kwargs) -> None:
        raise ValueError("bad copy")

    def fake_eject(card_root: Path) -> None:
        eject_calls.append(card_root)

    monkeypatch.setattr(memory_card_ingest, "verify_copied_files", fail_verification)
    monkeypatch.setattr(memory_card_ingest, "eject_memory_card", fake_eject)

    with pytest.raises(ValueError, match="bad copy"):
        memory_card_ingest.run_memory_card_ingest(
            destination_dir=tmp_path / "input",
            mount_root=mount_root,
        )

    assert source_video.exists()
    assert eject_calls == []


def test_run_memory_card_ingest_should_support_no_eject_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip card eject when no-eject mode is requested."""
    mount_root = tmp_path / "Volumes"
    card_root = mount_root / "SDCARD"
    source_dir = card_root / "DCIM" / "100MEDIA"
    source_dir.mkdir(parents=True)
    source_video = source_dir / "clip.AVI"
    source_video.write_bytes(b"video-bytes")
    eject_calls: list[Path] = []

    def fake_eject(card_root: Path) -> None:
        eject_calls.append(card_root)

    monkeypatch.setattr(memory_card_ingest, "eject_memory_card", fake_eject)

    result = memory_card_ingest.run_memory_card_ingest(
        destination_dir=tmp_path / "input",
        mount_root=mount_root,
        eject_card=False,
    )

    assert result is not None
    assert result.ejected is False
    assert eject_calls == []


def test_iter_memory_card_media_files_should_ignore_hidden_paths(tmp_path: Path) -> None:
    """Exclude files located under hidden card directories."""
    card_root = tmp_path / "SDCARD"
    visible_dir = card_root / "DCIM" / "100MEDIA"
    hidden_dir = card_root / ".Trashes" / "100MEDIA"
    visible_dir.mkdir(parents=True)
    hidden_dir.mkdir(parents=True)

    visible_file = visible_dir / "clip.AVI"
    hidden_file = hidden_dir / "clip.AVI"
    visible_file.write_bytes(b"video")
    hidden_file.write_bytes(b"video")

    files = memory_card_ingest.iter_memory_card_media_files(card_root)

    assert files == [visible_file]
