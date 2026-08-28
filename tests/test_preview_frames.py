"""Tests for SpeciesNet candidate selection helpers."""

from pathlib import Path

import pytest

pytest.importorskip("cv2")

from preview_frames import TopFrameRecord, process_record_for_preview, select_speciesnet_top_class


def test_select_speciesnet_top_class_should_skip_generic_bird_label() -> None:
    """Promote the next most likely specific candidate when bird is configured as generic."""
    candidates = [
        {
            "raw_class": "id;aves;;;;;bird",
            "label": "bird",
            "score": 0.6051611304283142,
        },
        {
            "raw_class": "id;aves;passeriformes;corvidae;cyanocitta;stelleri;steller's jay",
            "label": "steller's jay",
            "score": 0.32681676745414734,
        },
    ]

    selected = select_speciesnet_top_class(candidates, generic_labels_to_skip=["bird"])

    assert selected is not None
    assert selected.label == "steller's jay"
    assert selected.raw_class.endswith("steller's jay")


def test_select_speciesnet_top_class_should_fallback_when_all_candidates_generic() -> None:
    """Keep the best generic label when no more specific candidate exists."""
    candidates = [
        {
            "raw_class": "id;aves;;;;;bird",
            "label": "bird",
            "score": 0.7,
        },
        {
            "raw_class": "id;animal;;;;;animal",
            "label": "animal",
            "score": 0.2,
        },
    ]

    selected = select_speciesnet_top_class(candidates, generic_labels_to_skip=["bird", "animal"])

    assert selected is not None
    assert selected.label == "bird"


def test_process_record_for_preview_should_copy_image_sources(tmp_path: Path) -> None:
    """Copy image inputs directly into the preview output path."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    source = input_dir / "photo.jpg"
    source.write_bytes(b"image-bytes")

    record = TopFrameRecord(
        relative_path="photo.jpg",
        output_relative_path="20260710-photo.jpg",
        top_frame=0,
        top_confidence=0.9,
        bucket="interesting",
        top_bbox=None,
    )

    status, preview_path, classification_target, message = process_record_for_preview(
        record=record,
        input_dir=input_dir,
        output_dir=output_dir,
        speciesnet_use_crops=False,
        species_crop_output_dir=None,
        species_crop_padding=0.15,
    )

    assert status == "extracted"
    assert preview_path is not None
    assert preview_path.read_bytes() == b"image-bytes"
    assert classification_target == preview_path.resolve()
    assert "Wrote" in message
